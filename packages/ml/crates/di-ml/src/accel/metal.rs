//! Metal backend for Apple GPUs (M-series, including M4 Max).
//!
//! Uses shared unified-memory buffers and compute shaders. GEMM is the
//! training hot path; elementwise ops share the same command queue.

use std::collections::HashMap;
use std::ffi::c_void;
use std::sync::Mutex;

use metal::{
    Buffer, CommandQueue, CompileOptions, ComputeCommandEncoderRef, ComputePipelineState, Device,
    MTLResourceOptions, MTLSize, NSUInteger,
};

use crate::error::{Error, Result};
use crate::tensor::{broadcast_shape, broadcast_to, Tensor};

use super::cpu::{binop_cpu, from_vec, matmul_cpu, softmax_last_cpu, unary_cpu};
use super::{Accel, AccelKind, BinOp, UnaryOp};

/// GPU launch overhead dominates below these sizes on unified-memory Apple GPUs.
const GEMM_GPU_FLOPS: usize = 8_192;
const ELEM_GPU: usize = 2_048;

#[link(name = "CoreGraphics", kind = "framework")]
unsafe extern "C" {}

const SHADERS: &str = include_str!("shaders.metal");

pub struct MetalAccel {
    device: Device,
    queue: Mutex<CommandQueue>,
    unary: ComputePipelineState,
    binop: ComputePipelineState,
    matmul: ComputePipelineState,
    softmax: ComputePipelineState,
    matmul_batched: ComputePipelineState,
    sdpa: ComputePipelineState,
    pinned: Mutex<HashMap<String, Buffer>>,
    name: String,
}

impl MetalAccel {
    pub fn try_new() -> Option<Self> {
        let device = Device::system_default().or_else(|| Device::all().into_iter().next())?;
        let name = device.name().to_string();
        let queue = device.new_command_queue();
        let library = device
            .new_library_with_source(SHADERS, &CompileOptions::new())
            .ok()?;
        let pipeline = |fn_name: &str| -> Option<ComputePipelineState> {
            let function = library.get_function(fn_name, None).ok()?;
            device
                .new_compute_pipeline_state_with_function(&function)
                .ok()
        };
        Some(Self {
            unary: pipeline("unary_f32")?,
            binop: pipeline("binop_f32")?,
            matmul: pipeline("matmul_f32")?,
            softmax: pipeline("softmax_rows_f32")?,
            matmul_batched: pipeline("matmul_batched_f32")?,
            sdpa: pipeline("sdpa_f32")?,
            pinned: Mutex::new(HashMap::new()),
            device,
            queue: Mutex::new(queue),
            name,
        })
    }

    fn upload(&self, data: &[f32]) -> Buffer {
        let bytes = (data.len().max(1) * std::mem::size_of::<f32>()) as NSUInteger;
        self.device.new_buffer_with_data(
            data.as_ptr() as *const c_void,
            bytes,
            MTLResourceOptions::StorageModeShared,
        )
    }

    fn alloc(&self, len: usize) -> Buffer {
        let bytes = (len.max(1) * std::mem::size_of::<f32>()) as NSUInteger;
        self.device
            .new_buffer(bytes, MTLResourceOptions::StorageModeShared)
    }

    fn download(buf: &Buffer, len: usize) -> Vec<f32> {
        if len == 0 {
            return Vec::new();
        }
        let ptr = buf.contents() as *const f32;
        unsafe { std::slice::from_raw_parts(ptr, len) }.to_vec()
    }

    fn host(t: &Tensor) -> Vec<f32> {
        t.iter().copied().collect()
    }

    fn dispatch_1d(
        &self,
        pipeline: &ComputePipelineState,
        n: usize,
        bind: impl FnOnce(&ComputeCommandEncoderRef),
    ) -> Result<()> {
        if n == 0 {
            return Ok(());
        }
        let queue = self
            .queue
            .lock()
            .unwrap_or_else(|err| err.into_inner());
        let cmd = queue.new_command_buffer();
        let enc = cmd.new_compute_command_encoder();
        enc.set_compute_pipeline_state(pipeline);
        bind(enc);
        let tpt = 256u64;
        let groups = (n as u64).div_ceil(tpt).max(1);
        enc.dispatch_thread_groups(MTLSize::new(groups, 1, 1), MTLSize::new(tpt, 1, 1));
        enc.end_encoding();
        cmd.commit();
        cmd.wait_until_completed();
        Ok(())
    }
}

impl Accel for MetalAccel {
    fn kind(&self) -> AccelKind {
        AccelKind::Metal
    }

    fn label(&self) -> String {
        format!("metal ({})", self.name)
    }

    fn matmul(&self, a: &Tensor, b: &Tensor) -> Result<Tensor> {
        if a.ndim() != 2 || b.ndim() != 2 {
            return matmul_cpu(a, b);
        }
        let m = a.shape()[0];
        let k = a.shape()[1];
        let k2 = b.shape()[0];
        let n = b.shape()[1];
        if k != k2 {
            return Err(Error::fail(format!(
                "MatMul inner dimensions do not match: {:?} @ {:?}",
                a.shape(),
                b.shape()
            )));
        }
        if m == 0 || n == 0 || k == 0 {
            return from_vec(&[m, n], vec![0.0; m * n]);
        }
        if m.saturating_mul(n).saturating_mul(k) < GEMM_GPU_FLOPS {
            return matmul_cpu(a, b);
        }

        let a_host = Self::host(a);
        let b_host = Self::host(b);
        let a_buf = self.upload(&a_host);
        let b_buf = self.upload(&b_host);
        let c_buf = self.alloc(m * n);
        let m_u = m as u32;
        let k_u = k as u32;
        let n_u = n as u32;

        let queue = self
            .queue
            .lock()
            .unwrap_or_else(|err| err.into_inner());
        let cmd = queue.new_command_buffer();
        let enc = cmd.new_compute_command_encoder();
        enc.set_compute_pipeline_state(&self.matmul);
        enc.set_buffer(0, Some(&a_buf), 0);
        enc.set_buffer(1, Some(&b_buf), 0);
        enc.set_buffer(2, Some(&c_buf), 0);
        enc.set_bytes(3, 4, &m_u as *const u32 as *const c_void);
        enc.set_bytes(4, 4, &k_u as *const u32 as *const c_void);
        enc.set_bytes(5, 4, &n_u as *const u32 as *const c_void);
        let tg = 16u64;
        enc.dispatch_thread_groups(
            MTLSize::new((n as u64).div_ceil(tg).max(1), (m as u64).div_ceil(tg).max(1), 1),
            MTLSize::new(tg, tg, 1),
        );
        enc.end_encoding();
        cmd.commit();
        cmd.wait_until_completed();
        drop(queue);

        from_vec(&[m, n], Self::download(&c_buf, m * n))
    }

    fn binop(&self, a: &Tensor, b: &Tensor, op: BinOp) -> Result<Tensor> {
        let shape = broadcast_shape(a.shape(), b.shape())?;
        let a = broadcast_to(a, &shape)?;
        let b = broadcast_to(b, &shape)?;
        let n = a.len();
        if n == 0 {
            return from_vec(&shape, Vec::new());
        }
        if n < ELEM_GPU {
            return binop_cpu(&a, &b, op);
        }

        let a_buf = self.upload(&Self::host(&a));
        let b_buf = self.upload(&Self::host(&b));
        let y_buf = self.alloc(n);
        let op_u = op.code();
        let n_u = n as u32;
        let pipeline = self.binop.clone();
        self.dispatch_1d(&pipeline, n, |enc| {
            enc.set_buffer(0, Some(&a_buf), 0);
            enc.set_buffer(1, Some(&b_buf), 0);
            enc.set_buffer(2, Some(&y_buf), 0);
            enc.set_bytes(3, 4, &op_u as *const u32 as *const c_void);
            enc.set_bytes(4, 4, &n_u as *const u32 as *const c_void);
        })?;
        from_vec(&shape, Self::download(&y_buf, n))
    }

    fn unary(&self, x: &Tensor, op: UnaryOp) -> Result<Tensor> {
        let n = x.len();
        if n == 0 {
            return Ok(x.clone());
        }
        if n < ELEM_GPU || !op.metal_ready() {
            return Ok(unary_cpu(x, op));
        }
        let x_buf = self.upload(&Self::host(x));
        let y_buf = self.alloc(n);
        let (op_u, scale) = op.code();
        let n_u = n as u32;
        let pipeline = self.unary.clone();
        self.dispatch_1d(&pipeline, n, |enc| {
            enc.set_buffer(0, Some(&x_buf), 0);
            enc.set_buffer(1, Some(&y_buf), 0);
            enc.set_bytes(2, 4, &op_u as *const u32 as *const c_void);
            enc.set_bytes(3, 4, &scale as *const f32 as *const c_void);
            enc.set_bytes(4, 4, &n_u as *const u32 as *const c_void);
        })?;
        from_vec(x.shape(), Self::download(&y_buf, n))
    }

    fn softmax_last(&self, x: &Tensor) -> Result<Tensor> {
        if x.ndim() != 2 {
            return Ok(softmax_last_cpu(x));
        }
        let rows = x.shape()[0];
        let cols = x.shape()[1];
        if rows == 0 || cols == 0 || rows * cols < ELEM_GPU {
            return Ok(softmax_last_cpu(x));
        }
        let x_buf = self.upload(&Self::host(x));
        let y_buf = self.alloc(rows * cols);
        let c_u = cols as u32;
        let pipeline = self.softmax.clone();
        self.dispatch_1d(&pipeline, rows, |enc| {
            enc.set_buffer(0, Some(&x_buf), 0);
            enc.set_buffer(1, Some(&y_buf), 0);
            enc.set_bytes(2, 4, &c_u as *const u32 as *const c_void);
        })?;
        from_vec(x.shape(), Self::download(&y_buf, rows * cols))
    }

    fn batched_matmul(&self, a: &Tensor, b: &Tensor) -> Result<Tensor> {
        if a.ndim() == 2 && b.ndim() == 2 {
            return self.matmul(a, b);
        }
        if a.ndim() < 2 || b.ndim() < 2 {
            return crate::kernels::matmul_nd_cpu(a, b);
        }
        let nda = a.ndim();
        let ndb = b.ndim();
        let m = a.shape()[nda - 2];
        let k = a.shape()[nda - 1];
        let k2 = b.shape()[ndb - 2];
        let n = b.shape()[ndb - 1];
        if k != k2 {
            return Err(Error::fail(format!(
                "MatMul inner dimensions do not match: {:?} @ {:?}",
                a.shape(),
                b.shape()
            )));
        }
        let a_batch = &a.shape()[..nda - 2];
        let b_batch = &b.shape()[..ndb - 2];
        let batch_shape = crate::tensor::broadcast_shape(a_batch, b_batch)?;
        let batch = batch_shape.iter().product::<usize>().max(1);
        if batch * m * n * k < GEMM_GPU_FLOPS {
            return crate::kernels::matmul_nd_cpu(a, b);
        }
        let mut a_shape = batch_shape.clone();
        a_shape.extend_from_slice(&[m, k]);
        let mut b_shape = batch_shape.clone();
        b_shape.extend_from_slice(&[k, n]);
        let a = crate::tensor::broadcast_to(a, &a_shape)?;
        let b = crate::tensor::broadcast_to(b, &b_shape)?;
        let a_buf = self.upload(&Self::host(&a));
        let b_buf = self.upload(&Self::host(&b));
        let c_buf = self.alloc(batch * m * n);
        let m_u = m as u32;
        let k_u = k as u32;
        let n_u = n as u32;
        let b_u = batch as u32;
        let queue = self.queue.lock().unwrap_or_else(|err| err.into_inner());
        let cmd = queue.new_command_buffer();
        let enc = cmd.new_compute_command_encoder();
        enc.set_compute_pipeline_state(&self.matmul_batched);
        enc.set_buffer(0, Some(&a_buf), 0);
        enc.set_buffer(1, Some(&b_buf), 0);
        enc.set_buffer(2, Some(&c_buf), 0);
        enc.set_bytes(3, 4, &m_u as *const u32 as *const c_void);
        enc.set_bytes(4, 4, &k_u as *const u32 as *const c_void);
        enc.set_bytes(5, 4, &n_u as *const u32 as *const c_void);
        enc.set_bytes(6, 4, &b_u as *const u32 as *const c_void);
        let tg = 8u64;
        enc.dispatch_thread_groups(
            MTLSize::new(
                (n as u64).div_ceil(tg).max(1),
                (m as u64).div_ceil(tg).max(1),
                batch as u64,
            ),
            MTLSize::new(tg, tg, 1),
        );
        enc.end_encoding();
        cmd.commit();
        cmd.wait_until_completed();
        drop(queue);
        let mut out_shape = batch_shape;
        out_shape.extend_from_slice(&[m, n]);
        from_vec(&out_shape, Self::download(&c_buf, batch * m * n))
    }

    fn sdpa(
        &self,
        q: &Tensor,
        k: &Tensor,
        v: &Tensor,
        mask: Option<&Tensor>,
        scale: f32,
    ) -> Result<Tensor> {
        let q = q.as_standard_layout().to_owned();
        let k = k.as_standard_layout().to_owned();
        let v = v.as_standard_layout().to_owned();
        if q.ndim() < 2 {
            return crate::kernels::sdpa_fwd(&q, &k, &v, mask, scale);
        }
        let (b, h, tq, d, tk) = sdpa_dims(&q, &k)?;
        if tq * tk * d < GEMM_GPU_FLOPS {
            return crate::kernels::sdpa_fwd(&q, &k, &v, mask, scale);
        }
        let q4 = reshape_sdpa(&q, b, h, tq, d)?;
        let k4 = reshape_sdpa(&k, b, h, tk, d)?;
        let v4 = reshape_sdpa(&v, b, h, tk, d)?;
        let q_buf = self.upload(&Self::host(&q4));
        let k_buf = self.upload(&Self::host(&k4));
        let v_buf = self.upload(&Self::host(&v4));
        let mask_host = if let Some(m) = mask {
            let m = crate::kernels::prepare_mask(m)?;
            let want = [b, h, tq, tk];
            crate::tensor::broadcast_to(&m, &want)?
        } else {
            Tensor::zeros(ndarray::IxDyn(&[1]))
        };
        let m_buf = self.upload(&Self::host(&mask_host));
        let o_buf = self.alloc(b * h * tq * d);
        let b_u = b as u32;
        let h_u = h as u32;
        let tq_u = tq as u32;
        let tk_u = tk as u32;
        let d_u = d as u32;
        let has = u32::from(mask.is_some());
        let queue = self.queue.lock().unwrap_or_else(|err| err.into_inner());
        let cmd = queue.new_command_buffer();
        let enc = cmd.new_compute_command_encoder();
        enc.set_compute_pipeline_state(&self.sdpa);
        enc.set_buffer(0, Some(&q_buf), 0);
        enc.set_buffer(1, Some(&k_buf), 0);
        enc.set_buffer(2, Some(&v_buf), 0);
        enc.set_buffer(3, Some(&m_buf), 0);
        enc.set_buffer(4, Some(&o_buf), 0);
        enc.set_bytes(5, 4, &b_u as *const u32 as *const c_void);
        enc.set_bytes(6, 4, &h_u as *const u32 as *const c_void);
        enc.set_bytes(7, 4, &tq_u as *const u32 as *const c_void);
        enc.set_bytes(8, 4, &tk_u as *const u32 as *const c_void);
        enc.set_bytes(9, 4, &d_u as *const u32 as *const c_void);
        enc.set_bytes(10, 4, &scale as *const f32 as *const c_void);
        enc.set_bytes(11, 4, &has as *const u32 as *const c_void);
        let tpt = 32u64;
        enc.dispatch_thread_groups(
            MTLSize::new(
                (tq as u64).div_ceil(tpt).max(1),
                h as u64,
                b as u64,
            ),
            MTLSize::new(tpt, 1, 1),
        );
        enc.end_encoding();
        cmd.commit();
        cmd.wait_until_completed();
        drop(queue);
        from_vec(q.shape(), Self::download(&o_buf, b * h * tq * d))
    }

    fn pin_readonly(&self, name: &str, t: &Tensor) {
        let buf = self.upload(&Self::host(t));
        self.pinned
            .lock()
            .unwrap_or_else(|err| err.into_inner())
            .insert(name.to_string(), buf);
    }
}

fn sdpa_dims(q: &Tensor, k: &Tensor) -> Result<(usize, usize, usize, usize, usize)> {
    match q.ndim() {
        2 => {
            let tq = q.shape()[0];
            let d = q.shape()[1];
            let tk = k.shape()[0];
            Ok((1, 1, tq, d, tk))
        }
        3 => {
            let b = q.shape()[0];
            let tq = q.shape()[1];
            let d = q.shape()[2];
            let tk = k.shape()[k.ndim() - 2];
            Ok((b, 1, tq, d, tk))
        }
        4 => {
            let b = q.shape()[0];
            let h = q.shape()[1];
            let tq = q.shape()[2];
            let d = q.shape()[3];
            let tk = k.shape()[2];
            Ok((b, h, tq, d, tk))
        }
        n => Err(Error::fail(format!("SDPA rank {n} is not supported"))),
    }
}

fn reshape_sdpa(t: &Tensor, b: usize, h: usize, seq: usize, d: usize) -> Result<Tensor> {
    t.clone()
        .into_shape_with_order(ndarray::IxDyn(&[b, h, seq, d]))
        .map_err(|err| Error::fail(err.to_string()))
}
