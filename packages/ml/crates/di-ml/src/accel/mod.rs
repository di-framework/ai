//! Platform-selected compute backend.
//!
//! Graph and training code call [`crate::tensor`] helpers; those dispatch here.
//! New GPU targets (CUDA, ROCm, DirectML) plug in as another `Accel` impl
//! behind an OS/`cfg` gate in [`select`].

use std::env;
use std::sync::atomic::{AtomicU8, AtomicU64, AtomicUsize, Ordering};
use std::sync::OnceLock;

use crate::error::Result;
use crate::tensor::Tensor;
use crate::workspace::MixedPrecision;

mod cpu;
#[cfg(target_os = "macos")]
mod metal;

use cpu::CpuAccel;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum AccelKind {
    Cpu,
    Metal,
}

impl AccelKind {
    pub fn as_str(self) -> &'static str {
        match self {
            Self::Cpu => "cpu",
            Self::Metal => "metal",
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq)]
pub enum BinOp {
    Add,
    Sub,
    Mul,
    Div,
}

impl BinOp {
    pub fn as_fn(self) -> fn(f32, f32) -> f32 {
        match self {
            Self::Add => |x, y| x + y,
            Self::Sub => |x, y| x - y,
            Self::Mul => |x, y| x * y,
            Self::Div => |x, y| x / y,
        }
    }

    pub fn code(self) -> u32 {
        match self {
            Self::Add => 0,
            Self::Sub => 1,
            Self::Mul => 2,
            Self::Div => 3,
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq)]
pub enum UnaryOp {
    Relu,
    Sigmoid,
    Tanh,
    Neg,
    Scale(f32),
    Silu,
    Gelu,
    Sqrt,
    Exp,
    Log,
    Abs,
    Reciprocal,
    Erf,
    Sin,
    Cos,
}

impl UnaryOp {
    pub fn code(self) -> (u32, f32) {
        match self {
            Self::Relu => (0, 1.0),
            Self::Sigmoid => (1, 1.0),
            Self::Tanh => (2, 1.0),
            Self::Neg => (3, 1.0),
            Self::Scale(s) => (4, s),
            Self::Silu => (5, 1.0),
            Self::Gelu => (6, 1.0),
            Self::Sqrt => (7, 1.0),
            Self::Exp => (8, 1.0),
            Self::Log => (9, 1.0),
            Self::Abs => (10, 1.0),
            Self::Reciprocal => (11, 1.0),
            Self::Erf => (12, 1.0),
            Self::Sin => (13, 1.0),
            Self::Cos => (14, 1.0),
        }
    }

    pub fn metal_ready(self) -> bool {
        matches!(
            self,
            Self::Relu | Self::Sigmoid | Self::Tanh | Self::Neg | Self::Scale(_)
        )
    }
}

pub trait Accel: Send + Sync {
    fn kind(&self) -> AccelKind;
    fn label(&self) -> String;
    fn matmul(&self, a: &Tensor, b: &Tensor) -> Result<Tensor>;
    fn binop(&self, a: &Tensor, b: &Tensor, op: BinOp) -> Result<Tensor>;
    fn unary(&self, x: &Tensor, op: UnaryOp) -> Result<Tensor>;
    fn softmax_last(&self, x: &Tensor) -> Result<Tensor>;

    fn batched_matmul(&self, a: &Tensor, b: &Tensor) -> Result<Tensor> {
        crate::kernels::matmul_nd_cpu(a, b)
    }

    fn sdpa(
        &self,
        q: &Tensor,
        k: &Tensor,
        v: &Tensor,
        mask: Option<&Tensor>,
        scale: f32,
    ) -> Result<Tensor> {
        crate::kernels::sdpa_fwd(q, k, v, mask, scale)
    }

    fn pin_readonly(&self, _name: &str, _t: &Tensor) {}

    fn note_bytes(&self, n: usize) {
        note_bytes(n);
    }
}

static ACCEL: OnceLock<Box<dyn Accel>> = OnceLock::new();
static COMPUTE: AtomicU8 = AtomicU8::new(0);
static PEAK_BYTES: AtomicUsize = AtomicUsize::new(0);
static LAST_STEP_NS: AtomicU64 = AtomicU64::new(0);

pub fn set_mixed_precision(mp: MixedPrecision) {
    let code = match mp {
        MixedPrecision::Off => 0,
        MixedPrecision::Fp16 => 1,
        MixedPrecision::Bf16 => 2,
    };
    COMPUTE.store(code, Ordering::Relaxed);
}

pub fn mixed_precision() -> MixedPrecision {
    match COMPUTE.load(Ordering::Relaxed) {
        1 => MixedPrecision::Fp16,
        2 => MixedPrecision::Bf16,
        _ => MixedPrecision::Off,
    }
}

pub fn maybe_quantize(t: &Tensor) -> Tensor {
    match mixed_precision() {
        MixedPrecision::Off => t.clone(),
        MixedPrecision::Fp16 => crate::kernels::quantize_f16(t),
        MixedPrecision::Bf16 => crate::kernels::quantize_bf16(t),
    }
}

pub fn note_bytes(n: usize) {
    PEAK_BYTES.fetch_max(n, Ordering::Relaxed);
}

pub fn take_peak_bytes() -> usize {
    PEAK_BYTES.load(Ordering::Relaxed)
}

pub fn record_step_ns(ns: u64) {
    LAST_STEP_NS.store(ns, Ordering::Relaxed);
}

pub fn last_step_ms() -> f32 {
    LAST_STEP_NS.load(Ordering::Relaxed) as f32 / 1_000_000.0
}

/// Process-wide backend, chosen once on first use.
pub fn current() -> &'static dyn Accel {
    ACCEL.get_or_init(select).as_ref()
}

pub fn current_kind() -> AccelKind {
    current().kind()
}

pub fn current_label() -> String {
    current().label()
}

fn requested() -> Option<AccelKind> {
    match env::var("DI_ML_ACCEL") {
        Ok(value) => match value.to_ascii_lowercase().as_str() {
            "cpu" => Some(AccelKind::Cpu),
            "metal" => Some(AccelKind::Metal),
            "auto" | "" => None,
            other => {
                eprintln!("warning: unknown DI_ML_ACCEL={other:?}; using auto");
                None
            }
        },
        Err(_) => None,
    }
}

fn select() -> Box<dyn Accel> {
    match requested() {
        Some(AccelKind::Cpu) => Box::new(CpuAccel),
        Some(AccelKind::Metal) => try_metal()
            .unwrap_or_else(|| panic!("DI_ML_ACCEL=metal but no Metal GPU is available on this host")),
        None => auto(),
    }
}

fn auto() -> Box<dyn Accel> {
    if let Some(gpu) = try_metal() {
        return gpu;
    }
    // Future gates:
    // #[cfg(all(target_os = "linux", feature = "cuda"))]
    // #[cfg(all(target_os = "windows", feature = "cuda"))]
    Box::new(CpuAccel)
}

#[cfg(target_os = "macos")]
fn try_metal() -> Option<Box<dyn Accel>> {
    metal::MetalAccel::try_new().map(|gpu| Box::new(gpu) as Box<dyn Accel>)
}

#[cfg(not(target_os = "macos"))]
fn try_metal() -> Option<Box<dyn Accel>> {
    None
}

#[cfg(test)]
mod tests {
    use super::*;
    use ndarray::array;

    #[test]
    fn cpu_matmul_agrees_with_known_values() {
        let cpu = CpuAccel;
        let a = array![[1.0, 2.0], [3.0, 4.0]].into_dyn();
        let b = array![[5.0], [6.0]].into_dyn();
        let y = cpu.matmul(&a, &b).unwrap();
        assert_eq!(y.iter().copied().collect::<Vec<_>>(), vec![17.0, 39.0]);
    }

    #[cfg(target_os = "macos")]
    #[test]
    fn metal_init_on_macos() {
        let gpu = metal::MetalAccel::try_new();
        assert!(
            gpu.is_some(),
            "expected a Metal device on macOS (M-series / Apple GPU)"
        );
        let gpu = gpu.unwrap();
        assert_eq!(gpu.kind(), AccelKind::Metal);
        assert!(gpu.label().starts_with("metal"));
    }

    #[cfg(target_os = "macos")]
    #[test]
    fn metal_matmul_matches_cpu() {
        let Some(gpu) = metal::MetalAccel::try_new() else {
            return;
        };
        let cpu = CpuAccel;
        let a = ndarray::Array2::<f32>::from_shape_fn((64, 48), |(i, j)| (i + j) as f32 * 0.01)
            .into_dyn();
        let b = ndarray::Array2::<f32>::from_shape_fn((48, 32), |(i, j)| (i as i32 - j as i32) as f32 * 0.02)
            .into_dyn();
        let got = gpu.matmul(&a, &b).unwrap();
        let expect = cpu.matmul(&a, &b).unwrap();
        for (g, e) in got.iter().zip(expect.iter()) {
            assert!((g - e).abs() < 1e-4, "{g} vs {e}");
        }
    }

    #[cfg(target_os = "macos")]
    #[test]
    fn auto_selects_metal_on_macos() {
        if requested() == Some(AccelKind::Cpu) {
            return;
        }
        assert_eq!(current_kind(), AccelKind::Metal);
    }
}
