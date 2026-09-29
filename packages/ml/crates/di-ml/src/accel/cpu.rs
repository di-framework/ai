use ndarray::{Array, Axis, IxDyn, Zip};

use crate::error::{Error, Result};
use crate::tensor::{broadcast_shape, broadcast_to, Tensor};

use super::{Accel, AccelKind, BinOp, UnaryOp};

pub struct CpuAccel;

impl Accel for CpuAccel {
    fn kind(&self) -> AccelKind {
        AccelKind::Cpu
    }

    fn label(&self) -> String {
        "cpu".into()
    }

    fn matmul(&self, a: &Tensor, b: &Tensor) -> Result<Tensor> {
        matmul_cpu(a, b)
    }

    fn binop(&self, a: &Tensor, b: &Tensor, op: BinOp) -> Result<Tensor> {
        binop_cpu(a, b, op)
    }

    fn unary(&self, x: &Tensor, op: UnaryOp) -> Result<Tensor> {
        Ok(unary_cpu(x, op))
    }

    fn softmax_last(&self, x: &Tensor) -> Result<Tensor> {
        Ok(softmax_last_cpu(x))
    }

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
}

pub(crate) fn matmul_cpu(a: &Tensor, b: &Tensor) -> Result<Tensor> {
    if a.ndim() != 2 || b.ndim() != 2 {
        return Err(Error::fail(format!(
            "MatMul/Gemm expects rank-2 tensors, got {:?} and {:?}",
            a.shape(),
            b.shape()
        )));
    }
    let a2 = a
        .view()
        .into_dimensionality::<ndarray::Ix2>()
        .map_err(|err| Error::fail(err.to_string()))?;
    let b2 = b
        .view()
        .into_dimensionality::<ndarray::Ix2>()
        .map_err(|err| Error::fail(err.to_string()))?;
    if a2.ncols() != b2.nrows() {
        return Err(Error::fail(format!(
            "MatMul inner dimensions do not match: {:?} @ {:?}",
            a.shape(),
            b.shape()
        )));
    }
    Ok(a2.dot(&b2).into_dyn())
}

pub(crate) fn binop_cpu(a: &Tensor, b: &Tensor, op: BinOp) -> Result<Tensor> {
    let shape = broadcast_shape(a.shape(), b.shape())?;
    let a = broadcast_to(a, &shape)?;
    let b = broadcast_to(b, &shape)?;
    let f = op.as_fn();
    Ok(Zip::from(&a).and(&b).map_collect(|x, y| f(*x, *y)))
}

pub(crate) fn unary_cpu(x: &Tensor, op: UnaryOp) -> Tensor {
    match op {
        UnaryOp::Relu => x.mapv(|v| v.max(0.0)),
        UnaryOp::Sigmoid => x.mapv(sigmoid),
        UnaryOp::Tanh => x.mapv(|v| v.tanh()),
        UnaryOp::Neg => x.mapv(|v| -v),
        UnaryOp::Scale(s) => x.mapv(|v| v * s),
        UnaryOp::Silu => crate::kernels::silu(x),
        UnaryOp::Gelu => crate::kernels::gelu(x),
        UnaryOp::Sqrt => x.mapv(|v| v.max(0.0).sqrt()),
        UnaryOp::Exp => x.mapv(f32::exp),
        UnaryOp::Log => x.mapv(|v| v.max(1e-12).ln()),
        UnaryOp::Abs => x.mapv(|v| v.abs()),
        UnaryOp::Reciprocal => x.mapv(|v| 1.0 / v),
        UnaryOp::Erf => crate::kernels::erf_elem(x),
        UnaryOp::Sin => x.mapv(|v| v.sin()),
        UnaryOp::Cos => x.mapv(|v| v.cos()),
    }
}

pub(crate) fn softmax_last_cpu(x: &Tensor) -> Tensor {
    let axis = Axis(x.ndim() - 1);
    let max = x.map_axis(axis, |lane| {
        lane.iter().copied().fold(f32::NEG_INFINITY, f32::max)
    });
    let max_b = broadcast_to(&max.insert_axis(axis), x.shape()).expect("broadcast");
    let shifted = x - &max_b;
    let exp = shifted.mapv(f32::exp);
    let sum = exp.sum_axis(axis);
    let sum_b = broadcast_to(&sum.insert_axis(axis), x.shape()).expect("broadcast");
    exp / sum_b
}

pub(crate) fn from_vec(shape: &[usize], data: Vec<f32>) -> Result<Tensor> {
    Array::from_shape_vec(IxDyn(shape), data).map_err(|err| Error::fail(err.to_string()))
}

fn sigmoid(x: f32) -> f32 {
    1.0 / (1.0 + (-x).exp())
}
