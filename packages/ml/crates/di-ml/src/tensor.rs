use ndarray::{ArrayD, Axis, IxDyn};

use crate::accel::{current, BinOp, UnaryOp};
use crate::error::{Error, Result};
use crate::kernels;

pub type Tensor = ArrayD<f32>;

pub fn broadcast_shape(a: &[usize], b: &[usize]) -> Result<Vec<usize>> {
    let n = a.len().max(b.len());
    let mut out = vec![0; n];
    for i in 0..n {
        let da = dim_from_right(a, n, i);
        let db = dim_from_right(b, n, i);
        if da == db || da == 1 || db == 1 {
            out[i] = da.max(db);
        } else {
            return Err(Error::fail(format!(
                "cannot broadcast shapes {a:?} and {b:?}"
            )));
        }
    }
    Ok(out)
}

fn dim_from_right(shape: &[usize], out_rank: usize, i: usize) -> usize {
    let pad = out_rank - shape.len();
    if i < pad {
        1
    } else {
        shape[i - pad]
    }
}

pub fn broadcast_to(t: &Tensor, shape: &[usize]) -> Result<Tensor> {
    t.broadcast(IxDyn(shape))
        .map(|view| view.to_owned())
        .ok_or_else(|| {
            Error::fail(format!(
                "cannot broadcast {:?} to {shape:?}",
                t.shape()
            ))
        })
}

/// Reshape/broadcast `target` so it matches `shape` (e.g. labels `[N]` vs pred `[N, 1]`).
pub fn align_to(target: &Tensor, shape: &[usize]) -> Result<Tensor> {
    if target.shape() == shape {
        return Ok(target.clone());
    }
    if let Ok(t) = broadcast_to(target, shape) {
        return Ok(t);
    }
    let mut dims = target.shape().to_vec();
    while dims.len() < shape.len() {
        dims.push(1);
        let Ok(reshaped) = target.clone().into_shape_with_order(IxDyn(&dims)) else {
            break;
        };
        if reshaped.shape() == shape {
            return Ok(reshaped);
        }
        if let Ok(t) = broadcast_to(&reshaped, shape) {
            return Ok(t);
        }
    }
    Err(Error::fail(format!(
        "cannot align label shape {:?} to prediction {shape:?}",
        target.shape()
    )))
}

pub fn binop(a: &Tensor, b: &Tensor, op: BinOp) -> Result<Tensor> {
    current().binop(a, b, op)
}

pub fn add(a: &Tensor, b: &Tensor) -> Result<Tensor> {
    binop(a, b, BinOp::Add)
}

pub fn mul(a: &Tensor, b: &Tensor) -> Result<Tensor> {
    binop(a, b, BinOp::Mul)
}

pub fn div(a: &Tensor, b: &Tensor) -> Result<Tensor> {
    binop(a, b, BinOp::Div)
}

pub fn relu(x: &Tensor) -> Result<Tensor> {
    current().unary(x, UnaryOp::Relu)
}

pub fn sigmoid(x: &Tensor) -> Result<Tensor> {
    current().unary(x, UnaryOp::Sigmoid)
}

pub fn tanh(x: &Tensor) -> Result<Tensor> {
    current().unary(x, UnaryOp::Tanh)
}

pub fn neg(x: &Tensor) -> Result<Tensor> {
    current().unary(x, UnaryOp::Neg)
}

pub fn scale(x: &Tensor, s: f32) -> Result<Tensor> {
    if s == 1.0 {
        return Ok(x.clone());
    }
    current().unary(x, UnaryOp::Scale(s))
}

/// Reduce `grad` so it matches `target_shape` after a broadcasted forward op.
pub fn unbroadcast(grad: &Tensor, target_shape: &[usize]) -> Tensor {
    let mut g = grad.clone();
    while g.ndim() > target_shape.len() {
        g = g.sum_axis(Axis(0));
    }
    for i in 0..g.ndim() {
        let target = target_shape[i];
        if target == 1 && g.shape()[i] > 1 {
            g = g.sum_axis(Axis(i)).insert_axis(Axis(i));
        }
    }
    if g.shape() != target_shape {
        g = g
            .clone()
            .into_shape_with_order(IxDyn(target_shape))
            .unwrap_or(g);
    }
    g
}

pub fn matmul(a: &Tensor, b: &Tensor) -> Result<Tensor> {
    let a = crate::accel::maybe_quantize(a);
    let b = crate::accel::maybe_quantize(b);
    if a.ndim() == 2 && b.ndim() == 2 {
        current().matmul(&a, &b)
    } else {
        current().batched_matmul(&a, &b)
    }
}

pub fn transpose_2d(t: &Tensor) -> Result<Tensor> {
    if t.ndim() != 2 {
        return Err(Error::fail(format!(
            "expected rank-2 tensor to transpose, got {:?}",
            t.shape()
        )));
    }
    Ok(t.t().as_standard_layout().to_owned().into_dyn())
}

pub fn softmax_last(x: &Tensor) -> Tensor {
    current().softmax_last(x).expect("softmax")
}

pub fn softmax_axis(x: &Tensor, axis: usize) -> Result<Tensor> {
    kernels::softmax_axis(x, axis)
}

pub fn silu(x: &Tensor) -> Result<Tensor> {
    current().unary(x, UnaryOp::Silu)
}

pub fn gelu(x: &Tensor) -> Result<Tensor> {
    current().unary(x, UnaryOp::Gelu)
}

pub fn sqrt(x: &Tensor) -> Result<Tensor> {
    current().unary(x, UnaryOp::Sqrt)
}

pub fn exp(x: &Tensor) -> Result<Tensor> {
    current().unary(x, UnaryOp::Exp)
}

pub fn log(x: &Tensor) -> Result<Tensor> {
    current().unary(x, UnaryOp::Log)
}

pub fn abs(x: &Tensor) -> Result<Tensor> {
    current().unary(x, UnaryOp::Abs)
}

pub fn sdpa(
    q: &Tensor,
    k: &Tensor,
    v: &Tensor,
    mask: Option<&Tensor>,
    scale: f32,
) -> Result<Tensor> {
    current().sdpa(q, k, v, mask, scale)
}

pub fn add_assign(dst: &mut Tensor, src: &Tensor) {
    if dst.shape() == src.shape() {
        *dst += src;
        return;
    }
    let reduced = unbroadcast(src, dst.shape());
    *dst += &reduced;
}

#[cfg(test)]
mod tests {
    use super::*;
    use ndarray::array;

    #[test]
    fn unbroadcast_sums_leading_and_unit_dims() {
        let grad = array![[1.0, 2.0], [3.0, 4.0]].into_dyn();
        let out = unbroadcast(&grad, &[1, 2]);
        assert_eq!(out.shape(), &[1, 2]);
        assert_eq!(out.iter().copied().collect::<Vec<_>>(), vec![4.0, 6.0]);
    }

    #[test]
    fn matmul_2d() {
        let a = array![[1.0, 2.0], [3.0, 4.0]].into_dyn();
        let b = array![[5.0], [6.0]].into_dyn();
        let y = matmul(&a, &b).unwrap();
        assert_eq!(y.iter().copied().collect::<Vec<_>>(), vec![17.0, 39.0]);
    }

    #[test]
    fn align_appends_unit_dim() {
        let label = array![1.0, 0.0].into_dyn();
        let out = align_to(&label, &[2, 1]).unwrap();
        assert_eq!(out.shape(), &[2, 1]);
    }

    #[test]
    fn batched_matmul_3d() {
        let a = ndarray::Array3::<f32>::from_shape_fn((2, 3, 4), |(b, i, k)| (b + i + k) as f32)
            .into_dyn();
        let b = ndarray::Array3::<f32>::from_shape_fn((2, 4, 5), |(b, k, j)| (b + k + j) as f32 * 0.1)
            .into_dyn();
        let y = matmul(&a, &b).unwrap();
        assert_eq!(y.shape(), &[2, 3, 5]);
        let y0 = crate::kernels::matmul_nd_cpu(&a, &b).unwrap();
        for (g, e) in y.iter().zip(y0.iter()) {
            assert!((g - e).abs() < 1e-4);
        }
    }

    #[test]
    fn sdpa_shape() {
        let q = ndarray::Array4::<f32>::from_elem((1, 2, 4, 8), 0.1).into_dyn();
        let k = q.clone();
        let v = q.clone();
        let y = sdpa(&q, &k, &v, None, 0.35).unwrap();
        assert_eq!(y.shape(), &[1, 2, 4, 8]);
    }
}
