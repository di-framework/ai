//! Backend-agnostic tensor kernels used by the interpreter and Accel defaults.
//! Graph code calls [`crate::tensor`] / these helpers — never Metal APIs.

use ndarray::{Array, ArrayD, Axis, Dimension, Ix2, IxDyn, Zip};

use crate::error::{Error, Result};
use crate::tensor::{broadcast_shape, broadcast_to, unbroadcast, Tensor};

pub fn softmax_axis(x: &Tensor, axis: usize) -> Result<Tensor> {
    if axis >= x.ndim() {
        return Err(Error::fail(format!(
            "softmax axis {axis} out of range for {:?}",
            x.shape()
        )));
    }
    if axis == x.ndim() - 1 {
        return crate::accel::current().softmax_last(x);
    }
    let last = x.ndim() - 1;
    let mut perm: Vec<usize> = (0..x.ndim()).collect();
    perm.swap(axis, last);
    let t = x.clone().permuted_axes(IxDyn(&perm)).as_standard_layout().to_owned();
    let y = crate::accel::current().softmax_last(&t)?;
    let mut inv = vec![0; perm.len()];
    for (i, p) in perm.iter().enumerate() {
        inv[*p] = i;
    }
    Ok(y.permuted_axes(IxDyn(&inv)).as_standard_layout().to_owned())
}

pub fn softmax_vjp(p: &Tensor, dp: &Tensor) -> Result<Tensor> {
    let axis = Axis(p.ndim().saturating_sub(1));
    let dot = (p * dp).sum_axis(axis);
    let dot_b = broadcast_to(&dot.insert_axis(axis), p.shape())?;
    Ok(p * &(dp - &dot_b))
}

pub fn transpose_last2(t: &Tensor) -> Result<Tensor> {
    let nd = t.ndim();
    if nd < 2 {
        return Err(Error::fail(format!(
            "transpose_last2 expects rank >= 2, got {:?}",
            t.shape()
        )));
    }
    let mut perm: Vec<usize> = (0..nd).collect();
    perm.swap(nd - 2, nd - 1);
    Ok(t.clone().permuted_axes(IxDyn(&perm)).as_standard_layout().to_owned())
}

pub fn matmul_nd_cpu(a: &Tensor, b: &Tensor) -> Result<Tensor> {
    let nda = a.ndim();
    let ndb = b.ndim();
    if nda < 2 || ndb < 2 {
        return Err(Error::fail(format!(
            "MatMul expects rank >= 2, got {:?} and {:?}",
            a.shape(),
            b.shape()
        )));
    }
    let m = a.shape()[nda - 2];
    let ka = a.shape()[nda - 1];
    let kb = b.shape()[ndb - 2];
    let n = b.shape()[ndb - 1];
    if ka != kb {
        return Err(Error::fail(format!(
            "MatMul inner dimensions do not match: {:?} @ {:?}",
            a.shape(),
            b.shape()
        )));
    }
    let a_batch = &a.shape()[..nda - 2];
    let b_batch = &b.shape()[..ndb - 2];
    let batch_shape = broadcast_shape(a_batch, b_batch)?;
    let mut a_shape = batch_shape.clone();
    a_shape.push(m);
    a_shape.push(ka);
    let mut b_shape = batch_shape.clone();
    b_shape.push(kb);
    b_shape.push(n);
    let a_b = broadcast_to(a, &a_shape)?;
    let b_b = broadcast_to(b, &b_shape)?;
    let batch = batch_shape.iter().product::<usize>().max(1);
    let a3 = a_b
        .to_shape((batch, m, ka))
        .map_err(|err| Error::fail(err.to_string()))?;
    let b3 = b_b
        .to_shape((batch, ka, n))
        .map_err(|err| Error::fail(err.to_string()))?;
    let mut c = Array::zeros((batch, m, n));
    for i in 0..batch {
        let av = a3
            .index_axis(Axis(0), i)
            .into_dimensionality::<Ix2>()
            .map_err(|err| Error::fail(err.to_string()))?;
        let bv = b3
            .index_axis(Axis(0), i)
            .into_dimensionality::<Ix2>()
            .map_err(|err| Error::fail(err.to_string()))?;
        c.index_axis_mut(Axis(0), i).assign(&av.dot(&bv));
    }
    let mut out_shape = batch_shape;
    out_shape.push(m);
    out_shape.push(n);
    c.into_shape_with_order(IxDyn(&out_shape))
        .map_err(|err| Error::fail(err.to_string()))
}

pub fn silu(x: &Tensor) -> Tensor {
    x.mapv(|v| v * sigmoid_f32(v))
}

pub fn silu_vjp(x: &Tensor, dy: &Tensor) -> Tensor {
    Zip::from(x)
        .and(dy)
        .map_collect(|x, g| {
            let s = sigmoid_f32(*x);
            *g * (s + *x * s * (1.0 - s))
        })
}

pub fn gelu(x: &Tensor) -> Tensor {
    // tanh approximation used by most ONNX Gelu(approximate="tanh") exporters
    x.mapv(gelu_tanh)
}

pub fn gelu_vjp(x: &Tensor, dy: &Tensor) -> Tensor {
    Zip::from(x)
        .and(dy)
        .map_collect(|x, g| *g * gelu_tanh_grad(*x))
}

fn gelu_tanh(x: f32) -> f32 {
    const K: f32 = 0.7978845608; // sqrt(2/pi)
    let u = K * (x + 0.044715 * x * x * x);
    0.5 * x * (1.0 + u.tanh())
}

fn gelu_tanh_grad(x: f32) -> f32 {
    const K: f32 = 0.7978845608;
    let u = K * (x + 0.044715 * x * x * x);
    let t = u.tanh();
    let sech2 = 1.0 - t * t;
    let du = K * (1.0 + 3.0 * 0.044715 * x * x);
    0.5 * (1.0 + t) + 0.5 * x * sech2 * du
}

pub fn erf_elem(x: &Tensor) -> Tensor {
    x.mapv(|v| erf_f32(v))
}

pub fn erf_vjp(x: &Tensor, dy: &Tensor) -> Tensor {
    // d/dx erf(x) = 2/sqrt(pi) * exp(-x^2)
    const C: f32 = 1.1283791671;
    Zip::from(x)
        .and(dy)
        .map_collect(|x, g| *g * C * (-x * x).exp())
}

fn erf_f32(x: f32) -> f32 {
    // Abramowitz & Stegun 7.1.26
    let sign = if x < 0.0 { -1.0 } else { 1.0 };
    let x = x.abs();
    let t = 1.0 / (1.0 + 0.3275911 * x);
    let y = 1.0
        - (((((1.061405429 * t + -1.453152027) * t) + 1.421413741) * t + -0.284496736) * t
            + 0.254829592)
            * t
            * (-x * x).exp();
    sign * y
}

fn sigmoid_f32(x: f32) -> f32 {
    1.0 / (1.0 + (-x).exp())
}

#[derive(Debug, Clone, Copy)]
pub enum ReduceKind {
    Sum,
    Mean,
    Max,
}

pub fn reduce(x: &Tensor, axes: &[usize], keepdims: bool, kind: ReduceKind) -> Result<Tensor> {
    let mut axes: Vec<usize> = axes.to_vec();
    axes.sort_unstable();
    axes.dedup();
    let mut y = x.clone();
    for (i, axis) in axes.iter().enumerate() {
        let ax = Axis(*axis - i);
        y = match kind {
            ReduceKind::Sum => y.sum_axis(ax),
            ReduceKind::Mean => y.mean_axis(ax).ok_or_else(|| Error::fail("reduce mean on empty"))?,
            ReduceKind::Max => y.fold_axis(ax, f32::NEG_INFINITY, |a, b| a.max(*b)),
        };
        if keepdims {
            y = y.insert_axis(ax);
        }
    }
    Ok(y)
}

pub fn reduce_axes_or_all(rank: usize, axes: Option<&[i64]>) -> Result<Vec<usize>> {
    match axes {
        None | Some([]) => Ok((0..rank).collect()),
        Some(ax) => ax
            .iter()
            .map(|a| normalize_axis(*a, rank))
            .collect(),
    }
}

pub fn normalize_axis(axis: i64, rank: usize) -> Result<usize> {
    let a = if axis < 0 { axis + rank as i64 } else { axis };
    if a < 0 || a as usize >= rank {
        Err(Error::fail(format!("axis {axis} out of range for rank {rank}")))
    } else {
        Ok(a as usize)
    }
}

pub fn normalize_axis_inclusive(axis: i64, rank: usize) -> Result<usize> {
    let a = if axis < 0 { axis + rank as i64 } else { axis };
    if a < 0 || a as usize > rank {
        Err(Error::fail(format!("axis {axis} out of range for rank {rank}")))
    } else {
        Ok(a as usize)
    }
}

pub fn gather(data: &Tensor, indices: &ArrayD<i64>, axis: usize) -> Result<Tensor> {
    if axis >= data.ndim() {
        return Err(Error::fail(format!(
            "Gather axis {axis} out of range for {:?}",
            data.shape()
        )));
    }
    let dim = data.shape()[axis];
    let mut out_shape: Vec<usize> = Vec::new();
    out_shape.extend_from_slice(&data.shape()[..axis]);
    out_shape.extend_from_slice(indices.shape());
    out_shape.extend_from_slice(&data.shape()[axis + 1..]);
    let pre = data.shape()[..axis].iter().product::<usize>().max(1);
    let post = data.shape()[axis + 1..].iter().product::<usize>().max(1);
    let idx_len = indices.len();
    let data_std = data.as_standard_layout();
    let src = data_std.as_slice().ok_or_else(|| Error::fail("gather: data is not contiguous"))?;
    let mut out = vec![0.0f32; pre * idx_len * post];
    let idx: Vec<usize> = indices
        .iter()
        .map(|i| wrap_index(*i, dim))
        .collect::<Result<Vec<_>>>()?;
    for p in 0..pre {
        for (j, &ix) in idx.iter().enumerate() {
            let dst = (p * idx_len + j) * post;
            let s = (p * dim + ix) * post;
            out[dst..dst + post].copy_from_slice(&src[s..s + post]);
        }
    }
    Array::from_shape_vec(IxDyn(&out_shape), out).map_err(|err| Error::fail(err.to_string()))
}

pub fn gather_vjp(data_shape: &[usize], indices: &ArrayD<i64>, dy: &Tensor, axis: usize) -> Result<Tensor> {
    let dim = data_shape[axis];
    let pre = data_shape[..axis].iter().product::<usize>().max(1);
    let post = data_shape[axis + 1..].iter().product::<usize>().max(1);
    let idx_len = indices.len();
    let dy_std = dy.as_standard_layout();
    let g = dy_std.as_slice().ok_or_else(|| Error::fail("gather vjp: grad is not contiguous"))?;
    let mut out = vec![0.0f32; pre * dim * post];
    let idx: Vec<usize> = indices
        .iter()
        .map(|i| wrap_index(*i, dim))
        .collect::<Result<Vec<_>>>()?;
    for p in 0..pre {
        for (j, &ix) in idx.iter().enumerate() {
            let src = (p * idx_len + j) * post;
            let dst = (p * dim + ix) * post;
            for k in 0..post {
                out[dst + k] += g[src + k];
            }
        }
    }
    Array::from_shape_vec(IxDyn(data_shape), out).map_err(|err| Error::fail(err.to_string()))
}

pub fn gather_elements(data: &Tensor, indices: &ArrayD<i64>, axis: usize) -> Result<Tensor> {
    if data.ndim() != indices.ndim() {
        return Err(Error::fail("GatherElements: data and indices must have the same rank"));
    }
    if axis >= data.ndim() {
        return Err(Error::fail("GatherElements axis out of range"));
    }
    let mut out = Tensor::zeros(indices.raw_dim());
    for (idx, &sel) in indices.indexed_iter() {
        let mut coord: Vec<usize> = idx.as_array_view().iter().copied().collect();
        let dim = data.shape()[axis];
        coord[axis] = wrap_index(sel, dim)?;
        out[idx] = data[IxDyn(&coord)];
    }
    Ok(out)
}

pub fn gather_elements_vjp(
    data_shape: &[usize],
    indices: &ArrayD<i64>,
    dy: &Tensor,
    axis: usize,
) -> Result<Tensor> {
    let mut out = Tensor::zeros(IxDyn(data_shape));
    for (idx, &sel) in indices.indexed_iter() {
        let mut coord: Vec<usize> = idx.as_array_view().iter().copied().collect();
        let dim = data_shape[axis];
        coord[axis] = wrap_index(sel, dim)?;
        out[IxDyn(&coord)] += dy[idx];
    }
    Ok(out)
}

fn wrap_index(i: i64, dim: usize) -> Result<usize> {
    let dim_i = dim as i64;
    let v = if i < 0 { i + dim_i } else { i };
    if v < 0 || v >= dim_i {
        Err(Error::fail(format!("index {i} out of range for dim {dim}")))
    } else {
        Ok(v as usize)
    }
}

pub fn unsqueeze(x: &Tensor, axes: &[i64]) -> Result<Tensor> {
    let out_rank = x.ndim() + axes.len();
    let mut axes_n: Vec<usize> = axes
        .iter()
        .map(|a| normalize_axis_inclusive(*a, out_rank))
        .collect::<Result<Vec<_>>>()?;
    axes_n.sort_unstable();
    let mut shape = x.shape().to_vec();
    for ax in axes_n {
        if ax > shape.len() {
            return Err(Error::fail("Unsqueeze axis out of range"));
        }
        shape.insert(ax, 1);
    }
    x.clone()
        .into_shape_with_order(IxDyn(&shape))
        .map_err(|err| Error::fail(err.to_string()))
}

pub fn squeeze(x: &Tensor, axes: Option<&[i64]>) -> Result<Tensor> {
    let mut shape = x.shape().to_vec();
    match axes {
        None | Some([]) => {
            shape.retain(|&d| d != 1);
        }
        Some(ax) => {
            let mut drop: Vec<usize> = ax
                .iter()
                .map(|a| normalize_axis(*a, x.ndim()))
                .collect::<Result<Vec<_>>>()?;
            drop.sort_unstable();
            drop.dedup();
            for (i, d) in drop.iter().enumerate() {
                let idx = *d - i;
                if shape.get(idx) != Some(&1) {
                    return Err(Error::fail(format!(
                        "Squeeze axis {d} is not 1 (shape {:?})",
                        x.shape()
                    )));
                }
                shape.remove(idx);
            }
        }
    }
    x.clone()
        .into_shape_with_order(IxDyn(&shape))
        .map_err(|err| Error::fail(err.to_string()))
}

pub fn concat(xs: &[&Tensor], axis: usize) -> Result<Tensor> {
    if xs.is_empty() {
        return Err(Error::fail("Concat with no inputs"));
    }
    let views: Vec<_> = xs.iter().map(|t| t.view()).collect();
    ndarray::concatenate(Axis(axis), &views).map_err(|err| Error::fail(err.to_string()))
}

pub fn split(x: &Tensor, axis: usize, sizes: &[usize]) -> Result<Vec<Tensor>> {
    let dim = x.shape()[axis];
    let sum: usize = sizes.iter().sum();
    if sum != dim {
        return Err(Error::fail(format!(
            "Split sizes {sizes:?} do not sum to axis dim {dim}"
        )));
    }
    let mut out = Vec::new();
    let mut start = 0;
    for &sz in sizes {
        let sl = x.slice_axis(Axis(axis), ndarray::Slice::from(start..start + sz));
        out.push(sl.to_owned());
        start += sz;
    }
    Ok(out)
}

pub fn where_op(cond: &ArrayD<bool>, x: &Tensor, y: &Tensor) -> Result<Tensor> {
    let shape = broadcast_shape(
        cond.shape(),
        &broadcast_shape(x.shape(), y.shape())?,
    )?;
    let c = broadcast_bool(cond, &shape)?;
    let x = broadcast_to(x, &shape)?;
    let y = broadcast_to(y, &shape)?;
    Ok(Zip::from(&c).and(&x).and(&y).map_collect(|c, x, y| if *c { *x } else { *y }))
}

pub fn broadcast_bool(t: &ArrayD<bool>, shape: &[usize]) -> Result<ArrayD<bool>> {
    t.broadcast(IxDyn(shape))
        .map(|v| v.to_owned())
        .ok_or_else(|| Error::fail(format!("cannot broadcast bool {:?} to {shape:?}", t.shape())))
}

pub fn rms_norm(x: &Tensor, scale: &Tensor, epsilon: f32, axis: usize) -> Result<Tensor> {
    let (ms, rms) = rms_stats(x, axis, epsilon)?;
    let _ = ms;
    let inv = rms.mapv(|v| 1.0 / v);
    let inv_b = broadcast_to(&inv, x.shape())?;
    let y = x * &inv_b;
    let scale_b = broadcast_to(scale, x.shape())?;
    Ok(y * &scale_b)
}

fn rms_stats(x: &Tensor, axis: usize, epsilon: f32) -> Result<(Tensor, Tensor)> {
    let axes: Vec<usize> = (axis..x.ndim()).collect();
    let sq = x.mapv(|v| v * v);
    let ms = reduce(&sq, &axes, true, ReduceKind::Mean)?;
    let rms = ms.mapv(|v| (v + epsilon).sqrt());
    Ok((ms, rms))
}

pub fn rms_norm_vjp(
    x: &Tensor,
    scale: &Tensor,
    dy: &Tensor,
    epsilon: f32,
    axis: usize,
) -> Result<(Tensor, Tensor)> {
    let n = x.shape()[axis..].iter().product::<usize>().max(1) as f32;
    let (_ms, rms) = rms_stats(x, axis, epsilon)?;
    let inv = rms.mapv(|v| 1.0 / v);
    let inv_b = broadcast_to(&inv, x.shape())?;
    let xhat = x * &inv_b;
    let scale_b = broadcast_to(scale, x.shape())?;
    let d_scale = unbroadcast(&(dy * &xhat), scale.shape());
    let dy_xhat = dy * &scale_b;
    // y = x * inv; inv = (mean(x^2)+eps)^(-1/2)
    // d_inv from sum(dy_xhat * x) over reduced axes
    let d_inv_full = dy_xhat.clone() * x;
    let d_inv = reduce(&d_inv_full, &(axis..x.ndim()).collect::<Vec<_>>(), true, ReduceKind::Sum)?;
    let d_rms = &d_inv * &inv.mapv(|v| -v * v); // d(1/rms)/d(rms) = -1/rms^2, chain through stored inv
    // rms = (ms+eps)^0.5 ; d_ms = d_rms * 0.5 / rms
    let d_ms = &d_rms * &(inv.mapv(|v| 0.5 * v));
    let d_sq = broadcast_to(&d_ms, x.shape())?.mapv(|v| v / n);
    let dx = &dy_xhat * &inv_b + x * &d_sq * 2.0;
    Ok((dx, d_scale))
}

pub fn layer_norm(
    x: &Tensor,
    scale: Option<&Tensor>,
    bias: Option<&Tensor>,
    epsilon: f32,
    axis: usize,
) -> Result<Tensor> {
    let (mean, inv_std, xhat) = layer_norm_stats(x, epsilon, axis)?;
    let _ = (mean, inv_std);
    let mut y = xhat;
    if let Some(s) = scale {
        y = y * &broadcast_to(s, x.shape())?;
    }
    if let Some(b) = bias {
        y = y + &broadcast_to(b, x.shape())?;
    }
    Ok(y)
}

fn layer_norm_stats(x: &Tensor, epsilon: f32, axis: usize) -> Result<(Tensor, Tensor, Tensor)> {
    let axes: Vec<usize> = (axis..x.ndim()).collect();
    let mean = reduce(x, &axes, true, ReduceKind::Mean)?;
    let mean_b = broadcast_to(&mean, x.shape())?;
    let xc = x - &mean_b;
    let var = reduce(&xc.mapv(|v| v * v), &axes, true, ReduceKind::Mean)?;
    let inv_std = var.mapv(|v| 1.0 / (v + epsilon).sqrt());
    let inv_b = broadcast_to(&inv_std, x.shape())?;
    let xhat = &xc * &inv_b;
    Ok((mean, inv_std, xhat))
}

pub fn layer_norm_vjp_full(
    x: &Tensor,
    scale: Option<&Tensor>,
    bias: Option<&Tensor>,
    dy: &Tensor,
    epsilon: f32,
    axis: usize,
) -> Result<(Tensor, Option<Tensor>, Option<Tensor>)> {
    let n = x.shape()[axis..].iter().product::<usize>().max(1) as f32;
    let (_mean, inv_std, xhat) = layer_norm_stats(x, epsilon, axis)?;
    let inv_b = broadcast_to(&inv_std, x.shape())?;
    let d_scale = scale.map(|s| unbroadcast(&(dy * &xhat), s.shape()));
    let d_bias = bias.map(|b| unbroadcast(dy, b.shape()));
    let dy_hat = if let Some(s) = scale {
        dy * &broadcast_to(s, x.shape())?
    } else {
        dy.clone()
    };
    let sum_dy = reduce(&dy_hat, &(axis..x.ndim()).collect::<Vec<_>>(), true, ReduceKind::Sum)?;
    let sum_dy_xhat = reduce(
        &(&dy_hat * &xhat),
        &(axis..x.ndim()).collect::<Vec<_>>(),
        true,
        ReduceKind::Sum,
    )?;
    let sum_dy_b = broadcast_to(&sum_dy, x.shape())?;
    let sum_dy_xhat_b = broadcast_to(&sum_dy_xhat, x.shape())?;
    let dx = &inv_b * &(&dy_hat - &sum_dy_b / n - &xhat * &sum_dy_xhat_b / n);
    Ok((dx, d_scale, d_bias))
}

/// Q,K,V as [B, H, T, D] (or broadcastable). Mask additive [..., Tq, Tk] or 0/1.
pub fn sdpa_fwd(
    q: &Tensor,
    k: &Tensor,
    v: &Tensor,
    mask: Option<&Tensor>,
    scale: f32,
) -> Result<Tensor> {
    let scores = matmul_nd_cpu(q, &transpose_last2(k)?)?;
    let mut scores = scores.mapv(|s| s * scale);
    if let Some(m) = mask {
        let m = prepare_mask(m)?;
        let m = broadcast_to(&m, scores.shape())?;
        scores = scores + m;
    }
    let p = softmax_axis(&scores, scores.ndim() - 1)?;
    matmul_nd_cpu(&p, v)
}

pub fn sdpa_bwd(
    q: &Tensor,
    k: &Tensor,
    v: &Tensor,
    mask: Option<&Tensor>,
    dy: &Tensor,
    scale: f32,
) -> Result<(Tensor, Tensor, Tensor)> {
    let kt = transpose_last2(k)?;
    let mut scores = matmul_nd_cpu(q, &kt)?.mapv(|s| s * scale);
    if let Some(m) = mask {
        let m = prepare_mask(m)?;
        let m_b = broadcast_to(&m, scores.shape())?;
        scores = scores + m_b;
    }
    let p = softmax_axis(&scores, scores.ndim() - 1)?;
    let dv = matmul_nd_cpu(&transpose_last2(&p)?, dy)?;
    let dp = matmul_nd_cpu(dy, &transpose_last2(v)?)?;
    let ds = softmax_vjp(&p, &dp)?;
    let dq = matmul_nd_cpu(&ds, k)?.mapv(|g| g * scale);
    let dk = matmul_nd_cpu(&transpose_last2(&ds)?, q)?.mapv(|g| g * scale);
    Ok((dq, dk, dv))
}

pub fn prepare_mask(mask: &Tensor) -> Result<Tensor> {
    let mut min = f32::INFINITY;
    let mut max = f32::NEG_INFINITY;
    for v in mask.iter() {
        min = min.min(*v);
        max = max.max(*v);
    }
    if min >= 0.0 && max <= 1.0 + 1e-4 {
        Ok(mask.mapv(|v| if v > 0.5 { 0.0 } else { -1.0e9 }))
    } else {
        Ok(mask.clone())
    }
}

/// Rotate pairs on the last dim. `cos`/`sin` broadcast onto `x`.
pub fn rope(x: &Tensor, cos: &Tensor, sin: &Tensor) -> Result<Tensor> {
    let d = *x.shape().last().ok_or_else(|| Error::fail("RoPE on scalar"))?;
    if d % 2 != 0 {
        return Err(Error::fail("RoPE last dim must be even"));
    }
    let cos = broadcast_to(cos, x.shape())?;
    let sin = broadcast_to(sin, x.shape())?;
    let mut y = x.clone();
    let n = x.len();
    let xs = x.as_standard_layout();
    let cs = cos.as_standard_layout();
    let sn = sin.as_standard_layout();
    let xsl = xs.as_slice().unwrap();
    let csl = cs.as_slice().unwrap();
    let ssl = sn.as_slice().unwrap();
    let ys = y.as_slice_mut().unwrap();
    let mut i = 0;
    while i < n {
        let x0 = xsl[i];
        let x1 = xsl[i + 1];
        let c = csl[i];
        let s = ssl[i];
        ys[i] = x0 * c - x1 * s;
        ys[i + 1] = x0 * s + x1 * c;
        i += 2;
    }
    Ok(y)
}

pub fn rope_vjp(x: &Tensor, cos: &Tensor, sin: &Tensor, dy: &Tensor) -> Result<Tensor> {
    // inverse rotate: [c, s; -s, c]^T = [c, -s; s, c] wait
    // y0 = x0 c - x1 s; y1 = x0 s + x1 c
    // dx0 = dy0 c + dy1 s; dx1 = -dy0 s + dy1 c
    let cos = broadcast_to(cos, x.shape())?;
    let sin = broadcast_to(sin, x.shape())?;
    let mut dx = x.clone();
    let n = x.len();
    let dstd = dy.as_standard_layout();
    let cs = cos.as_standard_layout();
    let sn = sin.as_standard_layout();
    let dsl = dstd.as_slice().unwrap();
    let csl = cs.as_slice().unwrap();
    let ssl = sn.as_slice().unwrap();
    let out = dx.as_slice_mut().unwrap();
    let mut i = 0;
    while i < n {
        let dy0 = dsl[i];
        let dy1 = dsl[i + 1];
        let c = csl[i];
        let s = ssl[i];
        out[i] = dy0 * c + dy1 * s;
        out[i + 1] = -dy0 * s + dy1 * c;
        i += 2;
    }
    Ok(dx)
}

pub fn l2_normalize(x: &Tensor, eps: f32) -> Result<(Tensor, Tensor)> {
    let axis = Axis(x.ndim() - 1);
    let sq = x.mapv(|v| v * v).sum_axis(axis);
    let n = sq.mapv(|v| (v + eps).sqrt());
    let n_b = broadcast_to(&n.clone().insert_axis(axis), x.shape())?;
    let y = x / &n_b;
    Ok((y, n.insert_axis(axis)))
}

pub fn l2_normalize_vjp(y: &Tensor, n: &Tensor, dy: &Tensor) -> Result<Tensor> {
    // y = x / n; dx = (dy - y * sum(y*dy)) / n
    let axis = Axis(y.ndim() - 1);
    let dot = (y * dy).sum_axis(axis);
    let dot_b = broadcast_to(&dot.insert_axis(axis), y.shape())?;
    let n_b = broadcast_to(n, y.shape())?;
    Ok((dy - &(y * &dot_b)) / &n_b)
}

pub fn last_token_gather(x: &Tensor, mask: Option<&Tensor>) -> Result<(Tensor, Vec<usize>)> {
    if x.ndim() < 2 {
        return Err(Error::fail(format!(
            "last-token pool expects [B, T, ...] got {:?}",
            x.shape()
        )));
    }
    let b = x.shape()[0];
    let t = x.shape()[1];
    let mut idx = vec![t.saturating_sub(1); b];
    if let Some(m) = mask {
        let m = m.as_standard_layout();
        for i in 0..b {
            let mut last = 0;
            for j in 0..t {
                let v = if m.ndim() == 2 {
                    m[[i, j]]
                } else if m.ndim() == 1 && m.len() == t {
                    m[[j]]
                } else {
                    let flat = m.as_slice().unwrap_or(&[]);
                    *flat.get(i * t + j).unwrap_or(&1.0)
                };
                if v > 0.5 {
                    last = j;
                }
            }
            idx[i] = last;
        }
    }
    let rest: usize = x.shape()[2..].iter().product::<usize>().max(1);
    let mut out_shape = vec![b];
    out_shape.extend_from_slice(&x.shape()[2..]);
    let std = x.as_standard_layout();
    let src = std.as_slice().unwrap();
    let mut out = vec![0.0f32; b * rest];
    for i in 0..b {
        let s = (i * t + idx[i]) * rest;
        let d = i * rest;
        out[d..d + rest].copy_from_slice(&src[s..s + rest]);
    }
    let y = Array::from_shape_vec(IxDyn(&out_shape), out).map_err(|err| Error::fail(err.to_string()))?;
    Ok((y, idx))
}

pub fn last_token_vjp(x_shape: &[usize], idx: &[usize], dy: &Tensor) -> Result<Tensor> {
    let b = x_shape[0];
    let t = x_shape[1];
    let rest: usize = x_shape[2..].iter().product::<usize>().max(1);
    let mut out = vec![0.0f32; b * t * rest];
    let g = dy.as_standard_layout();
    let gs = g.as_slice().unwrap();
    for i in 0..b {
        let d = (i * t + idx[i]) * rest;
        let s = i * rest;
        out[d..d + rest].copy_from_slice(&gs[s..s + rest]);
    }
    Array::from_shape_vec(IxDyn(x_shape), out).map_err(|err| Error::fail(err.to_string()))
}

pub fn mean_pool(x: &Tensor, mask: Option<&Tensor>) -> Result<Tensor> {
    if x.ndim() < 2 {
        return Err(Error::fail("mean pool expects [B, T, ...]"));
    }
    let b = x.shape()[0];
    let t = x.shape()[1];
    let rest: usize = x.shape()[2..].iter().product::<usize>().max(1);
    let std = x.as_standard_layout();
    let src = std.as_slice().unwrap();
    let mut out = vec![0.0f32; b * rest];
    for i in 0..b {
        let mut denom = 0.0f32;
        for j in 0..t {
            let w = match mask {
                Some(m) => {
                    let m = m.as_standard_layout();
                    if m.ndim() == 2 {
                        m[[i, j]]
                    } else {
                        1.0
                    }
                }
                None => 1.0,
            };
            if w == 0.0 {
                continue;
            }
            denom += w;
            let s = (i * t + j) * rest;
            for k in 0..rest {
                out[i * rest + k] += src[s + k] * w;
            }
        }
        let d = denom.max(1.0);
        for k in 0..rest {
            out[i * rest + k] /= d;
        }
    }
    let mut shape = vec![b];
    shape.extend_from_slice(&x.shape()[2..]);
    Array::from_shape_vec(IxDyn(&shape), out).map_err(|err| Error::fail(err.to_string()))
}

pub fn quantize_f16(x: &Tensor) -> Tensor {
    x.mapv(|v| half::f16::from_f32(v).to_f32())
}

pub fn quantize_bf16(x: &Tensor) -> Tensor {
    x.mapv(|v| half::bf16::from_f32(v).to_f32())
}

#[cfg(test)]
mod tests {
    use super::*;
    use ndarray::array;

    #[test]
    fn gather_rows() {
        let data = array![[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]].into_dyn();
        let idx = array![2, 0].into_dyn();
        let y = gather(&data, &idx, 0).unwrap();
        assert_eq!(y.shape(), &[2, 2]);
        assert_eq!(y.iter().copied().collect::<Vec<_>>(), vec![5.0, 6.0, 1.0, 2.0]);
    }

    #[test]
    fn rms_norm_unit_scale() {
        let x = array![[3.0, 4.0]].into_dyn();
        let g = array![1.0, 1.0].into_dyn();
        let y = rms_norm(&x, &g, 0.0, 1).unwrap();
        // rms =  sqrt((9+16)/2) = sqrt(12.5)
        let rms = (12.5f32).sqrt();
        assert!((y[[0, 0]] - 3.0 / rms).abs() < 1e-5);
    }

    #[test]
    fn squeeze_removes_unit_axes() {
        let row = Array::from_shape_vec(IxDyn(&[1, 2, 1]), vec![3.0, 4.0]).unwrap();
        let y = squeeze(&row, None).unwrap();
        assert_eq!(y.shape(), &[2]);
        assert_eq!(y.iter().copied().collect::<Vec<_>>(), vec![3.0, 4.0]);

        let scalar = Array::from_shape_vec(IxDyn(&[1, 1]), vec![5.0]).unwrap();
        let y = squeeze(&scalar, None).unwrap();
        assert!(y.shape().is_empty());
        assert_eq!(y.iter().copied().collect::<Vec<_>>(), vec![5.0]);
    }

    #[test]
    fn silu_zero() {
        let x = array![0.0, 2.0].into_dyn();
        let y = silu(&x);
        assert!((y[0] - 0.0).abs() < 1e-6);
        let s = 1.0 / (1.0 + (-2.0f32).exp());
        assert!((y[1] - 2.0 * s).abs() < 1e-5);
    }
}
