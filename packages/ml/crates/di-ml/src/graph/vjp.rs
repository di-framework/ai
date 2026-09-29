use ndarray::{Axis, IxDyn, Zip};

use crate::error::{Error, Result};
use crate::kernels::{
    self, concat, gather_elements_vjp, gather_vjp, layer_norm_vjp_full, rms_norm_vjp, rope_vjp,
    sdpa_bwd, softmax_vjp, split, transpose_last2, ReduceKind,
};
use crate::onnx::TensorValue;
use crate::tensor::{
    add, broadcast_to, div, matmul, mul, neg, scale, transpose_2d, unbroadcast, Tensor,
};

use super::{ExecGraph, Tape, UnaryKind};

impl ExecGraph {
    pub(super) fn backward_entry(&mut self, entry: &Tape) -> Result<()> {
        match entry {
            Tape::Gemm {
                a,
                b,
                c,
                y,
                trans_a,
                trans_b,
                alpha,
                beta,
            } => {
                let dy = match self.grads.get(y) {
                    Some(g) => g.clone(),
                    None => return Ok(()),
                };
                let mut a_t = self.f32(a)?.clone();
                let mut b_t = self.f32(b)?.clone();
                if *trans_a {
                    a_t = transpose_2d(&a_t)?;
                }
                if *trans_b {
                    b_t = transpose_2d(&b_t)?;
                }
                let da_op = scale(&matmul(&dy, &transpose_2d(&b_t)?)?, *alpha)?;
                let db_op = scale(&matmul(&transpose_2d(&a_t)?, &dy)?, *alpha)?;
                let da = if *trans_a {
                    transpose_2d(&da_op)?
                } else {
                    da_op
                };
                let db = if *trans_b {
                    transpose_2d(&db_op)?
                } else {
                    db_op
                };
                self.add_grad(a, da);
                self.add_grad(b, db);
                if let Some(c) = c {
                    let c_shape = self.f32(c)?.shape().to_vec();
                    let dc = scale(&unbroadcast(&dy, &c_shape), *beta)?;
                    self.add_grad(c, dc);
                }
            }
            Tape::MatMul { a, b, y } => {
                let dy = match self.grads.get(y) {
                    Some(g) => g.clone(),
                    None => return Ok(()),
                };
                let a_t = self.f32(a)?.clone();
                let b_t = self.f32(b)?.clone();
                let da = matmul(&dy, &transpose_last2(&b_t)?)?;
                let db = matmul(&transpose_last2(&a_t)?, &dy)?;
                self.add_grad(a, unbroadcast(&da, a_t.shape()));
                self.add_grad(b, unbroadcast(&db, b_t.shape()));
            }
            Tape::Add { a, b, y } => {
                let dy = match self.grads.get(y) {
                    Some(g) => g.clone(),
                    None => return Ok(()),
                };
                let da = unbroadcast(&dy, self.f32(a)?.shape());
                let db = unbroadcast(&dy, self.f32(b)?.shape());
                self.add_grad(a, da);
                self.add_grad(b, db);
            }
            Tape::Sub { a, b, y } => {
                let dy = match self.grads.get(y) {
                    Some(g) => g.clone(),
                    None => return Ok(()),
                };
                let da = unbroadcast(&dy, self.f32(a)?.shape());
                let db = unbroadcast(&neg(&dy)?, self.f32(b)?.shape());
                self.add_grad(a, da);
                self.add_grad(b, db);
            }
            Tape::Mul { a, b, y } => {
                let dy = match self.grads.get(y) {
                    Some(g) => g.clone(),
                    None => return Ok(()),
                };
                let a_t = self.f32(a)?.clone();
                let b_t = self.f32(b)?.clone();
                let da = unbroadcast(&mul(&dy, &b_t)?, a_t.shape());
                let db = unbroadcast(&mul(&dy, &a_t)?, b_t.shape());
                self.add_grad(a, da);
                self.add_grad(b, db);
            }
            Tape::Div { a, b, y } => {
                let dy = match self.grads.get(y) {
                    Some(g) => g.clone(),
                    None => return Ok(()),
                };
                let a_t = self.f32(a)?.clone();
                let b_t = self.f32(b)?.clone();
                let da = unbroadcast(&div(&dy, &b_t)?, a_t.shape());
                let db_full = neg(&div(&mul(&dy, &a_t)?, &mul(&b_t, &b_t)?)?)?;
                let db = unbroadcast(&db_full, b_t.shape());
                self.add_grad(a, da);
                self.add_grad(b, db);
            }
            Tape::Relu { x, y } => {
                let dy = match self.grads.get(y) {
                    Some(g) => g.clone(),
                    None => return Ok(()),
                };
                let x_t = self.f32(x)?;
                let dx = Zip::from(x_t).and(&dy).map_collect(|x, g| {
                    if *x > 0.0 {
                        *g
                    } else {
                        0.0
                    }
                });
                self.add_grad(x, dx);
            }
            Tape::Sigmoid { x, y } => {
                let dy = match self.grads.get(y) {
                    Some(g) => g.clone(),
                    None => return Ok(()),
                };
                let y_t = self.f32(y)?;
                let dx = Zip::from(y_t)
                    .and(&dy)
                    .map_collect(|y, g| *g * *y * (1.0 - *y));
                self.add_grad(x, dx);
            }
            Tape::Tanh { x, y } => {
                let dy = match self.grads.get(y) {
                    Some(g) => g.clone(),
                    None => return Ok(()),
                };
                let y_t = self.f32(y)?;
                let dx = Zip::from(y_t)
                    .and(&dy)
                    .map_collect(|y, g| *g * (1.0 - *y * *y));
                self.add_grad(x, dx);
            }
            Tape::Softmax { x, y, axis } => {
                let dy = match self.grads.get(y) {
                    Some(g) => g.clone(),
                    None => return Ok(()),
                };
                let s = self.f32(y)?;
                let dx = if *axis == s.ndim() - 1 {
                    softmax_vjp(s, &dy)?
                } else {
                    let last = s.ndim() - 1;
                    let mut perm: Vec<usize> = (0..s.ndim()).collect();
                    perm.swap(*axis, last);
                    let sp = s.clone().permuted_axes(IxDyn(&perm)).as_standard_layout().to_owned();
                    let dp = dy.clone().permuted_axes(IxDyn(&perm)).as_standard_layout().to_owned();
                    let dxp = softmax_vjp(&sp, &dp)?;
                    let mut inv = vec![0; perm.len()];
                    for (i, p) in perm.iter().enumerate() {
                        inv[*p] = i;
                    }
                    dxp.permuted_axes(IxDyn(&inv)).as_standard_layout().to_owned()
                };
                self.add_grad(x, dx);
            }
            Tape::Identity { x, y } | Tape::Flatten { x, y } | Tape::Reshape { x, y } | Tape::Expand { x, y } => {
                let dy = match self.grads.get(y) {
                    Some(g) => g.clone(),
                    None => return Ok(()),
                };
                let xv = match self.values.get(x) {
                    Some(TensorValue::F32(t)) => t.shape().to_vec(),
                    Some(_) => return Ok(()),
                    None => return Ok(()),
                };
                let dx = if dy.shape() == xv.as_slice() {
                    dy
                } else if dy.len() == xv.iter().product::<usize>() {
                    dy.into_shape_with_order(IxDyn(&xv))
                        .map_err(|err| Error::fail(err.to_string()))?
                } else {
                    unbroadcast(&dy, &xv)
                };
                self.add_grad(x, dx);
            }
            Tape::Neg { x, y } => {
                let dy = match self.grads.get(y) {
                    Some(g) => g.clone(),
                    None => return Ok(()),
                };
                self.add_grad(x, neg(&dy)?);
            }
            Tape::Transpose { x, y, perm } => {
                let dy = match self.grads.get(y) {
                    Some(g) => g.clone(),
                    None => return Ok(()),
                };
                let mut inv = vec![0; perm.len()];
                for (i, p) in perm.iter().enumerate() {
                    inv[*p] = i;
                }
                let dx = dy.permuted_axes(IxDyn(&inv)).as_standard_layout().to_owned();
                self.add_grad(x, dx);
            }
            Tape::Reduce {
                x,
                y,
                axes,
                keepdims,
                kind,
                x_shape,
            } => {
                let dy = match self.grads.get(y) {
                    Some(g) => g.clone(),
                    None => return Ok(()),
                };
                let mut g = dy;
                if !*keepdims {
                    let mut restored = x_shape.to_vec();
                    for &ax in axes {
                        restored[ax] = 1;
                    }
                    // insert reduced axes as 1
                    let shape = g.shape().to_vec();
                    let mut acc = shape;
                    for &ax in axes {
                        if ax <= acc.len() {
                            acc.insert(ax, 1);
                        }
                    }
                    g = g
                        .clone()
                        .into_shape_with_order(IxDyn(&acc))
                        .unwrap_or(g);
                    let _ = restored;
                }
                let g = broadcast_to(&g, x_shape)?;
                let dx = match kind {
                    ReduceKind::Sum => g,
                    ReduceKind::Mean => {
                        let n = axes.iter().map(|&a| x_shape[a]).product::<usize>().max(1) as f32;
                        g.mapv(|v| v / n)
                    }
                    ReduceKind::Max => {
                        let x_t = self.f32(x)?;
                        let y_t = self.f32(y)?;
                        let yb = if *keepdims {
                            broadcast_to(y_t, x_shape)?
                        } else {
                            let mut acc = y_t.clone();
                            for &ax in axes {
                                acc = acc.insert_axis(Axis(ax));
                            }
                            broadcast_to(&acc, x_shape)?
                        };
                        Zip::from(x_t)
                            .and(&yb)
                            .and(&g)
                            .map_collect(|x, y, g| if (*x - *y).abs() < 1e-6 { *g } else { 0.0 })
                    }
                };
                self.add_grad(x, dx);
            }
            Tape::Where { c, x, y, out } => {
                let dy = match self.grads.get(out) {
                    Some(g) => g.clone(),
                    None => return Ok(()),
                };
                let cond = self.value(c)?.to_bool()?;
                let dx_full = Zip::from(&broadcast_bool_to(&cond, dy.shape())?)
                    .and(&dy)
                    .map_collect(|c, g| if *c { *g } else { 0.0 });
                let dy_full = Zip::from(&broadcast_bool_to(&cond, dy.shape())?)
                    .and(&dy)
                    .map_collect(|c, g| if *c { 0.0 } else { *g });
                let da = unbroadcast(&dx_full, self.f32(x)?.shape());
                let db = unbroadcast(&dy_full, self.f32(y)?.shape());
                self.add_grad(x, da);
                self.add_grad(y, db);
            }
            Tape::Gather {
                data,
                indices,
                y,
                axis,
            } => {
                let dy = match self.grads.get(y) {
                    Some(g) => g.clone(),
                    None => return Ok(()),
                };
                let shape = self.f32(data)?.shape().to_vec();
                let idx = self.i64(indices)?;
                let dx = gather_vjp(&shape, &idx, &dy, *axis)?;
                self.add_grad(data, dx);
            }
            Tape::GatherElements {
                data,
                indices,
                y,
                axis,
            } => {
                let dy = match self.grads.get(y) {
                    Some(g) => g.clone(),
                    None => return Ok(()),
                };
                let shape = self.f32(data)?.shape().to_vec();
                let idx = self.i64(indices)?;
                let dx = gather_elements_vjp(&shape, &idx, &dy, *axis)?;
                self.add_grad(data, dx);
            }
            Tape::Concat { xs, y, axis, sizes } => {
                let dy = match self.grads.get(y) {
                    Some(g) => g.clone(),
                    None => return Ok(()),
                };
                let parts = split(&dy, *axis, sizes)?;
                for (name, g) in xs.iter().zip(parts) {
                    self.add_grad(name, g);
                }
            }
            Tape::Split { x, ys, axis, sizes } => {
                let grads: Vec<Tensor> = ys
                    .iter()
                    .map(|n| {
                        self.grads.get(n).cloned().unwrap_or_else(|| {
                            let t = self.values.get(n).and_then(|v| v.as_f32().ok());
                            match t {
                                Some(t) => Tensor::zeros(t.raw_dim()),
                                None => Tensor::zeros(IxDyn(&[0])),
                            }
                        })
                    })
                    .collect();
                if grads.iter().all(|g| g.len() == 0) {
                    return Ok(());
                }
                let refs: Vec<&Tensor> = grads.iter().collect();
                let dx = concat(&refs, *axis)?;
                let _ = sizes;
                self.add_grad(x, dx);
            }
            Tape::Slice {
                x,
                y,
                starts,
                ends,
                axes,
                steps,
            } => {
                let dy = match self.grads.get(y) {
                    Some(g) => g.clone(),
                    None => return Ok(()),
                };
                let x_t = self.f32(x)?;
                let mut dx = Tensor::zeros(x_t.raw_dim());
                // only step=1
                if steps.iter().any(|s| *s != 1) {
                    return Err(Error::fail("Slice VJP requires step=1"));
                }
                copy_into_slice(&mut dx, &dy, starts, ends, axes)?;
                self.add_grad(x, dx);
            }
            Tape::LayerNorm {
                x,
                scale,
                bias,
                y,
                axis,
                epsilon,
            } => {
                let dy = match self.grads.get(y) {
                    Some(g) => g.clone(),
                    None => return Ok(()),
                };
                let x_t = self.f32(x)?.clone();
                let s = match scale {
                    Some(n) if !n.is_empty() => Some(self.f32(n)?.clone()),
                    _ => None,
                };
                let b = match bias {
                    Some(n) if !n.is_empty() => Some(self.f32(n)?.clone()),
                    _ => None,
                };
                let (dx, ds, db) =
                    layer_norm_vjp_full(&x_t, s.as_ref(), b.as_ref(), &dy, *epsilon, *axis)?;
                self.add_grad(x, dx);
                if let (Some(n), Some(g)) = (scale, ds) {
                    if !n.is_empty() {
                        self.add_grad(n, g);
                    }
                }
                if let (Some(n), Some(g)) = (bias, db) {
                    if !n.is_empty() {
                        self.add_grad(n, g);
                    }
                }
            }
            Tape::RmsNorm {
                x,
                scale,
                y,
                axis,
                epsilon,
            } => {
                let dy = match self.grads.get(y) {
                    Some(g) => g.clone(),
                    None => return Ok(()),
                };
                let x_t = self.f32(x)?.clone();
                let s = self.f32(scale)?.clone();
                let (dx, ds) = rms_norm_vjp(&x_t, &s, &dy, *epsilon, *axis)?;
                self.add_grad(x, dx);
                self.add_grad(scale, ds);
            }
            Tape::SkipRmsNorm {
                x,
                skip,
                scale,
                bias,
                y,
                residual,
                epsilon,
                axis,
            } => {
                let mut dy = match self.grads.get(y) {
                    Some(g) => g.clone(),
                    None => Tensor::zeros(self.f32(y)?.raw_dim()),
                };
                if let Some(res) = residual {
                    if let Some(g) = self.grads.get(res).cloned() {
                        dy = add(&dy, &g)?;
                    }
                }
                let pre = add(self.f32(x)?, self.f32(skip)?)?;
                let s = self.f32(scale)?.clone();
                let (dx_pre, ds) = rms_norm_vjp(&pre, &s, &dy, *epsilon, *axis)?;
                if let Some(b) = bias {
                    // y = rms + bias; extra db = unbroadcast(dy)
                    let db = unbroadcast(&dy, self.f32(b)?.shape());
                    self.add_grad(b, db);
                }
                self.add_grad(scale, ds);
                self.add_grad(x, dx_pre.clone());
                self.add_grad(skip, dx_pre);
            }
            Tape::Silu { x, y } => {
                let dy = match self.grads.get(y) {
                    Some(g) => g.clone(),
                    None => return Ok(()),
                };
                let dx = kernels::silu_vjp(self.f32(x)?, &dy);
                self.add_grad(x, dx);
            }
            Tape::Gelu { x, y } => {
                let dy = match self.grads.get(y) {
                    Some(g) => g.clone(),
                    None => return Ok(()),
                };
                let dx = kernels::gelu_vjp(self.f32(x)?, &dy);
                self.add_grad(x, dx);
            }
            Tape::Pow { a, b, y } => {
                let dy = match self.grads.get(y) {
                    Some(g) => g.clone(),
                    None => return Ok(()),
                };
                let a_t = self.f32(a)?.clone();
                let b_t = self.f32(b)?.clone();
                let y_t = self.f32(y)?.clone();
                let shape = dy.shape();
                let a_b = broadcast_to(&a_t, shape)?;
                let b_b = broadcast_to(&b_t, shape)?;
                let da_full = Zip::from(&dy)
                    .and(&b_b)
                    .and(&a_b)
                    .map_collect(|g, p, x| *g * *p * x.powf(*p - 1.0));
                let db_full = Zip::from(&dy)
                    .and(&y_t)
                    .and(&a_b)
                    .map_collect(|g, y, x| *g * *y * x.max(1e-12).ln());
                self.add_grad(a, unbroadcast(&da_full, a_t.shape()));
                self.add_grad(b, unbroadcast(&db_full, b_t.shape()));
            }
            Tape::Sqrt { x, y } => {
                let dy = match self.grads.get(y) {
                    Some(g) => g.clone(),
                    None => return Ok(()),
                };
                let y_t = self.f32(y)?;
                let dx = Zip::from(y_t)
                    .and(&dy)
                    .map_collect(|y, g| *g * 0.5 / y.max(1e-12));
                self.add_grad(x, dx);
            }
            Tape::Clip { x, y, min, max } => {
                let dy = match self.grads.get(y) {
                    Some(g) => g.clone(),
                    None => return Ok(()),
                };
                let x_t = self.f32(x)?;
                let dx = Zip::from(x_t).and(&dy).map_collect(|x, g| {
                    if *x > *min && *x < *max {
                        *g
                    } else {
                        0.0
                    }
                });
                self.add_grad(x, dx);
            }
            Tape::UnaryElem { x, y, op } => {
                let dy = match self.grads.get(y) {
                    Some(g) => g.clone(),
                    None => return Ok(()),
                };
                let x_t = self.f32(x)?;
                let dx = match op {
                    UnaryKind::Exp => {
                        let y_t = self.f32(y)?;
                        y_t * &dy
                    }
                    UnaryKind::Log => Zip::from(x_t)
                        .and(&dy)
                        .map_collect(|x, g| *g / x.max(1e-12)),
                    UnaryKind::Abs => Zip::from(x_t).and(&dy).map_collect(|x, g| {
                        if *x > 0.0 {
                            *g
                        } else if *x < 0.0 {
                            -*g
                        } else {
                            0.0
                        }
                    }),
                    UnaryKind::Reciprocal => {
                        let y_t = self.f32(y)?;
                        Zip::from(y_t)
                            .and(&dy)
                            .map_collect(|y, g| -*g * *y * *y)
                    }
                    UnaryKind::Erf => kernels::erf_vjp(x_t, &dy),
                    UnaryKind::Sin => Zip::from(x_t)
                        .and(&dy)
                        .map_collect(|x, g| *g * x.cos()),
                    UnaryKind::Cos => Zip::from(x_t)
                        .and(&dy)
                        .map_collect(|x, g| *g * (-x.sin())),
                    UnaryKind::Softplus => {
                        // d/dx = sigmoid(x)
                        Zip::from(x_t)
                            .and(&dy)
                            .map_collect(|x, g| *g / (1.0 + (-*x).exp()))
                    }
                };
                self.add_grad(x, dx);
            }
            Tape::Sdpa { q, k, v, mask, y, scale } => {
                let dy = match self.grads.get(y) {
                    Some(g) => g.clone(),
                    None => return Ok(()),
                };
                let q_t = self.f32(q)?.clone();
                let k_t = self.f32(k)?.clone();
                let v_t = self.f32(v)?.clone();
                let m = match mask {
                    Some(n) => Some(self.f32(n)?.clone()),
                    None => None,
                };
                let (dq, dk, dv) = sdpa_bwd(&q_t, &k_t, &v_t, m.as_ref(), &dy, *scale)?;
                self.add_grad(q, dq);
                self.add_grad(k, dk);
                self.add_grad(v, dv);
            }
            Tape::Rope { x, cos, sin, y } => {
                let dy = match self.grads.get(y) {
                    Some(g) => g.clone(),
                    None => return Ok(()),
                };
                let dx = rope_vjp(self.f32(x)?, self.f32(cos)?, self.f32(sin)?, &dy)?;
                self.add_grad(x, dx);
            }
            Tape::Segment {
                range: _,
                nodes,
                saved,
            } => {
                for (k, v) in saved {
                    self.values.insert(k.clone(), v.clone());
                }
                let was = self.recording;
                self.recording = true;
                let start = self.tape.len();
                let nodes = nodes.clone();
                for node in &nodes {
                    self.eval_node(node)?;
                }
                let local: Vec<Tape> = self.tape.drain(start..).collect();
                self.recording = was;
                for e in local.iter().rev() {
                    self.backward_entry(e)?;
                }
            }
        }
        Ok(())
    }
}

fn broadcast_bool_to(
    t: &ndarray::ArrayD<bool>,
    shape: &[usize],
) -> Result<ndarray::ArrayD<bool>> {
    t.broadcast(IxDyn(shape))
        .map(|v| v.to_owned())
        .ok_or_else(|| Error::fail("cannot broadcast condition"))
}

fn copy_into_slice(
    dx: &mut Tensor,
    dy: &Tensor,
    starts: &[i64],
    ends: &[i64],
    axes: &[i64],
) -> Result<()> {
    // Materialize a simple rank-aware copy for the common 1-axis slice.
    if axes.len() != 1 {
        // fallback: iterate coordinates of dy
        return copy_into_slice_nd(dx, dy, starts, ends, axes);
    }
    let axis = kernels::normalize_axis(axes[0], dx.ndim())?;
    let dim = dx.shape()[axis] as i64;
    let mut start = starts[0];
    if start < 0 {
        start += dim;
    }
    start = start.clamp(0, dim);
    let mut sl = dx.slice_axis_mut(
        Axis(axis),
        ndarray::Slice::from((start as usize)..(start as usize + dy.shape()[axis])),
    );
    sl.assign(&dy.view());
    Ok(())
}

fn copy_into_slice_nd(
    dx: &mut Tensor,
    dy: &Tensor,
    starts: &[i64],
    ends: &[i64],
    axes: &[i64],
) -> Result<()> {
    let _ = (ends, axes, starts);
    if dx.len() == dy.len() {
        *dx = dy.clone();
        return Ok(());
    }
    Err(Error::fail("multi-axis Slice VJP is not implemented"))
}
