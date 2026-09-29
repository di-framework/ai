//! Post-graph pooling and L2. Lives next to the interpreter so exporters do
//! not have to bake last-token gather perfectly.

use crate::error::Result;
use crate::kernels::{l2_normalize, l2_normalize_vjp, last_token_gather, last_token_vjp, mean_pool};
use crate::tensor::Tensor;
use crate::workspace::PoolKind;

#[derive(Debug, Clone)]
pub struct PoolTape {
    pub kind: PoolKind,
    pub l2: bool,
    pub src_shape: Vec<usize>,
    pub last_idx: Vec<usize>,
    pub l2_norm: Option<Tensor>,
    pub pooled: Tensor,
}

pub fn apply(
    y: &Tensor,
    mask: Option<&Tensor>,
    pool: PoolKind,
    l2: bool,
) -> Result<(Tensor, PoolTape)> {
    let src_shape = y.shape().to_vec();
    let (pooled, last_idx) = match pool {
        PoolKind::None => {
            if y.ndim() == 3 {
                // default: last-token if rank-3 embeddings
                last_token_gather(y, mask)?
            } else {
                (y.clone(), Vec::new())
            }
        }
        PoolKind::LastToken => last_token_gather(y, mask)?,
        PoolKind::Mean => (mean_pool(y, mask)?, Vec::new()),
        PoolKind::Cls => {
            let idx = vec![0; y.shape()[0]];
            (take_index(y, 0)?, idx)
        }
    };
    let (out, l2_norm) = if l2 {
        let (z, n) = l2_normalize(&pooled, 1e-12)?;
        (z, Some(n))
    } else {
        (pooled.clone(), None)
    };
    Ok((
        out.clone(),
        PoolTape {
            kind: pool,
            l2,
            src_shape,
            last_idx,
            l2_norm,
            pooled,
        },
    ))
}

fn take_index(y: &Tensor, t: usize) -> Result<Tensor> {
    let sl = y.slice_axis(ndarray::Axis(1), ndarray::Slice::from(t..t + 1));
    let mut shape = y.shape().to_vec();
    shape.remove(1);
    sl.to_owned()
        .into_shape_with_order(ndarray::IxDyn(&shape))
        .map_err(|err| crate::error::Error::fail(err.to_string()))
}

pub fn vjp(dy: &Tensor, tape: &PoolTape) -> Result<Tensor> {
    let mut g = dy.clone();
    if tape.l2 {
        if let Some(n) = &tape.l2_norm {
            // y_pooled after l2 is the output; vjp through l2 uses normalized y
            // reconstruct y_hat = pooled / n
            let yhat = {
                let n_b = crate::tensor::broadcast_to(n, tape.pooled.shape())?;
                &tape.pooled / &n_b
            };
            g = l2_normalize_vjp(&yhat, n, &g)?;
        }
    }
    match tape.kind {
        PoolKind::None if tape.last_idx.is_empty() => {
            if g.shape() == tape.src_shape.as_slice() {
                Ok(g)
            } else {
                g.into_shape_with_order(ndarray::IxDyn(&tape.src_shape))
                    .map_err(|err| crate::error::Error::fail(err.to_string()))
            }
        }
        PoolKind::None | PoolKind::LastToken | PoolKind::Cls => {
            last_token_vjp(&tape.src_shape, &tape.last_idx, &g)
        }
        PoolKind::Mean => mean_pool_vjp(&tape.src_shape, &g),
    }
}

fn mean_pool_vjp(src_shape: &[usize], dy: &Tensor) -> Result<Tensor> {
    let b = src_shape[0];
    let t = src_shape[1];
    let rest: usize = src_shape[2..].iter().product::<usize>().max(1);
    let scale = 1.0 / t as f32;
    let mut out = vec![0.0f32; b * t * rest];
    let gs = dy.as_standard_layout();
    let g = gs.as_slice().unwrap();
    for i in 0..b {
        for j in 0..t {
            for k in 0..rest {
                out[(i * t + j) * rest + k] = g[i * rest + k] * scale;
            }
        }
    }
    ndarray::Array::from_shape_vec(ndarray::IxDyn(src_shape), out)
        .map_err(|err| crate::error::Error::fail(err.to_string()))
}
