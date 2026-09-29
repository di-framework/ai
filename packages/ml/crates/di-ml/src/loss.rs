//! Graph-level losses. Consume named ONNX outputs + batch metadata.
//! InfoNCE is *not* baked into the ONNX file.

use ndarray::{IxDyn, Zip};
use serde::Serialize;

use crate::data::{Batched, Label};
use crate::error::{Error, Result};
use crate::hooks::{self, PoolTape};
use crate::tensor::{align_to, softmax_last, Tensor};
use crate::workspace::{LossKind, PoolKind, TrainConfig};

#[derive(Debug, Clone)]
pub struct LossOut {
    pub loss: f32,
    pub output_grads: Vec<(String, Tensor)>,
}

#[derive(Debug, Clone, Serialize)]
pub struct LossMeta {
    pub kind: String,
}

pub fn compute(
    named: &[(String, Tensor)],
    batch: &Batched,
    cfg: &TrainConfig,
) -> Result<LossOut> {
    let (name, pred) = select_output(named, cfg)?;
    let mask = batch.attention_mask.as_ref();
    let pool = if pred.ndim() >= 3 && cfg.pool == PoolKind::None && cfg.loss.is_embedding() {
        PoolKind::LastToken
    } else {
        cfg.pool
    };
    let l2 = cfg.l2_normalize || cfg.loss.is_embedding();
    let (emb, tape) = if pred.ndim() >= 3 || cfg.pool != PoolKind::None || cfg.l2_normalize {
        hooks::apply(pred, mask, pool, l2)?
    } else {
        (
            pred.clone(),
            PoolTape {
                kind: PoolKind::None,
                l2: false,
                src_shape: pred.shape().to_vec(),
                last_idx: Vec::new(),
                l2_norm: None,
                pooled: pred.clone(),
            },
        )
    };

    match cfg.loss {
        LossKind::Mse | LossKind::L1 | LossKind::BceLogits | LossKind::CrossEntropy => {
            let target = match &batch.label {
                Label::Float(t) => t,
                Label::Class(_) => {
                    return Err(Error::fail("internal: class labels should be stacked as floats"));
                }
                Label::None => {
                    return Err(Error::usage(
                        "pointwise losses need a label field (label, labels, or y)",
                    ));
                }
            };
            let (loss, g_pred) = pointwise(pred, target, cfg.loss, batch.class_indices)?;
            Ok(LossOut {
                loss,
                output_grads: vec![(name, g_pred)],
            })
        }
        LossKind::InfoNce | LossKind::SupCon => infonce(&name, &emb, &tape, batch, cfg.temperature, pred),
        LossKind::Cosine => cosine(&name, &emb, &tape, batch, pred),
        LossKind::Triplet => triplet(&name, &emb, &tape, batch, cfg.triplet_margin, pred),
        LossKind::Distill => distill(&name, &emb, &tape, batch, cfg, pred),
    }
}

fn select_output<'a>(
    named: &'a [(String, Tensor)],
    cfg: &TrainConfig,
) -> Result<(String, &'a Tensor)> {
    if let Some(want) = &cfg.embedding_output {
        named
            .iter()
            .find(|(n, _)| n == want)
            .map(|(n, t)| (n.clone(), t))
            .ok_or_else(|| Error::usage(format!("embedding-output {want:?} is not a graph output")))
    } else {
        named
            .first()
            .map(|(n, t)| (n.clone(), t))
            .ok_or_else(|| Error::fail("model has no outputs"))
    }
}

fn pointwise(
    pred: &Tensor,
    target: &Tensor,
    loss: LossKind,
    class_indices: bool,
) -> Result<(f32, Tensor)> {
    match loss {
        LossKind::Mse => {
            let target = align_to(target, pred.shape())?;
            let diff = pred - &target;
            let n = pred.len().max(1) as f32;
            let loss = diff.iter().map(|v| v * v).sum::<f32>() / n;
            Ok((loss, &diff * (2.0 / n)))
        }
        LossKind::L1 => {
            let target = align_to(target, pred.shape())?;
            let diff = pred - &target;
            let n = pred.len().max(1) as f32;
            let loss = diff.iter().map(|v| v.abs()).sum::<f32>() / n;
            let grad = diff.mapv(|v| {
                if v > 0.0 {
                    1.0 / n
                } else if v < 0.0 {
                    -1.0 / n
                } else {
                    0.0
                }
            });
            Ok((loss, grad))
        }
        LossKind::BceLogits => {
            let target = align_to(target, pred.shape())?;
            let n = pred.len().max(1) as f32;
            let mut loss = 0.0f32;
            let mut grad = pred.clone();
            Zip::from(&mut grad)
                .and(pred)
                .and(&target)
                .for_each(|g, &x, &y| {
                    let relu_x = x.max(0.0);
                    loss += relu_x - x * y + ((-x.abs()).exp() + 1.0).ln();
                    *g = (sigmoid(x) - y) / n;
                });
            Ok((loss / n, grad))
        }
        LossKind::CrossEntropy => {
            if class_indices {
                cross_entropy_indices(pred, target)
            } else {
                cross_entropy_soft(pred, target)
            }
        }
        _ => unreachable!(),
    }
}

fn sigmoid(x: f32) -> f32 {
    1.0 / (1.0 + (-x).exp())
}

fn cross_entropy_indices(pred: &Tensor, target: &Tensor) -> Result<(f32, Tensor)> {
    if pred.ndim() != 2 {
        return Err(Error::usage(format!(
            "cross-entropy with class labels expects rank-2 logits [N, C], got {:?}",
            pred.shape()
        )));
    }
    let n = pred.shape()[0];
    let c = pred.shape()[1];
    if target.len() != n {
        return Err(Error::usage(format!(
            "class-label count {} does not match batch {n}",
            target.len()
        )));
    }
    let probs = softmax_last(pred);
    let mut loss = 0.0f32;
    let mut grad = probs.clone();
    for (i, class) in target.iter().enumerate() {
        let k = *class as i64;
        if k < 0 || k as usize >= c {
            return Err(Error::usage(format!(
                "class index {k} is out of range for {c} logits"
            )));
        }
        let k = k as usize;
        let p = probs[IxDyn(&[i, k])].max(1e-12);
        loss -= p.ln();
        grad[IxDyn(&[i, k])] -= 1.0;
    }
    let n = n as f32;
    grad.mapv_inplace(|v| v / n);
    Ok((loss / n, grad))
}

fn cross_entropy_soft(pred: &Tensor, target: &Tensor) -> Result<(f32, Tensor)> {
    let target = align_to(target, pred.shape())?;
    let probs = softmax_last(pred);
    let axis = pred.ndim() - 1;
    let n = pred.len() / pred.shape()[axis];
    let log_p = probs.mapv(|p| p.max(1e-12).ln());
    let per = &target * &log_p;
    let loss = -per.iter().sum::<f32>() / n.max(1) as f32;
    let grad = (&probs - &target).mapv(|v| v / n.max(1) as f32);
    Ok((loss, grad))
}

/// Packed batch: first B rows queries, next B positives (in-batch negatives).
fn infonce(
    name: &str,
    emb: &Tensor,
    tape: &PoolTape,
    batch: &Batched,
    temperature: f32,
    raw: &Tensor,
) -> Result<LossOut> {
    let (q, p) = split_pairs(emb, batch)?;
    let b = q.shape()[0];
    let d = q.len() / b;
    let q2 = q
        .clone()
        .into_shape_with_order(IxDyn(&[b, d]))
        .map_err(|e| Error::fail(e.to_string()))?;
    let p2 = p
        .clone()
        .into_shape_with_order(IxDyn(&[b, d]))
        .map_err(|e| Error::fail(e.to_string()))?;
    // logits = q @ p.T / temp
    let mut logits = ndarray::Array2::<f32>::zeros((b, b));
    for i in 0..b {
        for j in 0..b {
            let mut s = 0.0f32;
            for k in 0..d {
                s += q2[[i, k]] * p2[[j, k]];
            }
            logits[[i, j]] = s / temperature.max(1e-8);
        }
    }
    let logits_t = logits.clone().into_dyn();
    let probs = softmax_last(&logits_t);
    let mut loss = 0.0f32;
    let mut dlogits = probs.clone();
    for i in 0..b {
        let pii = probs[IxDyn(&[i, i])].max(1e-12);
        loss -= pii.ln();
        dlogits[IxDyn(&[i, i])] -= 1.0;
    }
    loss /= b as f32;
    dlogits.mapv_inplace(|v| v / (b as f32 * temperature.max(1e-8)));
    // dQ = dlogits @ P ; dP = dlogits.T @ Q
    let mut dq = ndarray::Array2::<f32>::zeros((b, d));
    let mut dp = ndarray::Array2::<f32>::zeros((b, d));
    for i in 0..b {
        for k in 0..d {
            let mut gq = 0.0;
            let mut gp = 0.0;
            for j in 0..b {
                gq += dlogits[IxDyn(&[i, j])] * p2[[j, k]];
                gp += dlogits[IxDyn(&[j, i])] * q2[[j, k]];
            }
            dq[[i, k]] = gq;
            dp[[i, k]] = gp;
        }
    }
    let g_emb = stack_pairs(&dq.into_dyn(), &dp.into_dyn(), emb)?;
    let g_raw = hooks::vjp(&g_emb, tape)?;
    let g_raw = match_raw(&g_raw, raw)?;
    let mut loss_out = loss;
    let mut g_raw = g_raw;
    if let Some(replay) = &batch.replay_emb {
        let (extra, g) = replay_l2(emb, replay)?;
        loss_out += extra;
        let g_r = hooks::vjp(&g, tape)?;
        g_raw = g_raw + match_raw(&g_r, raw)?;
    }
    Ok(LossOut {
        loss: loss_out,
        output_grads: vec![(name.to_string(), g_raw)],
    })
}

fn cosine(
    name: &str,
    emb: &Tensor,
    tape: &PoolTape,
    batch: &Batched,
    raw: &Tensor,
) -> Result<LossOut> {
    let (q, p) = split_pairs(emb, batch)?;
    let b = q.shape()[0].min(p.shape()[0]);
    let d = q.len() / q.shape()[0];
    let target = match &batch.label {
        Label::Float(t) => t.clone(),
        Label::Class(_) | Label::None => ndarray::Array::from_elem(IxDyn(&[b]), 1.0),
    };
    let mut loss = 0.0f32;
    let mut dq = Tensor::zeros(q.raw_dim());
    let mut dp = Tensor::zeros(p.raw_dim());
    for i in 0..b {
        let mut dot = 0.0;
        for k in 0..d {
            let qi = q.as_slice().unwrap()[i * d + k];
            let pi = p.as_slice().unwrap()[i * d + k];
            dot += qi * pi;
        }
        let y = target.iter().nth(i).copied().unwrap_or(1.0);
        let diff = dot - y;
        loss += diff * diff;
        let g = 2.0 * diff / b as f32;
        let dqs = dq.as_slice_mut().unwrap();
        let dps = dp.as_slice_mut().unwrap();
        let qs = q.as_slice().unwrap();
        let ps = p.as_slice().unwrap();
        for k in 0..d {
            dqs[i * d + k] += g * ps[i * d + k];
            dps[i * d + k] += g * qs[i * d + k];
        }
    }
    loss /= b as f32;
    let g_emb = stack_pairs(&dq, &dp, emb)?;
    let g_raw = match_raw(&hooks::vjp(&g_emb, tape)?, raw)?;
    Ok(LossOut {
        loss,
        output_grads: vec![(name.to_string(), g_raw)],
    })
}

fn triplet(
    name: &str,
    emb: &Tensor,
    tape: &PoolTape,
    batch: &Batched,
    margin: f32,
    raw: &Tensor,
) -> Result<LossOut> {
    let (q, p) = split_pairs(emb, batch)?;
    // in-batch: negative is the next positive (circular)
    let b = q.shape()[0];
    let d = q.len() / b;
    let qs = q.as_slice().unwrap();
    let ps = p.as_slice().unwrap();
    let mut dq = vec![0.0f32; b * d];
    let mut dp = vec![0.0f32; b * d];
    let mut loss = 0.0f32;
    for i in 0..b {
        let j = (i + 1) % b;
        let mut pos = 0.0;
        let mut neg = 0.0;
        for k in 0..d {
            pos += qs[i * d + k] * ps[i * d + k];
            neg += qs[i * d + k] * ps[j * d + k];
        }
        // maximize pos, minimize neg → loss = relu(margin - pos + neg) on cosine (higher is closer)
        let hinge = (margin - pos + neg).max(0.0);
        loss += hinge;
        if hinge > 0.0 {
            for k in 0..d {
                dq[i * d + k] += (-ps[i * d + k] + ps[j * d + k]) / b as f32;
                dp[i * d + k] += (-qs[i * d + k]) / b as f32;
                dp[j * d + k] += qs[i * d + k] / b as f32;
            }
        }
    }
    loss /= b as f32;
    let dq = Tensor::from_shape_vec(q.raw_dim(), dq).map_err(|e| Error::fail(e.to_string()))?;
    let dp = Tensor::from_shape_vec(p.raw_dim(), dp).map_err(|e| Error::fail(e.to_string()))?;
    let g_emb = stack_pairs(&dq, &dp, emb)?;
    let g_raw = match_raw(&hooks::vjp(&g_emb, tape)?, raw)?;
    Ok(LossOut {
        loss,
        output_grads: vec![(name.to_string(), g_raw)],
    })
}

fn distill(
    name: &str,
    emb: &Tensor,
    tape: &PoolTape,
    batch: &Batched,
    cfg: &TrainConfig,
    raw: &Tensor,
) -> Result<LossOut> {
    let mut out = infonce(name, emb, tape, batch, cfg.temperature, raw)?;
    if let (Some(tp), Some(tn)) = (&batch.teacher_pos, &batch.teacher_neg) {
        // listwise: student sim vs teacher scores
        let (q, p) = split_pairs(emb, batch)?;
        let b = q.shape()[0];
        let d = q.len() / b;
        let mut extra = 0.0f32;
        let qs = q.as_slice().unwrap();
        let ps = p.as_slice().unwrap();
        for i in 0..b.min(tp.len()) {
            let mut s = 0.0;
            for k in 0..d {
                s += qs[i * d + k] * ps[i * d + k];
            }
            let diff = s - tp[i];
            extra += diff * diff;
        }
        extra /= b.max(1) as f32;
        out.loss += cfg.distill_weight.max(1.0) * extra;
        let _ = tn;
    }
    Ok(out)
}

fn replay_l2(emb: &Tensor, replay: &Tensor) -> Result<(f32, Tensor)> {
    let t = align_to(replay, emb.shape())?;
    let diff = emb - &t;
    let n = emb.len().max(1) as f32;
    let loss = diff.iter().map(|v| v * v).sum::<f32>() / n;
    Ok((loss, &diff * (2.0 / n)))
}

fn split_pairs(emb: &Tensor, batch: &Batched) -> Result<(Tensor, Tensor)> {
    if let Some(half) = batch.pair_split {
        if emb.ndim() == 0 || emb.shape()[0] < half * 2 && emb.shape()[0] != half * 2 {
            // allow exact 2B
        }
        let b = half;
        if emb.shape()[0] < 2 * b {
            return Err(Error::usage(format!(
                "InfoNCE packed batch expected 2*{b} rows, got {:?}",
                emb.shape()
            )));
        }
        let q = emb
            .slice_axis(ndarray::Axis(0), ndarray::Slice::from(0..b))
            .to_owned();
        let p = emb
            .slice_axis(ndarray::Axis(0), ndarray::Slice::from(b..2 * b))
            .to_owned();
        return Ok((q, p));
    }
    // single tower: treat even/odd or the whole batch as queries with label targets unused
    if emb.shape()[0] % 2 == 0 {
        let b = emb.shape()[0] / 2;
        let q = emb
            .slice_axis(ndarray::Axis(0), ndarray::Slice::from(0..b))
            .to_owned();
        let p = emb
            .slice_axis(ndarray::Axis(0), ndarray::Slice::from(b..2 * b))
            .to_owned();
        return Ok((q, p));
    }
    Err(Error::usage(
        "InfoNCE needs a packed (query, positive) batch with even batch size",
    ))
}

fn stack_pairs(dq: &Tensor, dp: &Tensor, emb: &Tensor) -> Result<Tensor> {
    let views = [dq.view(), dp.view()];
    let stacked = ndarray::concatenate(ndarray::Axis(0), &views)
        .map_err(|e| Error::fail(e.to_string()))?;
    if stacked.shape() == emb.shape() {
        Ok(stacked)
    } else {
        stacked
            .into_shape_with_order(emb.raw_dim())
            .map_err(|e| Error::fail(e.to_string()))
    }
}

fn match_raw(g: &Tensor, raw: &Tensor) -> Result<Tensor> {
    if g.shape() == raw.shape() {
        Ok(g.clone())
    } else if g.len() == raw.len() {
        g.clone()
            .into_shape_with_order(raw.raw_dim())
            .map_err(|e| Error::fail(e.to_string()))
    } else {
        Ok(g.clone())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::data::{Batched, Label};
    use ndarray::array;
    use std::collections::HashMap;

    #[test]
    fn infonce_perfect_pairs_low_loss() {
        let q = array![[1.0, 0.0], [0.0, 1.0]];
        let p = array![[1.0, 0.0], [0.0, 1.0]];
        let emb = ndarray::concatenate(ndarray::Axis(0), &[q.view().into_dyn(), p.view().into_dyn()]).unwrap();
        let batch = Batched {
            inputs: HashMap::new(),
            label: Label::Float(array![1.0, 1.0].into_dyn()),
            class_indices: false,
            pair_split: Some(2),
            attention_mask: None,
            teacher_pos: None,
            teacher_neg: None,
            replay_emb: None,
            scores: None,
        };
        let tape = PoolTape {
            kind: PoolKind::None,
            l2: false,
            src_shape: emb.shape().to_vec(),
            last_idx: Vec::new(),
            l2_norm: None,
            pooled: emb.clone(),
        };
        let out = infonce("embedding", &emb, &tape, &batch, 0.07, &emb).unwrap();
        assert!(out.loss < 0.05, "loss {}", out.loss);
    }
}
