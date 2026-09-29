//! Proxy MTEB-style gates: STS Spearman and nDCG on `data/eval.jsonl`.

use crate::data::Dataset;
use crate::error::Result;
use crate::graph::ExecGraph;
use crate::hooks;
use crate::onnx::TensorValue;
use crate::tensor::Tensor;
use crate::workspace::{EvalMetric, PoolKind, TrainConfig};

pub fn proxy(
    graph: &mut ExecGraph,
    data: &Dataset,
    cfg: &TrainConfig,
) -> Result<(Option<f32>, Option<f32>)> {
    match cfg.eval_metric {
        EvalMetric::Loss => Ok((None, None)),
        EvalMetric::Spearman => Ok((Some(spearman_eval(graph, data, cfg)?), None)),
        EvalMetric::Ndcg => Ok((None, Some(ndcg_eval(graph, data, cfg)?))),
    }
}

fn spearman_eval(graph: &mut ExecGraph, data: &Dataset, cfg: &TrainConfig) -> Result<f32> {
    let mut xs = Vec::new();
    let mut ys = Vec::new();
    for ex in &data.examples {
        let Some(score) = ex.score else {
            continue;
        };
        let a = embed(graph, &ex.inputs, cfg, ex.attention_mask.as_ref())?;
        let b_in = ex.pair.as_ref().unwrap_or(&ex.inputs);
        let b = embed(graph, b_in, cfg, ex.pair_mask.as_ref())?;
        xs.push(cosine_vec(&a, &b));
        ys.push(score);
    }
    if xs.len() < 3 {
        return Ok(0.0);
    }
    Ok(spearman(&xs, &ys))
}

fn ndcg_eval(graph: &mut ExecGraph, data: &Dataset, cfg: &TrainConfig) -> Result<f32> {
    // Each example: query embedding vs positive (rel=1) — toy nDCG@1.
    let mut vals = Vec::new();
    for ex in &data.examples {
        let rel = ex.score.unwrap_or(1.0);
        let q = embed(graph, &ex.inputs, cfg, ex.attention_mask.as_ref())?;
        if let Some(p) = &ex.pair {
            let d = embed(graph, p, cfg, ex.pair_mask.as_ref())?;
            let s = cosine_vec(&q, &d);
            vals.push((s, rel));
        }
    }
    if vals.is_empty() {
        return Ok(0.0);
    }
    vals.sort_by(|a, b| b.0.partial_cmp(&a.0).unwrap_or(std::cmp::Ordering::Equal));
    let k = vals.len().min(10);
    Ok(ndcg_at_k(&vals.iter().map(|v| v.1).collect::<Vec<_>>(), k))
}

fn embed(
    graph: &mut ExecGraph,
    inputs: &std::collections::HashMap<String, TensorValue>,
    cfg: &TrainConfig,
    mask: Option<&Tensor>,
) -> Result<Tensor> {
    let out = graph.forward_values(inputs)?;
    let t = if let Some(name) = &cfg.embedding_output {
        out.get(name).cloned()
    } else {
        out.values().next().cloned()
    }
    .ok_or_else(|| crate::error::Error::fail("no embedding output"))?;
    let pool = if t.ndim() >= 3 && cfg.pool == PoolKind::None {
        PoolKind::LastToken
    } else {
        cfg.pool
    };
    let (emb, _) = hooks::apply(&t, mask, pool, true)?;
    Ok(emb)
}

fn cosine_vec(a: &Tensor, b: &Tensor) -> f32 {
    let mut dot = 0.0;
    let mut na = 0.0;
    let mut nb = 0.0;
    for (x, y) in a.iter().zip(b.iter()) {
        dot += x * y;
        na += x * x;
        nb += y * y;
    }
    dot / ((na.sqrt() * nb.sqrt()).max(1e-8))
}

pub fn spearman(x: &[f32], y: &[f32]) -> f32 {
    pearson(&ranks(x), &ranks(y))
}

fn ranks(x: &[f32]) -> Vec<f32> {
    let mut idx: Vec<usize> = (0..x.len()).collect();
    idx.sort_by(|i, j| x[*i].partial_cmp(&x[*j]).unwrap_or(std::cmp::Ordering::Equal));
    let mut r = vec![0.0; x.len()];
    for (rank, i) in idx.into_iter().enumerate() {
        r[i] = rank as f32;
    }
    r
}

fn pearson(x: &[f32], y: &[f32]) -> f32 {
    let n = x.len().min(y.len()) as f32;
    if n < 2.0 {
        return 0.0;
    }
    let mx = x.iter().sum::<f32>() / n;
    let my = y.iter().sum::<f32>() / n;
    let mut num = 0.0;
    let mut dx = 0.0;
    let mut dy = 0.0;
    for (a, b) in x.iter().zip(y.iter()) {
        let da = *a - mx;
        let db = *b - my;
        num += da * db;
        dx += da * da;
        dy += db * db;
    }
    num / (dx.sqrt() * dy.sqrt()).max(1e-12)
}

pub fn ndcg_at_k(rels: &[f32], k: usize) -> f32 {
    let k = k.min(rels.len());
    if k == 0 {
        return 0.0;
    }
    let dcg: f32 = rels
        .iter()
        .take(k)
        .enumerate()
        .map(|(i, r)| r / (i as f32 + 2.0).log2())
        .sum();
    let mut ideal = rels.to_vec();
    ideal.sort_by(|a, b| b.partial_cmp(a).unwrap_or(std::cmp::Ordering::Equal));
    let idcg: f32 = ideal
        .iter()
        .take(k)
        .enumerate()
        .map(|(i, r)| r / (i as f32 + 2.0).log2())
        .sum();
    if idcg == 0.0 {
        0.0
    } else {
        dcg / idcg
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn spearman_monotonic() {
        let x = [1.0, 2.0, 3.0, 4.0];
        let y = [0.1, 0.2, 0.3, 0.4];
        assert!((spearman(&x, &y) - 1.0).abs() < 1e-5);
    }

    #[test]
    fn ndcg_perfect_is_one() {
        assert!((ndcg_at_k(&[3.0, 2.0, 1.0], 3) - 1.0).abs() < 1e-5);
        assert!(ndcg_at_k(&[0.0, 1.0], 2) < 1.0);
    }
}
