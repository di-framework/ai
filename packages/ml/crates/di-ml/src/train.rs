use std::collections::{HashMap, HashSet};
use std::path::Path;
use std::time::Instant;

use ndarray::Zip;
use rand::rngs::StdRng;
use rand::seq::SliceRandom;
use rand::SeedableRng;
use serde::{Deserialize, Serialize};

use crate::data::{Batched, Dataset};
use crate::error::{Error, Result};
use crate::graph::ExecGraph;
use crate::ops::{is_embedding_name, matches_trainable};
use crate::tensor::Tensor;
use crate::workspace::{OptimizerKind, TrainConfig};

#[derive(Debug, Clone, Serialize)]
#[serde(rename_all = "kebab-case")]
pub struct StepMetric {
    pub step: usize,
    pub train_loss: f32,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub eval_loss: Option<f32>,
}

#[derive(Debug, Clone, Serialize)]
#[serde(rename_all = "kebab-case")]
pub struct TrainReport {
    pub loss: String,
    pub optimizer: String,
    pub learning_rate: f32,
    pub steps: usize,
    pub batch_size: usize,
    pub trainable: Vec<String>,
    pub accel: String,
    pub initial_train_loss: f32,
    pub final_train_loss: f32,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub final_eval_loss: Option<f32>,
    pub history: Vec<StepMetric>,
    pub seed: u64,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub config_hash: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub data_hash: Option<String>,
    pub peak_bytes: usize,
    pub step_time_ms: f32,
    pub mixed_precision: String,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub eval_spearman: Option<f32>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub eval_ndcg: Option<f32>,
}

#[derive(Debug, Clone, Default)]
pub struct TrainExtras {
    pub config_hash: Option<String>,
    pub data_hash: Option<String>,
    pub resume_from: Option<std::path::PathBuf>,
    pub save_optimizer: Option<std::path::PathBuf>,
}

pub fn train(
    graph: &mut ExecGraph,
    train_data: &Dataset,
    eval_data: Option<&Dataset>,
    cfg: &TrainConfig,
    frozen: &[String],
    trainable: &[String],
) -> Result<TrainReport> {
    train_with(graph, train_data, eval_data, cfg, frozen, trainable, TrainExtras::default())
}

pub fn train_with(
    graph: &mut ExecGraph,
    train_data: &Dataset,
    eval_data: Option<&Dataset>,
    cfg: &TrainConfig,
    frozen: &[String],
    trainable: &[String],
    extras: TrainExtras,
) -> Result<TrainReport> {
    crate::accel::set_mixed_precision(cfg.mixed_precision);
    graph.set_checkpointing(cfg.activation_checkpointing, cfg.checkpoint_every);

    let trainable = resolve_trainable(graph, frozen, trainable, cfg)?;
    let trainable_set: HashSet<String> = trainable.iter().cloned().collect();
    graph.skip_grad = graph
        .weight_names
        .iter()
        .filter(|n| !trainable_set.contains(*n))
        .cloned()
        .collect();
    for name in &graph.skip_grad {
        if let Ok(t) = graph.param(name) {
            crate::accel::current().pin_readonly(name, t);
        }
    }

    let mut opt = Optimizer::new(cfg.optimizer, &trainable, graph)?;
    if let Some(path) = &extras.resume_from {
        if path.is_file() {
            opt.load(path)?;
        }
    }
    let mut rng = StdRng::seed_from_u64(cfg.seed);

    let n = train_data.examples.len();
    let batch_size = cfg.batch_size.min(n);
    let micro = cfg.microbatch_size.unwrap_or(batch_size).min(batch_size).max(1);
    let accum = cfg.grad_accum_steps.max(1);
    let pack = cfg.loss.is_embedding();
    let mut order: Vec<usize> = (0..n).collect();
    let mut cursor = n;

    let initial_train_loss = mean_loss(graph, train_data, cfg)?;
    let mut last_train = initial_train_loss;
    let mut last_eval = match eval_data {
        Some(ds) => Some(mean_loss(graph, ds, cfg)?),
        None => None,
    };
    let mut history = Vec::new();
    record(
        &mut history,
        0,
        last_train,
        last_eval,
        cfg.log_every,
        cfg.max_steps,
    );

    let mut step_times = Vec::new();

    for step in 1..=cfg.max_steps {
        let t0 = Instant::now();
        graph.zero_grad();
        let mut step_loss_acc = 0.0f32;
        let mut step_n = 0usize;

        for _ in 0..accum {
            if cursor + batch_size > n {
                order.shuffle(&mut rng);
                cursor = 0;
            }
            let end = (cursor + batch_size).min(n);
            let logical = &order[cursor..end];
            cursor = end;

            for chunk in logical.chunks(micro) {
                let batch = train_data.batch_pack(chunk, pack)?;
                let (loss, grads) = step_loss(graph, &batch, cfg)?;
                if !loss.is_finite() {
                    return Err(Error::fail(format!("non-finite train loss at step {step}")));
                }
                step_loss_acc += loss * chunk.len() as f32;
                step_n += chunk.len();
                graph.backward_acc(grads, true)?;
            }
        }
        last_train = step_loss_acc / step_n.max(1) as f32;
        opt.step(graph, &trainable, cfg.learning_rate)?;
        let dt = t0.elapsed();
        crate::accel::record_step_ns(dt.as_nanos() as u64);
        step_times.push(dt.as_secs_f32() * 1000.0);

        let should_log = cfg.log_every > 0 && step % cfg.log_every == 0;
        if should_log || step == cfg.max_steps {
            if let Some(ds) = eval_data {
                last_eval = Some(mean_loss(graph, ds, cfg)?);
            }
            record(
                &mut history,
                step,
                last_train,
                last_eval,
                cfg.log_every,
                cfg.max_steps,
            );
        }
    }

    if let Some(path) = extras.save_optimizer {
        if let Some(parent) = path.parent() {
            let _ = std::fs::create_dir_all(parent);
        }
        opt.save(&path)?;
    }

    let (eval_spearman, eval_ndcg) = match eval_data {
        Some(ds) => crate::eval::proxy(graph, ds, cfg)?,
        None => (None, None),
    };

    let step_time_ms = if step_times.is_empty() {
        0.0
    } else {
        step_times.iter().sum::<f32>() / step_times.len() as f32
    };

    Ok(TrainReport {
        loss: cfg.loss.as_str().to_string(),
        optimizer: cfg.optimizer.as_str().to_string(),
        learning_rate: cfg.learning_rate,
        steps: cfg.max_steps,
        batch_size,
        trainable,
        accel: crate::accel::current_label(),
        initial_train_loss,
        final_train_loss: last_train,
        final_eval_loss: last_eval,
        history,
        seed: cfg.seed,
        config_hash: extras.config_hash,
        data_hash: extras.data_hash,
        peak_bytes: graph.peak_bytes.max(crate::accel::take_peak_bytes()),
        step_time_ms,
        mixed_precision: cfg.mixed_precision.as_str().to_string(),
        eval_spearman,
        eval_ndcg,
    })
}

fn record(
    history: &mut Vec<StepMetric>,
    step: usize,
    train_loss: f32,
    eval_loss: Option<f32>,
    log_every: usize,
    max_steps: usize,
) {
    if step == 0 || step == max_steps || (log_every > 0 && step % log_every == 0) {
        history.push(StepMetric {
            step,
            train_loss,
            eval_loss,
        });
    }
}

fn resolve_trainable(
    graph: &ExecGraph,
    frozen: &[String],
    trainable: &[String],
    cfg: &TrainConfig,
) -> Result<Vec<String>> {
    let all: Vec<String> = graph.weight_names.iter().cloned().collect();
    let mut names: Vec<String> = if trainable.is_empty() {
        if cfg.lora_rank.is_some() && all.iter().any(|n| n.starts_with("lora_")) {
            all.iter()
                .filter(|n| n.starts_with("lora_"))
                .cloned()
                .collect()
        } else {
            all.clone()
        }
    } else {
        let mut matched = Vec::new();
        for pat in trainable {
            let mut hit = false;
            for name in &all {
                if matches_trainable(pat, name) {
                    matched.push(name.clone());
                    hit = true;
                }
            }
            if !hit && all.iter().any(|n| n == pat) {
                matched.push(pat.clone());
            } else if !hit {
                return Err(Error::usage(format!(
                    "{pat:?} matched no f32 initializer; available: {}",
                    all.join(", ")
                )));
            }
        }
        matched
    };
    names.retain(|n| {
        !frozen.iter().any(|f| matches_trainable(f, n))
    });
    if cfg.freeze_embeddings {
        names.retain(|n| !is_embedding_name(n));
    }
    names.sort();
    names.dedup();
    if names.is_empty() {
        return Err(Error::usage(
            "no trainable initializers remain after applying frozen/trainable filters",
        ));
    }
    Ok(names)
}

fn mean_loss(graph: &mut ExecGraph, data: &Dataset, cfg: &TrainConfig) -> Result<f32> {
    let n = data.examples.len();
    let mut total = 0.0f32;
    let mut seen = 0usize;
    let mut start = 0;
    let pack = cfg.loss.is_embedding();
    while start < n {
        let end = (start + cfg.batch_size).min(n);
        let idx: Vec<usize> = (start..end).collect();
        let batch = data.batch_pack(&idx, pack)?;
        let (loss, _) = step_loss(graph, &batch, cfg)?;
        total += loss * idx.len() as f32;
        seen += idx.len();
        start = end;
    }
    Ok(total / seen.max(1) as f32)
}

fn step_loss(
    graph: &mut ExecGraph,
    batch: &Batched,
    cfg: &TrainConfig,
) -> Result<(f32, HashMap<String, Tensor>)> {
    let outputs = graph.forward_values(&batch.inputs)?;
    let named: Vec<(String, Tensor)> = graph
        .outputs
        .iter()
        .filter_map(|n| outputs.get(n).map(|t| (n.clone(), t.clone())))
        .collect();
    let out = crate::loss::compute(&named, batch, cfg)?;
    Ok((out.loss, out.output_grads.into_iter().collect()))
}

struct Optimizer {
    kind: OptimizerKind,
    m: HashMap<String, Tensor>,
    v: HashMap<String, Tensor>,
    t: usize,
}

#[derive(Serialize, Deserialize)]
struct OptDisk {
    t: usize,
    kind: String,
    m: HashMap<String, SerTensor>,
    v: HashMap<String, SerTensor>,
}

#[derive(Serialize, Deserialize)]
struct SerTensor {
    shape: Vec<usize>,
    data: Vec<f32>,
}

impl Optimizer {
    fn new(kind: OptimizerKind, names: &[String], graph: &ExecGraph) -> Result<Self> {
        let mut m = HashMap::new();
        let mut v = HashMap::new();
        if kind == OptimizerKind::Adamw {
            for name in names {
                let w = graph.param(name)?;
                m.insert(name.clone(), Tensor::zeros(w.raw_dim()));
                v.insert(name.clone(), Tensor::zeros(w.raw_dim()));
            }
        }
        Ok(Self { kind, m, v, t: 0 })
    }

    fn save(&self, path: &Path) -> Result<()> {
        let to_ser = |map: &HashMap<String, Tensor>| {
            map.iter()
                .map(|(k, t)| {
                    (
                        k.clone(),
                        SerTensor {
                            shape: t.shape().to_vec(),
                            data: t.iter().copied().collect(),
                        },
                    )
                })
                .collect()
        };
        let disk = OptDisk {
            t: self.t,
            kind: self.kind.as_str().to_string(),
            m: to_ser(&self.m),
            v: to_ser(&self.v),
        };
        let json = serde_json::to_string(&disk).map_err(|e| Error::fail(e.to_string()))?;
        std::fs::write(path, json)?;
        Ok(())
    }

    fn load(&mut self, path: &Path) -> Result<()> {
        let disk: OptDisk = serde_json::from_str(&std::fs::read_to_string(path)?)
            .map_err(|e| Error::fail(format!("invalid optimizer checkpoint: {e}")))?;
        self.t = disk.t;
        let from_ser = |map: HashMap<String, SerTensor>| {
            map.into_iter()
                .filter_map(|(k, s)| {
                    Tensor::from_shape_vec(ndarray::IxDyn(&s.shape), s.data)
                        .ok()
                        .map(|t| (k, t))
                })
                .collect()
        };
        self.m = from_ser(disk.m);
        self.v = from_ser(disk.v);
        Ok(())
    }

    fn step(&mut self, graph: &mut ExecGraph, names: &[String], lr: f32) -> Result<()> {
        self.t += 1;
        match self.kind {
            OptimizerKind::Sgd => {
                for name in names {
                    let Some(g) = graph.grad(name).cloned() else {
                        continue;
                    };
                    let mut w = graph.param(name)?.clone();
                    Zip::from(&mut w).and(&g).for_each(|w, g| *w -= lr * *g);
                    graph.set_weight(name, w)?;
                }
            }
            OptimizerKind::Adamw => {
                const BETA1: f32 = 0.9;
                const BETA2: f32 = 0.999;
                const EPS: f32 = 1e-8;
                let t = self.t as i32;
                let bc1 = 1.0 - BETA1.powi(t);
                let bc2 = 1.0 - BETA2.powi(t);
                for name in names {
                    let Some(g) = graph.grad(name).cloned() else {
                        continue;
                    };
                    let mut w = graph.param(name)?.clone();
                    let m = self
                        .m
                        .get_mut(name)
                        .ok_or_else(|| Error::fail(format!("missing adam m for {name}")))?;
                    let v = self
                        .v
                        .get_mut(name)
                        .ok_or_else(|| Error::fail(format!("missing adam v for {name}")))?;
                    Zip::from(&mut *m)
                        .and(&mut *v)
                        .and(&g)
                        .and(&mut w)
                        .for_each(|m, v, g, w| {
                            *m = BETA1 * *m + (1.0 - BETA1) * *g;
                            *v = BETA2 * *v + (1.0 - BETA2) * *g * *g;
                            let mhat = *m / bc1;
                            let vhat = *v / bc2;
                            *w -= lr * mhat / (vhat.sqrt() + EPS);
                        });
                    graph.set_weight(name, w)?;
                }
            }
        }
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::data::{Dataset, Example, Label};
    use crate::onnx::{generate_xor_mlp, TensorValue};
    use ndarray::array;

    fn xor_dataset() -> Dataset {
        let rows = [
            ([0.0, 0.0], 0.0),
            ([0.0, 1.0], 1.0),
            ([1.0, 0.0], 1.0),
            ([1.0, 1.0], 0.0),
        ];
        Dataset {
            examples: rows
                .into_iter()
                .map(|(x, y)| Example {
                    inputs: HashMap::from([(
                        "input".into(),
                        TensorValue::F32(array![x[0], x[1]].into_dyn()),
                    )]),
                    label: Label::Float(array![y].into_dyn()),
                    pair: None,
                    negatives: Vec::new(),
                    attention_mask: None,
                    pair_mask: None,
                    teacher_pos: None,
                    teacher_neg: None,
                    replay_emb: None,
                    score: None,
                    text: None,
                    query: None,
                    positive: None,
                })
                .collect(),
        }
    }

    fn pair_ids(q: [i64; 2], p: [i64; 2], teacher_pos: Option<f32>) -> Example {
        Example {
            inputs: HashMap::from([("input_ids".into(), TensorValue::I64(array![q[0], q[1]].into_dyn()))]),
            label: Label::None,
            pair: Some(HashMap::from([(
                "input_ids".into(),
                TensorValue::I64(array![p[0], p[1]].into_dyn()),
            )])),
            negatives: Vec::new(),
            attention_mask: None,
            pair_mask: None,
            teacher_pos,
            teacher_neg: None,
            replay_emb: None,
            score: None,
            text: None,
            query: None,
            positive: None,
        }
    }

    fn tiny_pair_data(teacher: bool) -> Dataset {
        Dataset {
            examples: (0..8)
                .map(|i| {
                    let q = [i as i64, (i + 1) % 16];
                    pair_ids(q, q, teacher.then_some(1.0))
                })
                .collect(),
        }
    }

    #[test]
    fn xor_mse_drops() {
        let mut graph = ExecGraph::load(generate_xor_mlp(1)).unwrap();
        let data = xor_dataset();
        let cfg = TrainConfig {
            max_steps: 250,
            learning_rate: 0.1,
            batch_size: 4,
            log_every: 250,
            ..TrainConfig::default()
        };
        let report = train(&mut graph, &data, None, &cfg, &[], &[]).unwrap();
        assert!(
            report.final_train_loss < report.initial_train_loss * 0.4,
            "initial={} final={}",
            report.initial_train_loss,
            report.final_train_loss
        );
        assert!(
            report.final_train_loss < 0.12,
            "final loss {}",
            report.final_train_loss
        );
        #[cfg(target_os = "macos")]
        if crate::accel::current_kind() == crate::AccelKind::Metal {
            assert!(
                report.accel.starts_with("metal"),
                "expected metal accel, got {}",
                report.accel
            );
        }
    }

    #[test]
    fn lora_on_xor_drops_loss() {
        let mut graph = ExecGraph::load(generate_xor_mlp(1)).unwrap();
        let cfg = TrainConfig {
            max_steps: 400,
            learning_rate: 0.1,
            batch_size: 4,
            log_every: 400,
            lora_rank: Some(4),
            lora_alpha: 4.0,
            lora_targets: vec!["W2".into(), "gemm2".into()],
            ..TrainConfig::default()
        };
        crate::lora::inject(&mut graph, &cfg).unwrap();
        assert!(
            graph.weight_names.iter().any(|n| n.starts_with("lora_")),
            "expected LoRA tensors"
        );
        let data = xor_dataset();
        let report = train(&mut graph, &data, None, &cfg, &[], &[]).unwrap();
        assert!(
            report.trainable.iter().all(|n| n.starts_with("lora_")),
            "optimizer should only touch LoRA: {:?}",
            report.trainable
        );
        assert!(
            report.final_train_loss < report.initial_train_loss,
            "initial={} final={}",
            report.initial_train_loss,
            report.final_train_loss
        );
    }

    #[test]
    fn infonce_tiny_embedder() {
        let mut graph = ExecGraph::load(crate::onnx::generate_tiny_embedder(2, 16, 8)).unwrap();
        let cfg = TrainConfig {
            loss: crate::workspace::LossKind::InfoNce,
            max_steps: 40,
            learning_rate: 0.05,
            batch_size: 4,
            log_every: 40,
            pool: crate::workspace::PoolKind::LastToken,
            l2_normalize: true,
            temperature: 0.07,
            ..TrainConfig::default()
        };
        let report = train(&mut graph, &tiny_pair_data(false), None, &cfg, &[], &[]).unwrap();
        assert!(
            report.final_train_loss <= report.initial_train_loss + 0.05,
            "initial={} final={}",
            report.initial_train_loss,
            report.final_train_loss
        );
    }

    #[test]
    fn xor_sgd_runs() {
        let mut graph = ExecGraph::load(generate_xor_mlp(1)).unwrap();
        let cfg = TrainConfig {
            optimizer: OptimizerKind::Sgd,
            max_steps: 40,
            learning_rate: 0.1,
            batch_size: 4,
            log_every: 40,
            ..TrainConfig::default()
        };
        let report = train(&mut graph, &xor_dataset(), None, &cfg, &[], &[]).unwrap();
        assert!(report.final_train_loss.is_finite());
        assert_eq!(report.optimizer, "sgd");
    }

    #[test]
    fn mixed_precision_fp16_xor_trains() {
        let mut graph = ExecGraph::load(generate_xor_mlp(1)).unwrap();
        let cfg = TrainConfig {
            max_steps: 80,
            learning_rate: 0.1,
            batch_size: 4,
            log_every: 80,
            mixed_precision: crate::workspace::MixedPrecision::Fp16,
            ..TrainConfig::default()
        };
        let report = train(&mut graph, &xor_dataset(), None, &cfg, &[], &[]).unwrap();
        assert!(report.final_train_loss.is_finite());
        assert_eq!(report.mixed_precision, "fp16");
        assert!(
            report.final_train_loss < report.initial_train_loss,
            "initial={} final={}",
            report.initial_train_loss,
            report.final_train_loss
        );
    }

    #[test]
    fn xor_trains_with_checkpointing_and_microbatch() {
        let mut graph = ExecGraph::load(generate_xor_mlp(1)).unwrap();
        let cfg = TrainConfig {
            max_steps: 40,
            learning_rate: 0.1,
            batch_size: 4,
            log_every: 40,
            microbatch_size: Some(2),
            grad_accum_steps: 2,
            activation_checkpointing: true,
            checkpoint_every: 1,
            ..TrainConfig::default()
        };
        let report = train(&mut graph, &xor_dataset(), None, &cfg, &[], &[]).unwrap();
        assert!(report.final_train_loss.is_finite());
    }

    #[test]
    fn embedding_losses_finite_on_tiny_embedder() {
        let losses = [
            crate::workspace::LossKind::Cosine,
            crate::workspace::LossKind::Triplet,
            crate::workspace::LossKind::SupCon,
            crate::workspace::LossKind::Distill,
        ];
        for loss in losses {
            let mut graph = ExecGraph::load(crate::onnx::generate_tiny_embedder(2, 16, 8)).unwrap();
            let cfg = TrainConfig {
                loss,
                max_steps: 8,
                learning_rate: 0.05,
                batch_size: 4,
                log_every: 8,
                pool: crate::workspace::PoolKind::LastToken,
                l2_normalize: true,
                distill_weight: 0.5,
                ..TrainConfig::default()
            };
            let data = tiny_pair_data(loss == crate::workspace::LossKind::Distill);
            let report = train(&mut graph, &data, None, &cfg, &[], &[]).unwrap();
            assert!(
                report.final_train_loss.is_finite(),
                "{:?} loss not finite: {}",
                loss,
                report.final_train_loss
            );
        }
    }
}
