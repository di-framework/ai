use std::fs;
use std::path::{Path, PathBuf};

use sha2::{Digest, Sha256};

use crate::data::Dataset;
use crate::error::{Error, Result};
use crate::graph::ExecGraph;
use crate::onnx::{self, write_updated_model};
use crate::train::{train_with, TrainExtras, TrainReport};
use crate::workspace::{self, LoraExport, Workspace};

#[derive(Debug, Clone)]
pub struct RunResult {
    pub workspace: PathBuf,
    pub dist_model: PathBuf,
    pub dist_metrics: PathBuf,
    pub report: TrainReport,
}

pub fn run_workspace(root: impl AsRef<Path>) -> Result<RunResult> {
    let ws = workspace::load(root)?;
    run_loaded(&ws)
}

fn run_loaded(ws: &Workspace) -> Result<RunResult> {
    let bytes = onnx::load_bytes(&ws.model)?;
    let mut graph = ExecGraph::load(bytes)?;
    crate::lora::inject(&mut graph, &ws.config)?;

    let tok = crate::tokenize::Tokenizer::load(&ws.config, &ws.root)?;
    let train_data = Dataset::load(
        &ws.train_data,
        &graph.inputs,
        &graph.input_is_int,
        ws.config.loss,
        tok.as_ref(),
    )?;
    let eval_data = match &ws.eval_data {
        Some(path) => Some(Dataset::load(
            path,
            &graph.inputs,
            &graph.input_is_int,
            ws.config.loss,
            tok.as_ref(),
        )?),
        None => None,
    };

    let data_bytes = fs::read(&ws.train_data)?;
    let extras = TrainExtras {
        config_hash: Some(sha256(&ws.train_toml_bytes)),
        data_hash: Some(sha256(&data_bytes)),
        resume_from: ws.config.resume.then(|| ws.dist_optimizer()),
        save_optimizer: Some(ws.dist_optimizer()),
    };

    let report = train_with(
        &mut graph,
        &train_data,
        eval_data.as_ref(),
        &ws.config,
        &ws.frozen,
        &ws.trainable,
        extras,
    )?;

    fs::create_dir_all(ws.dist_dir())?;
    let dist_model = ws.dist_model();
    let dist_metrics = ws.dist_metrics();

    match ws.config.lora_export {
        LoraExport::Adapter if !graph.lora.is_empty() => {
            write_updated_model(&graph.original_bytes, &graph.weights(), &dist_model)?;
            write_adapter(&graph, &ws.dist_adapter())?;
        }
        _ => {
            crate::lora::merge_into_weights(&mut graph)?;
            write_updated_model(&graph.original_bytes, &graph.weights(), &dist_model)?;
        }
    }

    let json = serde_json::to_string_pretty(&report).map_err(|err| Error::fail(err.to_string()))?;
    fs::write(&dist_metrics, json)?;

    Ok(RunResult {
        workspace: ws.root.clone(),
        dist_model,
        dist_metrics,
        report,
    })
}

fn sha256(bytes: &[u8]) -> String {
    let mut h = Sha256::new();
    h.update(bytes);
    h.finalize().iter().map(|b| format!("{b:02x}")).collect()
}

fn write_adapter(graph: &ExecGraph, dest: &Path) -> Result<()> {
    let weights = crate::lora::adapter_weights(graph);
    let json_path = dest.with_extension("json");
    let mut obj = serde_json::Map::new();
    for (k, t) in &weights {
        obj.insert(
            k.clone(),
            serde_json::json!({
                "shape": t.shape(),
                "data": t.iter().copied().collect::<Vec<f32>>(),
            }),
        );
    }
    fs::write(
        &json_path,
        serde_json::to_string_pretty(&obj).map_err(|e| Error::fail(e.to_string()))?,
    )?;
    // Tiny ONNX: Identity graph over each adapter tensor as an initializer/output.
    write_adapter_onnx(&weights, dest)
}

fn write_adapter_onnx(
    weights: &std::collections::HashMap<String, crate::tensor::Tensor>,
    dest: &Path,
) -> Result<()> {
    use onnx_rs::ast::*;
    let names: Vec<&'static str> = weights
        .keys()
        .map(|k| Box::leak(k.clone().into_boxed_str()) as &'static str)
        .collect();
    let mut initializer = Vec::new();
    let mut output = Vec::new();
    let mut node = Vec::new();
    for name in &names {
        let t = &weights[*name];
        let dims: Vec<i64> = t.shape().iter().map(|d| *d as i64).collect();
        let data: Vec<f32> = t.iter().copied().collect();
        initializer.push(TensorProto::from_f32(name, dims.clone(), data));
        output.push(ValueInfo {
            name,
            r#type: Some(TypeProto {
                value: Some(TypeValue::Tensor(TensorTypeProto {
                    elem_type: DataType::Float,
                    shape: Some(TensorShape {
                        dim: dims
                            .into_iter()
                            .map(|value| TensorShapeDimension {
                                value: Dimension::Value(value),
                                denotation: "",
                            })
                            .collect(),
                    }),
                })),
                denotation: "",
            }),
            ..Default::default()
        });
        node.push(Node {
            name,
            op_type: OpType::Identity,
            input: vec![*name],
            output: vec![*name],
            ..Default::default()
        });
    }
    let model = Model {
        ir_version: 9,
        producer_name: "di-ml",
        producer_version: "0.1.0",
        opset_import: vec![OperatorSetId {
            domain: "",
            version: 17,
        }],
        graph: Some(Graph {
            name: "lora_adapter",
            node,
            initializer,
            output,
            ..Default::default()
        }),
        ..Default::default()
    };
    fs::write(dest, onnx_rs::encode(&model))?;
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::graph::ExecGraph;
    use crate::scaffold::{scaffold, InitSpec};
    use crate::workspace::TrainConfig;
    use std::collections::HashMap;
    use tempfile::tempdir;

    #[test]
    fn xor_workspace_writes_dist() {
        let dir = tempdir().unwrap();
        let root = dir.path().join("ws");
        scaffold(&InitSpec {
            root: root.clone(),
            config: TrainConfig {
                max_steps: 200,
                learning_rate: 0.1,
                batch_size: 4,
                log_every: 200,
                ..TrainConfig::default()
            },
            overwrite: false,
        })
        .unwrap();

        let result = run_workspace(&root).unwrap();
        assert!(result.dist_model.is_file());
        assert!(result.dist_metrics.is_file());
        assert!(
            result.report.final_train_loss < 0.15,
            "final loss {}",
            result.report.final_train_loss
        );
        let exported = onnx::load_bytes(&result.dist_model).unwrap();
        assert!(exported.len() > 32);
        assert!(result.report.config_hash.is_some());
        assert!(result.report.data_hash.is_some());
    }

    #[test]
    fn train_then_infer_xor_from_dist() {
        let dir = tempdir().unwrap();
        let root = dir.path().join("ws");
        scaffold(&InitSpec {
            root: root.clone(),
            config: TrainConfig {
                max_steps: 400,
                learning_rate: 0.1,
                batch_size: 4,
                log_every: 400,
                ..TrainConfig::default()
            },
            overwrite: false,
        })
        .unwrap();

        let result = run_workspace(&root).unwrap();
        let mut session = crate::Session::from_path(&result.dist_model).unwrap();
        assert_eq!(session.inputs(), &["input".to_string()]);
        assert_eq!(session.outputs(), &["output".to_string()]);

        let rows = [
            ([0.0f32, 0.0], 0),
            ([0.0, 1.0], 1),
            ([1.0, 0.0], 1),
            ([1.0, 1.0], 0),
        ];
        let mut feeds = HashMap::new();
        feeds.insert(
            "input".into(),
            ndarray::array![[0.0f32, 0.0], [0.0, 1.0], [1.0, 0.0], [1.0, 1.0]].into_dyn(),
        );
        let out = session.run_f32(&feeds).unwrap();
        let y = &out["output"];
        assert_eq!(y.shape(), &[4, 1]);
        for (i, (_, want)) in rows.iter().enumerate() {
            let pred = y[[i, 0]];
            let got = if pred >= 0.5 { 1 } else { 0 };
            assert_eq!(
                got, *want,
                "XOR row {i}: pred={pred:.4} want class {want}"
            );
        }
    }

    #[test]
    fn lora_merge_dist_has_no_adapter_nodes() {
        let dir = tempdir().unwrap();
        let root = dir.path().join("ws");
        scaffold(&InitSpec {
            root: root.clone(),
            config: TrainConfig {
                max_steps: 200,
                learning_rate: 0.1,
                batch_size: 4,
                log_every: 200,
                lora_rank: Some(4),
                lora_alpha: 4.0,
                lora_targets: vec!["W2".into(), "gemm2".into()],
                ..TrainConfig::default()
            },
            overwrite: false,
        })
        .unwrap();

        let result = run_workspace(&root).unwrap();
        assert!(
            result.report.trainable.iter().all(|n| n.starts_with("lora_")),
            "expected LoRA-only train: {:?}",
            result.report.trainable
        );

        let mut session = crate::Session::from_path(&result.dist_model).unwrap();
        assert_eq!(session.inputs(), &["input".to_string()]);
        assert_eq!(session.outputs(), &["output".to_string()]);
        let bytes = onnx::load_bytes(&result.dist_model).unwrap();
        let graph = ExecGraph::load(bytes).unwrap();
        assert!(
            graph.weight_names.iter().all(|n| !n.starts_with("lora_")),
            "merged export must not keep adapter initializers: {:?}",
            graph.weight_names
        );
        assert!(graph.lora.is_empty());

        let mut feeds = HashMap::new();
        feeds.insert(
            "input".into(),
            ndarray::array![[0.0f32, 1.0]].into_dyn(),
        );
        let out = session.run_f32(&feeds).unwrap();
        assert_eq!(out["output"].shape(), &[1, 1]);
        assert!(out["output"][[0, 0]].is_finite());
    }

    fn write_tiny_embedder_workspace(root: &std::path::Path, cfg: TrainConfig) {
        fs::create_dir_all(root.join(workspace::DATA_DIR)).unwrap();
        fs::write(
            root.join(workspace::MODEL_FILE),
            crate::onnx::generate_tiny_embedder(1, 16, 8),
        )
        .unwrap();
        fs::write(root.join(workspace::TRAIN_TOML), cfg.to_toml().unwrap()).unwrap();
        let mut train = String::new();
        let mut eval = String::new();
        for i in 0..8 {
            let q = format!("[{},{}]", i, (i + 1) % 16);
            train.push_str(&format!(
                "{{\"input_ids\":{q},\"positive\":{q}}}\n"
            ));
            let score = i as f32 / 7.0;
            eval.push_str(&format!(
                "{{\"input_ids\":{q},\"positive\":{q},\"score\":{score}}}\n"
            ));
        }
        fs::write(root.join(workspace::DATA_DIR).join(workspace::TRAIN_JSONL), train).unwrap();
        fs::write(root.join(workspace::DATA_DIR).join(workspace::EVAL_JSONL), eval).unwrap();
    }

    #[test]
    fn spearman_written_to_metrics() {
        let dir = tempdir().unwrap();
        let root = dir.path().join("ws");
        write_tiny_embedder_workspace(
            &root,
            TrainConfig {
                loss: crate::workspace::LossKind::InfoNce,
                max_steps: 12,
                learning_rate: 0.05,
                batch_size: 4,
                log_every: 12,
                pool: crate::workspace::PoolKind::LastToken,
                l2_normalize: true,
                eval_metric: crate::workspace::EvalMetric::Spearman,
                ..TrainConfig::default()
            },
        );
        let result = run_workspace(&root).unwrap();
        assert!(result.report.eval_spearman.is_some());
        let rho = result.report.eval_spearman.unwrap();
        assert!(rho.is_finite(), "spearman {rho}");
        let metrics = fs::read_to_string(&result.dist_metrics).unwrap();
        assert!(metrics.contains("eval-spearman"));
    }

    #[test]
    fn lora_adapter_export_writes_file() {
        let dir = tempdir().unwrap();
        let root = dir.path().join("ws");
        scaffold(&InitSpec {
            root: root.clone(),
            config: TrainConfig {
                max_steps: 40,
                learning_rate: 0.1,
                batch_size: 4,
                log_every: 40,
                lora_rank: Some(4),
                lora_alpha: 4.0,
                lora_targets: vec!["W2".into(), "gemm2".into()],
                lora_export: crate::workspace::LoraExport::Adapter,
                ..TrainConfig::default()
            },
            overwrite: false,
        })
        .unwrap();

        let result = run_workspace(&root).unwrap();
        let adapter = root.join(workspace::DIST_DIR).join("adapter.onnx");
        assert!(
            adapter.is_file(),
            "expected {}",
            adapter.display()
        );
        assert!(result.dist_model.is_file());
        let graph = ExecGraph::load(onnx::load_bytes(&result.dist_model).unwrap()).unwrap();
        assert!(
            graph.weight_names.iter().all(|n| !n.starts_with("lora_")),
            "adapter export should leave base graph without lora_* initializers: {:?}",
            graph.weight_names
        );
    }

    #[test]
    fn resume_reloads_optimizer() {
        let dir = tempdir().unwrap();
        let root = dir.path().join("ws");
        let mut cfg = TrainConfig {
            max_steps: 30,
            learning_rate: 0.1,
            batch_size: 4,
            log_every: 30,
            ..TrainConfig::default()
        };
        scaffold(&InitSpec {
            root: root.clone(),
            config: cfg.clone(),
            overwrite: false,
        })
        .unwrap();
        let first = run_workspace(&root).unwrap();
        assert!(root.join(workspace::DIST_DIR).join("optimizer.json").is_file());
        let loss_after_first = first.report.final_train_loss;

        cfg.resume = true;
        fs::write(root.join(workspace::TRAIN_TOML), cfg.to_toml().unwrap()).unwrap();
        let second = run_workspace(&root).unwrap();
        assert!(second.report.final_train_loss.is_finite());
        assert!(
            second.report.final_train_loss <= loss_after_first + 0.05,
            "resume should not reset training; first={} second={}",
            loss_after_first,
            second.report.final_train_loss
        );
    }
}
