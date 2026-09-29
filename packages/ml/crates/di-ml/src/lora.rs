//! LoRA injection on named Gemm/MatMul nodes.

use rand::rngs::StdRng;
use rand::{RngExt, SeedableRng};

use crate::error::{Error, Result};
use crate::graph::{ExecGraph, ExecNode, LoraSlot};
use crate::onnx::TensorValue;
use crate::tensor::Tensor;
use crate::workspace::TrainConfig;

pub fn inject(graph: &mut ExecGraph, cfg: &TrainConfig) -> Result<Vec<String>> {
    let Some(rank) = cfg.lora_rank else {
        return Ok(Vec::new());
    };
    let targets = cfg.lora_targets_effective();
    if targets.is_empty() {
        return Ok(Vec::new());
    }
    let scale = cfg.lora_alpha / rank as f32;
    let mut rng = StdRng::seed_from_u64(cfg.seed.wrapping_add(17));
    let mut added = Vec::new();
    let mut new_nodes = Vec::new();
    let nodes = graph.nodes.clone();

    for node in &nodes {
        if node.op != "Gemm" && node.op != "MatMul" {
            continue;
        }
        if !matches_target(node, graph, &targets) {
            continue;
        }
        let x_name = node.inputs.first().cloned().ok_or_else(|| {
            Error::fail(format!("LoRA: {} has no activation input", node.name))
        })?;
        let w_name = node.inputs.get(1).cloned().ok_or_else(|| {
            Error::fail(format!("LoRA: {} has no weight input", node.name))
        })?;
        let w = graph.param(&w_name)?;
        if w.ndim() != 2 {
            continue;
        }
        let trans_b = node.trans_b && node.op == "Gemm";
        let (in_f, out_f) = if trans_b {
            (w.shape()[1], w.shape()[0])
        } else {
            (w.shape()[0], w.shape()[1])
        };
        let a_name = format!("lora_A.{w_name}");
        let b_name = format!("lora_B.{w_name}");
        if graph.weight_names.contains(&a_name) {
            continue;
        }
        let a = kaiming(&mut rng, in_f, rank);
        let b = Tensor::zeros(ndarray::IxDyn(&[rank, out_f]));
        graph
            .values
            .insert(a_name.clone(), TensorValue::F32(a));
        graph
            .values
            .insert(b_name.clone(), TensorValue::F32(b));
        graph.weight_names.insert(a_name.clone());
        graph.weight_names.insert(b_name.clone());

        let y_orig = node
            .outputs
            .first()
            .cloned()
            .ok_or_else(|| Error::fail("LoRA gemm has no output"))?;
        let y_base = format!("{y_orig}__base");
        // rewrite original node output
        if let Some(n) = graph.nodes.iter_mut().find(|n| n.name == node.name) {
            n.outputs[0] = y_base.clone();
        }

        let mid = format!("{y_orig}__lora_mid");
        let down = format!("{y_orig}__lora_down");
        let scaled = format!("{y_orig}__lora");
        new_nodes.push(ExecNode {
            name: format!("lora_A/{}", node.name),
            op: "MatMul".into(),
            inputs: vec![x_name.clone(), a_name.clone()],
            outputs: vec![mid.clone()],
            trans_a: false,
            trans_b: false,
            alpha: 1.0,
            beta: 1.0,
            axis: 0,
            perm: None,
            axes: None,
            keepdims: true,
            epsilon: 1e-5,
            min: None,
            max: None,
            split: None,
            to: 1,
            approximate: false,
        });
        new_nodes.push(ExecNode {
            name: format!("lora_B/{}", node.name),
            op: "MatMul".into(),
            inputs: vec![mid, b_name.clone()],
            outputs: vec![down.clone()],
            trans_a: false,
            trans_b: false,
            alpha: 1.0,
            beta: 1.0,
            axis: 0,
            perm: None,
            axes: None,
            keepdims: true,
            epsilon: 1e-5,
            min: None,
            max: None,
            split: None,
            to: 1,
            approximate: false,
        });
        // scale via Mul with a constant initializer
        let scale_name = format!("lora_scale.{w_name}");
        graph.values.insert(
            scale_name.clone(),
            TensorValue::F32(Tensor::from_elem(ndarray::IxDyn(&[]), scale)),
        );
        new_nodes.push(ExecNode {
            name: format!("lora_scale/{}", node.name),
            op: "Mul".into(),
            inputs: vec![down, scale_name],
            outputs: vec![scaled.clone()],
            trans_a: false,
            trans_b: false,
            alpha: 1.0,
            beta: 1.0,
            axis: 0,
            perm: None,
            axes: None,
            keepdims: true,
            epsilon: 1e-5,
            min: None,
            max: None,
            split: None,
            to: 1,
            approximate: false,
        });
        new_nodes.push(ExecNode {
            name: format!("lora_add/{}", node.name),
            op: "Add".into(),
            inputs: vec![y_base, scaled],
            outputs: vec![y_orig],
            trans_a: false,
            trans_b: false,
            alpha: 1.0,
            beta: 1.0,
            axis: 0,
            perm: None,
            axes: None,
            keepdims: true,
            epsilon: 1e-5,
            min: None,
            max: None,
            split: None,
            to: 1,
            approximate: false,
        });

        graph.lora.push(LoraSlot {
            gemm: node.name.clone(),
            weight: w_name,
            a_name: a_name.clone(),
            b_name: b_name.clone(),
            trans_b,
            scale,
        });
        added.push(a_name);
        added.push(b_name);
    }

    // insert LoRA nodes immediately after their host gemm
    if !new_nodes.is_empty() {
        let mut merged = Vec::new();
        for node in graph.nodes.drain(..) {
            let name = node.name.clone();
            merged.push(node);
            let extras: Vec<ExecNode> = new_nodes
                .iter()
                .filter(|n| n.name.ends_with(&format!("/{name}")) || n.name.ends_with(&name))
                .cloned()
                .collect();
            merged.extend(extras);
        }
        // if filter missed, append remaining
        for n in new_nodes {
            if !merged.iter().any(|m| m.name == n.name) {
                merged.push(n);
            }
        }
        graph.nodes = merged;
    }
    Ok(added)
}

fn matches_target(node: &ExecNode, graph: &ExecGraph, targets: &[String]) -> bool {
    let hay = format!(
        "{} {} {}",
        node.name,
        node.inputs.get(1).cloned().unwrap_or_default(),
        node.outputs.first().cloned().unwrap_or_default()
    );
    let hay_l = hay.to_ascii_lowercase();
    targets.iter().any(|t| {
        let t = t.to_ascii_lowercase();
        hay_l.contains(&t) || crate::ops::matches_trainable(&t, &hay_l)
    }) && graph.weight_names.contains(node.inputs.get(1).map(|s| s.as_str()).unwrap_or(""))
}

fn kaiming(rng: &mut StdRng, rows: usize, cols: usize) -> Tensor {
    let std = (1.0 / rows as f32).sqrt();
    let data: Vec<f32> = (0..rows * cols)
        .map(|_| rng.random_range(-std..std))
        .collect();
    Tensor::from_shape_vec(ndarray::IxDyn(&[rows, cols]), data).expect("lora A shape")
}

/// Fold LoRA into the base Gemm weight: W += scale * A @ B (layout-aware).
pub fn merge_into_weights(graph: &mut ExecGraph) -> Result<()> {
    let slots = graph.lora.clone();
    for slot in slots {
        let a = graph.param(&slot.a_name)?.clone();
        let b = graph.param(&slot.b_name)?.clone();
        let delta = crate::tensor::matmul(&a, &b)?;
        let delta = crate::tensor::scale(&delta, slot.scale)?;
        let mut w = graph.param(&slot.weight)?.clone();
        if slot.trans_b {
            let dt = crate::tensor::transpose_2d(&delta)?;
            w = crate::tensor::add(&w, &dt)?;
        } else {
            w = crate::tensor::add(&w, &delta)?;
        }
        graph.set_weight(&slot.weight, w)?;
    }
    Ok(())
}

pub fn adapter_weights(graph: &ExecGraph) -> std::collections::HashMap<String, Tensor> {
    let mut out = std::collections::HashMap::new();
    for slot in &graph.lora {
        if let Ok(t) = graph.param(&slot.a_name) {
            out.insert(slot.a_name.clone(), t.clone());
        }
        if let Ok(t) = graph.param(&slot.b_name) {
            out.insert(slot.b_name.clone(), t.clone());
        }
    }
    out
}
