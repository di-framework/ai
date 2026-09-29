mod checkpoint;
mod eval;
mod tape;
mod vjp;

use std::collections::{HashMap, HashSet};

use ndarray::ArrayD;
use onnx_rs::ast::{OpType, DataType};

use crate::error::{Error, Result};
use crate::onnx::{
    attr_f, attr_f_opt, attr_i, attr_i_opt, attr_ints, input_elem_type, tensor_from_proto,
    TensorValue,
};
use crate::ops;
use crate::tensor::{add_assign, Tensor};

pub use tape::{Tape, UnaryKind};

#[derive(Debug, Clone)]
pub struct ExecNode {
    pub name: String,
    pub op: String,
    pub inputs: Vec<String>,
    pub outputs: Vec<String>,
    pub trans_a: bool,
    pub trans_b: bool,
    pub alpha: f32,
    pub beta: f32,
    pub axis: i64,
    pub perm: Option<Vec<i64>>,
    pub axes: Option<Vec<i64>>,
    pub keepdims: bool,
    pub epsilon: f32,
    pub min: Option<f32>,
    pub max: Option<f32>,
    pub split: Option<Vec<i64>>,
    pub to: i64,
    pub approximate: bool,
}

#[derive(Debug, Clone)]
pub struct LoraSlot {
    pub gemm: String,
    pub weight: String,
    pub a_name: String,
    pub b_name: String,
    pub trans_b: bool,
    pub scale: f32,
}

#[derive(Debug, Clone)]
pub struct ExecGraph {
    pub nodes: Vec<ExecNode>,
    pub values: HashMap<String, TensorValue>,
    pub inputs: Vec<String>,
    pub outputs: Vec<String>,
    pub weight_names: HashSet<String>,
    pub original_bytes: Vec<u8>,
    pub input_is_int: HashSet<String>,
    pub lora: Vec<LoraSlot>,
    pub lora_scale: f32,
    pub skip_grad: HashSet<String>,
    pub checkpoint: bool,
    pub checkpoint_every: usize,
    pub peak_bytes: usize,
    tape: Vec<Tape>,
    grads: HashMap<String, Tensor>,
    recording: bool,
    inference: bool,
}

impl ExecGraph {
    pub fn load(bytes: Vec<u8>) -> Result<Self> {
        let model = onnx_rs::parse(&bytes).map_err(|err| Error::fail(err.to_string()))?;
        let graph = model
            .graph
            .as_ref()
            .ok_or_else(|| Error::usage("ONNX model has no graph"))?;

        let report = ops::lint_nodes(&graph.node);
        ops::reject_if_missing(&report)?;

        let mut values = HashMap::new();
        let mut weight_names = HashSet::new();
        for init in &graph.initializer {
            match tensor_from_proto(init)? {
                TensorValue::F32(t) => {
                    weight_names.insert(init.name().to_string());
                    values.insert(init.name().to_string(), TensorValue::F32(t));
                }
                other => {
                    values.insert(init.name().to_string(), other);
                }
            }
        }

        let mut input_is_int = HashSet::new();
        let inputs: Vec<String> = graph
            .non_init_inputs()
            .iter()
            .map(|v| {
                if matches!(
                    input_elem_type(v),
                    Some(DataType::Int64 | DataType::Int32 | DataType::Bool)
                ) {
                    input_is_int.insert(v.name.to_string());
                }
                v.name.to_string()
            })
            .collect();
        if inputs.is_empty() {
            return Err(Error::usage(
                "ONNX graph has no runtime inputs (every input is an initializer)",
            ));
        }
        let outputs: Vec<String> = graph.output.iter().map(|v| v.name.to_string()).collect();
        if outputs.is_empty() {
            return Err(Error::usage("ONNX graph has no outputs"));
        }

        let mut nodes = Vec::new();
        for node in &graph.node {
            if matches!(node.op_type, OpType::Constant) {
                let value = constant_value(&node.attribute, node.name)?;
                if let Some(out) = node.output.first() {
                    values.insert((*out).to_string(), value);
                }
                continue;
            }
            nodes.push(parse_node(node));
        }

        Ok(Self {
            nodes,
            values,
            inputs,
            outputs,
            weight_names,
            original_bytes: bytes,
            input_is_int,
            lora: Vec::new(),
            lora_scale: 1.0,
            skip_grad: HashSet::new(),
            checkpoint: false,
            checkpoint_every: 8,
            peak_bytes: 0,
            tape: Vec::new(),
            grads: HashMap::new(),
            recording: true,
            inference: false,
        })
    }

    pub fn weights(&self) -> HashMap<String, Tensor> {
        self.weight_names
            .iter()
            .filter_map(|name| {
                self.values
                    .get(name)
                    .and_then(|v| v.as_f32().ok())
                    .map(|t| (name.clone(), t.clone()))
            })
            .collect()
    }

    pub fn param(&self, name: &str) -> Result<&Tensor> {
        self.f32(name)
    }

    pub fn set_weight(&mut self, name: &str, value: Tensor) -> Result<()> {
        self.values
            .insert(name.to_string(), TensorValue::F32(value));
        Ok(())
    }

    /// Forward-only: no tape, no checkpoint segments. Used after `dist/model.onnx` export.
    pub fn set_inference_mode(&mut self, on: bool) {
        self.inference = on;
        if on {
            self.checkpoint = false;
            self.tape.clear();
            self.grads.clear();
            self.recording = false;
        }
    }

    pub fn set_checkpointing(&mut self, on: bool, every: usize) {
        self.checkpoint = on;
        self.checkpoint_every = every.max(1);
    }

    pub fn zero_grad(&mut self) {
        self.grads.clear();
    }

    pub fn nbytes(&self) -> usize {
        let v: usize = self.values.values().map(|t| t.nbytes()).sum();
        let g: usize = self.grads.values().map(|t| t.len() * 4).sum();
        v + g
    }

    pub fn note_peak(&mut self) {
        let n = self.nbytes();
        if n > self.peak_bytes {
            self.peak_bytes = n;
        }
        crate::accel::note_bytes(n);
    }

    pub fn forward(&mut self, inputs: &HashMap<String, Tensor>) -> Result<HashMap<String, Tensor>> {
        let feeds: HashMap<String, TensorValue> = inputs
            .iter()
            .map(|(k, v)| (k.clone(), TensorValue::F32(v.clone())))
            .collect();
        self.forward_values(&feeds)
    }

    pub fn forward_values(
        &mut self,
        inputs: &HashMap<String, TensorValue>,
    ) -> Result<HashMap<String, Tensor>> {
        self.tape.clear();
        for name in &self.inputs {
            let t = inputs.get(name).ok_or_else(|| {
                Error::usage(format!(
                    "batch is missing model input {name:?}; expected {:?}",
                    self.inputs
                ))
            })?;
            self.values.insert(name.clone(), t.clone());
        }

        let nodes = self.nodes.clone();
        if self.inference {
            self.recording = false;
            for node in &nodes {
                self.eval_node(node)?;
            }
        } else if !self.checkpoint {
            self.recording = true;
            for node in &nodes {
                self.eval_node(node)?;
            }
        } else {
            self.recording = false;
            let segs = checkpoint::segments(&nodes, true, self.checkpoint_every);
            for range in segs {
                let needed = checkpoint::live_in(&nodes, range.clone());
                let mut saved = HashMap::new();
                for name in &needed {
                    if let Some(v) = self.values.get(name) {
                        saved.insert(name.clone(), v.clone());
                    }
                }
                let seg_nodes = nodes[range.clone()].to_vec();
                for node in &seg_nodes {
                    self.eval_node(node)?;
                }
                let keep = checkpoint::live_after(&nodes, range.end, &self.outputs);
                self.drop_dead(&keep);
                self.tape.push(Tape::Segment {
                    range,
                    nodes: seg_nodes,
                    saved,
                });
            }
            self.recording = true;
        }

        self.note_peak();

        let mut out = HashMap::new();
        for name in &self.outputs {
            out.insert(name.clone(), self.f32(name)?.clone());
        }
        Ok(out)
    }

    pub fn forward_any(
        &mut self,
        inputs: &HashMap<String, TensorValue>,
    ) -> Result<HashMap<String, TensorValue>> {
        let _ = self.forward_values(inputs)?;
        let mut out = HashMap::new();
        for name in &self.outputs {
            let v = self
                .values
                .get(name)
                .ok_or_else(|| Error::fail(format!("missing output {name}")))?
                .clone();
            out.insert(name.clone(), v);
        }
        Ok(out)
    }

    pub fn backward(&mut self, output_grads: HashMap<String, Tensor>) -> Result<()> {
        self.backward_acc(output_grads, false)
    }

    pub fn backward_acc(
        &mut self,
        output_grads: HashMap<String, Tensor>,
        accumulate: bool,
    ) -> Result<()> {
        if !accumulate {
            self.grads.clear();
        }
        for (name, g) in output_grads {
            self.add_grad(&name, g);
        }
        let tape = self.tape.clone();
        for entry in tape.iter().rev() {
            self.backward_entry(entry)?;
        }
        self.note_peak();
        Ok(())
    }

    pub fn grad(&self, name: &str) -> Option<&Tensor> {
        self.grads.get(name)
    }

    fn drop_dead(&mut self, keep: &HashSet<String>) {
        let weights = &self.weight_names;
        let inputs: HashSet<&str> = self.inputs.iter().map(|s| s.as_str()).collect();
        self.values.retain(|k, v| {
            if weights.contains(k) || inputs.contains(k.as_str()) || keep.contains(k) {
                return true;
            }
            !matches!(v, TensorValue::F32(_))
        });
    }

    fn push_tape(&mut self, t: Tape) {
        if self.recording {
            self.tape.push(t);
        }
    }

    pub(super) fn f32(&self, name: &str) -> Result<&Tensor> {
        self.values
            .get(name)
            .ok_or_else(|| Error::fail(format!("unknown tensor {name:?}")))?
            .as_f32()
    }

    pub(super) fn i64(&self, name: &str) -> Result<ArrayD<i64>> {
        let v = self
            .values
            .get(name)
            .ok_or_else(|| Error::fail(format!("unknown tensor {name:?}")))?;
        v.to_i64()
    }

    pub(super) fn value(&self, name: &str) -> Result<&TensorValue> {
        self.values
            .get(name)
            .ok_or_else(|| Error::fail(format!("unknown tensor {name:?}")))
    }

    pub(super) fn add_grad(&mut self, name: &str, g: Tensor) {
        if self.skip_grad.contains(name) && self.weight_names.contains(name) {
            return;
        }
        match self.grads.get_mut(name) {
            Some(existing) => add_assign(existing, &g),
            None => {
                self.grads.insert(name.to_string(), g);
            }
        }
    }

    pub(super) fn i64_shape(&self, name: &str, numel: usize) -> Result<Vec<usize>> {
        match self.values.get(name) {
            Some(TensorValue::I64(t)) => infer_shape(t.iter().copied().collect(), numel),
            Some(TensorValue::F32(t)) => infer_shape(t.iter().map(|v| *v as i64).collect(), numel),
            Some(TensorValue::Bool(_)) => Err(Error::fail(format!("shape tensor {name:?} is bool"))),
            None => Err(Error::fail(format!("unknown shape tensor {name:?}"))),
        }
    }

    pub(super) fn insert_f32(&mut self, name: &str, t: Tensor) {
        self.values.insert(name.to_string(), TensorValue::F32(t));
    }

    pub(super) fn insert_value(&mut self, name: &str, t: TensorValue) {
        self.values.insert(name.to_string(), t);
    }
}

fn parse_node(node: &onnx_rs::ast::Node<'_>) -> ExecNode {
    let op = node.op_type.as_str().to_string();
    let axis_default = match node.op_type {
        OpType::Softmax | OpType::LayerNormalization => -1,
        OpType::Flatten => 1,
        OpType::Concat => 1,
        OpType::Gather | OpType::GatherElements | OpType::Split => 0,
        _ => match op.as_str() {
            "RMSNorm" | "SimplifiedLayerNormalization" | "SiLU" | "Silu" => -1,
            _ => 0,
        },
    };
    ExecNode {
        name: node.name.to_string(),
        op,
        inputs: node
            .input
            .iter()
            .filter(|s| !s.is_empty())
            .map(|s| (*s).to_string())
            .collect(),
        outputs: node.output.iter().map(|s| (*s).to_string()).collect(),
        trans_a: attr_i(&node.attribute, "transA", 0) != 0,
        trans_b: attr_i(&node.attribute, "transB", 0) != 0,
        alpha: attr_f(&node.attribute, "alpha", 1.0),
        beta: attr_f(&node.attribute, "beta", 1.0),
        axis: attr_i_opt(&node.attribute, "axis").unwrap_or(axis_default),
        perm: attr_ints(&node.attribute, "perm"),
        axes: attr_ints(&node.attribute, "axes"),
        keepdims: attr_i_opt(&node.attribute, "keepdims").unwrap_or(1) != 0,
        epsilon: attr_f_opt(&node.attribute, "epsilon")
            .or_else(|| attr_f_opt(&node.attribute, "eps"))
            .unwrap_or(1e-5),
        min: attr_f_opt(&node.attribute, "min"),
        max: attr_f_opt(&node.attribute, "max"),
        split: attr_ints(&node.attribute, "split"),
        to: attr_i(&node.attribute, "to", 1),
        approximate: {
            let s = node
                .attribute
                .iter()
                .find(|a| a.name == "approximate")
                .and_then(|a| std::str::from_utf8(a.s).ok())
                .unwrap_or("");
            s.eq_ignore_ascii_case("tanh")
        },
    }
}

fn infer_shape(dims: Vec<i64>, numel: usize) -> Result<Vec<usize>> {
    let mut out = Vec::new();
    let mut infer = None;
    let mut known = 1usize;
    for (i, d) in dims.iter().enumerate() {
        if *d == -1 {
            infer = Some(i);
            out.push(1);
        } else if *d == 0 {
            return Err(Error::fail("reshape dim 0 is not supported"));
        } else {
            known *= *d as usize;
            out.push(*d as usize);
        }
    }
    if let Some(i) = infer {
        if known == 0 || numel % known != 0 {
            return Err(Error::fail("cannot infer reshape dimension"));
        }
        out[i] = numel / known;
    }
    Ok(out)
}

pub(super) fn req<'a>(inputs: &'a [String], idx: usize, node: &str) -> Result<&'a str> {
    inputs
        .get(idx)
        .map(|s| s.as_str())
        .ok_or_else(|| Error::fail(format!("node {node} is missing input {idx}")))
}

pub(super) fn flatten_dims(shape: &[usize], axis: usize) -> (usize, usize) {
    let d0 = shape[..axis].iter().product::<usize>().max(1);
    let d1 = shape[axis..].iter().product::<usize>().max(1);
    (d0, d1)
}

fn constant_value(attrs: &[onnx_rs::ast::Attribute<'_>], node: &str) -> Result<TensorValue> {
    let attr = attrs
        .iter()
        .find(|a| a.name == "value")
        .ok_or_else(|| Error::fail(format!("Constant node {node} has no value attribute")))?;
    let tensor = attr
        .t
        .as_ref()
        .ok_or_else(|| Error::fail(format!("Constant node {node} value is not a tensor")))?;
    tensor_from_proto(tensor)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::onnx::{generate_tiny_embedder, generate_xor_mlp, TensorValue};
    use ndarray::array;

    #[test]
    fn xor_forward_shapes() {
        let mut g = ExecGraph::load(generate_xor_mlp(1)).unwrap();
        let mut inputs = HashMap::new();
        inputs.insert(
            "input".into(),
            array![[0.0, 0.0], [0.0, 1.0]].into_dyn(),
        );
        let out = g.forward(&inputs).unwrap();
        assert_eq!(out["output"].shape(), &[2, 1]);
    }

    #[test]
    fn tiny_embedder_gather_and_rms() {
        let mut g = ExecGraph::load(generate_tiny_embedder(1, 8, 4)).unwrap();
        let mut feeds = HashMap::new();
        feeds.insert(
            "input_ids".into(),
            TensorValue::I64(array![[1i64, 2, 3], [0, 7, 4]].into_dyn()),
        );
        let out = g.forward_values(&feeds).unwrap();
        assert_eq!(out["embedding"].shape(), &[2, 3, 4]);
    }

    #[test]
    fn gather_vjp_hits_rows() {
        let mut g = ExecGraph::load(generate_tiny_embedder(1, 8, 4)).unwrap();
        let mut feeds = HashMap::new();
        feeds.insert(
            "input_ids".into(),
            TensorValue::I64(array![[1i64, 1]].into_dyn()),
        );
        let _ = g.forward_values(&feeds).unwrap();
        let mut grads = HashMap::new();
        grads.insert("embedding".into(), Tensor::ones(ndarray::IxDyn(&[1, 2, 4])));
        g.backward(grads).unwrap();
        let ge = g.grad("E").expect("embed table grad");
        let row1: f32 = ge.slice_axis(ndarray::Axis(0), ndarray::Slice::from(1..2)).iter().sum();
        let row0: f32 = ge.slice_axis(ndarray::Axis(0), ndarray::Slice::from(0..1)).iter().sum();
        assert!(row1 > 0.0);
        assert_eq!(row0, 0.0);
    }

    #[test]
    fn lint_lists_unknown_ops() {
        use onnx_rs::ast::*;
        let model = Model {
            ir_version: 9,
            producer_name: "test",
            opset_import: vec![OperatorSetId {
                domain: "",
                version: 17,
            }],
            graph: Some(Graph {
                name: "bad",
                node: vec![Node {
                    name: "c0",
                    op_type: OpType::Conv,
                    input: vec!["x", "w"],
                    output: vec!["y"],
                    ..Default::default()
                }],
                input: vec![ValueInfo {
                    name: "x",
                    ..Default::default()
                }],
                output: vec![ValueInfo {
                    name: "y",
                    ..Default::default()
                }],
                ..Default::default()
            }),
            ..Default::default()
        };
        let bytes = onnx_rs::encode(&model);
        let err = ExecGraph::load(bytes).unwrap_err();
        assert_eq!(err.exit_code(), 2);
        let msg = err.to_string();
        assert!(msg.contains("Conv"), "{msg}");
        assert!(msg.contains("P3") || msg.contains("roadmap"), "{msg}");
    }
}
