use std::collections::HashMap;
use std::fs;
use std::path::Path;

use ndarray::{Array, ArrayD, IxDyn};
use onnx_rs::ast::{
    Attribute, AttributeType, DataType, Dimension, Graph, Model, Node, OpType, OperatorSetId,
    TensorProto, TensorShape, TensorShapeDimension, TensorTypeProto, TypeProto, TypeValue,
    ValueInfo,
};
use rand::rngs::StdRng;
use rand::{RngExt, SeedableRng};

use crate::error::{Error, Result};
use crate::tensor::Tensor;

pub fn load_bytes(path: impl AsRef<Path>) -> Result<Vec<u8>> {
    fs::read(path.as_ref()).map_err(Error::from)
}

pub fn tensor_from_proto(t: &TensorProto<'_>) -> Result<TensorValue> {
    let shape: Vec<usize> = t.dims().iter().map(|d| *d as usize).collect();
    match t.data_type() {
        DataType::Float => {
            let data = t
                .as_f32()
                .ok_or_else(|| Error::fail(format!("initializer {} has no f32 data", t.name())))?
                .into_owned();
            let arr = Array::from_shape_vec(IxDyn(&shape), data)
                .map_err(|err| Error::fail(err.to_string()))?;
            Ok(TensorValue::F32(arr))
        }
        DataType::Int64 => {
            let data = t
                .as_i64()
                .ok_or_else(|| Error::fail(format!("initializer {} has no i64 data", t.name())))?
                .into_owned();
            let arr = Array::from_shape_vec(IxDyn(&shape), data)
                .map_err(|err| Error::fail(err.to_string()))?;
            Ok(TensorValue::I64(arr))
        }
        DataType::Int32 => {
            let data = t
                .as_i32()
                .ok_or_else(|| Error::fail(format!("initializer {} has no i32 data", t.name())))?
                .into_owned();
            let arr = Array::from_shape_vec(
                IxDyn(&shape),
                data.into_iter().map(|v| v as i64).collect(),
            )
            .map_err(|err| Error::fail(err.to_string()))?;
            Ok(TensorValue::I64(arr))
        }
        DataType::Bool => {
            let data = t
                .as_i32()
                .map(|d| d.iter().map(|v| *v != 0).collect::<Vec<_>>())
                .or_else(|| {
                    t.as_i64()
                        .map(|d| d.iter().map(|v| *v != 0).collect::<Vec<_>>())
                })
                .ok_or_else(|| Error::fail(format!("initializer {} has no bool data", t.name())))?;
            let arr = Array::from_shape_vec(IxDyn(&shape), data)
                .map_err(|err| Error::fail(err.to_string()))?;
            Ok(TensorValue::Bool(arr))
        }
        other => Err(Error::fail(format!(
            "unsupported tensor type {other:?} on {}",
            t.name()
        ))),
    }
}

#[derive(Debug, Clone)]
pub enum TensorValue {
    F32(Tensor),
    I64(ArrayD<i64>),
    Bool(ArrayD<bool>),
}

impl TensorValue {
    pub fn as_f32(&self) -> Result<&Tensor> {
        match self {
            Self::F32(t) => Ok(t),
            _ => Err(Error::fail("expected f32 tensor")),
        }
    }

    pub fn as_i64(&self) -> Result<&ArrayD<i64>> {
        match self {
            Self::I64(t) => Ok(t),
            _ => Err(Error::fail("expected i64 tensor")),
        }
    }

    pub fn as_bool(&self) -> Result<&ArrayD<bool>> {
        match self {
            Self::Bool(t) => Ok(t),
            _ => Err(Error::fail("expected bool tensor")),
        }
    }

    pub fn shape(&self) -> &[usize] {
        match self {
            Self::F32(t) => t.shape(),
            Self::I64(t) => t.shape(),
            Self::Bool(t) => t.shape(),
        }
    }

    pub fn nbytes(&self) -> usize {
        match self {
            Self::F32(t) => t.len() * 4,
            Self::I64(t) => t.len() * 8,
            Self::Bool(t) => t.len(),
        }
    }

    pub fn to_bool(&self) -> Result<ArrayD<bool>> {
        match self {
            Self::Bool(t) => Ok(t.clone()),
            Self::I64(t) => Ok(t.mapv(|v| v != 0)),
            Self::F32(t) => Ok(t.mapv(|v| v != 0.0)),
        }
    }

    pub fn to_i64(&self) -> Result<ArrayD<i64>> {
        match self {
            Self::I64(t) => Ok(t.clone()),
            Self::F32(t) => Ok(t.mapv(|v| v as i64)),
            Self::Bool(t) => Ok(t.mapv(|v| i64::from(v))),
        }
    }

    pub fn to_f32(&self) -> Result<Tensor> {
        match self {
            Self::F32(t) => Ok(t.clone()),
            Self::I64(t) => Ok(t.mapv(|v| v as f32)),
            Self::Bool(t) => Ok(t.mapv(|v| if v { 1.0 } else { 0.0 })),
        }
    }
}

pub fn generate_xor_mlp(seed: u64) -> Vec<u8> {
    let mut rng = StdRng::seed_from_u64(seed);
    let w1 = xavier(&mut rng, 2, 8);
    let b1 = vec![0.0f32; 8];
    let w2 = xavier(&mut rng, 8, 1);
    let b2 = vec![0.0f32; 1];

    let model = Model {
        ir_version: 9,
        producer_name: "di-ml",
        producer_version: "0.1.0",
        opset_import: vec![OperatorSetId {
            domain: "",
            version: 17,
        }],
        graph: Some(Graph {
            name: "xor_mlp",
            node: vec![
                gemm("gemm1", "input", "W1", "B1", "h1"),
                node("relu", OpType::Relu, &["h1"], &["h2"]),
                gemm("gemm2", "h2", "W2", "B2", "logits"),
                node("sigmoid", OpType::Sigmoid, &["logits"], &["output"]),
            ],
            initializer: vec![
                TensorProto::from_f32("W1", vec![2, 8], w1),
                TensorProto::from_f32("B1", vec![8], b1),
                TensorProto::from_f32("W2", vec![8, 1], w2),
                TensorProto::from_f32("B2", vec![1], b2),
            ],
            input: vec![value_info("input", vec![
                Dimension::Param("N"),
                Dimension::Value(2),
            ])],
            output: vec![value_info("output", vec![
                Dimension::Param("N"),
                Dimension::Value(1),
            ])],
            ..Default::default()
        }),
        ..Default::default()
    };
    onnx_rs::encode(&model)
}

pub fn xor_train_jsonl() -> String {
    [
        r#"{"input":[0.0,0.0],"label":[0.0]}"#,
        r#"{"input":[0.0,1.0],"label":[1.0]}"#,
        r#"{"input":[1.0,0.0],"label":[1.0]}"#,
        r#"{"input":[1.0,1.0],"label":[0.0]}"#,
    ]
    .join("\n")
        + "\n"
}

pub fn write_updated_model(
    original: &[u8],
    weights: &HashMap<String, Tensor>,
    dest: impl AsRef<Path>,
) -> Result<()> {
    let mut model = onnx_rs::parse(original).map_err(|err| Error::fail(err.to_string()))?;
    let graph = model
        .graph
        .as_mut()
        .ok_or_else(|| Error::fail("ONNX model has no graph"))?;
    let mut replaced = Vec::with_capacity(graph.initializer.len());
    for init in graph.initializer.iter() {
        if let Some(w) = weights.get(init.name()) {
            let dims = w.shape().iter().map(|&d| d as i64).collect();
            let data = w.iter().copied().collect();
            replaced.push(TensorProto::from_f32(init.name(), dims, data));
        } else {
            replaced.push(init.clone());
        }
    }
    graph.initializer = replaced;
    fs::write(dest.as_ref(), onnx_rs::encode(&model))?;
    Ok(())
}

fn xavier(rng: &mut StdRng, fan_in: usize, fan_out: usize) -> Vec<f32> {
    let scale = (6.0 / (fan_in + fan_out) as f32).sqrt();
    (0..fan_in * fan_out)
        .map(|_| rng.random_range(-scale..scale))
        .collect()
}

fn gemm<'a>(name: &'a str, a: &'a str, b: &'a str, c: &'a str, y: &'a str) -> Node<'a> {
    Node {
        name,
        op_type: OpType::Gemm,
        input: vec![a, b, c],
        output: vec![y],
        ..Default::default()
    }
}

fn node<'a>(name: &'a str, op: OpType<'a>, inputs: &[&'a str], outputs: &[&'a str]) -> Node<'a> {
    Node {
        name,
        op_type: op,
        input: inputs.to_vec(),
        output: outputs.to_vec(),
        ..Default::default()
    }
}

fn value_info(name: &'static str, dims: Vec<Dimension<'static>>) -> ValueInfo<'static> {
    ValueInfo {
        name,
        r#type: Some(TypeProto {
            value: Some(TypeValue::Tensor(TensorTypeProto {
                elem_type: DataType::Float,
                shape: Some(TensorShape {
                    dim: dims
                        .into_iter()
                        .map(|value| TensorShapeDimension {
                            value,
                            denotation: "",
                        })
                        .collect(),
                }),
            })),
            denotation: "",
        }),
        ..Default::default()
    }
}

pub fn attr_i(attrs: &[Attribute<'_>], name: &str, default: i64) -> i64 {
    attrs
        .iter()
        .find(|a| a.name == name)
        .map(|a| {
            if a.r#type == AttributeType::Int || a.i != 0 {
                a.i
            } else {
                default
            }
        })
        .unwrap_or(default)
}

pub fn attr_f(attrs: &[Attribute<'_>], name: &str, default: f32) -> f32 {
    attrs
        .iter()
        .find(|a| a.name == name)
        .map(|a| {
            if a.r#type == AttributeType::Float || a.f != 0.0 {
                a.f
            } else {
                default
            }
        })
        .unwrap_or(default)
}

pub fn attr_ints(attrs: &[Attribute<'_>], name: &str) -> Option<Vec<i64>> {
    attrs
        .iter()
        .find(|a| a.name == name)
        .map(|a| a.ints.clone())
}

pub fn attr_i_opt(attrs: &[Attribute<'_>], name: &str) -> Option<i64> {
    attrs.iter().find(|a| a.name == name).map(|a| a.i)
}

pub fn attr_f_opt(attrs: &[Attribute<'_>], name: &str) -> Option<f32> {
    attrs.iter().find(|a| a.name == name).map(|a| a.f)
}

pub fn attr_floats(attrs: &[Attribute<'_>], name: &str) -> Option<Vec<f32>> {
    attrs.iter().find(|a| a.name == name).map(|a| a.floats.clone())
}

pub fn input_elem_type(info: &ValueInfo<'_>) -> Option<DataType> {
    match info.r#type.as_ref()?.value.as_ref()? {
        TypeValue::Tensor(t) => Some(t.elem_type),
        _ => None,
    }
}

/// Tiny embedding table [V, D] + Gather. Used to prove last-token / InfoNCE
/// on a frozen-style ONNX file without a full transformer.
pub fn generate_tiny_embedder(seed: u64, vocab: usize, dim: usize) -> Vec<u8> {
    let mut rng = StdRng::seed_from_u64(seed);
    let table: Vec<f32> = (0..vocab * dim)
        .map(|_| rng.random_range(-0.5..0.5))
        .collect();
    let scale = vec![1.0f32; dim];
    let model = Model {
        ir_version: 9,
        producer_name: "di-ml",
        producer_version: "0.1.0",
        opset_import: vec![OperatorSetId {
            domain: "",
            version: 17,
        }],
        graph: Some(Graph {
            name: "tiny_embedder",
            node: vec![
                Node {
                    name: "emb",
                    op_type: OpType::Gather,
                    input: vec!["E", "input_ids"],
                    output: vec!["tok"],
                    ..Default::default()
                },
                Node {
                    name: "norm",
                    op_type: OpType::Custom("RMSNorm"),
                    input: vec!["tok", "rms_w"],
                    output: vec!["embedding"],
                    attribute: vec![Attribute {
                        name: "epsilon",
                        r#type: AttributeType::Float,
                        f: 1e-6,
                        ..Default::default()
                    }],
                    ..Default::default()
                },
            ],
            initializer: vec![
                TensorProto::from_f32("E", vec![vocab as i64, dim as i64], table),
                TensorProto::from_f32("rms_w", vec![dim as i64], scale),
            ],
            input: vec![value_info_i64(
                "input_ids",
                vec![Dimension::Param("N"), Dimension::Param("T")],
            )],
            output: vec![value_info(
                "embedding",
                vec![
                    Dimension::Param("N"),
                    Dimension::Param("T"),
                    Dimension::Value(dim as i64),
                ],
            )],
            ..Default::default()
        }),
        ..Default::default()
    };
    onnx_rs::encode(&model)
}

fn value_info_i64(name: &'static str, dims: Vec<Dimension<'static>>) -> ValueInfo<'static> {
    ValueInfo {
        name,
        r#type: Some(TypeProto {
            value: Some(TypeValue::Tensor(TensorTypeProto {
                elem_type: DataType::Int64,
                shape: Some(TensorShape {
                    dim: dims
                        .into_iter()
                        .map(|value| TensorShapeDimension {
                            value,
                            denotation: "",
                        })
                        .collect(),
                }),
            })),
            denotation: "",
        }),
        ..Default::default()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn xor_model_roundtrips() {
        let bytes = generate_xor_mlp(1);
        let model = onnx_rs::parse(&bytes).unwrap();
        let graph = model.graph.unwrap();
        assert_eq!(graph.node.len(), 4);
        assert_eq!(graph.initializer.len(), 4);
        assert_eq!(graph.non_init_inputs()[0].name, "input");
    }
}
