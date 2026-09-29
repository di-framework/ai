use std::collections::HashMap;
use std::fs::File;
use std::io::{BufRead, BufReader};
use std::path::Path;

use ndarray::{Array, ArrayD, Axis, IxDyn};
use serde_json::Value as Json;

use crate::error::{Error, Result};
use crate::onnx::TensorValue;
use crate::tensor::Tensor;
use crate::workspace::LossKind;

#[derive(Debug, Clone)]
pub struct Example {
    pub inputs: HashMap<String, TensorValue>,
    pub label: Label,
    pub pair: Option<HashMap<String, TensorValue>>,
    pub negatives: Vec<HashMap<String, TensorValue>>,
    pub attention_mask: Option<Tensor>,
    pub pair_mask: Option<Tensor>,
    pub teacher_pos: Option<f32>,
    pub teacher_neg: Option<Vec<f32>>,
    pub replay_emb: Option<Tensor>,
    pub score: Option<f32>,
    pub text: Option<String>,
    pub query: Option<String>,
    pub positive: Option<String>,
}

#[derive(Debug, Clone)]
pub enum Label {
    Float(Tensor),
    Class(i64),
    None,
}

#[derive(Debug, Clone)]
pub struct Dataset {
    pub examples: Vec<Example>,
}

impl Dataset {
    pub fn load(
        path: impl AsRef<Path>,
        input_names: &[String],
        input_is_int: &std::collections::HashSet<String>,
        loss: LossKind,
        tok: Option<&crate::tokenize::Tokenizer>,
    ) -> Result<Self> {
        let path = path.as_ref();
        let file = File::open(path)?;
        let reader = BufReader::new(file);
        let mut examples = Vec::new();
        for (idx, line) in reader.lines().enumerate() {
            let line = line?;
            if line.trim().is_empty() {
                continue;
            }
            let json: Json = serde_json::from_str(&line).map_err(|err| {
                Error::usage(format!("{}:{}: {err}", path.display(), idx + 1))
            })?;
            examples.push(
                parse_example(&json, input_names, input_is_int, loss, tok).map_err(|err| {
                    Error::usage(format!("{}:{}: {err}", path.display(), idx + 1))
                })?,
            );
        }
        if examples.is_empty() {
            return Err(Error::usage(format!(
                "{} contains no examples",
                path.display()
            )));
        }
        Ok(Self { examples })
    }

    pub fn batch(&self, indices: &[usize]) -> Result<Batched> {
        self.batch_pack(indices, false)
    }

    /// When `pack_pairs` is true and examples have `pair` inputs, stack
    /// [queries; positives] along the batch axis (in-batch negatives).
    pub fn batch_pack(&self, indices: &[usize], pack_pairs: bool) -> Result<Batched> {
        let mut q_inputs: HashMap<String, Vec<TensorValue>> = HashMap::new();
        let mut p_inputs: HashMap<String, Vec<TensorValue>> = HashMap::new();
        let mut float_labels: Vec<&Tensor> = Vec::new();
        let mut class_labels: Vec<i64> = Vec::new();
        let mut class_mode = false;
        let mut masks: Vec<Tensor> = Vec::new();
        let mut teacher_pos = Vec::new();
        let mut teacher_neg = Vec::new();
        let mut replay = Vec::new();
        let mut scores = Vec::new();
        let mut has_pair = false;

        for &i in indices {
            let ex = self
                .examples
                .get(i)
                .ok_or_else(|| Error::fail(format!("batch index {i} out of range")))?;
            for (name, tensor) in &ex.inputs {
                q_inputs.entry(name.clone()).or_default().push(tensor.clone());
            }
            if let Some(pair) = &ex.pair {
                has_pair = true;
                for (name, tensor) in pair {
                    p_inputs.entry(name.clone()).or_default().push(tensor.clone());
                }
            }
            match &ex.label {
                Label::Float(t) => float_labels.push(t),
                Label::Class(c) => {
                    class_mode = true;
                    class_labels.push(*c);
                }
                Label::None => {}
            }
            if let Some(m) = &ex.attention_mask {
                masks.push(m.clone());
            }
            if let Some(t) = ex.teacher_pos {
                teacher_pos.push(t);
            }
            if let Some(t) = &ex.teacher_neg {
                teacher_neg.push(t.clone());
            }
            if let Some(r) = &ex.replay_emb {
                replay.push(r);
            }
            if let Some(s) = ex.score {
                scores.push(s);
            }
        }

        let pack = pack_pairs && has_pair;
        let mut stacked = HashMap::new();
        if pack {
            for (name, qparts) in q_inputs {
                let pparts = p_inputs.get(&name).cloned().unwrap_or_default();
                let mut all = qparts;
                all.extend(pparts);
                stacked.insert(name, stack_values(&all)?);
            }
        } else {
            for (name, parts) in q_inputs {
                stacked.insert(name, stack_values(&parts)?);
            }
        }

        let label = if class_mode {
            if class_labels.len() != indices.len() {
                return Err(Error::fail(
                    "cannot mix class-index and float labels in one batch",
                ));
            }
            Label::Float(Array::from_shape_vec(IxDyn(&[class_labels.len()]), {
                class_labels.into_iter().map(|c| c as f32).collect()
            })
            .map_err(|err| Error::fail(err.to_string()))?)
        } else if float_labels.is_empty() {
            Label::None
        } else {
            Label::Float(stack(float_labels)?)
        };

        let attention_mask = if masks.is_empty() {
            None
        } else {
            let views: Vec<_> = masks.iter().map(|t| t.view()).collect();
            Some(ndarray::stack(Axis(0), &views).map_err(|e| Error::fail(e.to_string()))?)
        };

        Ok(Batched {
            inputs: stacked,
            label,
            class_indices: class_mode,
            pair_split: pack.then_some(indices.len()),
            attention_mask,
            teacher_pos: if teacher_pos.is_empty() {
                None
            } else {
                Some(teacher_pos)
            },
            teacher_neg: if teacher_neg.is_empty() {
                None
            } else {
                Some(teacher_neg.concat())
            },
            replay_emb: if replay.is_empty() {
                None
            } else {
                Some(stack(replay)?)
            },
            scores: if scores.is_empty() {
                None
            } else {
                Some(scores)
            },
        })
    }
}

#[derive(Debug, Clone)]
pub struct Batched {
    pub inputs: HashMap<String, TensorValue>,
    pub label: Label,
    pub class_indices: bool,
    pub pair_split: Option<usize>,
    pub attention_mask: Option<Tensor>,
    pub teacher_pos: Option<Vec<f32>>,
    pub teacher_neg: Option<Vec<f32>>,
    pub replay_emb: Option<Tensor>,
    pub scores: Option<Vec<f32>>,
}

fn stack(parts: Vec<&Tensor>) -> Result<Tensor> {
    if parts.is_empty() {
        return Err(Error::fail("empty batch"));
    }
    let views: Vec<_> = parts.iter().map(|t| t.view()).collect();
    ndarray::stack(Axis(0), &views).map_err(|err| Error::fail(err.to_string()))
}

fn stack_values(parts: &[TensorValue]) -> Result<TensorValue> {
    if parts.is_empty() {
        return Err(Error::fail("empty batch"));
    }
    match &parts[0] {
        TensorValue::F32(_) => {
            let ts: Vec<Tensor> = parts
                .iter()
                .map(|v| v.as_f32().cloned())
                .collect::<Result<Vec<_>>>()?;
            let pad = pad_stack_f32(&ts)?;
            Ok(TensorValue::F32(pad))
        }
        TensorValue::I64(_) => {
            let ts: Vec<ArrayD<i64>> = parts
                .iter()
                .map(|v| v.as_i64().cloned())
                .collect::<Result<Vec<_>>>()?;
            Ok(TensorValue::I64(pad_stack_i64(&ts)?))
        }
        TensorValue::Bool(_) => {
            let ts: Vec<ArrayD<bool>> = parts
                .iter()
                .map(|v| v.as_bool().cloned())
                .collect::<Result<Vec<_>>>()?;
            let views: Vec<_> = ts.iter().map(|t| t.view()).collect();
            Ok(TensorValue::Bool(
                ndarray::stack(Axis(0), &views).map_err(|e| Error::fail(e.to_string()))?,
            ))
        }
    }
}

fn pad_stack_f32(ts: &[Tensor]) -> Result<Tensor> {
    if ts.iter().all(|t| t.shape() == ts[0].shape()) {
        let views: Vec<_> = ts.iter().map(|t| t.view()).collect();
        return ndarray::stack(Axis(0), &views).map_err(|e| Error::fail(e.to_string()));
    }
    // variable-length: pad dim 0 of each example (sequence)
    let max_t = ts.iter().map(|t| t.shape().first().copied().unwrap_or(1)).max().unwrap_or(1);
    let rest: Vec<usize> = ts[0].shape().get(1..).unwrap_or(&[]).to_vec();
    let mut out_shape = vec![ts.len(), max_t];
    out_shape.extend_from_slice(&rest);
    let rest_n: usize = rest.iter().product::<usize>().max(1);
    let mut data = vec![0.0f32; ts.len() * max_t * rest_n];
    for (i, t) in ts.iter().enumerate() {
        let ti = t.shape().first().copied().unwrap_or(1);
        let std = t.as_standard_layout();
        let src = std.as_slice().unwrap();
        let dst = i * max_t * rest_n;
        data[dst..dst + ti * rest_n].copy_from_slice(&src[..ti * rest_n]);
    }
    Array::from_shape_vec(IxDyn(&out_shape), data).map_err(|e| Error::fail(e.to_string()))
}

fn pad_stack_i64(ts: &[ArrayD<i64>]) -> Result<ArrayD<i64>> {
    if ts.iter().all(|t| t.shape() == ts[0].shape()) {
        let views: Vec<_> = ts.iter().map(|t| t.view()).collect();
        return ndarray::stack(Axis(0), &views).map_err(|e| Error::fail(e.to_string()));
    }
    let max_t = ts.iter().map(|t| t.len()).max().unwrap_or(1);
    let mut data = vec![0i64; ts.len() * max_t];
    for (i, t) in ts.iter().enumerate() {
        let src = t.as_slice().unwrap_or(&[]);
        let dst = i * max_t;
        data[dst..dst + src.len()].copy_from_slice(src);
    }
    Array::from_shape_vec(IxDyn(&[ts.len(), max_t]), data).map_err(|e| Error::fail(e.to_string()))
}

fn parse_example(
    json: &Json,
    input_names: &[String],
    input_is_int: &std::collections::HashSet<String>,
    loss: LossKind,
    tok: Option<&crate::tokenize::Tokenizer>,
) -> Result<Example> {
    let obj = json
        .as_object()
        .ok_or_else(|| Error::usage("each JSONL row must be an object"))?;

    let mut inputs = HashMap::new();
    for name in input_names {
        if let Some(value) = obj.get(name) {
            inputs.insert(name.clone(), json_to_value(value, input_is_int.contains(name))?);
        }
    }

    // Tokenizer path: fill missing int inputs from text fields.
    if inputs.len() < input_names.len() {
        if let Some(tok) = tok {
            let query = obj
                .get("query")
                .or_else(|| obj.get("text"))
                .or_else(|| obj.get("sentence1"))
                .and_then(|v| v.as_str());
            if let Some(q) = query {
                let ids = tok.encode_query(q)?;
                fill_token_inputs(&mut inputs, input_names, input_is_int, &ids);
            }
        }
    }

    for name in input_names {
        if !inputs.contains_key(name) {
            return Err(Error::usage(format!(
                "missing input field {name:?}; known keys: {:?}",
                obj.keys().collect::<Vec<_>>()
            )));
        }
    }

    let pair = if loss.is_embedding() {
        if let Some(tok) = tok {
            if let Some(p) = obj.get("positive").or_else(|| obj.get("sentence2")).and_then(|v| v.as_str())
            {
                let ids = tok.encode_doc(p)?;
                let mut p_in = HashMap::new();
                fill_token_inputs(&mut p_in, input_names, input_is_int, &ids);
                Some(p_in)
            } else {
                parse_pair_object(obj, input_names, input_is_int)?
            }
        } else {
            parse_pair_object(obj, input_names, input_is_int)?
        }
    } else {
        None
    };

    let label_value = obj.get("label").or_else(|| obj.get("labels")).or_else(|| obj.get("y"));
    let label = match label_value {
        None if loss.is_embedding() => Label::None,
        None => {
            return Err(Error::usage("missing label field (label, labels, or y)"));
        }
        Some(label_value) => match loss {
            LossKind::CrossEntropy if label_value.is_i64() || label_value.is_u64() => {
                Label::Class(json_to_i64(label_value)?)
            }
            LossKind::CrossEntropy if label_value.is_f64() => {
                Label::Class(label_value.as_f64().unwrap() as i64)
            }
            _ => Label::Float(json_to_tensor(label_value)?),
        },
    };

    let attention_mask = obj
        .get("attention_mask")
        .map(json_to_tensor)
        .transpose()?;

    let teacher_pos = obj
        .get("teacher_pos")
        .and_then(|v| v.as_f64())
        .map(|v| v as f32);
    let teacher_neg = obj.get("teacher_neg").and_then(|v| {
        v.as_array().map(|a| {
            a.iter()
                .filter_map(|x| x.as_f64().map(|f| f as f32))
                .collect::<Vec<_>>()
        })
    });
    let replay_emb = obj.get("replay_emb").map(json_to_tensor).transpose()?;
    let score = obj
        .get("score")
        .and_then(|v| v.as_f64())
        .map(|v| v as f32);

    Ok(Example {
        inputs,
        label,
        pair,
        negatives: Vec::new(),
        attention_mask,
        pair_mask: None,
        teacher_pos,
        teacher_neg,
        replay_emb,
        score,
        text: obj.get("text").and_then(|v| v.as_str()).map(str::to_string),
        query: obj.get("query").and_then(|v| v.as_str()).map(str::to_string),
        positive: obj
            .get("positive")
            .and_then(|v| v.as_str())
            .map(str::to_string),
    })
}

fn fill_token_inputs(
    inputs: &mut HashMap<String, TensorValue>,
    names: &[String],
    input_is_int: &std::collections::HashSet<String>,
    ids: &[i64],
) {
    for name in names {
        if inputs.contains_key(name) {
            continue;
        }
        if name.contains("input_ids") || name.contains("tokens") || (input_is_int.contains(name) && name.contains("input"))
        {
            let arr = Array::from_shape_vec(IxDyn(&[ids.len()]), ids.to_vec()).unwrap();
            inputs.insert(name.clone(), TensorValue::I64(arr));
        } else if name.contains("attention_mask") || name.contains("mask") {
            let m: Vec<i64> = vec![1; ids.len()];
            let arr = Array::from_shape_vec(IxDyn(&[m.len()]), m).unwrap();
            inputs.insert(name.clone(), TensorValue::I64(arr));
        }
    }
}

fn parse_pair_object(
    obj: &serde_json::Map<String, Json>,
    input_names: &[String],
    input_is_int: &std::collections::HashSet<String>,
) -> Result<Option<HashMap<String, TensorValue>>> {
    if let Some(p) = obj.get("positive") {
        if p.is_object() {
            let mut map = HashMap::new();
            for name in input_names {
                if let Some(v) = p.get(name) {
                    map.insert(name.clone(), json_to_value(v, input_is_int.contains(name))?);
                }
            }
            if !map.is_empty() {
                return Ok(Some(map));
            }
        }
        if p.is_array() && input_names.len() == 1 {
            let mut map = HashMap::new();
            map.insert(
                input_names[0].clone(),
                json_to_value(p, input_is_int.contains(&input_names[0]))?,
            );
            return Ok(Some(map));
        }
    }
    // duplicate keys with _positive suffix
    let mut map = HashMap::new();
    for name in input_names {
        let key = format!("{name}_positive");
        if let Some(v) = obj.get(&key) {
            map.insert(name.clone(), json_to_value(v, input_is_int.contains(name))?);
        }
    }
    Ok((!map.is_empty()).then_some(map))
}

fn json_to_i64(value: &Json) -> Result<i64> {
    value
        .as_i64()
        .or_else(|| value.as_u64().map(|v| v as i64))
        .ok_or_else(|| Error::usage(format!("expected integer class label, got {value}")))
}

fn json_to_value(value: &Json, want_int: bool) -> Result<TensorValue> {
    if want_int {
        Ok(TensorValue::I64(json_to_i64_tensor(value)?))
    } else {
        Ok(TensorValue::F32(json_to_tensor(value)?))
    }
}

fn json_to_i64_tensor(value: &Json) -> Result<ArrayD<i64>> {
    match value {
        Json::Number(n) => {
            let v = n.as_i64().or_else(|| n.as_f64().map(|f| f as i64)).unwrap_or(0);
            Ok(Array::from_elem(IxDyn(&[]), v))
        }
        Json::Array(_) => {
            let (shape, data) = flatten_i64(value)?;
            Array::from_shape_vec(IxDyn(&shape), data).map_err(|err| Error::fail(err.to_string()))
        }
        other => Err(Error::usage(format!("expected int or array, got {other}"))),
    }
}

fn flatten_i64(value: &Json) -> Result<(Vec<usize>, Vec<i64>)> {
    match value {
        Json::Number(n) => Ok((
            vec![],
            vec![n.as_i64().or_else(|| n.as_f64().map(|f| f as i64)).unwrap_or(0)],
        )),
        Json::Array(items) => {
            if items.iter().all(|i| i.is_number()) {
                let data = items
                    .iter()
                    .map(|i| i.as_i64().or_else(|| i.as_f64().map(|f| f as i64)).unwrap_or(0))
                    .collect::<Vec<_>>();
                Ok((vec![data.len()], data))
            } else {
                let mut shape = None;
                let mut data = Vec::new();
                for item in items {
                    let (s, d) = flatten_i64(item)?;
                    match &shape {
                        None => shape = Some(s),
                        Some(expected) if expected == &s => {}
                        Some(expected) => {
                            return Err(Error::usage(format!("ragged array: {expected:?} vs {s:?}")));
                        }
                    }
                    data.extend(d);
                }
                let mut out_shape = vec![items.len()];
                out_shape.extend(shape.unwrap_or_default());
                Ok((out_shape, data))
            }
        }
        other => Err(Error::usage(format!("invalid int tensor JSON {other}"))),
    }
}

fn json_to_tensor(value: &Json) -> Result<Tensor> {
    match value {
        Json::Number(n) => {
            let v = n
                .as_f64()
                .ok_or_else(|| Error::usage("label number is not finite"))? as f32;
            Ok(Array::from_elem(IxDyn(&[]), v))
        }
        Json::Array(_) => {
            let (shape, data) = flatten_array(value)?;
            Array::from_shape_vec(IxDyn(&shape), data).map_err(|err| Error::fail(err.to_string()))
        }
        other => Err(Error::usage(format!(
            "expected number or nested array, got {other}"
        ))),
    }
}

fn flatten_array(value: &Json) -> Result<(Vec<usize>, Vec<f32>)> {
    match value {
        Json::Number(n) => Ok((
            vec![],
            vec![n.as_f64().ok_or_else(|| Error::usage("non-finite number"))? as f32],
        )),
        Json::Array(items) => {
            if items.is_empty() {
                return Ok((vec![0], vec![]));
            }
            if items.iter().all(|i| i.is_number()) {
                let data = items
                    .iter()
                    .map(|i| {
                        i.as_f64()
                            .map(|v| v as f32)
                            .ok_or_else(|| Error::usage("non-finite number"))
                    })
                    .collect::<Result<Vec<_>>>()?;
                Ok((vec![data.len()], data))
            } else {
                let mut shape = None;
                let mut data = Vec::new();
                for item in items {
                    let (s, d) = flatten_array(item)?;
                    match &shape {
                        None => shape = Some(s),
                        Some(expected) if expected == &s => {}
                        Some(expected) => {
                            return Err(Error::usage(format!(
                                "ragged array: {expected:?} vs {s:?}"
                            )));
                        }
                    }
                    data.extend(d);
                }
                let mut out_shape = vec![items.len()];
                out_shape.extend(shape.unwrap_or_default());
                Ok((out_shape, data))
            }
        }
        other => Err(Error::usage(format!("invalid tensor JSON {other}"))),
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::workspace::LossKind;

    #[test]
    fn parses_vector_row() {
        let json: Json = serde_json::from_str(r#"{"input":[0,1],"label":[1]}"#).unwrap();
        let ex = parse_example(
            &json,
            &["input".into()],
            &Default::default(),
            LossKind::Mse,
            None,
        )
        .unwrap();
        assert_eq!(ex.inputs["input"].shape(), &[2]);
    }
}
