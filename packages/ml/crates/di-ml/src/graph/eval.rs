use ndarray::{ArrayD, IxDyn, Zip};

use crate::accel::BinOp;
use crate::error::{Error, Result};
use crate::kernels::{
    self, concat, gather, gather_elements, layer_norm, normalize_axis, normalize_axis_inclusive,
    reduce, reduce_axes_or_all, rms_norm, rope, split, unsqueeze, where_op, ReduceKind,
};
use crate::onnx::TensorValue;
use crate::tensor::{
    add, binop, broadcast_to, gelu, matmul, neg, relu, scale, sigmoid, silu, softmax_axis,
    sqrt, tanh, transpose_2d, Tensor,
};

use super::{req, flatten_dims, ExecGraph, ExecNode, Tape, UnaryKind};

impl ExecGraph {
    pub(super) fn eval_node(&mut self, node: &ExecNode) -> Result<()> {
        match node.op.as_str() {
            "Gemm" => self.eval_gemm(node),
            "MatMul" => self.eval_matmul(node),
            "Add" | "Sub" | "Mul" | "Div" => self.eval_binop(node),
            "Relu" => self.eval_unary_act(node, "relu"),
            "Sigmoid" => self.eval_unary_act(node, "sigmoid"),
            "Tanh" => self.eval_unary_act(node, "tanh"),
            "Softmax" => self.eval_softmax(node),
            "Identity" | "Dropout" => self.eval_identity(node),
            "Cast" => self.eval_cast(node),
            "Neg" => self.eval_unary_act(node, "neg"),
            "Flatten" => self.eval_flatten(node),
            "Reshape" => self.eval_reshape(node),
            "Transpose" => self.eval_transpose(node),
            "ReduceSum" => self.eval_reduce(node, ReduceKind::Sum),
            "ReduceMean" => self.eval_reduce(node, ReduceKind::Mean),
            "ReduceMax" => self.eval_reduce(node, ReduceKind::Max),
            "Where" => self.eval_where(node),
            "Gather" | "Embedding" => self.eval_gather(node),
            "GatherElements" => self.eval_gather_elements(node),
            "Expand" => self.eval_expand(node),
            "Unsqueeze" => self.eval_unsqueeze(node),
            "Squeeze" => self.eval_squeeze(node),
            "Concat" => self.eval_concat(node),
            "Split" => self.eval_split(node),
            "Slice" => self.eval_slice(node),
            "LayerNormalization" | "SkipLayerNormalization" => self.eval_layer_norm(node),
            "RMSNorm" | "SimplifiedLayerNormalization" => self.eval_rms_norm(node),
            "SkipSimplifiedLayerNormalization" => self.eval_skip_rms(node),
            "SiLU" | "Silu" | "Swish" => self.eval_unary_act(node, "silu"),
            "Gelu" | "BiasGelu" | "FastGelu" | "QuickGelu" => self.eval_unary_act(node, "gelu"),
            "Pow" => self.eval_pow(node),
            "Sqrt" => self.eval_sqrt(node),
            "Clip" => self.eval_clip(node),
            "Exp" => self.eval_elem(node, UnaryKind::Exp),
            "Log" => self.eval_elem(node, UnaryKind::Log),
            "Abs" => self.eval_elem(node, UnaryKind::Abs),
            "Reciprocal" => self.eval_elem(node, UnaryKind::Reciprocal),
            "Erf" => self.eval_elem(node, UnaryKind::Erf),
            "Sin" => self.eval_elem(node, UnaryKind::Sin),
            "Cos" => self.eval_elem(node, UnaryKind::Cos),
            "Softplus" => self.eval_softplus(node),
            "Equal" | "Greater" | "GreaterOrEqual" | "Less" | "LessOrEqual" | "And" | "Or" => {
                self.eval_cmp(node)
            }
            "Not" => self.eval_not(node),
            "Shape" => self.eval_shape(node),
            "RoPE" | "RotaryEmbedding" => self.eval_rope(node),
            "ScaledDotProductAttention" | "SDPA" | "Attention" | "GroupQueryAttention"
            | "MultiHeadAttention" => self.eval_sdpa(node),
            other => Err(Error::fail(format!(
                "internal: unhandled op {other} on node {}",
                node.name
            ))),
        }
    }

    fn eval_gemm(&mut self, node: &ExecNode) -> Result<()> {
        let a_name = req(&node.inputs, 0, &node.name)?;
        let b_name = req(&node.inputs, 1, &node.name)?;
        let mut a = self.f32(a_name)?.clone();
        let mut b = self.f32(b_name)?.clone();
        if node.trans_a {
            a = transpose_2d(&a)?;
        }
        if node.trans_b {
            b = transpose_2d(&b)?;
        }
        let mut y = matmul(&a, &b)?;
        y = scale(&y, node.alpha)?;
        let c_name = node.inputs.get(2).cloned();
        if let Some(ref c) = c_name {
            let c_t = self.f32(c)?;
            let c_b = broadcast_to(c_t, y.shape())?;
            let c_b = scale(&c_b, node.beta)?;
            y = add(&y, &c_b)?;
        }
        let y_name = req(&node.outputs, 0, &node.name)?;
        self.push_tape(Tape::Gemm {
            a: a_name.to_string(),
            b: b_name.to_string(),
            c: c_name,
            y: y_name.to_string(),
            trans_a: node.trans_a,
            trans_b: node.trans_b,
            alpha: node.alpha,
            beta: node.beta,
        });
        self.insert_f32(y_name, y);
        Ok(())
    }

    fn eval_matmul(&mut self, node: &ExecNode) -> Result<()> {
        let a_name = req(&node.inputs, 0, &node.name)?;
        let b_name = req(&node.inputs, 1, &node.name)?;
        let y = matmul(self.f32(a_name)?, self.f32(b_name)?)?;
        let y_name = req(&node.outputs, 0, &node.name)?;
        self.push_tape(Tape::MatMul {
            a: a_name.to_string(),
            b: b_name.to_string(),
            y: y_name.to_string(),
        });
        self.insert_f32(y_name, y);
        Ok(())
    }

    fn eval_binop(&mut self, node: &ExecNode) -> Result<()> {
        let a_name = req(&node.inputs, 0, &node.name)?;
        let b_name = req(&node.inputs, 1, &node.name)?;
        let op = match node.op.as_str() {
            "Add" => BinOp::Add,
            "Sub" => BinOp::Sub,
            "Mul" => BinOp::Mul,
            _ => BinOp::Div,
        };
        let y = binop(self.f32(a_name)?, self.f32(b_name)?, op)?;
        let y_name = req(&node.outputs, 0, &node.name)?;
        let tape = match node.op.as_str() {
            "Add" => Tape::Add {
                a: a_name.to_string(),
                b: b_name.to_string(),
                y: y_name.to_string(),
            },
            "Sub" => Tape::Sub {
                a: a_name.to_string(),
                b: b_name.to_string(),
                y: y_name.to_string(),
            },
            "Mul" => Tape::Mul {
                a: a_name.to_string(),
                b: b_name.to_string(),
                y: y_name.to_string(),
            },
            _ => Tape::Div {
                a: a_name.to_string(),
                b: b_name.to_string(),
                y: y_name.to_string(),
            },
        };
        self.push_tape(tape);
        self.insert_f32(y_name, y);
        Ok(())
    }

    fn eval_unary_act(&mut self, node: &ExecNode, which: &str) -> Result<()> {
        let x_name = req(&node.inputs, 0, &node.name)?;
        let x = self.f32(x_name)?;
        let y = match which {
            "relu" => relu(x)?,
            "sigmoid" => sigmoid(x)?,
            "tanh" => tanh(x)?,
            "neg" => neg(x)?,
            "silu" => silu(x)?,
            "gelu" => gelu(x)?,
            _ => unreachable!(),
        };
        let y_name = req(&node.outputs, 0, &node.name)?;
        let tape = match which {
            "relu" => Tape::Relu {
                x: x_name.to_string(),
                y: y_name.to_string(),
            },
            "sigmoid" => Tape::Sigmoid {
                x: x_name.to_string(),
                y: y_name.to_string(),
            },
            "tanh" => Tape::Tanh {
                x: x_name.to_string(),
                y: y_name.to_string(),
            },
            "neg" => Tape::Neg {
                x: x_name.to_string(),
                y: y_name.to_string(),
            },
            "silu" => Tape::Silu {
                x: x_name.to_string(),
                y: y_name.to_string(),
            },
            "gelu" => Tape::Gelu {
                x: x_name.to_string(),
                y: y_name.to_string(),
            },
            _ => unreachable!(),
        };
        self.push_tape(tape);
        self.insert_f32(y_name, y);
        Ok(())
    }

    fn eval_softmax(&mut self, node: &ExecNode) -> Result<()> {
        let x_name = req(&node.inputs, 0, &node.name)?;
        let x = self.f32(x_name)?;
        let axis = normalize_axis(node.axis, x.ndim())?;
        let y = softmax_axis(x, axis)?;
        let y_name = req(&node.outputs, 0, &node.name)?;
        self.push_tape(Tape::Softmax {
            x: x_name.to_string(),
            y: y_name.to_string(),
            axis,
        });
        self.insert_f32(y_name, y);
        Ok(())
    }

    fn eval_identity(&mut self, node: &ExecNode) -> Result<()> {
        let x_name = req(&node.inputs, 0, &node.name)?;
        let y_name = req(&node.outputs, 0, &node.name)?;
        let v = self.value(x_name)?.clone();
        self.push_tape(Tape::Identity {
            x: x_name.to_string(),
            y: y_name.to_string(),
        });
        self.insert_value(y_name, v);
        Ok(())
    }

    fn eval_cast(&mut self, node: &ExecNode) -> Result<()> {
        let x_name = req(&node.inputs, 0, &node.name)?;
        let y_name = req(&node.outputs, 0, &node.name)?;
        let v = self.value(x_name)?;
        let out = match node.to {
            1 => TensorValue::F32(v.to_f32()?),
            6 | 7 => TensorValue::I64(v.to_i64()?),
            9 => TensorValue::Bool(v.to_bool()?),
            _ => v.clone(),
        };
        self.push_tape(Tape::Identity {
            x: x_name.to_string(),
            y: y_name.to_string(),
        });
        self.insert_value(y_name, out);
        Ok(())
    }

    fn eval_flatten(&mut self, node: &ExecNode) -> Result<()> {
        let x_name = req(&node.inputs, 0, &node.name)?;
        let x = self.f32(x_name)?;
        let axis = normalize_axis_inclusive(node.axis, x.ndim())?;
        let (d0, d1) = flatten_dims(x.shape(), axis);
        let y = x
            .clone()
            .into_shape_with_order(IxDyn(&[d0, d1]))
            .map_err(|err| Error::fail(err.to_string()))?;
        let y_name = req(&node.outputs, 0, &node.name)?;
        self.push_tape(Tape::Flatten {
            x: x_name.to_string(),
            y: y_name.to_string(),
        });
        self.insert_f32(y_name, y);
        Ok(())
    }

    fn eval_reshape(&mut self, node: &ExecNode) -> Result<()> {
        let x_name = req(&node.inputs, 0, &node.name)?;
        let shape_name = req(&node.inputs, 1, &node.name)?;
        let src = self.value(x_name)?.clone();
        let numel = match &src {
            TensorValue::F32(t) => t.len(),
            TensorValue::I64(t) => t.len(),
            TensorValue::Bool(t) => t.len(),
        };
        let new_shape = self.i64_shape(shape_name, numel)?;
        let y_name = req(&node.outputs, 0, &node.name)?;
        let out = match src {
            TensorValue::F32(x) => TensorValue::F32(
                x.into_shape_with_order(IxDyn(&new_shape))
                    .map_err(|err| Error::fail(err.to_string()))?,
            ),
            TensorValue::I64(x) => TensorValue::I64(
                x.into_shape_with_order(IxDyn(&new_shape))
                    .map_err(|err| Error::fail(err.to_string()))?,
            ),
            TensorValue::Bool(x) => TensorValue::Bool(
                x.into_shape_with_order(IxDyn(&new_shape))
                    .map_err(|err| Error::fail(err.to_string()))?,
            ),
        };
        self.push_tape(Tape::Reshape {
            x: x_name.to_string(),
            y: y_name.to_string(),
        });
        self.insert_value(y_name, out);
        Ok(())
    }

    fn eval_transpose(&mut self, node: &ExecNode) -> Result<()> {
        let x_name = req(&node.inputs, 0, &node.name)?;
        let x = self.f32(x_name)?;
        let perm = match &node.perm {
            Some(p) if !p.is_empty() => p.iter().map(|i| *i as usize).collect(),
            _ => (0..x.ndim()).rev().collect::<Vec<_>>(),
        };
        let y = x.clone().permuted_axes(IxDyn(&perm));
        let y = y.as_standard_layout().to_owned();
        let y_name = req(&node.outputs, 0, &node.name)?;
        self.push_tape(Tape::Transpose {
            x: x_name.to_string(),
            y: y_name.to_string(),
            perm,
        });
        self.insert_f32(y_name, y);
        Ok(())
    }

    fn eval_reduce(&mut self, node: &ExecNode, kind: ReduceKind) -> Result<()> {
        let x_name = req(&node.inputs, 0, &node.name)?;
        let x = self.f32(x_name)?.clone();
        let axes_raw = if let Some(ax) = &node.axes {
            if ax.is_empty() {
                None
            } else {
                Some(ax.clone())
            }
        } else if node.inputs.len() > 1 {
            Some(self.i64(&node.inputs[1])?.iter().copied().collect())
        } else {
            None
        };
        let axes = reduce_axes_or_all(x.ndim(), axes_raw.as_deref())?;
        let y = reduce(&x, &axes, node.keepdims, kind)?;
        let y_name = req(&node.outputs, 0, &node.name)?;
        self.push_tape(Tape::Reduce {
            x: x_name.to_string(),
            y: y_name.to_string(),
            axes,
            keepdims: node.keepdims,
            kind,
            x_shape: x.shape().to_vec(),
        });
        self.insert_f32(y_name, y);
        Ok(())
    }

    fn eval_where(&mut self, node: &ExecNode) -> Result<()> {
        let c_name = req(&node.inputs, 0, &node.name)?;
        let x_name = req(&node.inputs, 1, &node.name)?;
        let y_name_in = req(&node.inputs, 2, &node.name)?;
        let cond = self.value(c_name)?.to_bool()?;
        let y = where_op(&cond, self.f32(x_name)?, self.f32(y_name_in)?)?;
        let out = req(&node.outputs, 0, &node.name)?;
        self.push_tape(Tape::Where {
            c: c_name.to_string(),
            x: x_name.to_string(),
            y: y_name_in.to_string(),
            out: out.to_string(),
        });
        self.insert_f32(out, y);
        Ok(())
    }

    fn eval_gather(&mut self, node: &ExecNode) -> Result<()> {
        let data_name = req(&node.inputs, 0, &node.name)?;
        let idx_name = req(&node.inputs, 1, &node.name)?;
        let data = self.f32(data_name)?;
        let axis = normalize_axis(node.axis, data.ndim())?;
        let idx = self.i64(idx_name)?;
        let y = gather(data, &idx, axis)?;
        let y_name = req(&node.outputs, 0, &node.name)?;
        self.push_tape(Tape::Gather {
            data: data_name.to_string(),
            indices: idx_name.to_string(),
            y: y_name.to_string(),
            axis,
        });
        self.insert_f32(y_name, y);
        Ok(())
    }

    fn eval_gather_elements(&mut self, node: &ExecNode) -> Result<()> {
        let data_name = req(&node.inputs, 0, &node.name)?;
        let idx_name = req(&node.inputs, 1, &node.name)?;
        let data = self.f32(data_name)?;
        let axis = normalize_axis(node.axis, data.ndim())?;
        let idx = self.i64(idx_name)?;
        let y = gather_elements(data, &idx, axis)?;
        let y_name = req(&node.outputs, 0, &node.name)?;
        self.push_tape(Tape::GatherElements {
            data: data_name.to_string(),
            indices: idx_name.to_string(),
            y: y_name.to_string(),
            axis,
        });
        self.insert_f32(y_name, y);
        Ok(())
    }

    fn eval_expand(&mut self, node: &ExecNode) -> Result<()> {
        let x_name = req(&node.inputs, 0, &node.name)?;
        let shape_name = req(&node.inputs, 1, &node.name)?;
        let x = self.f32(x_name)?;
        let shape: Vec<usize> = self.i64(shape_name)?.iter().map(|d| *d as usize).collect();
        let y = broadcast_to(x, &shape)?;
        let y_name = req(&node.outputs, 0, &node.name)?;
        self.push_tape(Tape::Expand {
            x: x_name.to_string(),
            y: y_name.to_string(),
        });
        self.insert_f32(y_name, y);
        Ok(())
    }

    fn eval_unsqueeze(&mut self, node: &ExecNode) -> Result<()> {
        let x_name = req(&node.inputs, 0, &node.name)?;
        let axes = if let Some(ax) = &node.axes {
            ax.clone()
        } else if node.inputs.len() > 1 {
            self.i64(&node.inputs[1])?.iter().copied().collect()
        } else {
            return Err(Error::fail(format!(
                "Unsqueeze node {} needs axes",
                node.name
            )));
        };
        let src = self.value(x_name)?.clone();
        let y_name = req(&node.outputs, 0, &node.name)?;
        let out = match src {
            TensorValue::F32(x) => TensorValue::F32(unsqueeze(&x, &axes)?),
            TensorValue::I64(x) => {
                let xf = x.mapv(|v| v as f32);
                let y = unsqueeze(&xf, &axes)?;
                TensorValue::I64(y.mapv(|v| v as i64))
            }
            TensorValue::Bool(x) => {
                let xf = x.mapv(|v| if v { 1.0 } else { 0.0 });
                let y = unsqueeze(&xf, &axes)?;
                TensorValue::Bool(y.mapv(|v| v != 0.0))
            }
        };
        self.push_tape(Tape::Reshape {
            x: x_name.to_string(),
            y: y_name.to_string(),
        });
        self.insert_value(y_name, out);
        Ok(())
    }

    fn eval_squeeze(&mut self, node: &ExecNode) -> Result<()> {
        let x_name = req(&node.inputs, 0, &node.name)?;
        let axes = if let Some(ax) = &node.axes {
            Some(ax.clone())
        } else if node.inputs.len() > 1 {
            Some(self.i64(&node.inputs[1])?.iter().copied().collect())
        } else {
            None
        };
        let x = self.f32(x_name)?;
        let y = kernels::squeeze(x, axes.as_deref())?;
        let y_name = req(&node.outputs, 0, &node.name)?;
        self.push_tape(Tape::Reshape {
            x: x_name.to_string(),
            y: y_name.to_string(),
        });
        self.insert_f32(y_name, y);
        Ok(())
    }

    fn eval_concat(&mut self, node: &ExecNode) -> Result<()> {
        let ts: Vec<Tensor> = node
            .inputs
            .iter()
            .map(|n| self.f32(n).cloned())
            .collect::<Result<Vec<_>>>()?;
        let axis = normalize_axis(node.axis, ts[0].ndim())?;
        let refs: Vec<&Tensor> = ts.iter().collect();
        let sizes: Vec<usize> = ts.iter().map(|t| t.shape()[axis]).collect();
        let y = concat(&refs, axis)?;
        let y_name = req(&node.outputs, 0, &node.name)?;
        self.push_tape(Tape::Concat {
            xs: node.inputs.clone(),
            y: y_name.to_string(),
            axis,
            sizes,
        });
        self.insert_f32(y_name, y);
        Ok(())
    }

    fn eval_split(&mut self, node: &ExecNode) -> Result<()> {
        let x_name = req(&node.inputs, 0, &node.name)?;
        let x = self.f32(x_name)?;
        let axis = normalize_axis(node.axis, x.ndim())?;
        let sizes: Vec<usize> = if let Some(s) = &node.split {
            s.iter().map(|v| *v as usize).collect()
        } else if node.inputs.len() > 1 {
            self.i64(&node.inputs[1])?
                .iter()
                .map(|v| *v as usize)
                .collect()
        } else {
            let n = node.outputs.len();
            let dim = x.shape()[axis];
            if dim % n != 0 {
                return Err(Error::fail("Split: dim not divisible by number of outputs"));
            }
            vec![dim / n; n]
        };
        let parts = split(x, axis, &sizes)?;
        if parts.len() != node.outputs.len() {
            return Err(Error::fail("Split: output count mismatch"));
        }
        self.push_tape(Tape::Split {
            x: x_name.to_string(),
            ys: node.outputs.clone(),
            axis,
            sizes,
        });
        for (name, t) in node.outputs.iter().zip(parts) {
            self.insert_f32(name, t);
        }
        Ok(())
    }

    fn eval_slice(&mut self, node: &ExecNode) -> Result<()> {
        let x_name = req(&node.inputs, 0, &node.name)?;
        let x = self.f32(x_name)?.clone();
        let starts: Vec<i64> = self.i64(req(&node.inputs, 1, &node.name)?)?.iter().copied().collect();
        let ends: Vec<i64> = self.i64(req(&node.inputs, 2, &node.name)?)?.iter().copied().collect();
        let axes: Vec<i64> = if node.inputs.len() > 3 {
            self.i64(&node.inputs[3])?.iter().copied().collect()
        } else {
            (0..starts.len() as i64).collect()
        };
        let steps: Vec<i64> = if node.inputs.len() > 4 {
            self.i64(&node.inputs[4])?.iter().copied().collect()
        } else {
            vec![1; starts.len()]
        };
        let y = slice_tensor(&x, &starts, &ends, &axes, &steps)?;
        let y_name = req(&node.outputs, 0, &node.name)?;
        self.push_tape(Tape::Slice {
            x: x_name.to_string(),
            y: y_name.to_string(),
            starts,
            ends,
            axes,
            steps,
        });
        self.insert_f32(y_name, y);
        Ok(())
    }

    fn eval_layer_norm(&mut self, node: &ExecNode) -> Result<()> {
        let skip = node.op.contains("Skip");
        let x_name = req(&node.inputs, 0, &node.name)?;
        let mut x = self.f32(x_name)?.clone();
        let mut idx = 1;
        if skip {
            let skip_name = req(&node.inputs, 1, &node.name)?;
            x = add(&x, self.f32(skip_name)?)?;
            idx = 2;
        }
        let scale_n = node.inputs.get(idx).cloned();
        let bias_n = node.inputs.get(idx + 1).cloned();
        let axis = normalize_axis(node.axis, x.ndim())?;
        let s = match &scale_n {
            Some(n) => Some(self.f32(n)?.clone()),
            None => None,
        };
        let b = match &bias_n {
            Some(n) => Some(self.f32(n)?.clone()),
            None => None,
        };
        let y = layer_norm(&x, s.as_ref(), b.as_ref(), node.epsilon, axis)?;
        let y_name = req(&node.outputs, 0, &node.name)?;
        if skip {
            if let Some(res) = node.outputs.get(1) {
                self.insert_f32(res, x.clone());
            }
            let skip_name = node.inputs[1].clone();
            let pre = format!("{}__pre", node.name);
            self.insert_f32(&pre, x);
            self.push_tape(Tape::Add {
                a: x_name.to_string(),
                b: skip_name,
                y: pre.clone(),
            });
            self.push_tape(Tape::LayerNorm {
                x: pre,
                scale: scale_n,
                bias: bias_n,
                y: y_name.to_string(),
                axis,
                epsilon: node.epsilon,
            });
        } else {
            self.push_tape(Tape::LayerNorm {
                x: x_name.to_string(),
                scale: scale_n,
                bias: bias_n,
                y: y_name.to_string(),
                axis,
                epsilon: node.epsilon,
            });
        }
        self.insert_f32(y_name, y);
        Ok(())
    }

    fn eval_rms_norm(&mut self, node: &ExecNode) -> Result<()> {
        let x_name = req(&node.inputs, 0, &node.name)?;
        let scale_name = req(&node.inputs, 1, &node.name)?;
        let x = self.f32(x_name)?;
        let axis = normalize_axis(node.axis, x.ndim())?;
        let y = rms_norm(x, self.f32(scale_name)?, node.epsilon, axis)?;
        let y_name = req(&node.outputs, 0, &node.name)?;
        self.push_tape(Tape::RmsNorm {
            x: x_name.to_string(),
            scale: scale_name.to_string(),
            y: y_name.to_string(),
            axis,
            epsilon: node.epsilon,
        });
        self.insert_f32(y_name, y);
        Ok(())
    }

    fn eval_skip_rms(&mut self, node: &ExecNode) -> Result<()> {
        let x_name = req(&node.inputs, 0, &node.name)?;
        let skip_name = req(&node.inputs, 1, &node.name)?;
        let scale_name = req(&node.inputs, 2, &node.name)?;
        let bias = node.inputs.get(3).cloned();
        let x = add(self.f32(x_name)?, self.f32(skip_name)?)?;
        let axis = normalize_axis(node.axis, x.ndim())?;
        let mut y = rms_norm(&x, self.f32(scale_name)?, node.epsilon, axis)?;
        if let Some(ref b) = bias {
            y = add(&y, self.f32(b)?)?;
        }
        let y_name = req(&node.outputs, 0, &node.name)?;
        if let Some(res) = node.outputs.get(1) {
            self.insert_f32(res, x.clone());
        }
        let pre = format!("{}__pre", node.name);
        self.insert_f32(&pre, x);
        self.push_tape(Tape::SkipRmsNorm {
            x: x_name.to_string(),
            skip: skip_name.to_string(),
            scale: scale_name.to_string(),
            bias,
            y: y_name.to_string(),
            residual: node.outputs.get(1).cloned(),
            epsilon: node.epsilon,
            axis,
        });
        self.insert_f32(y_name, y);
        Ok(())
    }

    fn eval_pow(&mut self, node: &ExecNode) -> Result<()> {
        let a_name = req(&node.inputs, 0, &node.name)?;
        let b_name = req(&node.inputs, 1, &node.name)?;
        let a = self.f32(a_name)?;
        let b = self.f32(b_name)?;
        let shape = crate::tensor::broadcast_shape(a.shape(), b.shape())?;
        let a_b = broadcast_to(a, &shape)?;
        let b_b = broadcast_to(b, &shape)?;
        let y = Zip::from(&a_b)
            .and(&b_b)
            .map_collect(|x, p| x.powf(*p));
        let y_name = req(&node.outputs, 0, &node.name)?;
        self.push_tape(Tape::Pow {
            a: a_name.to_string(),
            b: b_name.to_string(),
            y: y_name.to_string(),
        });
        self.insert_f32(y_name, y);
        Ok(())
    }

    fn eval_sqrt(&mut self, node: &ExecNode) -> Result<()> {
        let x_name = req(&node.inputs, 0, &node.name)?;
        let y = sqrt(self.f32(x_name)?)?;
        let y_name = req(&node.outputs, 0, &node.name)?;
        self.push_tape(Tape::Sqrt {
            x: x_name.to_string(),
            y: y_name.to_string(),
        });
        self.insert_f32(y_name, y);
        Ok(())
    }

    fn eval_clip(&mut self, node: &ExecNode) -> Result<()> {
        let x_name = req(&node.inputs, 0, &node.name)?;
        let x = self.f32(x_name)?;
        let min = if node.inputs.len() > 1 {
            scalar_f32(self.value(&node.inputs[1])?)?
        } else {
            node.min.unwrap_or(f32::NEG_INFINITY)
        };
        let max = if node.inputs.len() > 2 {
            scalar_f32(self.value(&node.inputs[2])?)?
        } else {
            node.max.unwrap_or(f32::INFINITY)
        };
        let y = x.mapv(|v| v.clamp(min, max));
        let y_name = req(&node.outputs, 0, &node.name)?;
        self.push_tape(Tape::Clip {
            x: x_name.to_string(),
            y: y_name.to_string(),
            min,
            max,
        });
        self.insert_f32(y_name, y);
        Ok(())
    }

    fn eval_elem(&mut self, node: &ExecNode, op: UnaryKind) -> Result<()> {
        let x_name = req(&node.inputs, 0, &node.name)?;
        let x = self.f32(x_name)?;
        let y = match op {
            UnaryKind::Exp => x.mapv(f32::exp),
            UnaryKind::Log => x.mapv(|v| v.max(1e-12).ln()),
            UnaryKind::Abs => x.mapv(|v| v.abs()),
            UnaryKind::Reciprocal => x.mapv(|v| 1.0 / v),
            UnaryKind::Erf => kernels::erf_elem(x),
            UnaryKind::Sin => x.mapv(|v| v.sin()),
            UnaryKind::Cos => x.mapv(|v| v.cos()),
            UnaryKind::Softplus => x.mapv(|v| (1.0 + v.exp()).ln()),
        };
        let y_name = req(&node.outputs, 0, &node.name)?;
        self.push_tape(Tape::UnaryElem {
            x: x_name.to_string(),
            y: y_name.to_string(),
            op,
        });
        self.insert_f32(y_name, y);
        Ok(())
    }

    fn eval_softplus(&mut self, node: &ExecNode) -> Result<()> {
        let x_name = req(&node.inputs, 0, &node.name)?;
        let x = self.f32(x_name)?;
        let y = x.mapv(|v| (1.0 + v.exp()).ln());
        let y_name = req(&node.outputs, 0, &node.name)?;
        self.push_tape(Tape::UnaryElem {
            x: x_name.to_string(),
            y: y_name.to_string(),
            op: UnaryKind::Softplus,
        });
        self.insert_f32(y_name, y);
        Ok(())
    }

    fn eval_cmp(&mut self, node: &ExecNode) -> Result<()> {
        let a = self.value(req(&node.inputs, 0, &node.name)?)?.to_f32()?;
        let b = self.value(req(&node.inputs, 1, &node.name)?)?.to_f32()?;
        let shape = crate::tensor::broadcast_shape(a.shape(), b.shape())?;
        let a = broadcast_to(&a, &shape)?;
        let b = broadcast_to(&b, &shape)?;
        let y = match node.op.as_str() {
            "Equal" => Zip::from(&a).and(&b).map_collect(|x, y| (*x - *y).abs() < 1e-6),
            "Greater" => Zip::from(&a).and(&b).map_collect(|x, y| x > y),
            "GreaterOrEqual" => Zip::from(&a).and(&b).map_collect(|x, y| x >= y),
            "Less" => Zip::from(&a).and(&b).map_collect(|x, y| x < y),
            "LessOrEqual" => Zip::from(&a).and(&b).map_collect(|x, y| x <= y),
            "And" => {
                let ab = a.mapv(|v| v != 0.0);
                let bb = b.mapv(|v| v != 0.0);
                Zip::from(&ab).and(&bb).map_collect(|x, y| *x && *y)
            }
            "Or" => {
                let ab = a.mapv(|v| v != 0.0);
                let bb = b.mapv(|v| v != 0.0);
                Zip::from(&ab).and(&bb).map_collect(|x, y| *x || *y)
            }
            _ => unreachable!(),
        };
        let y_name = req(&node.outputs, 0, &node.name)?;
        self.insert_value(y_name, TensorValue::Bool(y));
        Ok(())
    }

    fn eval_not(&mut self, node: &ExecNode) -> Result<()> {
        let x = self.value(req(&node.inputs, 0, &node.name)?)?.to_bool()?;
        let y = x.mapv(|v| !v);
        let y_name = req(&node.outputs, 0, &node.name)?;
        self.insert_value(y_name, TensorValue::Bool(y));
        Ok(())
    }

    fn eval_shape(&mut self, node: &ExecNode) -> Result<()> {
        let x_name = req(&node.inputs, 0, &node.name)?;
        let shape: Vec<i64> = self.value(x_name)?.shape().iter().map(|d| *d as i64).collect();
        let arr = ArrayD::from_shape_vec(IxDyn(&[shape.len()]), shape)
            .map_err(|err| Error::fail(err.to_string()))?;
        let y_name = req(&node.outputs, 0, &node.name)?;
        self.insert_value(y_name, TensorValue::I64(arr));
        Ok(())
    }

    fn eval_rope(&mut self, node: &ExecNode) -> Result<()> {
        let x_name = req(&node.inputs, 0, &node.name)?;
        let x = self.f32(x_name)?.clone();
        let (cos, sin, cos_n, sin_n) = if node.inputs.len() >= 3 {
            (
                self.f32(&node.inputs[1])?.clone(),
                self.f32(&node.inputs[2])?.clone(),
                node.inputs[1].clone(),
                node.inputs[2].clone(),
            )
        } else {
            let pos = self.i64(req(&node.inputs, 1, &node.name)?)?;
            let (c, s) = rope_from_pos(&x, &pos)?;
            let cn = format!("{}__cos", node.name);
            let sn = format!("{}__sin", node.name);
            self.insert_f32(&cn, c.clone());
            self.insert_f32(&sn, s.clone());
            (c, s, cn, sn)
        };
        let y = rope(&x, &cos, &sin)?;
        let y_name = req(&node.outputs, 0, &node.name)?;
        self.push_tape(Tape::Rope {
            x: x_name.to_string(),
            cos: cos_n,
            sin: sin_n,
            y: y_name.to_string(),
        });
        self.insert_f32(y_name, y);
        Ok(())
    }

    fn eval_sdpa(&mut self, node: &ExecNode) -> Result<()> {
        let q_name = req(&node.inputs, 0, &node.name)?;
        let k_name = req(&node.inputs, 1, &node.name)?;
        let v_name = req(&node.inputs, 2, &node.name)?;
        let mask = node.inputs.get(3).cloned();
        let q = self.f32(q_name)?.clone();
        let k = self.f32(k_name)?.clone();
        let v = self.f32(v_name)?.clone();
        let d = *q.shape().last().unwrap_or(&1) as f32;
        let scale = if node.alpha != 1.0 && node.alpha != 0.0 {
            node.alpha
        } else {
            1.0 / d.sqrt()
        };
        let m = match &mask {
            Some(n) => Some(self.f32(n)?.clone()),
            None => None,
        };
        let y = crate::tensor::sdpa(&q, &k, &v, m.as_ref(), scale)?;
        let y_name = req(&node.outputs, 0, &node.name)?;
        self.push_tape(Tape::Sdpa {
            q: q_name.to_string(),
            k: k_name.to_string(),
            v: v_name.to_string(),
            mask,
            y: y_name.to_string(),
            scale,
        });
        self.insert_f32(y_name, y);
        Ok(())
    }
}

fn scalar_f32(v: &TensorValue) -> Result<f32> {
    Ok(*v.to_f32()?.iter().next().unwrap_or(&0.0))
}

fn slice_tensor(
    x: &Tensor,
    starts: &[i64],
    ends: &[i64],
    axes: &[i64],
    steps: &[i64],
) -> Result<Tensor> {
    let mut y = x.clone();
    for (i, axis) in axes.iter().enumerate() {
        let ax = normalize_axis(*axis, x.ndim())?;
        let dim = x.shape()[ax] as i64;
        let step = steps.get(i).copied().unwrap_or(1);
        if step != 1 {
            return Err(Error::fail("Slice step != 1 is not implemented"));
        }
        let mut start = starts[i];
        let mut end = ends[i];
        if start < 0 {
            start += dim;
        }
        if end < 0 {
            end += dim;
        }
        start = start.clamp(0, dim);
        end = end.clamp(0, dim);
        y = y
            .slice_axis(ndarray::Axis(ax), ndarray::Slice::from((start as usize)..(end as usize)))
            .to_owned();
    }
    Ok(y)
}

fn rope_from_pos(x: &Tensor, pos: &ArrayD<i64>) -> Result<(Tensor, Tensor)> {
    let d = *x.shape().last().unwrap_or(&2);
    let half = d / 2;
    let t = if pos.ndim() == 0 {
        1
    } else {
        *pos.shape().last().unwrap_or(&1)
    };
    let mut cos = vec![0.0f32; t * d];
    let mut sin = vec![0.0f32; t * d];
    let theta = 10000.0f32;
    for (ti, p) in pos.iter().enumerate() {
        let p = *p as f32;
        for i in 0..half {
            let freq = 1.0 / theta.powf((2 * i) as f32 / d as f32);
            let ang = p * freq;
            cos[ti * d + 2 * i] = ang.cos();
            cos[ti * d + 2 * i + 1] = ang.cos();
            sin[ti * d + 2 * i] = ang.sin();
            sin[ti * d + 2 * i + 1] = ang.sin();
        }
    }
    let cshape = vec![t, d];
    let c = Tensor::from_shape_vec(IxDyn(&cshape), cos).map_err(|e| Error::fail(e.to_string()))?;
    let s = Tensor::from_shape_vec(IxDyn(&cshape), sin).map_err(|e| Error::fail(e.to_string()))?;
    Ok((c, s))
}
