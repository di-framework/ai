use std::collections::HashMap;
use std::ops::Range;

use crate::kernels::ReduceKind;
use crate::onnx::TensorValue;

use super::ExecNode;

#[derive(Debug, Clone)]
pub enum Tape {
    Gemm {
        a: String,
        b: String,
        c: Option<String>,
        y: String,
        trans_a: bool,
        trans_b: bool,
        alpha: f32,
        beta: f32,
    },
    MatMul {
        a: String,
        b: String,
        y: String,
    },
    Add {
        a: String,
        b: String,
        y: String,
    },
    Sub {
        a: String,
        b: String,
        y: String,
    },
    Mul {
        a: String,
        b: String,
        y: String,
    },
    Div {
        a: String,
        b: String,
        y: String,
    },
    Relu {
        x: String,
        y: String,
    },
    Sigmoid {
        x: String,
        y: String,
    },
    Tanh {
        x: String,
        y: String,
    },
    Softmax {
        x: String,
        y: String,
        axis: usize,
    },
    Identity {
        x: String,
        y: String,
    },
    Neg {
        x: String,
        y: String,
    },
    Flatten {
        x: String,
        y: String,
    },
    Reshape {
        x: String,
        y: String,
    },
    Transpose {
        x: String,
        y: String,
        perm: Vec<usize>,
    },
    Reduce {
        x: String,
        y: String,
        axes: Vec<usize>,
        keepdims: bool,
        kind: ReduceKind,
        x_shape: Vec<usize>,
    },
    Where {
        c: String,
        x: String,
        y: String,
        out: String,
    },
    Gather {
        data: String,
        indices: String,
        y: String,
        axis: usize,
    },
    GatherElements {
        data: String,
        indices: String,
        y: String,
        axis: usize,
    },
    Expand {
        x: String,
        y: String,
    },
    Concat {
        xs: Vec<String>,
        y: String,
        axis: usize,
        sizes: Vec<usize>,
    },
    Split {
        x: String,
        ys: Vec<String>,
        axis: usize,
        sizes: Vec<usize>,
    },
    Slice {
        x: String,
        y: String,
        starts: Vec<i64>,
        ends: Vec<i64>,
        axes: Vec<i64>,
        steps: Vec<i64>,
    },
    LayerNorm {
        x: String,
        scale: Option<String>,
        bias: Option<String>,
        y: String,
        axis: usize,
        epsilon: f32,
    },
    RmsNorm {
        x: String,
        scale: String,
        y: String,
        axis: usize,
        epsilon: f32,
    },
    SkipRmsNorm {
        x: String,
        skip: String,
        scale: String,
        bias: Option<String>,
        y: String,
        residual: Option<String>,
        epsilon: f32,
        axis: usize,
    },
    Silu {
        x: String,
        y: String,
    },
    Gelu {
        x: String,
        y: String,
    },
    Pow {
        a: String,
        b: String,
        y: String,
    },
    Sqrt {
        x: String,
        y: String,
    },
    Clip {
        x: String,
        y: String,
        min: f32,
        max: f32,
    },
    UnaryElem {
        x: String,
        y: String,
        op: UnaryKind,
    },
    Sdpa {
        q: String,
        k: String,
        v: String,
        mask: Option<String>,
        y: String,
        scale: f32,
    },
    Rope {
        x: String,
        cos: String,
        sin: String,
        y: String,
    },
    Segment {
        range: Range<usize>,
        nodes: Vec<ExecNode>,
        saved: HashMap<String, TensorValue>,
    },
}

#[derive(Debug, Clone, Copy)]
pub enum UnaryKind {
    Exp,
    Log,
    Abs,
    Reciprocal,
    Erf,
    Sin,
    Cos,
    Softplus,
}
