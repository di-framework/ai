//! Supported ONNX ops and import lint.
//!
//! Graphs with unknown ops are rejected *before* epoch 0. The missing-op list
//! is the roadmap: P0 attention/pooling, P1 QKV layout, P2 fused SDPA/RoPE,
//! P3 vision / older BERT contrib.

use onnx_rs::ast::OpType;

use crate::error::{Error, Result};

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum OpPriority {
    Ready,
    P2,
    P3,
    Unknown,
}

pub fn op_name(op: &OpType<'_>) -> String {
    op.as_str().to_string()
}

pub fn is_supported(op: &OpType<'_>) -> bool {
    classify(op).0
}

pub fn classify(op: &OpType<'_>) -> (bool, OpPriority) {
    match op {
        OpType::Gemm
        | OpType::MatMul
        | OpType::Add
        | OpType::Sub
        | OpType::Mul
        | OpType::Div
        | OpType::Relu
        | OpType::Sigmoid
        | OpType::Tanh
        | OpType::Softmax
        | OpType::Identity
        | OpType::Dropout
        | OpType::Cast
        | OpType::Neg
        | OpType::Flatten
        | OpType::Reshape
        | OpType::Transpose
        | OpType::Constant
        | OpType::ReduceSum
        | OpType::ReduceMean
        | OpType::ReduceMax
        | OpType::Where
        | OpType::Gather
        | OpType::GatherElements
        | OpType::Expand
        | OpType::Unsqueeze
        | OpType::Squeeze
        | OpType::Concat
        | OpType::Split
        | OpType::LayerNormalization
        | OpType::Gelu
        | OpType::Pow
        | OpType::Sqrt
        | OpType::Clip
        | OpType::Erf
        | OpType::Exp
        | OpType::Log
        | OpType::Abs
        | OpType::Reciprocal
        | OpType::Sin
        | OpType::Cos
        | OpType::Equal
        | OpType::Greater
        | OpType::GreaterOrEqual
        | OpType::Less
        | OpType::LessOrEqual
        | OpType::Not
        | OpType::And
        | OpType::Or
        | OpType::Shape
        | OpType::Slice
        | OpType::Softplus
        | OpType::Attention => (true, OpPriority::Ready),
        OpType::Custom(name) => match *name {
            "RMSNorm"
            | "SimplifiedLayerNormalization"
            | "SkipSimplifiedLayerNormalization"
            | "SkipLayerNormalization"
            | "SiLU"
            | "Silu"
            | "Swish"
            | "BiasGelu"
            | "FastGelu"
            | "QuickGelu"
            | "RoPE"
            | "RotaryEmbedding"
            | "ScaledDotProductAttention"
            | "SDPA"
            | "GroupQueryAttention"
            | "MultiHeadAttention"
            | "Embedding" => (true, OpPriority::Ready),
            other if other.eq_ignore_ascii_case("rmsnorm") => (true, OpPriority::Ready),
            _ => (false, OpPriority::Unknown),
        },
        OpType::Conv
        | OpType::ConvTranspose
        | OpType::AveragePool
        | OpType::MaxPool
        | OpType::GlobalAveragePool
        | OpType::GlobalMaxPool
        | OpType::BatchNormalization
        | OpType::HardSigmoid
        | OpType::HardSwish
        | OpType::GroupNormalization
        | OpType::InstanceNormalization => (false, OpPriority::P3),
        OpType::Einsum | OpType::Pad | OpType::Tile | OpType::Range | OpType::OneHot => {
            (false, OpPriority::P2)
        }
        _ => (false, OpPriority::Unknown),
    }
}

pub fn supported_names() -> &'static [&'static str] {
    &[
        "Gemm",
        "MatMul",
        "Add",
        "Sub",
        "Mul",
        "Div",
        "Relu",
        "Sigmoid",
        "Tanh",
        "Softmax",
        "Identity",
        "Dropout",
        "Cast",
        "Neg",
        "Flatten",
        "Reshape",
        "Transpose",
        "Constant",
        "ReduceSum",
        "ReduceMean",
        "ReduceMax",
        "Where",
        "Gather",
        "GatherElements",
        "Expand",
        "Unsqueeze",
        "Squeeze",
        "Concat",
        "Split",
        "Slice",
        "LayerNormalization",
        "RMSNorm",
        "SiLU",
        "GELU",
        "Pow",
        "Sqrt",
        "Clip",
        "Erf",
        "Sin",
        "Cos",
        "Equal",
        "Greater",
        "Where",
        "Shape",
        "RoPE",
        "ScaledDotProductAttention",
        "Embedding",
    ]
}

#[derive(Debug, Clone)]
pub struct LintReport {
    pub missing: Vec<MissingOp>,
}

#[derive(Debug, Clone)]
pub struct MissingOp {
    pub node: String,
    pub op: String,
    pub domain: String,
    pub priority: OpPriority,
}

impl LintReport {
    pub fn ok(&self) -> bool {
        self.missing.is_empty()
    }

    pub fn format_error(&self) -> String {
        let mut lines = vec!["unsupported ONNX ops (rejected before epoch 0):".to_string()];
        for m in &self.missing {
            let tag = match m.priority {
                OpPriority::P2 => " [roadmap P2: fuse or decompose]",
                OpPriority::P3 => " [roadmap P3: vision / older BERT]",
                OpPriority::Unknown => " [not on the transformer roadmap]",
                OpPriority::Ready => "",
            };
            let domain = if m.domain.is_empty() {
                String::new()
            } else {
                format!(" ({})", m.domain)
            };
            lines.push(format!("  - {} {}{}{}", m.op, m.node, domain, tag));
        }
        lines.push(format!(
            "di-ml trains: {}.",
            supported_names().join(", ")
        ));
        lines.join("\n")
    }
}

pub fn lint_nodes<'a, I>(nodes: I) -> LintReport
where
    I: IntoIterator<Item = &'a onnx_rs::ast::Node<'a>>,
{
    let mut missing = Vec::new();
    for node in nodes {
        let (ok, priority) = classify(&node.op_type);
        if !ok {
            missing.push(MissingOp {
                node: node.name.to_string(),
                op: node.op_type.as_str().to_string(),
                domain: node.domain.to_string(),
                priority,
            });
        }
    }
    missing.sort_by(|a, b| a.op.cmp(&b.op).then(a.node.cmp(&b.node)));
    missing.dedup_by(|a, b| a.op == b.op && a.node == b.node);
    LintReport { missing }
}

pub fn reject_if_missing(report: &LintReport) -> Result<()> {
    if report.ok() {
        Ok(())
    } else {
        Err(Error::usage(report.format_error()))
    }
}

pub fn matches_trainable(pattern: &str, name: &str) -> bool {
    if pattern == name {
        return true;
    }
    if let Ok(re) = regex::Regex::new(pattern) {
        if re.is_match(name) {
            return true;
        }
    }
    let glob = glob_to_regex(pattern);
    regex::Regex::new(&glob)
        .map(|re| re.is_match(name))
        .unwrap_or(false)
}

fn glob_to_regex(pattern: &str) -> String {
    let mut out = String::from("^");
    for ch in pattern.chars() {
        match ch {
            '*' => out.push_str(".*"),
            '?' => out.push('.'),
            '.' | '+' | '(' | ')' | '|' | '[' | ']' | '{' | '}' | '^' | '$' | '\\' => {
                out.push('\\');
                out.push(ch);
            }
            other => out.push(other),
        }
    }
    out.push('$');
    out
}

pub fn is_embedding_name(name: &str) -> bool {
    let n = name.to_ascii_lowercase();
    n.contains("embed")
        || n.contains("wte")
        || n.contains("tok_emb")
        || n.contains("word_emb")
        || n.contains("token_emb")
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn glob_and_regex_trainable() {
        assert!(matches_trainable("lora_.*", "lora_A.W2"));
        assert!(matches_trainable(".*attn.*out_proj.*", "model.layers.0.self_attn.out_proj.weight"));
        assert!(matches_trainable("W*", "W1"));
        assert!(!matches_trainable("W2", "W1"));
    }

    #[test]
    fn freeze_embeddings_matches_table_names() {
        assert!(is_embedding_name("model.embed_tokens.weight"));
        assert!(is_embedding_name("tok_emb"));
        assert!(is_embedding_name("wte"));
        assert!(!is_embedding_name("q_proj"));
        assert!(!is_embedding_name("E"));
        assert!(!is_embedding_name("rms_w"));
    }
}
