use std::collections::HashSet;
use std::ops::Range;

use super::ExecNode;

/// Split the node list into checkpoint segments (attention/MLP blocks when
/// names contain a layer index; otherwise chunks of `every` nodes).
pub fn segments(nodes: &[ExecNode], enabled: bool, every: usize) -> Vec<Range<usize>> {
    if !enabled || nodes.is_empty() {
        return vec![0..nodes.len()];
    }
    let every = every.max(1);
    let mut out: Vec<Range<usize>> = Vec::new();
    let mut start = 0;
    let mut prev_layer: Option<i64> = layer_id(&nodes[0].name);
    for (i, node) in nodes.iter().enumerate().skip(1) {
        let layer = layer_id(&node.name);
        let split = match (prev_layer, layer) {
            (Some(a), Some(b)) if a != b => true,
            _ if i - start >= every => true,
            _ => false,
        };
        if split {
            out.push(start..i);
            start = i;
            prev_layer = layer;
        }
    }
    out.push(start..nodes.len());
    out
}

fn layer_id(name: &str) -> Option<i64> {
    // model.layers.12.mlp / /layer.3/ /h.0/
    let re = regex::Regex::new(r"(?i)(?:layers?|h|block)[./_](\d+)").ok()?;
    re.captures(name)
        .and_then(|c| c.get(1))
        .and_then(|m| m.as_str().parse().ok())
}

pub fn live_in(nodes: &[ExecNode], range: Range<usize>) -> HashSet<String> {
    let mut produced = HashSet::new();
    let mut needed = HashSet::new();
    for node in &nodes[range] {
        for inp in &node.inputs {
            if !produced.contains(inp) {
                needed.insert(inp.clone());
            }
        }
        for out in &node.outputs {
            produced.insert(out.clone());
        }
    }
    needed
}

pub fn live_after(nodes: &[ExecNode], end: usize, outputs: &[String]) -> HashSet<String> {
    let mut used = HashSet::new();
    for o in outputs {
        used.insert(o.clone());
    }
    for node in &nodes[end..] {
        for inp in &node.inputs {
            used.insert(inp.clone());
        }
    }
    used
}
