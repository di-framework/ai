//! Forward-only session over a trained ONNX graph.
//!
//! This is the runtime half of the train → `dist/model.onnx` → infer pipeline.
//! It loads bytes (no workspace, no optimizer, no tape) so the same API can
//! later sit in WASM without the CLI or filesystem layout.

use std::collections::HashMap;
use std::path::Path;

use crate::error::Result;
use crate::graph::ExecGraph;
use crate::onnx::{self, TensorValue};
use crate::tensor::Tensor;

pub struct Session {
    graph: ExecGraph,
}

impl Session {
    pub fn from_bytes(bytes: Vec<u8>) -> Result<Self> {
        let mut graph = ExecGraph::load(bytes)?;
        graph.set_inference_mode(true);
        Ok(Self { graph })
    }

    pub fn from_path(path: impl AsRef<Path>) -> Result<Self> {
        Self::from_bytes(onnx::load_bytes(path)?)
    }

    pub fn inputs(&self) -> &[String] {
        &self.graph.inputs
    }

    pub fn outputs(&self) -> &[String] {
        &self.graph.outputs
    }

    pub fn run(
        &mut self,
        feeds: &HashMap<String, TensorValue>,
    ) -> Result<HashMap<String, TensorValue>> {
        self.graph.forward_any(feeds)
    }

    pub fn run_f32(
        &mut self,
        feeds: &HashMap<String, Tensor>,
    ) -> Result<HashMap<String, Tensor>> {
        self.graph.forward(feeds)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::onnx::generate_xor_mlp;
    use ndarray::array;

    #[test]
    fn xor_session_preserves_io_names_and_shape() {
        let mut session = Session::from_bytes(generate_xor_mlp(1)).unwrap();
        assert_eq!(session.inputs(), &["input".to_string()]);
        assert_eq!(session.outputs(), &["output".to_string()]);
        let mut feeds = HashMap::new();
        feeds.insert(
            "input".into(),
            array![[0.0f32, 0.0], [0.0, 1.0], [1.0, 0.0], [1.0, 1.0]].into_dyn(),
        );
        let out = session.run_f32(&feeds).unwrap();
        assert_eq!(out["output"].shape(), &[4, 1]);
        assert!(out["output"].iter().all(|v| v.is_finite()));
    }

    #[test]
    fn ocr_rec_graph_is_rejected_with_vision_ops() {
        let path = std::path::PathBuf::from(env!("CARGO_MANIFEST_DIR"))
            .join("../../examples/ocr/models/en_PP-OCRv4_rec_mobile.onnx");
        if !path.is_file() {
            eprintln!(
                "skip ocr lint: {} missing (bun scripts/fetch-ocr-model.ts)",
                path.display()
            );
            return;
        }
        let bytes = std::fs::read(&path).unwrap();
        let err = crate::graph::ExecGraph::load(bytes).unwrap_err();
        assert_eq!(err.exit_code(), 2, "{err}");
        let msg = err.to_string();
        assert!(msg.contains("unsupported ONNX ops"), "{msg}");
        assert!(
            msg.contains("Conv"),
            "OCR rec is a conv graph; lint must name Conv:\n{msg}"
        );
    }
}
