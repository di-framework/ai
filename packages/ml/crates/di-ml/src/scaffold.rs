use std::fs;
use std::path::{Path, PathBuf};

use crate::error::{Error, Result};
use crate::onnx::{generate_xor_mlp, xor_train_jsonl};
use crate::workspace::{
    TrainConfig, DATA_DIR, EVAL_JSONL, MODEL_FILE, TRAIN_JSONL, TRAIN_TOML,
};

pub const DEFAULT_INIT_DIR: &str = "xor-workspace";

#[derive(Debug, Clone)]
pub struct InitSpec {
    pub root: PathBuf,
    pub config: TrainConfig,
    pub overwrite: bool,
}

impl InitSpec {
    pub fn xor_demo(root: impl Into<PathBuf>) -> Self {
        Self {
            root: root.into(),
            config: TrainConfig::default(),
            overwrite: false,
        }
    }
}

/// Write a workspace: XOR MLP `model.onnx`, `train.toml`, and `data/*.jsonl`.
pub fn scaffold(spec: &InitSpec) -> Result<PathBuf> {
    let root = spec.root.clone();
    if root.exists() && root.is_file() {
        return Err(Error::usage(format!(
            "{} exists and is a file",
            root.display()
        )));
    }
    fs::create_dir_all(&root)?;
    fs::create_dir_all(root.join(DATA_DIR))?;

    let model = root.join(MODEL_FILE);
    if model.is_file() && !spec.overwrite {
        return Err(Error::usage(format!(
            "{} already exists; pick another directory or remove it",
            model.display()
        )));
    }

    fs::write(&model, generate_xor_mlp(spec.config.seed))?;
    fs::write(root.join(TRAIN_TOML), spec.config.to_toml()?)?;
    let jsonl = xor_train_jsonl();
    fs::write(root.join(DATA_DIR).join(TRAIN_JSONL), &jsonl)?;
    fs::write(root.join(DATA_DIR).join(EVAL_JSONL), &jsonl)?;
    Ok(root)
}

pub fn default_init_dir() -> PathBuf {
    PathBuf::from(DEFAULT_INIT_DIR)
}

pub fn exists_model(root: impl AsRef<Path>) -> bool {
    root.as_ref().join(MODEL_FILE).is_file()
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::workspace::load;
    use tempfile::tempdir;

    #[test]
    fn writes_xor_layout() {
        let dir = tempdir().unwrap();
        let root = dir.path().join("ws");
        scaffold(&InitSpec::xor_demo(&root)).unwrap();
        let ws = load(&root).unwrap();
        assert!(ws.model.is_file());
        assert!(ws.train_data.is_file());
        assert!(ws.eval_data.is_some());
        assert_eq!(ws.config.loss.as_str(), "mse");
    }

    #[test]
    fn refuses_to_overwrite() {
        let dir = tempdir().unwrap();
        let root = dir.path().join("ws");
        scaffold(&InitSpec::xor_demo(&root)).unwrap();
        let err = scaffold(&InitSpec::xor_demo(&root)).unwrap_err();
        assert_eq!(err.exit_code(), 2);
    }
}
