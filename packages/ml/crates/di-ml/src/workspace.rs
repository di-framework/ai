use std::fs;
use std::path::{Path, PathBuf};

use serde::{Deserialize, Serialize};

use crate::error::{missing_file, Error, Result};

pub const MODEL_FILE: &str = "model.onnx";
pub const TRAIN_TOML: &str = "train.toml";
pub const DATA_DIR: &str = "data";
pub const TRAIN_JSONL: &str = "train.jsonl";
pub const EVAL_JSONL: &str = "eval.jsonl";
pub const FROZEN_FILE: &str = "frozen.txt";
pub const TRAINABLE_FILE: &str = "trainable.txt";
pub const DIST_DIR: &str = "dist";

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "kebab-case")]
pub enum LossKind {
    Mse,
    #[serde(alias = "cross-entropy", alias = "ce")]
    CrossEntropy,
    #[serde(alias = "bce-logits", alias = "bce")]
    BceLogits,
    L1,
    #[serde(alias = "info-nce", alias = "infonce", alias = "multiple-negatives", alias = "mnrl")]
    InfoNce,
    Cosine,
    Triplet,
    #[serde(alias = "supcon", alias = "supervised-contrastive")]
    SupCon,
    #[serde(alias = "distillation")]
    Distill,
}

impl LossKind {
    pub fn as_str(self) -> &'static str {
        match self {
            Self::Mse => "mse",
            Self::CrossEntropy => "cross-entropy",
            Self::BceLogits => "bce-logits",
            Self::L1 => "l1",
            Self::InfoNce => "infonce",
            Self::Cosine => "cosine",
            Self::Triplet => "triplet",
            Self::SupCon => "supcon",
            Self::Distill => "distill",
        }
    }

    pub fn parse(s: &str) -> Result<Self> {
        match s {
            "mse" => Ok(Self::Mse),
            "cross-entropy" | "ce" => Ok(Self::CrossEntropy),
            "bce-logits" | "bce" => Ok(Self::BceLogits),
            "l1" => Ok(Self::L1),
            "infonce" | "info-nce" | "multiple-negatives" | "mnrl" => Ok(Self::InfoNce),
            "cosine" => Ok(Self::Cosine),
            "triplet" => Ok(Self::Triplet),
            "supcon" | "supervised-contrastive" => Ok(Self::SupCon),
            "distill" | "distillation" => Ok(Self::Distill),
            other => Err(Error::usage(format!(
                "unknown loss {other:?}; expected mse, cross-entropy, bce-logits, l1, infonce, cosine, triplet, supcon, or distill"
            ))),
        }
    }

    pub fn is_embedding(self) -> bool {
        matches!(
            self,
            Self::InfoNce | Self::Cosine | Self::Triplet | Self::SupCon | Self::Distill
        )
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "kebab-case")]
pub enum OptimizerKind {
    Adamw,
    Sgd,
}

impl OptimizerKind {
    pub fn as_str(self) -> &'static str {
        match self {
            Self::Adamw => "adamw",
            Self::Sgd => "sgd",
        }
    }

    pub fn parse(s: &str) -> Result<Self> {
        match s {
            "adamw" => Ok(Self::Adamw),
            "sgd" => Ok(Self::Sgd),
            other => Err(Error::usage(format!(
                "unknown optimizer {other:?}; expected adamw or sgd"
            ))),
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize, Default)]
#[serde(rename_all = "kebab-case")]
pub enum PoolKind {
    #[default]
    None,
    #[serde(alias = "last_token", alias = "last")]
    LastToken,
    Mean,
    Cls,
}

impl PoolKind {
    pub fn as_str(self) -> &'static str {
        match self {
            Self::None => "none",
            Self::LastToken => "last-token",
            Self::Mean => "mean",
            Self::Cls => "cls",
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize, Default)]
#[serde(rename_all = "kebab-case")]
pub enum LoraExport {
    #[default]
    Merge,
    Adapter,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize, Default)]
#[serde(rename_all = "kebab-case")]
pub enum MixedPrecision {
    #[default]
    Off,
    Fp16,
    Bf16,
}

impl MixedPrecision {
    pub fn as_str(self) -> &'static str {
        match self {
            Self::Off => "off",
            Self::Fp16 => "fp16",
            Self::Bf16 => "bf16",
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize, Default)]
#[serde(rename_all = "kebab-case")]
pub enum EvalMetric {
    #[default]
    Loss,
    Spearman,
    Ndcg,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(rename_all = "kebab-case")]
pub struct TrainConfig {
    #[serde(default = "default_loss")]
    pub loss: LossKind,
    #[serde(default = "default_optimizer")]
    pub optimizer: OptimizerKind,
    #[serde(default = "default_lr")]
    pub learning_rate: f32,
    #[serde(default = "default_steps")]
    pub max_steps: usize,
    #[serde(default = "default_batch")]
    pub batch_size: usize,
    #[serde(default = "default_seed")]
    pub seed: u64,
    #[serde(default = "default_log_every")]
    pub log_every: usize,
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub frozen: Vec<String>,
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub trainable: Vec<String>,
    #[serde(default, skip_serializing_if = "std::ops::Not::not")]
    pub freeze_embeddings: bool,
    #[serde(default, skip_serializing_if = "Option::is_none", alias = "lora_rank")]
    pub lora_rank: Option<usize>,
    #[serde(default = "default_lora_alpha", skip_serializing_if = "is_default_lora_alpha", alias = "lora_alpha")]
    pub lora_alpha: f32,
    #[serde(default, skip_serializing_if = "Vec::is_empty", alias = "lora_targets")]
    pub lora_targets: Vec<String>,
    #[serde(default, skip_serializing_if = "is_default_lora_export", alias = "lora_export")]
    pub lora_export: LoraExport,
    #[serde(default = "default_temperature", skip_serializing_if = "is_default_temperature")]
    pub temperature: f32,
    #[serde(default, skip_serializing_if = "is_default_pool")]
    pub pool: PoolKind,
    #[serde(default, skip_serializing_if = "std::ops::Not::not", alias = "l2_normalize")]
    pub l2_normalize: bool,
    #[serde(default, skip_serializing_if = "Option::is_none", alias = "embedding_output")]
    pub embedding_output: Option<String>,
    #[serde(default, skip_serializing_if = "Option::is_none", alias = "microbatch_size")]
    pub microbatch_size: Option<usize>,
    #[serde(default = "default_one", skip_serializing_if = "is_one", alias = "grad_accum_steps")]
    pub grad_accum_steps: usize,
    #[serde(default, skip_serializing_if = "is_default_mixed", alias = "mixed_precision")]
    pub mixed_precision: MixedPrecision,
    #[serde(default, skip_serializing_if = "std::ops::Not::not", alias = "activation_checkpointing")]
    pub activation_checkpointing: bool,
    #[serde(default = "default_checkpoint_every", skip_serializing_if = "is_default_checkpoint_every", alias = "checkpoint_every")]
    pub checkpoint_every: usize,
    #[serde(default, skip_serializing_if = "is_zero_f32", alias = "replay_ratio")]
    pub replay_ratio: f32,
    #[serde(default, skip_serializing_if = "is_zero_f32", alias = "replay_kl_weight")]
    pub replay_kl_weight: f32,
    #[serde(default, skip_serializing_if = "is_zero_f32", alias = "distill_weight")]
    pub distill_weight: f32,
    #[serde(default = "default_triplet_margin", skip_serializing_if = "is_default_triplet_margin", alias = "triplet_margin")]
    pub triplet_margin: f32,
    #[serde(default, skip_serializing_if = "is_default_eval")]
    pub eval_metric: EvalMetric,
    #[serde(default, skip_serializing_if = "std::ops::Not::not")]
    pub resume: bool,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub tokenizer: Option<String>,
    #[serde(default, skip_serializing_if = "Option::is_none", alias = "instruction_query")]
    pub instruction_query: Option<String>,
    #[serde(default, skip_serializing_if = "Option::is_none", alias = "instruction_doc")]
    pub instruction_doc: Option<String>,
    #[serde(default, skip_serializing_if = "is_zero_usize", alias = "max_length")]
    pub max_length: usize,
    #[serde(default, skip_serializing_if = "is_zero_usize", alias = "hard_negatives")]
    pub hard_negatives: usize,
}

fn default_loss() -> LossKind {
    LossKind::Mse
}
fn default_optimizer() -> OptimizerKind {
    OptimizerKind::Adamw
}
fn default_lr() -> f32 {
    0.05
}
fn default_steps() -> usize {
    400
}
fn default_batch() -> usize {
    4
}
fn default_seed() -> u64 {
    1
}
fn default_log_every() -> usize {
    50
}
fn default_lora_alpha() -> f32 {
    32.0
}
fn is_default_lora_alpha(v: &f32) -> bool {
    (*v - 32.0).abs() < f32::EPSILON
}
fn is_default_lora_export(v: &LoraExport) -> bool {
    *v == LoraExport::Merge
}
fn default_temperature() -> f32 {
    0.07
}
fn is_default_temperature(v: &f32) -> bool {
    (*v - 0.07).abs() < 1e-6
}
fn is_default_pool(v: &PoolKind) -> bool {
    *v == PoolKind::None
}
fn default_one() -> usize {
    1
}
fn is_one(v: &usize) -> bool {
    *v == 1
}
fn is_default_mixed(v: &MixedPrecision) -> bool {
    *v == MixedPrecision::Off
}
fn default_checkpoint_every() -> usize {
    8
}
fn is_default_checkpoint_every(v: &usize) -> bool {
    *v == 8
}
fn is_zero_f32(v: &f32) -> bool {
    *v == 0.0
}
fn is_zero_usize(v: &usize) -> bool {
    *v == 0
}
fn default_triplet_margin() -> f32 {
    0.2
}
fn is_default_triplet_margin(v: &f32) -> bool {
    (*v - 0.2).abs() < 1e-6
}
fn is_default_eval(v: &EvalMetric) -> bool {
    *v == EvalMetric::Loss
}

impl Default for TrainConfig {
    fn default() -> Self {
        Self {
            loss: default_loss(),
            optimizer: default_optimizer(),
            learning_rate: default_lr(),
            max_steps: default_steps(),
            batch_size: default_batch(),
            seed: default_seed(),
            log_every: default_log_every(),
            frozen: Vec::new(),
            trainable: Vec::new(),
            freeze_embeddings: false,
            lora_rank: None,
            lora_alpha: default_lora_alpha(),
            lora_targets: default_lora_targets(),
            lora_export: LoraExport::Merge,
            temperature: default_temperature(),
            pool: PoolKind::None,
            l2_normalize: false,
            embedding_output: None,
            microbatch_size: None,
            grad_accum_steps: 1,
            mixed_precision: MixedPrecision::Off,
            activation_checkpointing: false,
            checkpoint_every: default_checkpoint_every(),
            replay_ratio: 0.0,
            replay_kl_weight: 0.0,
            distill_weight: 0.0,
            triplet_margin: default_triplet_margin(),
            eval_metric: EvalMetric::Loss,
            resume: false,
            tokenizer: None,
            instruction_query: None,
            instruction_doc: None,
            max_length: 0,
            hard_negatives: 0,
        }
    }
}

fn default_lora_targets() -> Vec<String> {
    Vec::new()
}

impl TrainConfig {
    pub fn to_toml(&self) -> Result<String> {
        toml::to_string_pretty(self).map_err(|err| Error::fail(err.to_string()))
    }

    pub fn default_lora_target_list() -> Vec<String> {
        ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]
            .into_iter()
            .map(str::to_string)
            .collect()
    }

    pub fn lora_targets_effective(&self) -> Vec<String> {
        if self.lora_targets.is_empty() && self.lora_rank.is_some() {
            Self::default_lora_target_list()
        } else {
            self.lora_targets.clone()
        }
    }
}

#[derive(Debug, Deserialize, Default)]
#[serde(default)]
struct TrainFile {
    #[serde(flatten)]
    root: TrainConfig,
    train: Option<TrainConfig>,
}

pub fn parse_train_toml(text: &str) -> Result<TrainConfig> {
    let file: TrainFile = toml::from_str(text).map_err(|err| Error::usage(format!("invalid train.toml: {err}")))?;
    Ok(file.train.unwrap_or(file.root))
}

#[derive(Debug, Clone)]
pub struct Workspace {
    pub root: PathBuf,
    pub model: PathBuf,
    pub train_data: PathBuf,
    pub eval_data: Option<PathBuf>,
    pub config: TrainConfig,
    pub frozen: Vec<String>,
    pub trainable: Vec<String>,
    pub train_toml_bytes: Vec<u8>,
}

impl Workspace {
    pub fn dist_dir(&self) -> PathBuf {
        self.root.join(DIST_DIR)
    }

    pub fn dist_model(&self) -> PathBuf {
        self.dist_dir().join(MODEL_FILE)
    }

    pub fn dist_metrics(&self) -> PathBuf {
        self.dist_dir().join("metrics.json")
    }

    pub fn dist_adapter(&self) -> PathBuf {
        self.dist_dir().join("adapter.onnx")
    }

    pub fn dist_optimizer(&self) -> PathBuf {
        self.dist_dir().join("optimizer.json")
    }
}

pub fn load(root: impl AsRef<Path>) -> Result<Workspace> {
    let root = root.as_ref().to_path_buf();
    if !root.exists() {
        return Err(Error::usage(format!(
            "workspace {} does not exist; run `di-ml init` to scaffold one",
            root.display()
        )));
    }
    if !root.is_dir() {
        return Err(Error::usage(format!(
            "{} is not a directory",
            root.display()
        )));
    }

    let model = root.join(MODEL_FILE);
    if !model.is_file() {
        return Err(missing_file(
            model,
            "Place a forward ONNX graph here, or run `di-ml init`.",
        ));
    }

    let train_toml = root.join(TRAIN_TOML);
    if !train_toml.is_file() {
        return Err(missing_file(
            train_toml,
            "Run `di-ml init` to write train.toml.",
        ));
    }
    let train_toml_bytes = fs::read(&train_toml)?;
    let text = String::from_utf8_lossy(&train_toml_bytes);
    let config = parse_train_toml(&text)?;

    if config.max_steps == 0 {
        return Err(Error::usage("train.toml max-steps must be > 0"));
    }
    if config.batch_size == 0 {
        return Err(Error::usage("train.toml batch-size must be > 0"));
    }
    if !config.learning_rate.is_finite() || config.learning_rate <= 0.0 {
        return Err(Error::usage("train.toml learning-rate must be > 0"));
    }
    if let Some(rank) = config.lora_rank {
        if rank == 0 {
            return Err(Error::usage("train.toml lora-rank must be > 0"));
        }
    }
    if let Some(micro) = config.microbatch_size {
        if micro == 0 {
            return Err(Error::usage("train.toml microbatch-size must be > 0"));
        }
    }
    if config.grad_accum_steps == 0 {
        return Err(Error::usage("train.toml grad-accum-steps must be > 0"));
    }

    let train_data = root.join(DATA_DIR).join(TRAIN_JSONL);
    if !train_data.is_file() {
        return Err(missing_file(
            train_data,
            "Add JSONL rows under data/train.jsonl, or run `di-ml init`.",
        ));
    }

    let eval_path = root.join(DATA_DIR).join(EVAL_JSONL);
    let eval_data = eval_path.is_file().then_some(eval_path);

    let mut frozen = config.frozen.clone();
    frozen.extend(read_name_list(&root.join(FROZEN_FILE))?);
    frozen.sort();
    frozen.dedup();

    let mut trainable = config.trainable.clone();
    trainable.extend(read_name_list(&root.join(TRAINABLE_FILE))?);
    trainable.sort();
    trainable.dedup();

    Ok(Workspace {
        root,
        model,
        train_data,
        eval_data,
        config,
        frozen,
        trainable,
        train_toml_bytes,
    })
}

fn read_name_list(path: &Path) -> Result<Vec<String>> {
    if !path.is_file() {
        return Ok(Vec::new());
    }
    let mut names = Vec::new();
    for (idx, line) in fs::read_to_string(path)?.lines().enumerate() {
        let line = line.trim();
        if line.is_empty() || line.starts_with('#') {
            continue;
        }
        if line.split_whitespace().nth(1).is_some() {
            return Err(Error::usage(format!(
                "{}:{}: expected one initializer name per line",
                path.display(),
                idx + 1
            )));
        }
        names.push(line.to_string());
    }
    Ok(names)
}

#[cfg(test)]
mod tests {
    use super::*;
    use tempfile::tempdir;

    #[test]
    fn load_rejects_empty_directory() {
        let dir = tempdir().unwrap();
        let err = load(dir.path()).unwrap_err();
        assert_eq!(err.exit_code(), 2);
        assert!(err.to_string().contains("model.onnx"));
    }

    #[test]
    fn load_reads_layout() {
        let dir = tempdir().unwrap();
        fs::write(dir.path().join(MODEL_FILE), b"onnx").unwrap();
        fs::write(
            dir.path().join(TRAIN_TOML),
            TrainConfig::default().to_toml().unwrap(),
        )
        .unwrap();
        fs::create_dir(dir.path().join(DATA_DIR)).unwrap();
        fs::write(dir.path().join(DATA_DIR).join(TRAIN_JSONL), "{}\n").unwrap();
        fs::write(dir.path().join(FROZEN_FILE), "W1\n# comment\nB1\n").unwrap();

        let ws = load(dir.path()).unwrap();
        assert_eq!(ws.frozen, vec!["B1".to_string(), "W1".to_string()]);
        assert_eq!(ws.config.loss, LossKind::Mse);
    }

    #[test]
    fn nested_train_table() {
        let text = r#"
[train]
loss = "infonce"
lora_rank = 16
trainable = ["lora_.*"]
"#;
        let cfg = parse_train_toml(text).unwrap();
        assert_eq!(cfg.loss, LossKind::InfoNce);
        assert_eq!(cfg.lora_rank, Some(16));
        assert_eq!(cfg.trainable, vec!["lora_.*".to_string()]);
    }
}
