use crate::error::{Error, Result};
use crate::workspace::TrainConfig;

pub struct Tokenizer {
    inner: tokenizers::Tokenizer,
    query_prefix: String,
    doc_prefix: String,
    max_length: usize,
}

impl Tokenizer {
    pub fn load(cfg: &TrainConfig, root: &std::path::Path) -> Result<Option<Self>> {
        let Some(path) = &cfg.tokenizer else {
            return Ok(None);
        };
        let path = if std::path::Path::new(path).is_absolute() {
            std::path::PathBuf::from(path)
        } else {
            root.join(path)
        };
        let inner = tokenizers::Tokenizer::from_file(&path).map_err(|err| {
            Error::usage(format!("failed to load tokenizer {}: {err}", path.display()))
        })?;
        Ok(Some(Self {
            inner,
            query_prefix: cfg.instruction_query.clone().unwrap_or_default(),
            doc_prefix: cfg.instruction_doc.clone().unwrap_or_default(),
            max_length: if cfg.max_length == 0 { 512 } else { cfg.max_length },
        }))
    }

    pub fn encode_query(&self, text: &str) -> Result<Vec<i64>> {
        self.encode(&format!("{}{text}", self.query_prefix))
    }

    pub fn encode_doc(&self, text: &str) -> Result<Vec<i64>> {
        self.encode(&format!("{}{text}", self.doc_prefix))
    }

    fn encode(&self, text: &str) -> Result<Vec<i64>> {
        let enc = self
            .inner
            .encode(text, true)
            .map_err(|err| Error::fail(err.to_string()))?;
        let mut ids: Vec<i64> = enc.get_ids().iter().map(|id| *id as i64).collect();
        if self.max_length > 0 && ids.len() > self.max_length {
            ids.truncate(self.max_length);
        }
        Ok(ids)
    }
}
