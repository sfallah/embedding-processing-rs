use crate::prelude::Serde;
use serde::Deserialize;

#[derive(Debug, Deserialize, Clone, PartialEq, Eq)]
#[allow(unused)]
pub struct ModelInfoConfig {
    pub path: String,
    pub n_ctx: i32,
    pub n_embd: i32,
}

impl Serde for ModelInfoConfig {}
