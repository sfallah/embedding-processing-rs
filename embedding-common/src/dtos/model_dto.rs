use serde::{Deserialize, Serialize};

#[derive(Serialize, Deserialize, Clone, Debug)]
pub struct ModelDto {
    pub model_id: u64,
    pub path: String,
    pub n_ctx: usize,
    pub n_embd: usize,
}

impl ModelDto {
    pub fn new(model_id: u64, path: String, n_ctx: usize, n_embd: usize) -> Self {
        Self {
            model_id,
            path,
            n_ctx,
            n_embd,
        }
    }
}
