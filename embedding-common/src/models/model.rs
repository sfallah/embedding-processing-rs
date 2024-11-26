use crate::prelude::Serde;
use crate::utils::hashing::DeterministicAHasher;
use serde::{Deserialize, Serialize};
use std::hash::Hasher;

#[derive(Serialize, Deserialize, Debug, PartialEq, Clone)]
pub struct Model {
    pub model_id: u64,
    pub path: String,
    pub n_ctx: i32,
    pub n_embd: i32,
}
impl Model {
    pub fn new(hasher: &DeterministicAHasher, path: String, n_ctx: i32, n_embd: i32) -> Self {
        let model_id = Self::hash(hasher, path.clone(), n_ctx, n_embd);
        Self {
            model_id,
            path,
            n_ctx,
            n_embd,
        }
    }

    pub fn hash(hasher: &DeterministicAHasher, path: String, n_ctx: i32, n_embd: i32) -> u64 {
        let mut hasher = hasher.get_hasher();
        hasher.write(path.as_bytes());
        hasher.write_i32(n_ctx);
        hasher.write_i32(n_embd);
        hasher.finish()
    }
}
impl Serde for Model {}
