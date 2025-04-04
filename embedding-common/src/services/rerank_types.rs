use crate::prelude::Serde;
use serde::{Deserialize, Serialize};

#[derive(Serialize, Deserialize, Debug)]
pub struct RerankRequest {
    pub query: String,
    pub texts: Vec<String>,
    pub return_text: bool,
}

impl RerankRequest {
    pub fn new(query: String, texts: Vec<String>, return_text: bool) -> Self {
        RerankRequest {
            query,
            texts,
            return_text,
        }
    }
}

impl Serde for RerankRequest {}

unsafe impl Sync for RerankRequest {}
unsafe impl Send for RerankRequest {}

#[derive(Serialize, Deserialize, Debug, Clone)]
pub struct Rank {
    pub index: usize,
    pub rank: Option<usize>,
    pub score: Option<f32>,
    pub text: Option<String>,
}

impl Rank {
    pub fn new(
        index: usize,
        rank: Option<usize>,
        score: Option<f32>,
        text: Option<String>,
    ) -> Self {
        Rank {
            index,
            rank,
            score,
            text,
        }
    }
}
unsafe impl Sync for Rank {}
unsafe impl Send for Rank {}

impl Serde for Rank {}

#[derive(Serialize, Deserialize, Debug)]
pub struct RerankResponse {
    pub model_id: Option<u64>,
    pub ranks: Vec<Rank>,
    pub error: Option<String>,
}

impl RerankResponse {
    pub fn new(model_id: Option<u64>, ranks: Vec<Rank>, error: Option<String>) -> Self {
        RerankResponse {
            model_id,
            ranks,
            error,
        }
    }
}

unsafe impl Sync for RerankResponse {}
unsafe impl Send for RerankResponse {}

impl Serde for RerankResponse {}
