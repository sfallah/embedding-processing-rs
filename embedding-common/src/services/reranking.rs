use serde::{Deserialize, Serialize};
use crate::prelude::Serde;

#[derive(Serialize, Deserialize, Debug)]
pub enum TruncationDirection {
    Left,
    Right,
    Both,
}

#[derive(Serialize, Deserialize, Debug)]
pub struct RerankRequest {
    pub req_id: usize,
    pub query: String,
    pub raw_scores: bool,
    pub return_text: bool,
    pub texts: Vec<String>,
    pub truncate: bool,
    pub truncation_direction: TruncationDirection,
}
impl RerankRequest {
    pub fn new(req_id: usize, query: String, texts: Vec<String>) -> Self {
        RerankRequest {
            req_id,
            query,
            raw_scores: false,
            return_text: true,
            texts,
            truncate: true,
            truncation_direction: TruncationDirection::Right,
        }
    }
}

impl Serde for RerankRequest {}

unsafe impl Sync for RerankRequest {}
unsafe impl Send for RerankRequest {}

#[derive(Serialize, Deserialize, Debug, Clone)]
pub struct Ranking {
    pub idx: usize,
    pub score: f32,
    pub text: String,
}

impl Ranking {
    pub fn new(idx: usize, score: f32, text: String) -> Self {
        Ranking { idx, score, text }
    }
}
unsafe impl Sync for Ranking {}
unsafe impl Send for Ranking {}
