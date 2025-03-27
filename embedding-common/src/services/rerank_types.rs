use crate::prelude::Serde;
use serde::{Deserialize, Serialize};
use serde_repr::{Deserialize_repr, Serialize_repr};

#[derive(Serialize_repr, Deserialize_repr, Debug, PartialEq, Clone, Copy, Default)]
#[repr(u8)]
pub enum TruncationDirection {
    Left = 1,
    #[default]
    Right = 2,
}

#[derive(Serialize, Deserialize, Debug)]
pub struct RerankRequest {
    pub req_id: Option<usize>,
    pub query: String,
    pub texts: Vec<String>,
    #[serde(default)]
    pub truncate: Option<bool>,
    #[serde(default)]
    pub truncation_direction: TruncationDirection,
    #[serde(default)]
    pub raw_scores: bool,
    #[serde(default)]
    pub return_text: bool,
}

impl RerankRequest {
    pub fn new(req_id: Option<usize>, query: String, texts: Vec<String>) -> Self {
        RerankRequest {
            req_id,
            query,
            raw_scores: false,
            return_text: true,
            texts,
            truncate: Some(true),
            truncation_direction: TruncationDirection::Right,
        }
    }
}

impl Serde for RerankRequest {}

unsafe impl Sync for RerankRequest {}
unsafe impl Send for RerankRequest {}

#[derive(Serialize, Deserialize, Debug, Clone)]
pub struct Rank {
    pub index: usize,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub text: Option<String>,
    pub score: f32,
}

impl Rank {
    pub fn new(index: usize, text: Option<String>, score: f32) -> Self {
        Rank { index, text, score }
    }
}
unsafe impl Sync for Rank {}
unsafe impl Send for Rank {}

impl Serde for Rank {}

#[derive(Serialize, Deserialize, Debug)]
pub struct RerankResponse {
    #[serde(skip_serializing_if = "Option::is_none")]
    pub req_id: Option<usize>,
    pub ranks: Vec<Rank>,
}

impl RerankResponse {
    pub fn new(req_id: Option<usize>, ranks: Vec<Rank>) -> Self {
        RerankResponse { req_id, ranks }
    }
}

unsafe impl Sync for RerankResponse {}
unsafe impl Send for RerankResponse {}

impl Serde for RerankResponse {}
