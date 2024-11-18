use std::fmt;
use std::fmt::{Display, Formatter};
use serde::Deserialize;

#[derive(Debug, Deserialize)]
#[allow(unused)]
pub struct IndexConfig {
    #[serde(default = "default_dir")]
    pub index_dir: String,
    pub dimensions: usize,
    #[serde(default)]
    pub metric_kind: MetricKind,
    #[serde(default)]
    pub scalar_kind: ScalarKind,
    #[serde(default)]
    pub connectivity: usize,
    #[serde(default)]
    pub expansion_add: usize,
    #[serde(default)]
    pub expansion_search: usize,
}

fn default_dir() -> String {
    "index_dir".to_string()
}

#[derive(Debug, Deserialize, Clone, Copy, PartialEq, Eq)]
pub enum MetricKind {
    /// The Inner Product metric, defined as `IP = 1 - sum(a[i] * b[i])`.
    IP,
    /// The squared Euclidean Distance metric, defined as `L2 = sum((a[i] - b[i])^2)`.
    L2sq,
    /// The Cosine Similarity metric, defined as `Cos = 1 - sum(a[i] * b[i]) / (sqrt(sum(a[i]^2) * sqrt(sum(b[i]^2)))`.
    Cos,
}

impl Default for MetricKind {
    fn default() -> Self {
        MetricKind::L2sq
    }
}

impl Display for MetricKind {
    fn fmt(&self, f: &mut Formatter) -> fmt::Result {
        match self {
            MetricKind::IP => write!(f, "IP"),
            MetricKind::L2sq => write!(f, "L2sq"),
            MetricKind::Cos => write!(f, "Cos"),
        }
    }
}

#[derive(Debug, Deserialize, Clone, Copy, PartialEq, Eq)]
pub enum ScalarKind {
    /// 64-bit double-precision IEEE 754 floating-point number.
    F64,
    /// 32-bit single-precision IEEE 754 floating-point number.
    F32,
    /// 16-bit half-precision IEEE 754 floating-point number (different from `bf16`).
    F16,
    /// 16-bit brain floating-point number.
    BF16,
    /// 8-bit signed integer.
    I8,
    /// 1-bit binary value, packed 8 per byte.
    B1,
}

impl Default for ScalarKind {
    fn default() -> Self {
        ScalarKind::F32
    }
}

impl Display for ScalarKind {
    fn fmt(&self, f: &mut Formatter) -> fmt::Result {
        match self {
            ScalarKind::F64 => write!(f, "F64"),
            ScalarKind::F32 => write!(f, "F32"),
            ScalarKind::F16 => write!(f, "F16"),
            ScalarKind::BF16 => write!(f, "BF16"),
            ScalarKind::I8 => write!(f, "I8"),
            ScalarKind::B1 => write!(f, "B1"),
        }
    }
}
