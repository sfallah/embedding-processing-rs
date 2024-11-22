use std::path::Path;
use ndarray::{Array, Array1};
use rand::Rng;
use rayon::prelude::*;
use usearch::{IndexOptions, MetricKind, ScalarKind};
use embedding_common::config::IndexConfig;

fn metric_kind(conf_metric: embedding_common::config::MetricKind) -> MetricKind {
    match conf_metric {
        embedding_common::config::MetricKind::IP => MetricKind::IP,
        embedding_common::config::MetricKind::L2sq => MetricKind::L2sq,
        embedding_common::config::MetricKind::Cos => MetricKind::Cos,
    }
}
fn scalar_kind(conf_scalar: embedding_common::config::ScalarKind) -> ScalarKind {
    match conf_scalar {
        embedding_common::config::ScalarKind::F64 => ScalarKind::F64,
        embedding_common::config::ScalarKind::F32 => ScalarKind::F32,
        embedding_common::config::ScalarKind::F16 => ScalarKind::F16,
        embedding_common::config::ScalarKind::BF16 => ScalarKind::BF16,
        embedding_common::config::ScalarKind::I8 => ScalarKind::I8,
        embedding_common::config::ScalarKind::B1 => ScalarKind::B1,
    }
}

pub(crate) fn from_config(index_config: &IndexConfig) -> IndexOptions {
    let metric = metric_kind(index_config.metric_kind);
    let scalar = scalar_kind(index_config.scalar_kind);
    IndexOptions {
        dimensions: index_config.dimensions,
        metric,
        quantization: scalar,
        connectivity: index_config.connectivity,
        expansion_add: index_config.expansion_add,
        expansion_search: index_config.expansion_search,
        multi: false,
    }
}

pub fn index_file(base_name: String,index_config: &IndexConfig) -> String {
    format!("{}/{}__{}_{}_{}_{}_{}_{}.usearch",
            index_config.index_dir,
            base_name,
            index_config.metric_kind.to_string(),
            index_config.scalar_kind.to_string(),
            index_config.dimensions,
            index_config.connectivity,
            index_config.expansion_add,
            index_config.expansion_search)
}

pub fn create_dir(dir: &str) -> anyhow::Result<bool> {
    let path = Path::new(dir);
    if !path.exists() {
        std::fs::create_dir_all(path)?;
        Ok(true)
    } else {
        Ok(false)
    }
}


pub fn norm(array: &Array1<f32>) -> f32 {
    array.pow2().sum().sqrt()
}

pub fn normalize(embeddings: &Vec<f32>) -> Vec<f32> {
    let embeddings: Array1<f32> = Array::from(embeddings.to_vec());
    let denom = norm(&embeddings).clamp(1e-12, f32::INFINITY);
    let normed = embeddings / denom;
    normed.flatten().to_vec()
}

pub fn generate_random_vectors(num_vectors: usize, dim: usize) -> Vec<Vec<f32>> {
    (0..num_vectors)
        .into_par_iter() // Use rayon's parallel iterator
        .map(|_| {
            let mut rng = rand::thread_rng();
            (0..dim).map(|_| rng.gen::<f32>()).collect()
        })
        .collect()
}