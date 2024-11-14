use ndarray::{Array, Array1};
use rand::Rng;
use rayon::prelude::*;

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