use anyhow::anyhow;
use rand::{thread_rng, Rng};
use usearch::{new_index, IndexOptions, MetricKind, ScalarKind};
use embedding_index::utils::generate_random_vectors;

fn main() -> anyhow::Result<()> {
    let options = IndexOptions {
        dimensions: 384, // necessary for most metric kinds
        metric: MetricKind::L2sq, // or ::L2sq, ::Cos ...
        quantization: ScalarKind::F32, // or ::F32, ::F16, ::I8, ::B1x8 ...
        connectivity: 0, // zero for auto
        expansion_add: 0, // zero for auto
        expansion_search: 0, // zero for auto
        multi: false,
    };

    let index = new_index(&options).map_err(|e| anyhow!("Failed to create index: {:?}", e))?;
    index.reserve(1000).map_err(|e| anyhow!("Failed to reserve index: {:?}", e))?;
    let binding = generate_random_vectors(1, 384);
    let embedding = binding.get(0).ok_or(anyhow!("No embeddings"))?;
    let embed_id: u64 = thread_rng().gen();
    index.add(embed_id, embedding)?;
    Ok(())
}