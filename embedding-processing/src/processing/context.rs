use embedding_common::utils::hashing::DeterministicAHasher;
use fast_text_splitter::config::SplitterLiteConfig;
use fast_text_splitter::hf_tokenizer::HFTokenizer;
use std::sync::Arc;

#[derive(Clone)]
pub struct ProcessingContext {
    pub embedding_endpoint: String,
    pub reranking_endpoint: Option<String>,
    pub zmq_context: Arc<zmq::Context>,
    pub splitter: Arc<SplitterLiteConfig<HFTokenizer>>,
    pub sentence_splitter: Arc<SplitterLiteConfig<HFTokenizer>>,
    pub hasher: Arc<DeterministicAHasher>,
    pub n_embd: usize,
    pub model_id: u64,
}
unsafe impl Sync for ProcessingContext {}
unsafe impl Send for ProcessingContext {}
