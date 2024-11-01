use crate::utils::hash_utils::DeterministicAHasher;
use fast_text_splitter::config::SplitterLiteConfig;
use fast_text_splitter::hf_tokenizer::HFTokenizer;
use std::sync::Arc;

#[derive(Clone)]
pub struct ProcessingContext {
    pub splitter: Arc<SplitterLiteConfig<HFTokenizer>>,
    pub sentence_splitter: Arc<SplitterLiteConfig<HFTokenizer>>,
    pub hasher: Arc<DeterministicAHasher>,
    pub n_embd: usize,
}
unsafe impl Sync for ProcessingContext {}
unsafe impl Send for ProcessingContext {}
