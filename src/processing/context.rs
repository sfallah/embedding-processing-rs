use crate::services::embeddings::EmbeddingsRequest;
use crate::utils::hash_utils::DeterministicAHasher;
use async_std::channel::Sender;
use fast_text_splitter::config::SplitterLiteConfig;
use fast_text_splitter::hf_tokenizer::HFTokenizer;
use std::sync::Arc;

#[derive(Clone)]
pub struct ProcessingContext {
    pub splitter: Arc<SplitterLiteConfig<HFTokenizer>>,
    pub sentence_splitter: Arc<SplitterLiteConfig<HFTokenizer>>,
    pub hasher: Arc<DeterministicAHasher>,
    pub embeddings_sender: Arc<Sender<EmbeddingsRequest>>,
    pub n_embd: usize,
    //pub embeddings_recv_task: Arc<JoinHandle<()>>,
}
unsafe impl Sync for ProcessingContext {}
unsafe impl Send for ProcessingContext {}
