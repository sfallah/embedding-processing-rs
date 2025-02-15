use embedding_common::utils::hashing::DeterministicAHasher;
use fast_text_splitter::config::SplitterLiteConfig;
use fast_text_splitter::hf_tokenizer::HFTokenizer;
use std::rc::Rc;

#[derive(Clone)]
pub struct ProcessingContext {
    pub model_endpoint: String,
    pub zmq_context: Rc<zmq::Context>,
    pub splitter: Rc<SplitterLiteConfig<HFTokenizer>>,
    pub sentence_splitter: Rc<SplitterLiteConfig<HFTokenizer>>,
    pub hasher: Rc<DeterministicAHasher>,
    pub n_embd: usize,
    pub model_id: u64,
}
unsafe impl Sync for ProcessingContext {}
unsafe impl Send for ProcessingContext {}
