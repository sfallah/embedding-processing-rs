use fast_text_splitter::config::SplitterLiteConfig;
use fast_text_splitter::hf_tokenizer::HFTokenizer;
use fast_text_splitter::splitter::split_node::utils::SplitResultLite;
use std::rc::Rc;
use tracing::trace;

#[tracing::instrument(skip(splitter, text))]
pub fn split_text(
    splitter: Rc<SplitterLiteConfig<HFTokenizer>>,
    text: Vec<u8>,
) -> anyhow::Result<Vec<SplitResultLite>> {
    trace!("Splitting text...");
    let splits =  splitter.hf_splits(text.as_slice());
    trace!("Number of splits: {}", splits.len());
    Ok(splits)
}
