use fast_text_splitter::config::SplitterLiteConfig;
use fast_text_splitter::hf_tokenizer::HFTokenizer;
use fast_text_splitter::splitter::split_node::utils::SplitResultLite;
use std::sync::Arc;
use tracing::trace;

#[tracing::instrument(skip(splitter, text))]
pub async fn split_text(
    splitter: Arc<SplitterLiteConfig<HFTokenizer>>,
    text: Vec<u8>,
) -> anyhow::Result<Vec<SplitResultLite>> {
    trace!("Splitting text...");
    let splits = tokio::spawn(async move { splitter.hf_splits(text.as_slice()) }).await?;
    trace!("Number of splits: {}", splits.len());
    Ok(splits)
}
