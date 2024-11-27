use crate::processing::context::ProcessingContext;
use crate::processing::splitter::split_text;
use crate::services::embeddings::{async_get_embeddings, EmbeddingsRequest};
use embedding_common::dtos::embedding_dto::EmbeddingDto;
use embedding_common::dtos::summary_dto::SummaryDto;
use fast_text_splitter::splitter::split_node::utils::SplitResultLite;
use std::sync::Arc;
use tracing::trace;
use crate::processing::utils;

#[tracing::instrument(skip(ctx, embed_sender, text))]
pub async fn process_summaries(
    ctx: Arc<ProcessingContext>,
    embed_sender: Arc<async_channel::Sender<EmbeddingsRequest>>,
    text: String,
    doc_id: u64,
    split_id: u64,
) -> anyhow::Result<Vec<SummaryDto>> {
    trace!("Processing summaries...");
    let splits = split_text(ctx.sentence_splitter.clone(), text.into_bytes()).await?;
    let splits: Vec<_> = filter_splits(&splits, 4);
    let sentences: Vec<_> = utils::splits_texts(&splits);
    trace!("Number of sentences: {}", sentences.len());

    let embeddings = async_get_embeddings(embed_sender.clone(), &sentences, ctx.n_embd).await?;

    let lx_ranks =
        lexrank_sentences(embeddings.clone(), sentences.len(), ctx.n_embd, None, None).await?;
    let no_tokens = tokens_num(splits);

    let summaries: Vec<_> = lx_ranks
        .into_iter()
        .take(2)
        .enumerate()
        .map(|(idx, (seq_id, score))| {
            let sentence = sentences.get(seq_id).unwrap().clone();
            let no_tokens = *no_tokens.get(seq_id).unwrap();
            let i = seq_id * ctx.n_embd;
            let j = (seq_id + 1) * ctx.n_embd;
            let embeddings: Vec<f32> = embeddings[i..j].to_vec();
            let sum_id = ctx
                .hasher
                .hash(&format!("{}{}{}", split_id, seq_id, no_tokens));
            let embedding = EmbeddingDto::new(sum_id, embeddings.clone(), ctx.model_id);
            SummaryDto::new(
                sum_id,
                doc_id,
                split_id,
                seq_id as i32,
                sentence.as_str(),
                no_tokens,
                score,
                Some(embedding),
                None,
            )
        })
        .collect();
    Ok(summaries)
}

fn tokens_num(splits: Vec<SplitResultLite>) -> Vec<usize> {
    splits
        .iter()
        .map(|sentence_split| sentence_split.tokens.len())
        .collect()
}

fn filter_splits(splits: &Vec<SplitResultLite>, ln: usize) -> Vec<SplitResultLite> {
    splits
        .iter()
        .filter_map(|split| {
            if split.tokens.len() >= ln {
                Some(split.clone())
            } else {
                None
            }
        })
        .collect()
}

#[tracing::instrument]
async fn lexrank_sentences(
    embeddings: Vec<f32>,
    len: usize,
    n_embd: usize,
    threshold: Option<f32>,
    max_iter: Option<usize>,
) -> anyhow::Result<Vec<(usize, f32)>> {
    tokio::spawn(async move {
        lexrank_ndarray::lexrank_array(
            &embeddings,
            len,
            n_embd,
            threshold,
            max_iter.unwrap_or(10000),
        )
    })
    .await?
}
