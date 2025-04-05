use crate::processing::context::ProcessingContext;
use crate::processing::splitter::split_text;
use crate::processing::utils;
use embedding_common::dtos::embedding_dto::EmbeddingDto;
use embedding_common::dtos::summary_dto::SummaryDto;
use fast_text_splitter::splitter::split_node::utils::SplitResultLite;
use std::sync::Arc;
use tracing::trace;

pub fn process_summaries(
    ctx: Arc<ProcessingContext>,
    sentences: &[String],
    sentences_no_tokens: &[usize],
    embeddings: &[f32],
    doc_id: u64,
    split_id: u64,
) -> anyhow::Result<Vec<SummaryDto>> {
    trace!("Processing summaries...");
    trace!("Number of sentences: {}", sentences.len());

    if sentences.len() == 0 {
        return Ok(Vec::new());
    }

    let lx_ranks = lexrank_sentences(embeddings.to_vec(), sentences.len(), ctx.n_embd, None, None)?;

    let summaries: Vec<_> = lx_ranks
        .into_iter()
        .take(2)
        .map(|(seq_id, score)| {
            let sentence = sentences.get(seq_id).unwrap().clone();
            let no_tokens = sentences_no_tokens[seq_id];
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
                None,
            )
        })
        .collect();
    Ok(summaries)
}

pub fn get_sentences(
    ctx: Arc<ProcessingContext>,
    text: String,
) -> anyhow::Result<(Vec<String>, Vec<usize>)> {
    let splits = split_text(ctx.sentence_splitter.clone(), text.into_bytes())?;
    let splits: Vec<_> = filter_splits(&splits, 4);
    let splits_no_tokens: Vec<_> = splits.iter().map(|res| res.tokens.len()).collect();
    let sentences: Vec<_> = utils::splits_texts(&splits);
    Ok((sentences, splits_no_tokens))
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

fn lexrank_sentences(
    embeddings: Vec<f32>,
    len: usize,
    n_embd: usize,
    threshold: Option<f32>,
    max_iter: Option<usize>,
) -> anyhow::Result<Vec<(usize, f32)>> {
    lexrank_ndarray::lexrank_array(
        &embeddings,
        len,
        n_embd,
        threshold,
        max_iter.unwrap_or(10000),
    )
}
