use crate::processing::context::ProcessingContext;
use crate::processing::splits::process_split;
use crate::processing::splitter::split_text;
use embedding_common::dtos::document_dto::DocumentDto;
use std::sync::Arc;
use rayon::prelude::*;
use tracing::{error, trace};

#[tracing::instrument(skip(ctx, text))]
pub fn process_document(
    ctx: Arc<ProcessingContext>,
    url: String,
    text: Vec<u8>,
) -> anyhow::Result<DocumentDto> {
    let splits = split_text(ctx.splitter.clone(), text)?;

    let doc_id = ctx.clone().hasher.hash(&url);
    trace!("Document ID: {}", doc_id);


    let split_dtos: Vec<_> = splits
        .par_iter()
        .enumerate()
        .filter_map(|(seq_id, split)| {
            let ctx = ctx.clone();
            let res = process_split(ctx.clone(), split, doc_id, seq_id as i32);
            match res {
                Ok(split_dto) => Some(split_dto),
                Err(e) => {
                    error!("Failed to process split: {}", e);
                    None
                }
            }
        })
        .collect();

    let summaries: Vec<_> = split_dtos
        .iter()
        .flat_map(|split| split.summaries.clone())
        .collect();

    Ok(DocumentDto::new(
        doc_id,
        url.as_str(),
        split_dtos,
        Some(summaries),
    ))
}
