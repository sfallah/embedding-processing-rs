use crate::processing::context::ProcessingContext;
use crate::processing::splits::process_split;
use crate::processing::splitter::split_text;
use embedding_common::dtos::document_dto::DocumentDto;
use std::sync::Arc;
use tracing::trace;

#[tracing::instrument(skip(ctx, embedding_addr, text))]
pub async fn process_document(
    ctx: Arc<ProcessingContext>,
    embedding_addr: String,
    url: String,
    text: Vec<u8>,
) -> anyhow::Result<DocumentDto> {
    let splits = split_text(ctx.splitter.clone(), text).await?;

    let mut split_dtos: Vec<_> = Vec::new();
    let doc_id = ctx.clone().hasher.hash(&url);
    trace!("Document ID: {}", doc_id);

    //let mut handles = Vec::new();

    let splits_futures = splits.iter().enumerate().map(|(seq_id, split)| {
        let split = Arc::new(split.clone());
        process_split(
            ctx.clone(),
            split,
            embedding_addr.clone(),
            doc_id,
            seq_id as i32,
        )
    });

    futures::future::join_all(splits_futures)
        .await
        .into_iter()
        .for_each(|res| {
            split_dtos.push(res.expect("Failed to process split"));
        });

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
