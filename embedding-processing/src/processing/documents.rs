use embedding_common::dtos::document_dto::DocumentDto;
use crate::processing::context::ProcessingContext;
use crate::processing::splits::process_split;
use crate::processing::splitter::split_text;
use crate::services::embeddings::EmbeddingsRequest;
use std::sync::Arc;
use tracing::trace;

#[tracing::instrument(skip(ctx, embed_sender, text))]
pub async fn process_document(
    ctx: Arc<ProcessingContext>,
    embed_sender: Arc<async_channel::Sender<EmbeddingsRequest>>,
    url: String,
    text: Vec<u8>,
) -> anyhow::Result<DocumentDto> {
    let splits = split_text(ctx.splitter.clone(), text).await?;

    let mut split_dtos: Vec<_> = Vec::new();
    let doc_id = ctx.clone().hasher.hash(&url);
    trace!("Document ID: {}", doc_id);

    let mut handles = Vec::new();

    for (seq_id, split) in splits.iter().enumerate() {
        let split = Arc::new(split.clone());
        let ctx = ctx.clone();
        let embed_sender = embed_sender.clone();
        let handle = tokio::spawn(async move {
            process_split(ctx.clone(), split, embed_sender, doc_id, seq_id as i32)
                .await
                .expect("Failed to process split")
        });
        handles.push(handle);
    }
    futures::future::join_all(handles.into_iter())
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
        summaries,
    ))
}
