use crate::dtos::document_new::DocumentDtoNew;
use crate::dtos::split_new::SplitDtoNew;
use crate::processing::context::ProcessingContext;
use crate::processing::splits::process_split;
use crate::processing::splitter::split_text;
use crate::services::embeddings::EmbeddingsRequest;
use std::sync::Arc;
use tokio::sync::mpsc::UnboundedSender;
use tokio::task::JoinSet;

pub async fn process_document(
    ctx: Arc<ProcessingContext>,
    embed_sender: Arc<UnboundedSender<EmbeddingsRequest>>,
    url: String,
    text: Vec<u8>,
) -> anyhow::Result<DocumentDtoNew> {
    let splits = split_text(ctx.splitter.clone(), text).await?;

    let doc_id = ctx.clone().hasher.hash(&url);

    let mut set = JoinSet::new();
    splits
        .into_iter()
        .enumerate()
        .for_each(|(seq_id, split_res)| {
            let ctx_clone = ctx.clone();
            let embed_sender = embed_sender.clone();
            set.spawn(async move {
                process_split(
                    ctx_clone,
                    Arc::from(split_res.clone()),
                    embed_sender,
                    doc_id,
                    seq_id as i32,
                )
                .await
                .unwrap()
            });
        });

    let split_dtos: Vec<SplitDtoNew> = set.join_all().await;

    let summaries: Vec<_> = split_dtos
        .iter()
        .flat_map(|split| split.summaries.clone())
        .collect();

    Ok(DocumentDtoNew::new(
        doc_id,
        url.as_str(),
        split_dtos,
        summaries,
    ))
}
