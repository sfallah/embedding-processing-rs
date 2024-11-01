#[cfg(test)]
mod tests {
    use crate::inference::llama_context::LlamaContext;
    use crate::processing::context::ProcessingContext;
    use crate::processing::documents::process_document;
    use crate::processing::embeddings::process_embedding;
    use crate::processing::splits::process_split;
    use crate::processing::splitter::split_text;
    use crate::processing::summaries::process_summaries;
    use crate::services::embeddings;
    use crate::services::embeddings::{
        async_embeddings_routine, async_get_embeddings, EmbeddingsRequest,
    };
    use crate::utils::app_utils::init;
    use crate::utils::hash_utils::DeterministicAHasher;
    use criterion::async_executor::AsyncExecutor;
    use fast_text_splitter::config::SplitterLiteConfig;
    use rstest::{fixture, rstest};
    use std::ops::Deref;
    use std::sync::Arc;
    use tokio::sync::broadcast::Sender;
    use tokio::sync::mpsc::UnboundedSender;
    use tokio::sync::{broadcast, mpsc};

    #[fixture]
    fn text() -> String {
        "October 2023\n\n\
        One of the most important things I didn't understand about the world when I was a child is the degree to which the returns for performance are superlinear.\n\n\
        Teachers and coaches implicitly told us the returns were linear. \"You get out,\" I heard a thousand times, \"what you put in.\" They meant well, but this is rarely true. \
        If your product is only half as good as your competitor's, you don't get half as many customers. You get no customers, and you go out of business.\n\n\
        It's obviously true that the returns for performance are superlinear in business. Some think this is a flaw of capitalism, and that if we changed the rules it would stop being true. \
        But superlinear returns for performance are a feature of the world, not an artifact of rules we've invented. We see the same pattern in fame, power, military victories, knowledge, and even benefit to humanity. \
        In all of these, the rich get richer. [1]\n\nYou can't understand the world without understanding the concept of superlinear returns. \
        And if you're ambitious you definitely should, because this will be the wave you surf on.\n\n\n\n\n\n\
        It may seem as if there are a lot of different situations with superlinear returns, but as far as I can tell they reduce to two fundamental causes: exponential growth and thresholds.\n\n\
        The most obvious case of superlinear returns is when you're working on something that grows exponentially. \
        For example, growing bacterial cultures. When they grow at all, they grow exponentially. But they're tricky to grow. \
        Which means the difference in outcome between someone who's adept at it and someone who's not is very great.\n\n\
        Startups can also grow exponentially, and we see the same pattern there. Some manage to achieve high growth rates. \
        Most don't. And as a result you get qualitatively different outcomes: the companies with high growth rates tend to become immensely valuable, while the ones with lower growth rates may not even survive.\n\n\
        Y Combinator encourages founders to focus on growth rate rather than absolute numbers. It prevents them from being discouraged early on, when the absolute numbers are still low. \
        It also helps them decide what to focus on: you can use growth rate as a compass to tell you how to evolve the company. But the main advantage is that by focusing on growth rate you tend to get something that grows exponentially.\n\n"
            .to_string()
    }

    #[fixture]
    async fn text_from_file(#[default("tests/test_data/superlinear.txt")] file: String) -> String {
        tokio::fs::read_to_string(file)
            .await
            .expect("Failed to read test text file")
    }

    #[fixture]
    async fn ctx() -> Arc<ProcessingContext> {
        let splitter_patterns = vec![
            vec!["\n\n".to_string()],
            vec!["\n".to_string()],
            vec![
                ".".to_string(),
                "!".to_string(),
                "?".to_string(),
                ". ".to_string(),
            ],
        ];
        let nw_splitter =
            SplitterLiteConfig::new_hf(splitter_patterns.clone(), Some(512), None, true, None);

        let sentence_splitter = SplitterLiteConfig::new_hf(
            splitter_patterns.clone(),
            Some(512),
            Some(splitter_patterns.len()),
            true,
            None,
        );

        let haser = DeterministicAHasher::new(None, None);
        Arc::new(ProcessingContext {
            splitter: Arc::new(nw_splitter),
            sentence_splitter: Arc::new(sentence_splitter),
            hasher: Arc::new(haser),
            n_embd: 384,
        })
    }

    #[tokio::test(flavor = "multi_thread")]
    async fn test_embeddings() -> anyhow::Result<()> {
        let (embed, shutdown, handle) = init().await?;

        let text = "This is a test text".to_string();
        let embeddings = async_get_embeddings(embed.clone(), &vec![text], 512).await?;
        println!("{:?}", embeddings);
        shutdown
            .send("shutdown".to_string())
            .expect("Failed to send shutdown signal");
        handle.await.expect("Failed to shutdown embeddings");
        Ok(())
    }

    #[rstest]
    #[tokio::test]
    async fn test_splitter(#[future] ctx: Arc<ProcessingContext>) -> anyhow::Result<()> {
        let text = "This is a test text".to_string();

        let splitter = ctx.await.splitter.clone();
        let splits = split_text(splitter, text.as_bytes().to_vec())
            .await
            .expect("Failed to split text");
        assert_eq!(splits.len(), 1);
        println!("{:?}", splits);
        Ok(())
    }

    #[rstest]
    #[tokio::test(flavor = "multi_thread")]
    async fn test_summaries_process(
        #[future] ctx: Arc<ProcessingContext>,
        text: String,
    ) -> anyhow::Result<()> {
        let (embed_sender, shutdown, handle) = init().await?;
        let ctx = ctx.await.clone();
        let text = text.clone();
        let embed_sender = embed_sender.clone();
        let summaries = process_summaries(ctx, embed_sender, text, 0, 0)
            .await
            .expect("Failed to process summaries");
        println!("{:?}", summaries);
        assert_eq!(summaries.len(), 2);
        let sum_texts = summaries
            .iter()
            .map(|sum| sum.text_content.clone())
            .collect::<Vec<_>>();
        println!("{}", sum_texts.join("\n"));
        shutdown
            .send("shutdown".to_string())
            .expect("Failed to send shutdown signal");
        handle.await.expect("Failed to shutdown embeddings");
        Ok(())
    }

    #[rstest]
    #[tokio::test(flavor = "multi_thread")]
    async fn test_split_process(
        #[future] ctx: Arc<ProcessingContext>,
        text: String,
    ) -> anyhow::Result<()> {
        let (embed_sender, shutdown, handle) = init().await?;
        let ctx = ctx.await.clone();
        let text = text.clone();
        let embed_sender = embed_sender.clone();
        let splitter = ctx.clone().splitter.clone();
        let splits = split_text(splitter, text.as_bytes().to_vec())
            .await
            .expect("Failed to split text");
        assert_eq!(splits.len(), 1);
        println!("{:?}", splits);

        let split = process_split(ctx, Arc::new(splits[0].clone()), embed_sender, 0, 0)
            .await
            .expect("Failed to process split");
        shutdown
            .send("shutdown".to_string())
            .expect("Failed to send shutdown signal");
        handle.await.expect("Failed to shutdown embeddings");
        println!("{:?}", split);
        Ok(())
    }

    #[rstest]
    #[tokio::test(flavor = "multi_thread",)]
    async fn test_document_process(
        #[future] ctx: Arc<ProcessingContext>,
        #[future] text_from_file: String,
    ) -> anyhow::Result<()> {
        let (embed_sender, shutdown, handle) = init().await?;
        let ctx = ctx.await.clone();
        let embed_sender = embed_sender.clone();
        let doc = process_document(
            ctx,
            embed_sender,
            "test_url".to_string(),
            text_from_file.await.as_bytes().to_vec(),
        )
        .await
        .expect("Failed to process document");

        println!("{:?}", doc);
        assert_eq!(doc.splits.len(), 11);

        let sum_texts = doc
            .splits
            .iter()
            .map(|split| split.text_content.clone())
            .collect::<Vec<_>>();
        println!("{}", sum_texts.join("\n"));

        shutdown
            .send("shutdown".to_string())
            .expect("Failed to send shutdown signal");
        handle.await.expect("Failed to shutdown embeddings");
        Ok(())
    }
}
