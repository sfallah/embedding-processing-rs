#[cfg(test)]
mod tests {
    use crate::inference::llama_context::LlamaContext;
    use crate::processing::context::ProcessingContext;
    use crate::processing::embeddings::process_embedding;
    use crate::processing::splits::process_split;
    use crate::processing::splitter::split_text;
    use crate::processing::summaries::process_summaries;
    use crate::services::embeddings::{async_embeddings_routine, EmbeddingsRequest};
    use crate::utils::hash_utils::DeterministicAHasher;
    use fast_text_splitter::config::SplitterLiteConfig;
    use rstest::{fixture, rstest};
    use std::sync::Arc;
    use crate::processing::documents::process_document;

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
        async_std::fs::read_to_string(file)
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

        let (embedding_sender, embedding_receiver) =
            async_std::channel::unbounded::<EmbeddingsRequest>();

        let embedding_sender = Arc::new(embedding_sender);
        let model_path = "models/all-minilm-l6-v2-q2_k.gguf";

        async_std::task::block_on(async {
            for _ in 0..2 {
                let model_instance = Arc::new(LlamaContext::new(model_path, 512, 1000));
                let receiver_clone = embedding_receiver.clone();
                async_std::task::spawn(async move {
                    async_embeddings_routine(model_instance, receiver_clone).await;
                });
            }
        });

        let haser = DeterministicAHasher::new(None, None);
        Arc::new(ProcessingContext {
            splitter: Arc::new(nw_splitter),
            sentence_splitter: Arc::new(sentence_splitter),
            hasher: Arc::new(haser),
            embeddings_sender: embedding_sender,
            n_embd: 384,
            //embeddings_recv_task: Arc::new(embd_task),
        })
    }

    #[rstest]
    //#[async_std::test]
    async fn test_embeddings(#[future] ctx: Arc<ProcessingContext>) {
        let text = "This is a test text".to_string();

        println!("Model loaded");

        let sender = ctx.await.embeddings_sender.clone();
        async_std::task::block_on(async move {
            let embeddings = process_embedding(sender, 0, vec![text.clone()])
                .await
                .unwrap();
            println!("{:?}", embeddings);
        });
    }

    #[rstest]
    async fn test_splitter(#[future] ctx: Arc<ProcessingContext>) {
        let text = "This is a test text".to_string();

        let splitter = ctx.await.splitter.clone();
        let splits = async_std::task::block_on(async move {
            split_text(splitter, text.as_bytes().to_vec())
                .await
                .expect("Failed to split text")
        });
        assert_eq!(splits.len(), 1);
        println!("{:?}", splits);
    }

    #[rstest]
    async fn test_split_process(#[future] ctx: Arc<ProcessingContext>, text: String) {
        let ctx = ctx.await.clone();
        let splitter = ctx.clone().splitter.clone();
        let splits = async_std::task::block_on(async move {
            split_text(splitter, text.as_bytes().to_vec())
                .await
                .expect("Failed to split text")
        });
        assert_eq!(splits.len(), 1);
        println!("{:?}", splits);

        let split = process_split(ctx, Arc::new(splits[0].clone()), 0, 0)
            .await
            .expect("Failed to process split");
        println!("{:?}", split);
    }

    #[rstest]
    async fn test_summaries_process(#[future] ctx: Arc<ProcessingContext>, text: String) {
        let ctx = ctx.await.clone();
        let summaries = process_summaries(ctx, text, 0, 0)
            .await
            .expect("Failed to process summaries");
        println!("{:?}", summaries);
        assert_eq!(summaries.len(), 2);
        let sum_texts = summaries.iter().map(|sum| sum.text_content.clone()).collect::<Vec<_>>();
        println!("{}", sum_texts.join("\n"));
    }

    #[rstest]
    async fn test_document_process(#[future] ctx: Arc<ProcessingContext>, #[future] text_from_file: String) {
        let ctx = ctx.await.clone();
        let doc = process_document(ctx, "test_url".to_string(), text_from_file.await.as_bytes().to_vec())
            .await
            .expect("Failed to process document");
        println!("{:?}", doc);
        assert_eq!(doc.splits.len(), 11);
        let sum_texts = doc.splits.iter().map(|split| split.text_content.clone()).collect::<Vec<_>>();
        println!("{}", sum_texts.join("\n"));
    }
}
