#[cfg(test)]
mod tests {
    use embedding_common::config::ModelConfig;
    use embedding_common::utils::tracting::setup_tracing;
    use embedding_processing::processing::documents::process_document;
    use embedding_processing::processing::embeddings::async_get_embeddings;
    use embedding_processing::processing::splits::process_split;
    use embedding_processing::processing::splitter::split_text;
    use embedding_processing::processing::summaries::process_summaries;
    use embedding_processing::utils::app_utils::init_ctx;
    use rstest::{fixture, rstest};
    use std::sync::Arc;
    use tracing::{debug, Level};

    const MODEL_PATH: &str = "../models/all-minilm-l6-v2-q2_k.gguf";

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
    fn text_from_file(#[default("tests/test_data/superlinear.txt")] file: String) -> String {
        std::fs::read_to_string(file).expect("Failed to read test text file")
    }

    fn test_embeddings() -> anyhow::Result<()> {
        let model_config = ModelConfig::new(MODEL_PATH.to_string(), 1, true);
        let ctx = init_ctx(510, Some(3), 384, 500);
        let text = "This is a test text".to_string();
        let embeddings = async_get_embeddings(
            ctx.zmq_context.clone(),
            ctx.model_endpoint.clone(),
            ctx.n_embd,
            vec![text],
            0,
        );
        println!("{:?}", embeddings);

        Ok(())
    }

    #[tokio::test]
    async fn test_splitter() -> anyhow::Result<()> {
        let text = "This is a test text".to_string();
        let ctx = init_ctx(512, None, 384, 0);
        let splitter = ctx.splitter.clone();
        let splits = split_text(splitter, text.as_bytes().to_vec()).expect("Failed to split text");
        assert_eq!(splits.len(), 1);
        println!("{:?}", splits);
        Ok(())
    }

    fn test_summaries_process(text: String) -> anyhow::Result<()> {
        let ctx = init_ctx(512, None, 384, 10);
        let text = text.clone();
        let summaries = process_summaries(ctx, text, 0, 0).expect("Failed to process summaries");
        println!("{:?}", summaries);
        assert_eq!(summaries.len(), 2);
        let sum_texts = summaries
            .iter()
            .map(|sum| sum.text_content.clone())
            .collect::<Vec<_>>();
        println!("{}", sum_texts.join("\n"));

        Ok(())
    }

    async fn test_split_process(text: String) -> anyhow::Result<()> {
        let ctx = init_ctx(512, None, 384, 3000);
        let text = text.clone();
        let splitter = ctx.clone().splitter.clone();
        let splits = split_text(splitter, text.as_bytes().to_vec()).expect("Failed to split text");
        assert_eq!(splits.len(), 1);
        println!("{:?}", splits);

        let split =
            process_split(ctx, Arc::new(splits[0].clone()), 0, 0).expect("Failed to process split");
        println!("{:?}", split);
        Ok(())
    }

    fn test_document_process(text_from_file: String) -> anyhow::Result<()> {
        setup_tracing(Level::DEBUG);
        let ctx = init_ctx(512, None, 384, 100);
        let doc = process_document(
            ctx,
            "test_url".to_string(),
            text_from_file.as_bytes().to_vec(),
        )
        .expect("Failed to process document");

        assert_eq!(doc.splits.len(), 11);
        debug!("{:?}", doc);

        let _sum_texts = doc
            .splits
            .iter()
            .map(|split| split.text_content.clone())
            .collect::<Vec<_>>();
        //println!("{}", sum_texts.join("\n"));
        Ok(())
    }

    #[rstest]
    #[tokio::test(flavor = "multi_thread")]
    async fn test_document_process_short() -> anyhow::Result<()> {
        let ctx = init_ctx(10, None, 384, 3000); // low max_tokens to test short text
        let doc = process_document(
            ctx,
            "test_url".to_string(),
            // Short text to test error -> with filter_splits 4, this will test 0 sentences and 1 sentence for 2 splits
            "a a. a a. a \n\n a a a. a".to_string().as_bytes().to_vec(),
        )
        .expect("Failed to process document");

        //assert_eq!(doc.splits.len(), 11);
        debug!("{:?}", doc);

        let _sum_texts = doc
            .splits
            .iter()
            .map(|split| split.text_content.clone())
            .collect::<Vec<_>>();
        //println!("{}", sum_texts.join("\n"));
        Ok(())
    }
}
