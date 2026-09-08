//! The manual smoke test from `PLAN-lean-storage.md`, step 3, as a program.
//!
//! Drives a running `embedding_server` over its own ZMQ protocol — insert, query in all three
//! modes, retrieve, delete, re-insert the same url — and checks the answers. It needs the two
//! model backends up as well, because insertion runs the whole embedding pipeline.
//!
//! ```shell
//! cargo run -r -F metal -p embedding-model    --bin model_cli     -- --config-file embedding-model/embedding_config.toml
//! cargo run -r -F metal -p reranking-model    --bin reranking_cli -- --config-file reranking-model/reranking_config.toml
//! cargo run -r -p embedding-server --bin embedding_server -- --config-file config.toml
//! cargo run -r -p embedding-server --bin smoke_client
//! ```
//!
//! Exits non-zero if any check fails, so it can gate a change rather than only inform one.

use clap::Parser;
use embedding_common::prelude::*;
use embedding_server::schema::document::{
    DocumentDeletionRequest, DocumentDeletionResponse, DocumentInsertionRequest,
    DocumentInsertionResponse, DocumentQueryRequest, DocumentQueryResponse,
    DocumentRetrievalRequest, DocumentRetrievalResponse,
};
use embedding_server::schema::document_status::{
    DeletionStatus, DocumentInsertionStatus, RetrievalStatus,
};
use embedding_server::schema::search_mode::SearchModeType;
use embedding_server::schema::zmq_message_header::{
    ZmqMessageHeader, ZmqMessageStatus, ZmqMessageType,
};
use std::process::ExitCode;
use uuid::Uuid;
use zmq::Socket;

#[derive(Parser, Debug, Clone)]
#[command(about = "Smoke test for a running embedding_server")]
struct Args {
    /// The server's frontend, as `[zmq] host` and `frontend_port` in config.toml give it.
    #[arg(long, default_value = "tcp://localhost:5556")]
    endpoint: String,

    /// How long to wait for one reply. Insertion runs the whole pipeline, so this is generous.
    #[arg(long, default_value_t = 120_000)]
    timeout_ms: i32,

    /// Leave the documents behind instead of deleting them at the end.
    #[arg(long)]
    keep: bool,
}

// ---------------------------------------------------------------------------
// Checks
// ---------------------------------------------------------------------------

#[derive(Default)]
struct Report {
    passed: usize,
    failed: Vec<String>,
}

impl Report {
    fn check(&mut self, what: &str, ok: bool, detail: impl FnOnce() -> String) {
        if ok {
            self.passed += 1;
            println!("  ok    {}", what);
        } else {
            let detail = detail();
            println!("  FAIL  {} -- {}", what, detail);
            self.failed.push(format!("{}: {}", what, detail));
        }
    }
}

// ---------------------------------------------------------------------------
// The wire
// ---------------------------------------------------------------------------

/// One request and its reply. The server answers `[header, body]` on success and `[header]` alone
/// on an error, the identity frame having been stripped by the frontend router.
fn call(
    socket: &Socket,
    message_type: ZmqMessageType,
    body: Option<Vec<u8>>,
) -> anyhow::Result<(ZmqMessageHeader, Option<Vec<u8>>)> {
    let header = ZmqMessageHeader::new(message_type);
    let mut frames = vec![header.pack()?];
    if let Some(body) = body {
        frames.push(body);
    }
    socket.send_multipart(frames, 0)?;

    let reply = socket.recv_multipart(0)?;
    let header: ZmqMessageHeader = ZmqMessageHeader::unpack(
        reply
            .first()
            .ok_or_else(|| anyhow::anyhow!("empty reply from the server"))?,
    )?;
    Ok((header, reply.get(1).cloned()))
}

fn succeeded(header: &ZmqMessageHeader) -> bool {
    header.status == Some(ZmqMessageStatus::Success)
}

fn error_of(header: &ZmqMessageHeader) -> String {
    header
        .error_message
        .clone()
        .unwrap_or_else(|| "no error message".to_string())
}

// ---------------------------------------------------------------------------
// The corpus
// ---------------------------------------------------------------------------

/// A document long enough to be split several times at `[splitter] max_tokens = 400`.
///
/// `marker` appears in one paragraph and nowhere else, so a query can prove that a particular
/// version of the document is or is not in the index.
fn document_text(paragraphs: usize, marker: &str) -> String {
    let mut out = String::new();
    for i in 0..paragraphs {
        if i > 0 {
            out.push_str("\n\n");
        }
        if i == paragraphs / 2 {
            out.push_str(marker);
            out.push(' ');
        }
        for sentence in 0..8 {
            out.push_str(&format!(
                "Paragraph {} sentence {} describes how the storage layer keeps a record log and \
                 two vector indexes together under one lock for workspace paragraph {}. ",
                i, sentence, i
            ));
        }
    }
    out
}

const FIRST_MARKER: &str = "Antimony calibrates the quarterly telemetry harness.";
const SECOND_MARKER: &str = "Bergamot reconciles the nightly ingestion ledger.";

// ---------------------------------------------------------------------------

fn main() -> ExitCode {
    let args = Args::parse();
    match run(&args) {
        Ok(report) => {
            println!();
            if report.failed.is_empty() {
                println!("{} checks passed", report.passed);
                ExitCode::SUCCESS
            } else {
                println!("{} passed, {} FAILED:", report.passed, report.failed.len());
                for failure in &report.failed {
                    println!("  - {}", failure);
                }
                ExitCode::FAILURE
            }
        }
        Err(e) => {
            eprintln!("smoke test could not run: {:#}", e);
            eprintln!("are the server and both model backends up?");
            ExitCode::FAILURE
        }
    }
}

fn run(args: &Args) -> anyhow::Result<Report> {
    let ctx = zmq::Context::new();
    let socket = ctx.socket(zmq::DEALER)?;
    socket.set_linger(0)?;
    socket.set_rcvtimeo(args.timeout_ms)?;
    socket.set_sndtimeo(5_000)?;
    socket.connect(&args.endpoint)?;
    println!("connected to {}", args.endpoint);

    let mut report = Report::default();
    let workspace_a = Uuid::new_v4();
    let workspace_b = Uuid::new_v4();
    let url = format!("smoke://document/{}", Uuid::new_v4());
    println!("workspace a       : {}", workspace_a);
    println!("workspace b       : {}", workspace_b);
    println!("document url      : {}", url);

    // -- health ------------------------------------------------------------
    println!("\nhealth");
    let (header, _) = call(&socket, ZmqMessageType::HealthCheck, None)?;
    report.check("health check answers", succeeded(&header), || {
        error_of(&header)
    });

    // -- insert ------------------------------------------------------------
    println!("\ninsert");
    let insert = DocumentInsertionRequest {
        input: document_text(9, FIRST_MARKER),
        model: "smoke".to_string(),
        workspace_id: workspace_a,
        doc_url: url.clone(),
        verbose: Some(true),
    };
    let (header, body) = call(
        &socket,
        ZmqMessageType::DocumentInsertion,
        Some(insert.pack()?),
    )?;
    report.check("insert succeeds", succeeded(&header), || error_of(&header));
    let inserted: DocumentInsertionResponse = DocumentInsertionResponse::unpack(
        body.as_deref()
            .ok_or_else(|| anyhow::anyhow!("insert returned no body: {}", error_of(&header)))?,
    )?;
    report.check(
        "insert reports success",
        inserted.status == DocumentInsertionStatus::Success,
        || format!("{:?}", inserted.status),
    );
    let document = inserted
        .document
        .ok_or_else(|| anyhow::anyhow!("verbose insert returned no document"))?;
    let document_id = document.document_id;
    println!(
        "  document {} with {} split(s), {} document-level summaries",
        document_id,
        document.splits.len(),
        document.summaries.as_ref().map(|s| s.len()).unwrap_or(0)
    );
    report.check("the document was split", document.splits.len() > 1, || {
        format!(
            "{} split(s); is [splitter] max_tokens smaller than the text?",
            document.splits.len()
        )
    });
    report.check(
        "every split carries summaries",
        document.splits.iter().all(|s| !s.summaries.is_empty()),
        || "some split has no summaries".to_string(),
    );

    // -- query, all three modes -------------------------------------------
    println!("\nquery");
    for mode in [
        SearchModeType::SplitOnly,
        SearchModeType::SummaryOnly,
        SearchModeType::SplitAndSummary,
    ] {
        let response = query(
            &socket,
            &QueryArgs {
                input: FIRST_MARKER.to_string(),
                workspaces: vec![workspace_a],
                doc_ids: vec![],
                mode: Some(mode),
                rerank: false,
                verbose: false,
                top_k: 5,
            },
        )?;
        report.check(
            &format!("query {} finds the document", mode),
            response
                .documents
                .iter()
                .any(|d| d.document_id == document_id),
            || format!("{} document(s) returned", response.documents.len()),
        );
    }

    let response = query(
        &socket,
        &QueryArgs {
            input: FIRST_MARKER.to_string(),
            workspaces: vec![workspace_a],
            doc_ids: vec![],
            mode: Some(SearchModeType::SplitAndSummary),
            rerank: true,
            verbose: false,
            top_k: 5,
        },
    )?;
    report.check(
        "query with rerank finds the document",
        response
            .documents
            .iter()
            .any(|d| d.document_id == document_id),
        || format!("{} document(s) returned", response.documents.len()),
    );
    report.check(
        "rerank attaches a rank to the document summaries",
        response
            .documents
            .iter()
            .filter(|d| d.document_id == document_id)
            .any(|d| {
                d.summaries
                    .as_ref()
                    .is_some_and(|s| s.iter().any(|summary| summary.rank.is_some()))
            }),
        || "no summary came back with a rank".to_string(),
    );

    let response = query(
        &socket,
        &QueryArgs {
            input: FIRST_MARKER.to_string(),
            workspaces: vec![workspace_a],
            doc_ids: vec![],
            mode: Some(SearchModeType::SplitAndSummary),
            rerank: false,
            verbose: true,
            top_k: 5,
        },
    )?;
    report.check(
        "verbose query returns the query embedding",
        response.query_embeddings.is_some_and(|e| !e.is_empty()),
        || "no query embedding".to_string(),
    );

    // A workspace nothing was written to contributes nothing, and must not be an error.
    let response = query(
        &socket,
        &QueryArgs {
            input: FIRST_MARKER.to_string(),
            workspaces: vec![workspace_b],
            doc_ids: vec![],
            mode: Some(SearchModeType::SplitAndSummary),
            rerank: false,
            verbose: false,
            top_k: 5,
        },
    )?;
    report.check(
        "an unwritten workspace returns nothing rather than failing",
        response.documents.is_empty(),
        || format!("{} document(s) returned", response.documents.len()),
    );

    // -- retrieval ---------------------------------------------------------
    println!("\nretrieval");
    let retrieved = retrieve(
        &socket,
        DocumentRetrievalRequest {
            document_id: Some(document_id),
            document_url: None,
            user: Some(workspace_a),
            verbose: Some(false),
        },
    )?;
    report.check(
        "retrieval by id returns the document",
        retrieved
            .as_ref()
            .is_some_and(|r| r.status == RetrievalStatus::Success),
        || "no document".to_string(),
    );
    if let Some(full) = retrieved.as_ref().and_then(|r| r.document.as_ref()) {
        report.check(
            "retrieval is fully nested",
            full.splits.len() == document.splits.len()
                && full.splits.iter().all(|s| !s.summaries.is_empty()),
            || {
                format!(
                    "{} split(s) against {} inserted",
                    full.splits.len(),
                    document.splits.len()
                )
            },
        );
    }

    let by_url = retrieve(
        &socket,
        DocumentRetrievalRequest {
            document_id: None,
            document_url: Some(url.clone()),
            user: Some(workspace_a),
            verbose: Some(false),
        },
    )?;
    report.check(
        "retrieval by url returns the same document",
        by_url
            .and_then(|r| r.document)
            .is_some_and(|d| d.document_id == document_id),
        || "url lookup did not find it".to_string(),
    );

    // Gap G4: without a workspace there is no shard to look in.
    let (header, _) = call(
        &socket,
        ZmqMessageType::DocumentRetrieval,
        Some(
            DocumentRetrievalRequest {
                document_id: Some(document_id),
                document_url: None,
                user: None,
                verbose: Some(false),
            }
            .pack()?,
        ),
    )?;
    report.check(
        "retrieval without a workspace is refused (G4)",
        !succeeded(&header),
        || "it was accepted".to_string(),
    );

    let (header, _) = call(
        &socket,
        ZmqMessageType::DocumentDeletion,
        Some(
            DocumentDeletionRequest {
                document_id,
                user: None,
            }
            .pack()?,
        ),
    )?;
    report.check(
        "deletion without a workspace is refused (G4)",
        !succeeded(&header),
        || "it was accepted".to_string(),
    );

    // -- re-insert the same url -------------------------------------------
    println!("\nre-insert");
    let shorter = DocumentInsertionRequest {
        input: document_text(4, SECOND_MARKER),
        model: "smoke".to_string(),
        workspace_id: workspace_a,
        doc_url: url.clone(),
        verbose: Some(true),
    };
    let (header, body) = call(
        &socket,
        ZmqMessageType::DocumentInsertion,
        Some(shorter.pack()?),
    )?;
    report.check("re-insert succeeds", succeeded(&header), || {
        error_of(&header)
    });
    let reinserted: DocumentInsertionResponse = DocumentInsertionResponse::unpack(
        body.as_deref()
            .ok_or_else(|| anyhow::anyhow!("re-insert returned no body"))?,
    )?;
    let second = reinserted
        .document
        .ok_or_else(|| anyhow::anyhow!("verbose re-insert returned no document"))?;
    report.check(
        "the same url is the same document id",
        second.document_id == document_id,
        || format!("{} against {}", second.document_id, document_id),
    );
    report.check(
        "the shorter version has fewer splits",
        second.splits.len() < document.splits.len(),
        || format!("{} against {}", second.splits.len(), document.splits.len()),
    );

    // Gap G8: the old version's splits used to be left behind in the index.
    let response = query(
        &socket,
        &QueryArgs {
            input: FIRST_MARKER.to_string(),
            workspaces: vec![workspace_a],
            doc_ids: vec![],
            mode: Some(SearchModeType::SplitOnly),
            rerank: false,
            verbose: false,
            top_k: 10,
        },
    )?;
    let orphan_ids: Vec<u64> = document
        .splits
        .iter()
        .map(|s| s.split_id)
        .filter(|id| !second.splits.iter().any(|s| s.split_id == *id))
        .collect();
    let still_there: Vec<u64> = response
        .documents
        .iter()
        .flat_map(|d| d.splits.iter().map(|s| s.split_id))
        .filter(|id| orphan_ids.contains(id))
        .collect();
    report.check(
        "the replaced version's splits are gone from the index (G8)",
        still_there.is_empty(),
        || format!("{:?} are still searchable", still_there),
    );

    // -- a second workspace, and the cross-shard merge ---------------------
    println!("\nsecond workspace");
    let url_b = format!("smoke://document/{}", Uuid::new_v4());
    let (header, body) = call(
        &socket,
        ZmqMessageType::DocumentInsertion,
        Some(
            DocumentInsertionRequest {
                input: document_text(6, FIRST_MARKER),
                model: "smoke".to_string(),
                workspace_id: workspace_b,
                doc_url: url_b.clone(),
                verbose: Some(true),
            }
            .pack()?,
        ),
    )?;
    report.check("insert into a second workspace", succeeded(&header), || {
        error_of(&header)
    });
    let doc_b_id = DocumentInsertionResponse::unpack::<DocumentInsertionResponse>(
        body.as_deref()
            .ok_or_else(|| anyhow::anyhow!("insert returned no body"))?,
    )?
    .document
    .map(|d| d.document_id)
    .ok_or_else(|| anyhow::anyhow!("verbose insert returned no document"))?;

    let response = query(
        &socket,
        &QueryArgs {
            input: FIRST_MARKER.to_string(),
            workspaces: vec![workspace_a],
            doc_ids: vec![],
            mode: Some(SearchModeType::SplitAndSummary),
            rerank: false,
            verbose: false,
            top_k: 5,
        },
    )?;
    report.check(
        "a query in one workspace never sees the other's document",
        response.documents.iter().all(|d| d.document_id != doc_b_id),
        || "workspace b's document came back from workspace a".to_string(),
    );

    let response = query(
        &socket,
        &QueryArgs {
            input: FIRST_MARKER.to_string(),
            workspaces: vec![workspace_a, workspace_b],
            doc_ids: vec![],
            mode: Some(SearchModeType::SplitAndSummary),
            rerank: false,
            verbose: false,
            top_k: 10,
        },
    )?;
    report.check(
        "a query over both workspaces returns both documents",
        response.documents.iter().any(|d| d.document_id == doc_b_id),
        || {
            format!(
                "{} document(s), ids {:?}",
                response.documents.len(),
                response
                    .documents
                    .iter()
                    .map(|d| d.document_id)
                    .collect::<Vec<_>>()
            )
        },
    );

    // The document filter still narrows within that.
    let response = query(
        &socket,
        &QueryArgs {
            input: FIRST_MARKER.to_string(),
            workspaces: vec![workspace_a, workspace_b],
            doc_ids: vec![doc_b_id],
            mode: Some(SearchModeType::SplitAndSummary),
            rerank: false,
            verbose: false,
            top_k: 10,
        },
    )?;
    report.check(
        "a document filter narrows the answer to that document",
        !response.documents.is_empty()
            && response.documents.iter().all(|d| d.document_id == doc_b_id),
        || {
            format!(
                "ids {:?}",
                response
                    .documents
                    .iter()
                    .map(|d| d.document_id)
                    .collect::<Vec<_>>()
            )
        },
    );

    // -- delete ------------------------------------------------------------
    if args.keep {
        println!("\n--keep: leaving both documents in place");
        return Ok(report);
    }
    println!("\ndelete");
    let status = delete(&socket, document_id, workspace_a)?;
    report.check(
        "delete removes the document",
        status == Some(DeletionStatus::Success),
        || format!("{:?}", status),
    );
    let status = delete(&socket, document_id, workspace_a)?;
    report.check(
        "deleting it again is NotFound, not an error",
        status == Some(DeletionStatus::NotFound),
        || format!("{:?}", status),
    );

    let retrieved = retrieve(
        &socket,
        DocumentRetrievalRequest {
            document_id: Some(document_id),
            document_url: None,
            user: Some(workspace_a),
            verbose: Some(false),
        },
    )?;
    report.check(
        "the deleted document cannot be retrieved",
        retrieved.is_some_and(|r| r.document.is_none()),
        || "it came back".to_string(),
    );

    let response = query(
        &socket,
        &QueryArgs {
            input: SECOND_MARKER.to_string(),
            workspaces: vec![workspace_a],
            doc_ids: vec![],
            mode: Some(SearchModeType::SplitAndSummary),
            rerank: false,
            verbose: false,
            top_k: 5,
        },
    )?;
    report.check(
        "the deleted document is out of the indexes too",
        response
            .documents
            .iter()
            .all(|d| d.document_id != document_id),
        || "it is still searchable".to_string(),
    );

    let status = delete(&socket, doc_b_id, workspace_b)?;
    report.check(
        "the second workspace's document deletes as well",
        status == Some(DeletionStatus::Success),
        || format!("{:?}", status),
    );

    Ok(report)
}

// ---------------------------------------------------------------------------
// Request helpers
// ---------------------------------------------------------------------------

struct QueryArgs {
    input: String,
    workspaces: Vec<Uuid>,
    doc_ids: Vec<u64>,
    mode: Option<SearchModeType>,
    rerank: bool,
    verbose: bool,
    top_k: i32,
}

fn query(socket: &Socket, args: &QueryArgs) -> anyhow::Result<DocumentQueryResponse> {
    let request = DocumentQueryRequest {
        input: args.input.clone(),
        model: "smoke".to_string(),
        search_mode: args.mode,
        top_k: Some(args.top_k),
        workspace_ids: args.workspaces.clone(),
        doc_ids: args.doc_ids.clone(),
        rerank: Some(args.rerank),
        verbose: Some(args.verbose),
    };
    let (header, body) = call(socket, ZmqMessageType::DocumentQuery, Some(request.pack()?))?;
    let body = body.ok_or_else(|| anyhow::anyhow!("query failed: {}", error_of(&header)))?;
    Ok(DocumentQueryResponse::unpack(&body)?)
}

fn retrieve(
    socket: &Socket,
    request: DocumentRetrievalRequest,
) -> anyhow::Result<Option<DocumentRetrievalResponse>> {
    let (header, body) = call(
        socket,
        ZmqMessageType::DocumentRetrieval,
        Some(request.pack()?),
    )?;
    match body {
        Some(body) => Ok(Some(DocumentRetrievalResponse::unpack(&body)?)),
        None => {
            println!("  (retrieval returned an error: {})", error_of(&header));
            Ok(None)
        }
    }
}

fn delete(
    socket: &Socket,
    document_id: u64,
    workspace: Uuid,
) -> anyhow::Result<Option<DeletionStatus>> {
    let request = DocumentDeletionRequest {
        document_id,
        user: Some(workspace),
    };
    let (header, body) = call(
        socket,
        ZmqMessageType::DocumentDeletion,
        Some(request.pack()?),
    )?;
    match body {
        Some(body) => {
            let response: DocumentDeletionResponse = DocumentDeletionResponse::unpack(&body)?;
            Ok(Some(response.status))
        }
        None => {
            println!("  (deletion returned an error: {})", error_of(&header));
            Ok(None)
        }
    }
}
