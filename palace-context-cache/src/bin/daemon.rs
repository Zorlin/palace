//! Palace Context Cache Daemon
//!
//! A high-performance daemon that manages context for the swarm consciousness.
//! Communicates via Unix socket for near-zero latency.
//!
//! Protocol:
//! - `DELTA ++1,2,3--4,5,6` - Apply context delta
//! - `GET_ACTIVE` - Get active block IDs
//! - `GET_CONTEXT` - Get full active context
//! - `GET_CLASSIFIER_INPUT` - Get input for 1b classifier
//! - `REGISTER <type> <summary>` - Register new block (returns ID)
//! - `STATS` - Get cache statistics
//! - `WATCH <path>` - Watch a path for changes

use palace_context_cache::{BlockType, ContextCache};
use std::path::PathBuf;
use tokio::io::{AsyncBufReadExt, AsyncWriteExt, BufReader};
use tokio::net::UnixListener;
use tracing::{error, info, Level};
use tracing_subscriber::FmtSubscriber;

const SOCKET_PATH: &str = "/tmp/palace-context-cache.sock";
const DEFAULT_CONTENT_DIR: &str = "/var/lib/palace/context";

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    // Initialize tracing
    let subscriber = FmtSubscriber::builder()
        .with_max_level(Level::INFO)
        .finish();
    tracing::subscriber::set_global_default(subscriber)?;

    info!("Palace Context Cache Daemon starting...");

    // Parse command line args
    let content_dir = std::env::args()
        .nth(1)
        .map(PathBuf::from)
        .unwrap_or_else(|| PathBuf::from(DEFAULT_CONTENT_DIR));

    // Ensure content directory exists
    std::fs::create_dir_all(&content_dir)?;

    // Create cache
    let cache = std::sync::Arc::new(ContextCache::new(content_dir.clone()));

    info!("Content directory: {:?}", content_dir);

    // Remove old socket if exists
    let _ = std::fs::remove_file(SOCKET_PATH);

    // Bind Unix socket
    let listener = UnixListener::bind(SOCKET_PATH)?;
    info!("Listening on {}", SOCKET_PATH);

    loop {
        let (stream, _) = listener.accept().await?;
        let cache = cache.clone();

        tokio::spawn(async move {
            let (reader, mut writer) = stream.into_split();
            let mut reader = BufReader::new(reader);
            let mut line = String::new();

            loop {
                line.clear();
                match reader.read_line(&mut line).await {
                    Ok(0) => break, // Connection closed
                    Ok(_) => {
                        let response = handle_command(&cache, line.trim());
                        if let Err(e) = writer.write_all(response.as_bytes()).await {
                            error!("Failed to write response: {}", e);
                            break;
                        }
                        if let Err(e) = writer.write_all(b"\n").await {
                            error!("Failed to write newline: {}", e);
                            break;
                        }
                    }
                    Err(e) => {
                        error!("Read error: {}", e);
                        break;
                    }
                }
            }
        });
    }
}

fn handle_command(cache: &ContextCache, command: &str) -> String {
    let parts: Vec<&str> = command.splitn(2, ' ').collect();
    let cmd = parts.first().map(|s| s.to_uppercase()).unwrap_or_default();
    let args = parts.get(1).copied().unwrap_or("");

    match cmd.as_str() {
        "DELTA" => match cache.apply_delta(args) {
            Ok(()) => {
                let stats = cache.stats();
                format!("OK active={}", stats.active_blocks)
            }
            Err(e) => format!("ERR {}", e),
        },

        "GET_ACTIVE" => {
            let ids = cache.get_active_ids();
            let ids_str: Vec<String> = ids.iter().map(|id| id.to_string()).collect();
            format!("OK {}", ids_str.join(","))
        }

        "GET_CONTEXT" => {
            let context = cache.get_active_context();
            format!("OK {}", context.replace('\n', "\\n"))
        }

        "GET_CLASSIFIER_INPUT" => {
            let input = cache.get_classifier_input();
            format!("OK {}", input.replace('\n', "\\n"))
        }

        "REGISTER" => {
            let reg_parts: Vec<&str> = args.splitn(2, ' ').collect();
            if reg_parts.len() < 2 {
                return "ERR Usage: REGISTER <type> <summary>".to_string();
            }

            let block_type = match reg_parts[0].to_uppercase().as_str() {
                "CURRENTFILE" => BlockType::CurrentFile,
                "ERROR" => BlockType::Error,
                "USERGOAL" => BlockType::UserGoal,
                "PREVIOUSATTEMPT" => BlockType::PreviousAttempt,
                "BUILDSTATUS" => BlockType::BuildStatus,
                "TESTOUTPUT" => BlockType::TestOutput,
                "RELATED" => BlockType::Related,
                "UNRELATED" => BlockType::Unrelated,
                "MODELRESPONSE" => BlockType::ModelResponse,
                "TOOLRESULT" => BlockType::ToolResult,
                _ => return format!("ERR Unknown block type: {}", reg_parts[0]),
            };

            // For now, create a placeholder file
            let summary = reg_parts[1];
            let id = std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .unwrap()
                .as_nanos();
            let path = std::env::temp_dir().join(format!("palace_block_{}.txt", id));

            if let Err(e) = std::fs::write(&path, summary) {
                return format!("ERR Failed to create block file: {}", e);
            }

            let block_id = cache.register_block(summary.to_string(), block_type, path);
            format!("OK {}", block_id)
        }

        "STATS" => {
            let stats = cache.stats();
            format!(
                "OK active={} total={} cached={}",
                stats.active_blocks, stats.total_blocks, stats.cached_content
            )
        }

        "PING" => "PONG".to_string(),

        "HELP" => {
            r#"OK Commands:
DELTA ++1,2,3--4,5,6 - Apply context delta
GET_ACTIVE - Get active block IDs
GET_CONTEXT - Get full active context
GET_CLASSIFIER_INPUT - Get input for 1b classifier
REGISTER <type> <summary> - Register new block
STATS - Get cache statistics
PING - Health check
HELP - Show this help"#
                .replace('\n', "\\n")
        }

        _ => format!("ERR Unknown command: {}", cmd),
    }
}
