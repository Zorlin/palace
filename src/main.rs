mod ai;
mod app;
mod config;
mod db;
mod input;
mod ui;

use anyhow::Result;
use clap::Parser;
use std::path::PathBuf;

#[derive(Parser, Debug)]
#[command(name = "palace")]
#[command(about = "Gamepad-native AI-assisted coding TUI")]
#[command(version)]
struct Cli {
    /// Use OpenCode HTTP API instead of Anthropic SDK
    #[arg(long = "opencode", short = 'o')]
    opencode: bool,

    /// Path to project directory
    #[arg(short, long, default_value = ".")]
    project: PathBuf,

    /// Path to config file
    #[arg(short, long)]
    config: Option<PathBuf>,

    #[command(subcommand)]
    command: Option<Command>,
}

#[derive(clap::Subcommand, Debug)]
enum Command {
    /// Switch to a different project
    Switch { path: PathBuf },
}

#[tokio::main]
async fn main() -> Result<()> {
    // Load .env file if present
    let _ = dotenvy::dotenv();

    // Initialize logging
    tracing_subscriber::fmt()
        .with_env_filter(
            tracing_subscriber::EnvFilter::from_default_env()
                .add_directive(tracing::Level::INFO.into()),
        )
        .init();

    let cli = Cli::parse();

    // Load config
    let config = config::Config::load(cli.config.as_deref())?;

    // Determine backend
    let backend = if cli.opencode {
        config::Backend::OpenCode
    } else {
        config::Backend::Anthropic
    };

    tracing::info!(?backend, project = ?cli.project, "Starting Palace");

    // Run the application
    let mut app = app::App::new(config, backend, cli.project)?;

    match cli.command {
        Some(Command::Switch { path }) => {
            app.switch_project(path)?;
            app.run().await?;
        }
        None => {
            app.run().await?;
        }
    }

    Ok(())
}
