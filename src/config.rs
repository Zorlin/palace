use anyhow::Result;
use serde::{Deserialize, Serialize};
use std::path::Path;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Backend {
    Anthropic,
    OpenCode,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Config {
    #[serde(default)]
    pub api: ApiConfig,

    #[serde(default)]
    pub ui: UiConfig,

    #[serde(default)]
    pub gamepad: GamepadConfig,

    #[serde(default)]
    pub execution: ExecutionConfig,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ApiConfig {
    #[serde(default = "default_backend")]
    pub backend: String,

    #[serde(default = "default_base_url")]
    pub base_url: String,

    #[serde(default = "default_opencode_url")]
    pub opencode_url: String,

    #[serde(default = "default_model")]
    pub model: String,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct UiConfig {
    #[serde(default = "default_theme")]
    pub theme: String,

    #[serde(default = "default_true")]
    pub show_complexity: bool,

    #[serde(default = "default_true")]
    pub mouse_enabled: bool,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct GamepadConfig {
    #[serde(default = "default_true")]
    pub enabled: bool,

    #[serde(default = "default_deadzone")]
    pub deadzone: f32,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ExecutionConfig {
    #[serde(default = "default_parallel_limit")]
    pub parallel_limit: usize,

    #[serde(default)]
    pub auto_commit: bool,
}

// Default value functions
fn default_backend() -> String {
    "anthropic".to_string()
}

fn default_base_url() -> String {
    "https://api.z.ai/api/anthropic".to_string()
}

fn default_opencode_url() -> String {
    "http://localhost:3000".to_string()
}

fn default_model() -> String {
    "claude-sonnet-4-5".to_string()
}

fn default_theme() -> String {
    "dark".to_string()
}

fn default_true() -> bool {
    true
}

fn default_deadzone() -> f32 {
    0.15
}

fn default_parallel_limit() -> usize {
    4
}

impl Default for Config {
    fn default() -> Self {
        Self {
            api: ApiConfig::default(),
            ui: UiConfig::default(),
            gamepad: GamepadConfig::default(),
            execution: ExecutionConfig::default(),
        }
    }
}

impl Default for ApiConfig {
    fn default() -> Self {
        Self {
            backend: default_backend(),
            base_url: default_base_url(),
            opencode_url: default_opencode_url(),
            model: default_model(),
        }
    }
}

impl Default for UiConfig {
    fn default() -> Self {
        Self {
            theme: default_theme(),
            show_complexity: true,
            mouse_enabled: true,
        }
    }
}

impl Default for GamepadConfig {
    fn default() -> Self {
        Self {
            enabled: true,
            deadzone: default_deadzone(),
        }
    }
}

impl Default for ExecutionConfig {
    fn default() -> Self {
        Self {
            parallel_limit: default_parallel_limit(),
            auto_commit: false,
        }
    }
}

impl Config {
    pub fn load(path: Option<&Path>) -> Result<Self> {
        // Try explicit path first
        if let Some(p) = path {
            if p.exists() {
                let contents = std::fs::read_to_string(p)?;
                return Ok(toml::from_str(&contents)?);
            }
        }

        // Try default config locations
        let default_paths = [
            dirs::config_dir().map(|p| p.join("palace/config.toml")),
            Some(std::path::PathBuf::from(".palace/config.toml")),
        ];

        for path in default_paths.into_iter().flatten() {
            if path.exists() {
                let contents = std::fs::read_to_string(&path)?;
                tracing::info!(?path, "Loaded config");
                return Ok(toml::from_str(&contents)?);
            }
        }

        // Return defaults
        tracing::info!("Using default config");
        Ok(Self::default())
    }
}
