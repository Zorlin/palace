use crate::renderer::ProjectStatus;
use anyhow::{Context, Result};
use serde::{Deserialize, Serialize};
use std::fs;
use std::path::PathBuf;

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Project {
    pub path: PathBuf,
    pub name: String,
    #[serde(default)]
    pub description: String,
    #[serde(default)]
    pub languages: Vec<String>,
    #[serde(skip)]
    pub status: ProjectStatus,
}

#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct ProjectsConfig {
    pub projects: Vec<Project>,
}

impl ProjectsConfig {
    /// Load projects from config file
    pub fn load() -> Result<Self> {
        let config_path = Self::config_path()?;

        if !config_path.exists() {
            return Ok(Self::default());
        }

        let contents = fs::read_to_string(&config_path)
            .context("Failed to read projects config")?;

        serde_json::from_str(&contents)
            .context("Failed to parse projects config")
    }

    /// Save projects to config file
    pub fn save(&self) -> Result<()> {
        let config_path = Self::config_path()?;

        // Ensure config directory exists
        if let Some(parent) = config_path.parent() {
            fs::create_dir_all(parent)
                .context("Failed to create config directory")?;
        }

        let contents = serde_json::to_string_pretty(self)
            .context("Failed to serialize projects config")?;

        fs::write(&config_path, contents)
            .context("Failed to write projects config")?;

        Ok(())
    }

    /// Add a project if not already present
    pub fn add_project(&mut self, path: PathBuf) -> bool {
        // Check if already exists
        if self.projects.iter().any(|p| p.path == path) {
            return false;
        }

        let name = path
            .file_name()
            .and_then(|n| n.to_str())
            .unwrap_or("Unknown")
            .to_string();

        let languages = detect_languages(&path);

        self.projects.push(Project {
            path,
            name,
            description: String::new(),
            languages,
            status: ProjectStatus::Unknown,
        });
        true
    }

    /// Get config file path
    fn config_path() -> Result<PathBuf> {
        let config_dir = dirs::config_dir()
            .or_else(|| dirs::home_dir().map(|h| h.join(".config")))
            .context("Could not determine config directory")?;

        Ok(config_dir.join("palace").join("projects.json"))
    }
}

/// Determine if we're in a project directory or home
pub fn detect_launch_context() -> LaunchContext {
    let cwd = std::env::current_dir().unwrap_or_default();
    let home = dirs::home_dir().unwrap_or_default();

    if cwd == home {
        LaunchContext::Home
    } else if is_project_directory(&cwd) {
        LaunchContext::Project(cwd)
    } else {
        // Not home, but not a recognized project - treat as project anyway
        LaunchContext::Project(cwd)
    }
}

#[derive(Debug)]
pub enum LaunchContext {
    Home,
    Project(PathBuf),
}

/// Check if a directory looks like a project
fn is_project_directory(path: &PathBuf) -> bool {
    // Check for common project markers
    let markers = [
        "Cargo.toml",
        "package.json",
        "pyproject.toml",
        "go.mod",
        ".git",
        "Makefile",
    ];

    markers.iter().any(|marker| path.join(marker).exists())
}

/// Detect programming languages used in a project
fn detect_languages(path: &PathBuf) -> Vec<String> {
    let mut languages = Vec::new();

    // Check for language-specific files
    if path.join("Cargo.toml").exists() {
        languages.push("Rust".to_string());
    }
    if path.join("package.json").exists() {
        // Check for framework hints
        if path.join("vue.config.js").exists()
            || path.join("vite.config.ts").exists()
            || path.join("nuxt.config.ts").exists()
        {
            languages.push("Vue".to_string());
        } else if path.join("next.config.js").exists() || path.join("next.config.ts").exists() {
            languages.push("React".to_string());
        } else {
            languages.push("TypeScript".to_string());
        }
    }
    if path.join("pyproject.toml").exists() || path.join("requirements.txt").exists() {
        languages.push("Python".to_string());
    }
    if path.join("go.mod").exists() {
        languages.push("Go".to_string());
    }
    if path.join("Gemfile").exists() {
        languages.push("Ruby".to_string());
    }
    if path.join("pom.xml").exists() || path.join("build.gradle").exists() {
        languages.push("Java".to_string());
    }
    if path.join("*.sln").exists() || path.join("*.csproj").exists() {
        languages.push("C#".to_string());
    }

    languages
}

/// Get color for a programming language
pub fn language_color(lang: &str) -> [f32; 4] {
    match lang.to_lowercase().as_str() {
        "rust" => [1.0, 0.4, 0.2, 1.0],      // Orange
        "typescript" | "javascript" => [0.2, 0.6, 1.0, 1.0], // Blue
        "vue" => [0.2, 0.8, 0.4, 1.0],       // Green
        "react" => [0.4, 0.8, 1.0, 1.0],     // Cyan
        "python" => [1.0, 0.8, 0.2, 1.0],    // Yellow
        "go" => [0.0, 0.7, 0.9, 1.0],        // Teal
        "ruby" => [0.9, 0.2, 0.2, 1.0],      // Red
        "java" => [0.9, 0.5, 0.2, 1.0],      // Orange-brown
        "c#" | "csharp" => [0.5, 0.2, 0.8, 1.0], // Purple
        _ => [0.6, 0.6, 0.6, 1.0],           // Gray
    }
}
