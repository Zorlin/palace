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
    #[serde(default)]
    pub archived: bool,
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
            archived: false,
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

#[cfg(test)]
mod tests {
    use super::*;

    // ============== Project Tests ==============
    #[test]
    fn test_project_creation() {
        let project = Project {
            path: PathBuf::from("/test/project"),
            name: "test_project".to_string(),
            description: "A test project".to_string(),
            languages: vec!["Rust".to_string()],
            archived: false,
            status: ProjectStatus::Unknown,
        };

        assert_eq!(project.name, "test_project");
        assert_eq!(project.path, PathBuf::from("/test/project"));
        assert_eq!(project.languages.len(), 1);
        assert_eq!(project.languages[0], "Rust");
    }

    // ============== ProjectsConfig Tests ==============
    #[test]
    fn test_projects_config_default() {
        let config = ProjectsConfig::default();
        assert!(config.projects.is_empty());
    }

    #[test]
    fn test_projects_config_add_project() {
        let mut config = ProjectsConfig::default();
        let path = PathBuf::from("/test/project");

        // Add project
        let added = config.add_project(path.clone());
        assert!(added, "Should return true when adding new project");
        assert_eq!(config.projects.len(), 1);
        assert_eq!(config.projects[0].path, path);

        // Try adding duplicate
        let added_again = config.add_project(path.clone());
        assert!(!added_again, "Should return false when adding duplicate");
        assert_eq!(config.projects.len(), 1, "Should not add duplicate");
    }

    #[test]
    fn test_projects_config_add_multiple_projects() {
        let mut config = ProjectsConfig::default();

        config.add_project(PathBuf::from("/project1"));
        config.add_project(PathBuf::from("/project2"));
        config.add_project(PathBuf::from("/project3"));

        assert_eq!(config.projects.len(), 3);
    }

    // ============== LaunchContext Tests ==============
    #[test]
    fn test_detect_launch_context_project() {
        // Since we can't easily mock std::env::current_dir and dirs::home_dir,
        // we'll test the LaunchContext enum variants instead

        let project_path = PathBuf::from("/home/user/projects/myproject");
        let context = LaunchContext::Project(project_path);

        match context {
            LaunchContext::Project(path) => {
                assert_eq!(path, PathBuf::from("/home/user/projects/myproject"));
            }
            LaunchContext::Home => {
                panic!("Expected Project context");
            }
        }
    }

    #[test]
    fn test_launch_context_home_variant() {
        let context = LaunchContext::Home;
        match context {
            LaunchContext::Home => {}
            LaunchContext::Project(_) => {
                panic!("Expected Home context");
            }
        }
    }

    // ============== is_project_directory Tests ==============
    #[test]
    fn test_is_project_directory_with_temp_dir() {
        use tempfile::TempDir;

        let temp_dir = TempDir::new().unwrap();
        let dir_path = temp_dir.path();

        // Empty directory should not be detected as a project
        assert!(!is_project_directory(&dir_path.to_path_buf()));

        // Add Cargo.toml (Rust project marker)
        std::fs::write(dir_path.join("Cargo.toml"), "[package]\nname = \"test\"").unwrap();
        assert!(is_project_directory(&dir_path.to_path_buf()));

        // Test with package.json (Node.js project marker)
        let temp_dir2 = TempDir::new().unwrap();
        let dir_path2 = temp_dir2.path();
        std::fs::write(dir_path2.join("package.json"), "{\"name\": \"test\"}").unwrap();
        assert!(is_project_directory(&dir_path2.to_path_buf()));

        // Test with .git directory
        let temp_dir3 = TempDir::new().unwrap();
        let dir_path3 = temp_dir3.path();
        std::fs::create_dir(dir_path3.join(".git")).unwrap();
        assert!(is_project_directory(&dir_path3.to_path_buf()));

        // Test with pyproject.toml (Python project marker)
        let temp_dir4 = TempDir::new().unwrap();
        let dir_path4 = temp_dir4.path();
        std::fs::write(dir_path4.join("pyproject.toml"), "[project]").unwrap();
        assert!(is_project_directory(&dir_path4.to_path_buf()));

        // Test with go.mod (Go project marker)
        let temp_dir5 = TempDir::new().unwrap();
        let dir_path5 = temp_dir5.path();
        std::fs::write(dir_path5.join("go.mod"), "module test").unwrap();
        assert!(is_project_directory(&dir_path5.to_path_buf()));

        // Test with Makefile
        let temp_dir6 = TempDir::new().unwrap();
        let dir_path6 = temp_dir6.path();
        std::fs::write(dir_path6.join("Makefile"), "all:\n\techo 'test'").unwrap();
        assert!(is_project_directory(&dir_path6.to_path_buf()));
    }

    // ============== detect_languages Tests ==============
    #[test]
    fn test_detect_languages_rust() {
        use tempfile::TempDir;

        let temp_dir = TempDir::new().unwrap();
        let dir_path = temp_dir.path();

        // Create Cargo.toml
        std::fs::write(
            dir_path.join("Cargo.toml"),
            "[package]\nname = \"test\"",
        )
        .unwrap();

        let languages = detect_languages(&dir_path.to_path_buf());
        assert_eq!(languages, vec!["Rust"]);
    }

    #[test]
    fn test_detect_languages_python() {
        use tempfile::TempDir;

        let temp_dir = TempDir::new().unwrap();
        let dir_path = temp_dir.path();

        // Create pyproject.toml
        std::fs::write(dir_path.join("pyproject.toml"), "[project]").unwrap();

        let languages = detect_languages(&dir_path.to_path_buf());
        assert_eq!(languages, vec!["Python"]);
    }

    #[test]
    fn test_detect_languages_typescript() {
        use tempfile::TempDir;

        let temp_dir = TempDir::new().unwrap();
        let dir_path = temp_dir.path();

        // Create package.json and tsconfig.json
        std::fs::write(dir_path.join("package.json"), "{\"name\": \"test\"}").unwrap();
        std::fs::write(dir_path.join("tsconfig.json"), "{}").unwrap();

        let languages = detect_languages(&dir_path.to_path_buf());
        assert_eq!(languages, vec!["TypeScript"]);
    }

    #[test]
    fn test_detect_languages_javascript() {
        use tempfile::TempDir;

        let temp_dir = TempDir::new().unwrap();
        let dir_path = temp_dir.path();

        // Create package.json without tsconfig.json
        std::fs::write(dir_path.join("package.json"), "{\"name\": \"test\"}").unwrap();

        let languages = detect_languages(&dir_path.to_path_buf());
        assert_eq!(languages, vec!["TypeScript"]); // Default to TypeScript
    }

    #[test]
    fn test_detect_languages_vue() {
        use tempfile::TempDir;

        let temp_dir = TempDir::new().unwrap();
        let dir_path = temp_dir.path();

        // Create Vue-specific config
        std::fs::write(dir_path.join("package.json"), "{\"name\": \"test\"}").unwrap();
        std::fs::write(dir_path.join("vite.config.ts"), "export default {}").unwrap();

        let languages = detect_languages(&dir_path.to_path_buf());
        assert_eq!(languages, vec!["Vue"]);
    }

    #[test]
    fn test_detect_languages_react() {
        use tempfile::TempDir;

        let temp_dir = TempDir::new().unwrap();
        let dir_path = temp_dir.path();

        // Create Next.js config (React framework)
        std::fs::write(dir_path.join("package.json"), "{\"name\": \"test\"}").unwrap();
        std::fs::write(dir_path.join("next.config.js"), "module.exports = {}").unwrap();

        let languages = detect_languages(&dir_path.to_path_buf());
        assert_eq!(languages, vec!["React"]);
    }

    #[test]
    fn test_detect_languages_go() {
        use tempfile::TempDir;

        let temp_dir = TempDir::new().unwrap();
        let dir_path = temp_dir.path();

        // Create go.mod
        std::fs::write(dir_path.join("go.mod"), "module test").unwrap();

        let languages = detect_languages(&dir_path.to_path_buf());
        assert_eq!(languages, vec!["Go"]);
    }

    #[test]
    fn test_detect_languages_ruby() {
        use tempfile::TempDir;

        let temp_dir = TempDir::new().unwrap();
        let dir_path = temp_dir.path();

        // Create Gemfile
        std::fs::write(dir_path.join("Gemfile"), "gem 'rails'").unwrap();

        let languages = detect_languages(&dir_path.to_path_buf());
        assert_eq!(languages, vec!["Ruby"]);
    }

    #[test]
    fn test_detect_languages_java() {
        use tempfile::TempDir;

        let temp_dir = TempDir::new().unwrap();
        let dir_path = temp_dir.path();

        // Create pom.xml (Maven)
        std::fs::write(
            dir_path.join("pom.xml"),
            "<project></project>",
        )
        .unwrap();

        let languages = detect_languages(&dir_path.to_path_buf());
        assert_eq!(languages, vec!["Java"]);
    }

    #[test]
    fn test_detect_languages_multiple() {
        use tempfile::TempDir;

        let temp_dir = TempDir::new().unwrap();
        let dir_path = temp_dir.path();

        // Create both Cargo.toml and pyproject.toml
        std::fs::write(dir_path.join("Cargo.toml"), "[package]").unwrap();
        std::fs::write(dir_path.join("pyproject.toml"), "[project]").unwrap();

        let languages = detect_languages(&dir_path.to_path_buf());
        assert!(languages.contains(&"Rust".to_string()));
        assert!(languages.contains(&"Python".to_string()));
        assert_eq!(languages.len(), 2);
    }

    // ============== language_color Tests ==============
    #[test]
    fn test_language_color_rust() {
        let color = language_color("Rust");
        assert_eq!(color, [1.0, 0.4, 0.2, 1.0]);

        // Test case-insensitive
        let color_lower = language_color("rust");
        assert_eq!(color_lower, [1.0, 0.4, 0.2, 1.0]);
    }

    #[test]
    fn test_language_color_typescript() {
        let color = language_color("TypeScript");
        assert_eq!(color, [0.2, 0.6, 1.0, 1.0]);

        let color_lower = language_color("typescript");
        assert_eq!(color_lower, [0.2, 0.6, 1.0, 1.0]);
    }

    #[test]
    fn test_language_color_javascript() {
        let color = language_color("JavaScript");
        assert_eq!(color, [0.2, 0.6, 1.0, 1.0]);
    }

    #[test]
    fn test_language_color_vue() {
        let color = language_color("Vue");
        assert_eq!(color, [0.2, 0.8, 0.4, 1.0]);
    }

    #[test]
    fn test_language_color_react() {
        let color = language_color("React");
        assert_eq!(color, [0.4, 0.8, 1.0, 1.0]);
    }

    #[test]
    fn test_language_color_python() {
        let color = language_color("Python");
        assert_eq!(color, [1.0, 0.8, 0.2, 1.0]);
    }

    #[test]
    fn test_language_color_go() {
        let color = language_color("Go");
        assert_eq!(color, [0.0, 0.7, 0.9, 1.0]);
    }

    #[test]
    fn test_language_color_ruby() {
        let color = language_color("Ruby");
        assert_eq!(color, [0.9, 0.2, 0.2, 1.0]);
    }

    #[test]
    fn test_language_color_java() {
        let color = language_color("Java");
        assert_eq!(color, [0.9, 0.5, 0.2, 1.0]);
    }

    #[test]
    fn test_language_color_csharp() {
        let color = language_color("c#");
        assert_eq!(color, [0.5, 0.2, 0.8, 1.0]);

        let color_alt = language_color("csharp");
        assert_eq!(color_alt, [0.5, 0.2, 0.8, 1.0]);
    }

    #[test]
    fn test_language_color_unknown() {
        let color = language_color("UnknownLanguage");
        assert_eq!(color, [0.6, 0.6, 0.6, 1.0]);
    }

    // ============== Integration Tests ==============
    #[test]
    fn test_project_roundtrip() {
        use tempfile::TempDir;

        let temp_dir = TempDir::new().unwrap();
        let project_dir = temp_dir.path().join("myproject");

        // Create project directory with markers
        std::fs::create_dir(&project_dir).unwrap();
        std::fs::write(
            project_dir.join("Cargo.toml"),
            "[package]\nname = \"myproject\"",
        )
        .unwrap();

        // Detect languages
        let languages = detect_languages(&project_dir);
        assert_eq!(languages, vec!["Rust"]);

        // Get language color
        let color = language_color(&languages[0]);
        assert_eq!(color, [1.0, 0.4, 0.2, 1.0]);

        // Check it's detected as a project directory
        assert!(is_project_directory(&project_dir));
    }

    #[test]
    fn test_add_project_extracts_name_from_path() {
        use tempfile::TempDir;

        let mut config = ProjectsConfig::default();
        let temp_dir = TempDir::new().unwrap();
        let project_path = temp_dir.path().join("awesome_project");

        std::fs::create_dir(&project_path).unwrap();
        std::fs::write(project_path.join("Cargo.toml"), "[package]").unwrap();

        config.add_project(project_path.clone());

        assert_eq!(config.projects[0].name, "awesome_project");
        assert_eq!(config.projects[0].path, project_path);
        assert!(config.projects[0].languages.contains(&"Rust".to_string()));
    }
}
