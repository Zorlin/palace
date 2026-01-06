use std::path::PathBuf;
use std::process::Command;
use std::fs;

/// Get the path where libanthropic.so should be cached
pub fn get_lib_cache_path() -> Result<PathBuf, Box<dyn std::error::Error>> {
    let config_dir = dirs::config_dir()
        .or_else(|| dirs::home_dir().map(|h| h.join(".config")))
        .ok_or("Could not determine config directory")?;

    let lib_dir = config_dir.join("palace").join("lib");
    Ok(lib_dir.join("libanthropic.so"))
}

/// Check if Go is available
pub fn go_available() -> bool {
    Command::new("go")
        .arg("version")
        .output()
        .map(|o| o.status.success())
        .unwrap_or(false)
}

/// Build libanthropic.so from the Go sources
/// Returns the path to the built library
pub fn build_library() -> Result<PathBuf, Box<dyn std::error::Error>> {
    tracing::info!("Building libanthropic.so from Go sources...");

    if !go_available() {
        return Err("Go SDK not found - cannot build libanthropic.so".into());
    }

    // Find the Go sources (relative to this crate)
    let manifest_dir = PathBuf::from(env!("CARGO_MANIFEST_DIR"));

    // Ensure go.mod is tidy
    let status = Command::new("go")
        .args(["mod", "tidy"])
        .current_dir(&manifest_dir)
        .status()?;

    if !status.success() {
        return Err("go mod tidy failed".into());
    }

    // Create output directory
    let lib_path = get_lib_cache_path()?;
    if let Some(parent) = lib_path.parent() {
        fs::create_dir_all(parent)?;
    }

    // Build the shared library
    tracing::info!("Compiling Go shared library to {:?}", lib_path);
    let status = Command::new("go")
        .args([
            "build",
            "-buildmode=c-shared",
            "-o",
            lib_path.to_str().unwrap(),
            "main.go",
        ])
        .current_dir(&manifest_dir)
        .status()?;

    if !status.success() {
        return Err("Failed to build Go shared library".into());
    }

    tracing::info!("Successfully built libanthropic.so");
    Ok(lib_path)
}

/// Ensure the library exists, building it if necessary
/// Returns the path to the library, or an error if it cannot be obtained
pub fn ensure_library() -> Result<PathBuf, Box<dyn std::error::Error>> {
    let lib_path = get_lib_cache_path()?;

    if lib_path.exists() {
        tracing::debug!("Using cached libanthropic.so from {:?}", lib_path);
        return Ok(lib_path);
    }

    tracing::info!("libanthropic.so not found, attempting to build...");
    build_library()
}
