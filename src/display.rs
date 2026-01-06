//! Display detection and UI scaling
//!
//! Handles:
//! - Detecting system UI scaling from Wayland/X11
//! - Estimating physical display size from model names
#![allow(dead_code)]
//! - Per-display scaling preferences
//! - Computing appropriate UI scale based on DPI

use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::path::PathBuf;
use std::process::Command;

/// Known display models and their approximate diagonal sizes in inches
/// This helps estimate DPI when the system doesn't report physical size
fn known_display_sizes() -> HashMap<&'static str, f32> {
    let mut m = HashMap::new();

    // Handheld gaming PCs
    m.insert("GPD Win 4", 6.0);
    m.insert("GPD Win Max 2", 10.1);
    m.insert("GPD Win Max", 8.0);
    m.insert("GPD Win 3", 5.5);
    m.insert("GPD Win 2", 6.0);
    m.insert("AYANEO 2", 7.0);
    m.insert("AYANEO AIR", 5.5);
    m.insert("AYANEO NEXT", 7.0);
    m.insert("ROG Ally", 7.0);
    m.insert("Steam Deck", 7.0);
    m.insert("Legion Go", 8.8);
    m.insert("ONEXPLAYER", 8.4);
    m.insert("AYN Loki", 6.0);

    // Common laptop panel sizes (by resolution patterns)
    m.insert("1920x1080@14", 14.0);  // Common 14" FHD
    m.insert("1920x1080@15", 15.6);  // Common 15.6" FHD
    m.insert("2560x1440@14", 14.0);  // QHD 14"
    m.insert("2560x1600@13", 13.3);  // MacBook-style
    m.insert("3840x2160@15", 15.6);  // 4K 15.6"

    m
}

/// Display scaling configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DisplayScaling {
    /// Per-display scale overrides (display name -> scale factor)
    #[serde(default)]
    pub display_scales: HashMap<String, f32>,

    /// Whether to auto-detect system scaling
    #[serde(default = "default_true")]
    pub use_system_scaling: bool,

    /// Whether to estimate scaling from display model
    #[serde(default = "default_true")]
    pub use_model_estimation: bool,

    /// Global scale multiplier (applied on top of detected scale)
    #[serde(default = "default_one")]
    pub global_multiplier: f32,
}

fn default_true() -> bool { true }
fn default_one() -> f32 { 1.0 }

impl Default for DisplayScaling {
    fn default() -> Self {
        Self {
            display_scales: HashMap::new(),
            use_system_scaling: true,
            use_model_estimation: true,
            global_multiplier: 1.0,
        }
    }
}

impl DisplayScaling {
    /// Load scaling config from file
    pub fn load() -> Self {
        let config_path = Self::config_path();
        if config_path.exists() {
            if let Ok(json) = std::fs::read_to_string(&config_path) {
                if let Ok(config) = serde_json::from_str(&json) {
                    return config;
                }
            }
        }
        Self::default()
    }

    /// Save scaling config to file
    pub fn save(&self) -> std::io::Result<()> {
        let config_path = Self::config_path();
        if let Some(parent) = config_path.parent() {
            std::fs::create_dir_all(parent)?;
        }
        let json = serde_json::to_string_pretty(self)?;
        std::fs::write(config_path, json)
    }

    fn config_path() -> PathBuf {
        dirs::config_dir()
            .unwrap_or_else(|| PathBuf::from("."))
            .join("palace")
            .join("display_scaling.json")
    }

    /// Get the scale factor for a specific display
    pub fn get_scale_for_display(&self, display_name: &str, width: u32, height: u32) -> f32 {
        // Priority 1: User-specified per-display scale
        if let Some(&scale) = self.display_scales.get(display_name) {
            tracing::info!("Using user-configured scale {} for {}", scale, display_name);
            return scale * self.global_multiplier;
        }

        // Priority 2: System scaling (Wayland/X11)
        if self.use_system_scaling {
            if let Some(scale) = detect_system_scaling(display_name) {
                tracing::info!("Using system scale {} for {}", scale, display_name);
                return scale * self.global_multiplier;
            }
        }

        // Priority 3: Model-based estimation
        if self.use_model_estimation {
            if let Some(scale) = estimate_scale_from_model(display_name, width, height) {
                tracing::info!("Estimated scale {} for {} based on model", scale, display_name);
                return scale * self.global_multiplier;
            }
        }

        // Fallback: resolution-based heuristic
        let scale = estimate_scale_from_resolution(width, height);
        tracing::info!("Using resolution-based scale {} for {}", scale, display_name);
        scale * self.global_multiplier
    }
}

/// Detect system UI scaling from Wayland or X11
pub fn detect_system_scaling(display_name: &str) -> Option<f32> {
    // Try Wayland first
    if std::env::var("WAYLAND_DISPLAY").is_ok() {
        if let Some(scale) = detect_wayland_scaling(display_name) {
            return Some(scale);
        }
    }

    // Fall back to X11
    if std::env::var("DISPLAY").is_ok() {
        if let Some(scale) = detect_x11_scaling(display_name) {
            return Some(scale);
        }
    }

    // Try GNOME settings (works on both)
    if let Some(scale) = detect_gnome_scaling() {
        return Some(scale);
    }

    // Try KDE settings
    if let Some(scale) = detect_kde_scaling() {
        return Some(scale);
    }

    None
}

/// Detect Wayland scaling using various methods
fn detect_wayland_scaling(display_name: &str) -> Option<f32> {
    // Try wlr-randr (wlroots compositors)
    if let Some(scale) = detect_wlr_randr_scaling(display_name) {
        return Some(scale);
    }

    // Try GNOME Mutter (via gdbus)
    if let Some(scale) = detect_gnome_mutter_scaling(display_name) {
        return Some(scale);
    }

    // Try KDE KScreen
    if let Some(scale) = detect_kscreen_scaling(display_name) {
        return Some(scale);
    }

    None
}

/// Detect scaling from wlr-randr (Sway, Hyprland, etc.)
fn detect_wlr_randr_scaling(display_name: &str) -> Option<f32> {
    let output = Command::new("wlr-randr")
        .output()
        .ok()?;

    if !output.status.success() {
        return None;
    }

    let stdout = String::from_utf8_lossy(&output.stdout);
    let mut current_output = "";

    for line in stdout.lines() {
        // Output names start at beginning of line
        if !line.starts_with(' ') && !line.starts_with('\t') {
            current_output = line.trim();
        }

        // Look for scale in current output section
        if current_output.contains(display_name) || display_name.is_empty() {
            if let Some(scale_str) = line.trim().strip_prefix("Scale: ") {
                if let Ok(scale) = scale_str.parse::<f32>() {
                    return Some(scale);
                }
            }
        }
    }

    None
}

/// Detect scaling from GNOME Mutter via D-Bus
fn detect_gnome_mutter_scaling(display_name: &str) -> Option<f32> {
    // Get current monitor configuration from Mutter
    let output = Command::new("gdbus")
        .args([
            "call", "--session",
            "--dest", "org.gnome.Mutter.DisplayConfig",
            "--object-path", "/org/gnome/Mutter/DisplayConfig",
            "--method", "org.gnome.Mutter.DisplayConfig.GetCurrentState",
        ])
        .output()
        .ok()?;

    if !output.status.success() {
        return None;
    }

    let stdout = String::from_utf8_lossy(&output.stdout);

    // Mutter GetCurrentState returns a complex structure:
    // (serial, monitors[], logical_monitors[], properties{})
    //
    // logical_monitors format: [(x, y, scale, transform, primary, outputs[], props{}), ...]
    // We need to find the scale value (third element) in the logical_monitors section
    //
    // Example: [(0, 0, 1.5, uint32 0, true, [('eDP-1', ...)], @a{sv} {})]

    // Strategy: Find the logical monitors section (after the monitors array)
    // Look for pattern: ", 1.5," or similar scale values

    // First, try to find the display in the output and get the scale from logical monitors
    // The logical monitors section contains tuples starting with (x, y, scale, ...)

    // Find sections that look like logical monitor entries: "(0, 0, X.X, uint32"
    // This regex-like pattern: comma, space, number with decimal, comma
    let mut best_scale: Option<f32> = None;

    // Look for pattern: ", X.X, uint32" which indicates scale in logical monitor tuple
    let chars: Vec<char> = stdout.chars().collect();
    let mut i = 0;
    while i < chars.len().saturating_sub(20) {
        // Look for ", X.X, uint32" pattern
        if chars[i] == ',' && chars[i + 1] == ' ' {
            // Try to parse a float followed by ", uint32"
            let rest: String = chars[i + 2..].iter().take(30).collect();
            if let Some(comma_pos) = rest.find(", uint32") {
                let potential_scale = &rest[..comma_pos];
                if let Ok(scale) = potential_scale.trim().parse::<f32>() {
                    if scale >= 0.5 && scale <= 4.0 {
                        // Check if this logical monitor contains our display
                        // Look ahead for display_name or accept first valid scale
                        let context: String = chars[i..].iter().take(200).collect();
                        if display_name.is_empty() || context.contains(display_name) {
                            best_scale = Some(scale);
                            break;
                        } else if best_scale.is_none() {
                            // Keep as fallback if no display_name match
                            best_scale = Some(scale);
                        }
                    }
                }
            }
        }
        i += 1;
    }

    best_scale
}

/// Detect scaling from KDE KScreen
fn detect_kscreen_scaling(display_name: &str) -> Option<f32> {
    let output = Command::new("kscreen-doctor")
        .arg("-o")
        .output()
        .ok()?;

    if !output.status.success() {
        return None;
    }

    let stdout = String::from_utf8_lossy(&output.stdout);
    let mut in_target_output = false;

    for line in stdout.lines() {
        // Output blocks start with "Output:"
        if line.starts_with("Output:") {
            in_target_output = line.contains(display_name) || display_name.is_empty();
        }

        if in_target_output {
            // Look for "Scale: X.X"
            if let Some(scale_str) = line.trim().strip_prefix("Scale: ") {
                if let Ok(scale) = scale_str.parse::<f32>() {
                    return Some(scale);
                }
            }
        }
    }

    None
}

/// Detect scaling from X11 using xrandr
fn detect_x11_scaling(_display_name: &str) -> Option<f32> {
    // X11 doesn't have native scaling, but we can detect:
    // 1. Xft.dpi setting
    // 2. GNOME/KDE scaling applied via Xresources

    // Check Xft.dpi from xrdb
    let output = Command::new("xrdb")
        .arg("-query")
        .output()
        .ok()?;

    if output.status.success() {
        let stdout = String::from_utf8_lossy(&output.stdout);
        for line in stdout.lines() {
            if line.starts_with("Xft.dpi:") {
                if let Some(dpi_str) = line.split(':').nth(1) {
                    if let Ok(dpi) = dpi_str.trim().parse::<f32>() {
                        // Standard DPI is 96, calculate scale from that
                        let scale = dpi / 96.0;
                        if scale >= 0.5 && scale <= 4.0 {
                            return Some(scale);
                        }
                    }
                }
            }
        }
    }

    // Check GDK_SCALE environment variable
    if let Ok(gdk_scale) = std::env::var("GDK_SCALE") {
        if let Ok(scale) = gdk_scale.parse::<f32>() {
            return Some(scale);
        }
    }

    // Check QT_SCALE_FACTOR
    if let Ok(qt_scale) = std::env::var("QT_SCALE_FACTOR") {
        if let Ok(scale) = qt_scale.parse::<f32>() {
            return Some(scale);
        }
    }

    None
}

/// Detect scaling from GNOME gsettings
fn detect_gnome_scaling() -> Option<f32> {
    let output = Command::new("gsettings")
        .args(["get", "org.gnome.desktop.interface", "text-scaling-factor"])
        .output()
        .ok()?;

    if output.status.success() {
        let stdout = String::from_utf8_lossy(&output.stdout);
        if let Ok(scale) = stdout.trim().parse::<f32>() {
            if scale > 0.5 && scale <= 3.0 {
                return Some(scale);
            }
        }
    }

    None
}

/// Detect scaling from KDE settings
fn detect_kde_scaling() -> Option<f32> {
    // KDE stores scaling in ~/.config/kdeglobals
    let config_path = dirs::config_dir()?.join("kdeglobals");
    let content = std::fs::read_to_string(config_path).ok()?;

    for line in content.lines() {
        if line.starts_with("ScaleFactor=") {
            if let Some(scale_str) = line.strip_prefix("ScaleFactor=") {
                if let Ok(scale) = scale_str.parse::<f32>() {
                    return Some(scale);
                }
            }
        }
    }

    None
}

/// Estimate scale from display model name
fn estimate_scale_from_model(display_name: &str, width: u32, height: u32) -> Option<f32> {
    let known = known_display_sizes();

    // Check if display name contains any known model
    for (model, diagonal_inches) in &known {
        if display_name.to_lowercase().contains(&model.to_lowercase()) {
            return Some(calculate_scale_for_size(width, height, *diagonal_inches));
        }
    }

    // Try to match resolution pattern (e.g., "1920x1080@15" for 15")
    let res_key = format!("{}x{}", width, height);
    for (pattern, diagonal_inches) in &known {
        if pattern.starts_with(&res_key) {
            return Some(calculate_scale_for_size(width, height, *diagonal_inches));
        }
    }

    None
}

/// Calculate appropriate UI scale for a given physical size
fn calculate_scale_for_size(width: u32, height: u32, diagonal_inches: f32) -> f32 {
    // Calculate DPI
    let diagonal_pixels = ((width * width + height * height) as f32).sqrt();
    let dpi = diagonal_pixels / diagonal_inches;

    // Target "comfortable" DPI is around 96-110 for desktop use
    // For handhelds held closer, we might want ~120-140 effective DPI
    let target_dpi = if diagonal_inches < 8.0 {
        130.0  // Handheld - held closer to eyes
    } else if diagonal_inches < 15.0 {
        110.0  // Laptop - medium distance
    } else {
        96.0   // Desktop monitor - further away
    };

    let scale = dpi / target_dpi;

    // Clamp to reasonable range and round to nearest 0.25
    let scale = scale.clamp(0.75, 3.0);
    (scale * 4.0).round() / 4.0
}

/// Fallback: estimate scale from resolution alone
fn estimate_scale_from_resolution(width: u32, height: u32) -> f32 {
    // Heuristic based on common resolutions
    // Assumes "standard" physical sizes for each resolution tier

    let pixels = width.max(height);

    if pixels >= 3840 {
        // 4K - likely needs 2x scaling
        2.0
    } else if pixels >= 2560 {
        // QHD/WQHD - likely needs 1.5x
        1.5
    } else if pixels >= 1920 {
        // FHD - check if it might be a small high-DPI screen
        // 1920 on a 6" screen (like GPD Win 4) needs ~1.5x
        // 1920 on a 15" screen is fine at 1x
        // Without physical size info, assume laptop/desktop (1x)
        1.0
    } else {
        // Lower resolution - probably fine at 1x
        1.0
    }
}

/// Get current display info from the system
pub fn get_display_info() -> Vec<DisplayInfo> {
    let displays = Vec::new();

    // Try Wayland first
    if std::env::var("WAYLAND_DISPLAY").is_ok() {
        if let Some(wayland_displays) = get_wayland_displays() {
            return wayland_displays;
        }
    }

    // Fall back to X11
    if let Some(x11_displays) = get_x11_displays() {
        return x11_displays;
    }

    displays
}

#[derive(Debug, Clone)]
pub struct DisplayInfo {
    pub name: String,
    pub width: u32,
    pub height: u32,
    pub physical_width_mm: Option<u32>,
    pub physical_height_mm: Option<u32>,
    pub scale: f32,
}

impl DisplayInfo {
    /// Calculate DPI if physical size is known
    pub fn dpi(&self) -> Option<f32> {
        let (pw, ph) = (self.physical_width_mm?, self.physical_height_mm?);
        if pw == 0 || ph == 0 {
            return None;
        }

        let diagonal_mm = ((pw * pw + ph * ph) as f32).sqrt();
        let diagonal_inches = diagonal_mm / 25.4;
        let diagonal_pixels = ((self.width * self.width + self.height * self.height) as f32).sqrt();

        Some(diagonal_pixels / diagonal_inches)
    }
}

fn get_wayland_displays() -> Option<Vec<DisplayInfo>> {
    // Try wlr-randr
    let output = Command::new("wlr-randr").output().ok()?;

    if !output.status.success() {
        return None;
    }

    let stdout = String::from_utf8_lossy(&output.stdout);
    let mut displays = Vec::new();
    let mut current: Option<DisplayInfo> = None;

    for line in stdout.lines() {
        let line = line.trim();

        // New output starts with name (no leading whitespace in original)
        if !line.starts_with(' ') && !line.is_empty() && !line.starts_with("Output") {
            if let Some(display) = current.take() {
                displays.push(display);
            }
            current = Some(DisplayInfo {
                name: line.to_string(),
                width: 0,
                height: 0,
                physical_width_mm: None,
                physical_height_mm: None,
                scale: 1.0,
            });
        }

        if let Some(ref mut display) = current {
            // Parse resolution: "1920x1080 px, ..."
            if line.contains("px,") {
                if let Some(res) = line.split("px").next() {
                    let parts: Vec<&str> = res.trim().split('x').collect();
                    if parts.len() == 2 {
                        display.width = parts[0].parse().unwrap_or(0);
                        display.height = parts[1].parse().unwrap_or(0);
                    }
                }
            }

            // Parse physical size: "... 310mm x 170mm"
            if line.contains("mm x") {
                let parts: Vec<&str> = line.split("mm").collect();
                if parts.len() >= 2 {
                    if let Some(w_str) = parts[0].split_whitespace().last() {
                        display.physical_width_mm = w_str.parse().ok();
                    }
                    if let Some(h_str) = parts[1].trim().strip_prefix("x ").and_then(|s| s.split_whitespace().next()) {
                        display.physical_height_mm = h_str.parse().ok();
                    }
                }
            }

            // Parse scale
            if let Some(scale_str) = line.strip_prefix("Scale: ") {
                display.scale = scale_str.parse().unwrap_or(1.0);
            }
        }
    }

    if let Some(display) = current {
        displays.push(display);
    }

    if displays.is_empty() {
        None
    } else {
        Some(displays)
    }
}

fn get_x11_displays() -> Option<Vec<DisplayInfo>> {
    let output = Command::new("xrandr")
        .arg("--query")
        .output()
        .ok()?;

    if !output.status.success() {
        return None;
    }

    let stdout = String::from_utf8_lossy(&output.stdout);
    let mut displays = Vec::new();

    for line in stdout.lines() {
        // Connected displays: "HDMI-1 connected primary 1920x1080+0+0 (normal...) 530mm x 300mm"
        if line.contains(" connected") {
            let parts: Vec<&str> = line.split_whitespace().collect();
            if parts.is_empty() {
                continue;
            }

            let name = parts[0].to_string();
            let mut width = 0u32;
            let mut height = 0u32;
            let mut physical_width_mm = None;
            let mut physical_height_mm = None;

            // Find resolution (NNNNxNNNN+X+Y format)
            for part in &parts {
                if part.contains('x') && part.contains('+') {
                    if let Some(res) = part.split('+').next() {
                        let dims: Vec<&str> = res.split('x').collect();
                        if dims.len() == 2 {
                            width = dims[0].parse().unwrap_or(0);
                            height = dims[1].parse().unwrap_or(0);
                        }
                    }
                }

                // Find physical size (NNNmm x NNNmm)
                if part.ends_with("mm") {
                    if let Some(mm_str) = part.strip_suffix("mm") {
                        if physical_width_mm.is_none() {
                            physical_width_mm = mm_str.parse().ok();
                        } else {
                            physical_height_mm = mm_str.parse().ok();
                        }
                    }
                }
            }

            if width > 0 && height > 0 {
                displays.push(DisplayInfo {
                    name,
                    width,
                    height,
                    physical_width_mm,
                    physical_height_mm,
                    scale: 1.0,  // X11 doesn't have native scaling
                });
            }
        }
    }

    if displays.is_empty() {
        None
    } else {
        Some(displays)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_calculate_scale_gpd_win4() {
        // GPD Win 4: 1920x1080, 6" diagonal
        let scale = calculate_scale_for_size(1920, 1080, 6.0);
        // DPI = 2203/6 = 367, target = 130, scale = 2.82 -> clamped/rounded to 2.75
        assert!(scale >= 2.5 && scale <= 3.0, "Expected ~2.75 for GPD Win 4, got {}", scale);
    }

    #[test]
    fn test_calculate_scale_laptop() {
        // Typical 15.6" FHD laptop
        let scale = calculate_scale_for_size(1920, 1080, 15.6);
        // DPI = 141, target = 110, scale = 1.28 -> rounded to 1.25
        assert!(scale >= 1.0 && scale <= 1.5, "Expected ~1.25 for 15.6\" FHD, got {}", scale);
    }

    #[test]
    fn test_calculate_scale_4k_desktop() {
        // 27" 4K desktop monitor
        let scale = calculate_scale_for_size(3840, 2160, 27.0);
        // DPI = 163, target = 96, scale = 1.7 -> rounded to 1.75
        assert!(scale >= 1.5 && scale <= 2.0, "Expected ~1.75 for 27\" 4K, got {}", scale);
    }

    #[test]
    fn test_resolution_heuristic() {
        assert_eq!(estimate_scale_from_resolution(3840, 2160), 2.0);
        assert_eq!(estimate_scale_from_resolution(2560, 1440), 1.5);
        assert_eq!(estimate_scale_from_resolution(1920, 1080), 1.0);
    }
}
