//! PalaceWindow - A single Palace window with its own renderer
//!
//! Each PalaceWindow represents one display/monitor with:
//! - A winit Window
//! - Its own Renderer (with shared GPU device/queue)
//! - Per-window UI state
//!
//! Multiple PalaceWindows act as a single application:
//! - macOS: Same app bundle groups windows in Dock/Cmd-Tab
//! - Linux: Same app_id groups windows in Wayland

use crate::renderer::{Renderer, SharedGpuResources};
use crate::VirtualViewport;
use anyhow::Result;
use winit::event_loop::ActiveEventLoop;
use winit::monitor::MonitorHandle;
use winit::window::{Fullscreen, Window, WindowId};

/// A Palace window - one per monitor
pub struct PalaceWindow {
    /// The winit window handle
    pub window: Window,

    /// Per-window renderer (owns surface, card/sprite renderers, etc.)
    pub renderer: Renderer,

    /// Monitor name for this window (for preferences/logging)
    pub monitor_name: String,

    /// Whether this window needs a redraw
    pub needs_redraw: bool,
}

impl PalaceWindow {
    /// Create a new Palace window on the specified monitor
    ///
    /// If `monitor` is None, creates on the primary/default monitor.
    /// If `parent` is provided, creates as a child window (groups with parent in Dock/taskbar).
    pub fn new(
        event_loop: &ActiveEventLoop,
        monitor: Option<MonitorHandle>,
        shared_gpu: &SharedGpuResources,
        virtual_viewport: &VirtualViewport,
    ) -> Result<Self> {
        Self::new_with_parent(event_loop, monitor, shared_gpu, virtual_viewport, None)
    }

    /// Create a new Palace window on a specific monitor
    ///
    /// Windows from the same application naturally group together:
    /// - macOS: Same Dock icon, Cmd-Tab groups them (via app bundle)
    /// - Linux: Same app_id groups them (Wayland)
    ///
    /// The `_parent` parameter is kept for API compatibility but not used.
    pub fn new_with_parent(
        event_loop: &ActiveEventLoop,
        monitor: Option<MonitorHandle>,
        shared_gpu: &SharedGpuResources,
        virtual_viewport: &VirtualViewport,
        _parent: Option<&Window>,
    ) -> Result<Self> {
        // Get requested monitor name (we'll verify after window creation)
        let requested_monitor_name = monitor
            .as_ref()
            .and_then(|m| m.name())
            .unwrap_or_else(|| "Primary".to_string());

        tracing::info!("Creating Palace window on monitor: {}", requested_monitor_name);

        // Create fullscreen window on the target monitor
        // NOTE: Use Exclusive fullscreen, NOT Borderless - borderless can span/merge displays on macOS
        // Find video mode matching the OS-configured resolution (monitor.size())
        let matching_mode = monitor.as_ref().and_then(|m| {
            let current_size = m.size();
            tracing::info!("Monitor current size: {}x{}", current_size.width, current_size.height);
            m.video_modes()
                .filter(|mode| mode.size() == current_size)
                .max_by_key(|mode| mode.refresh_rate_millihertz())
        });

        let fullscreen_mode = matching_mode
            .map(|mode| {
                tracing::info!("Using video mode: {}x{} @ {}Hz",
                    mode.size().width, mode.size().height,
                    mode.refresh_rate_millihertz() / 1000);
                Fullscreen::Exclusive(mode)
            })
            .or_else(|| {
                tracing::warn!("No matching video mode found, falling back to Borderless");
                Some(Fullscreen::Borderless(monitor.clone()))
            });

        let window_attrs = Window::default_attributes()
            .with_title("Palace")
            .with_fullscreen(fullscreen_mode);

        let window = event_loop
            .create_window(window_attrs)
            .map_err(|e| anyhow::anyhow!("Failed to create window: {}", e))?;

        tracing::info!("Window created: {:?}", window.inner_size());

        // Get actual monitor name from the window (more reliable than requested)
        let actual_monitor_name = window
            .current_monitor()
            .and_then(|m| m.name())
            .unwrap_or(requested_monitor_name);

        tracing::info!("Window placed on monitor: {}", actual_monitor_name);

        // Create renderer using shared GPU resources
        let mut renderer = Renderer::new_with_shared(&window, shared_gpu)?;

        // Apply virtual viewport settings
        renderer.set_virtual_viewport(virtual_viewport.clone());

        // Set monitor name for display identification
        renderer.set_monitor_name(actual_monitor_name.clone());

        Ok(Self {
            window,
            renderer,
            monitor_name: actual_monitor_name,
            needs_redraw: true,
        })
    }

    /// Get this window's ID
    pub fn id(&self) -> WindowId {
        self.window.id()
    }

    /// Request a redraw for this window
    pub fn request_redraw(&mut self) {
        self.needs_redraw = true;
        self.window.request_redraw();
    }

    /// Handle window resize
    pub fn resize(&mut self, new_size: winit::dpi::PhysicalSize<u32>) {
        if new_size.width > 0 && new_size.height > 0 {
            self.renderer.resize(new_size);
            self.request_redraw();
        }
    }

    /// Toggle fullscreen mode
    pub fn toggle_fullscreen(&self) {
        let is_fullscreen = self.window.fullscreen().is_some();
        if is_fullscreen {
            self.window.set_fullscreen(None);
        } else {
            // Get current monitor and go exclusive fullscreen matching OS resolution
            let monitor = self.window.current_monitor();
            let matching_mode = monitor.as_ref().and_then(|m| {
                let current_size = m.size();
                m.video_modes()
                    .filter(|mode| mode.size() == current_size)
                    .max_by_key(|mode| mode.refresh_rate_millihertz())
            });
            let fullscreen_mode = matching_mode
                .map(|mode| Fullscreen::Exclusive(mode))
                .or_else(|| Some(Fullscreen::Borderless(monitor)));
            self.window.set_fullscreen(fullscreen_mode);
        }
    }

    /// Check if this window is fullscreen
    pub fn is_fullscreen(&self) -> bool {
        self.window.fullscreen().is_some()
    }
}
