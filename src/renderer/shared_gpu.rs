//! Shared GPU resources for multi-window rendering
//!
//! These resources are created once and shared across all Palace windows.
//! The expensive GPU initialization (adapter, device, queue) happens here,
//! while per-window resources (surfaces, textures) are created separately.

use anyhow::{Context, Result};
use std::sync::Arc;

/// GPU resources shared across all Palace windows via Arc
///
/// Creating a wgpu Device and Queue is expensive - we do it once and share.
/// Each window creates its own Surface and per-window resources.
pub struct SharedGpuResources {
    /// WGPU instance - needed for creating surfaces
    pub instance: wgpu::Instance,

    /// GPU adapter info (for logging/debugging)
    pub adapter_info: wgpu::AdapterInfo,

    /// Shared GPU device - thread-safe
    pub device: Arc<wgpu::Device>,

    /// Shared GPU queue - thread-safe
    pub queue: Arc<wgpu::Queue>,

    /// Preferred surface format (sRGB)
    pub surface_format: wgpu::TextureFormat,
}

impl SharedGpuResources {
    /// Create shared GPU resources
    ///
    /// This performs the expensive GPU initialization once.
    /// Call this early at app startup, before creating any windows.
    pub async fn new() -> Result<Self> {
        // Create WGPU instance with all backends (Metal on macOS, Vulkan on Linux)
        let instance = wgpu::Instance::new(&wgpu::InstanceDescriptor {
            backends: wgpu::Backends::all(),
            ..Default::default()
        });

        // Request high-performance adapter without a specific surface
        // We'll verify surface compatibility per-window
        let adapter = instance
            .request_adapter(&wgpu::RequestAdapterOptions {
                power_preference: wgpu::PowerPreference::HighPerformance,
                compatible_surface: None, // Will check per-surface
                force_fallback_adapter: false,
            })
            .await
            .context("Failed to find suitable GPU adapter")?;

        let adapter_info = adapter.get_info();
        tracing::info!("Using GPU: {}", adapter_info.name);
        tracing::info!("Backend: {:?}", adapter_info.backend);

        // Request device with standard features
        let (device, queue) = adapter
            .request_device(&wgpu::DeviceDescriptor {
                label: Some("Palace Shared GPU Device"),
                required_features: wgpu::Features::empty(),
                required_limits: wgpu::Limits::default(),
                memory_hints: Default::default(),
                experimental_features: Default::default(),
                trace: Default::default(),
            })
            .await
            .context("Failed to create GPU device")?;

        // Prefer sRGB format - standard across macOS Metal and Linux Vulkan
        let surface_format = wgpu::TextureFormat::Bgra8UnormSrgb;

        Ok(Self {
            instance,
            adapter_info,
            device: Arc::new(device),
            queue: Arc::new(queue),
            surface_format,
        })
    }

    /// Create a new surface for a window
    ///
    /// # Safety
    /// The window must outlive the surface. In practice this is fine since
    /// we store them together in PalaceWindow.
    pub fn create_surface(&self, window: &winit::window::Window) -> Result<wgpu::Surface<'static>> {
        let surface = unsafe {
            self.instance.create_surface_unsafe(
                wgpu::SurfaceTargetUnsafe::from_window(window)
                    .context("Failed to create surface target")?,
            )
        }
        .context("Failed to create surface")?;

        Ok(surface)
    }

    /// Configure a surface for rendering
    pub fn configure_surface(
        &self,
        surface: &wgpu::Surface,
        width: u32,
        height: u32,
    ) -> wgpu::SurfaceConfiguration {
        let config = wgpu::SurfaceConfiguration {
            usage: wgpu::TextureUsages::RENDER_ATTACHMENT,
            format: self.surface_format,
            width,
            height,
            present_mode: wgpu::PresentMode::AutoVsync,
            alpha_mode: wgpu::CompositeAlphaMode::Auto,
            view_formats: vec![],
            desired_maximum_frame_latency: 2,
        };

        surface.configure(&self.device, &config);
        config
    }
}
