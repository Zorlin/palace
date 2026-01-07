#![allow(dead_code)]
use crate::projects::ProjectsConfig;
use crate::renderer::cards::{CardInstance, CardRenderer};
use crate::renderer::sprites::{SpriteInstance, SpriteRenderer, XboxButton};
use crate::renderer::text::{PreparedText, TextQueue};
use crate::state::{AppState, SuggestionCard, TaskStatus};
use anyhow::{Context, Result};
use glyphon::{
    Cache, FontSystem, Resolution, SwashCache, TextAtlas, TextRenderer, Viewport,
};
use winit::dpi::PhysicalSize;
use winit::window::Window;

// Embedded fonts - primary + fallback for symbols/emoji
const FONT_JETBRAINS: &[u8] = include_bytes!("fonts/JetBrainsMono-Regular.ttf");
const FONT_EMOJI: &[u8] = include_bytes!("fonts/NotoColorEmoji.ttf");

/// Simple fullscreen blit shader for copying screenshot texture to surface
const BLIT_SHADER: &str = r#"
struct VertexOutput {
    @builtin(position) position: vec4<f32>,
    @location(0) uv: vec2<f32>,
}

@vertex
fn vs_main(@builtin(vertex_index) vertex_index: u32) -> VertexOutput {
    // Fullscreen triangle
    var positions = array<vec2<f32>, 3>(
        vec2<f32>(-1.0, -1.0),
        vec2<f32>(3.0, -1.0),
        vec2<f32>(-1.0, 3.0)
    );
    var uvs = array<vec2<f32>, 3>(
        vec2<f32>(0.0, 1.0),
        vec2<f32>(2.0, 1.0),
        vec2<f32>(0.0, -1.0)
    );

    var out: VertexOutput;
    out.position = vec4<f32>(positions[vertex_index], 0.0, 1.0);
    out.uv = uvs[vertex_index];
    return out;
}

@group(0) @binding(0) var t_screenshot: texture_2d<f32>;
@group(0) @binding(1) var s_screenshot: sampler;

@fragment
fn fs_main(in: VertexOutput) -> @location(0) vec4<f32> {
    return textureSample(t_screenshot, s_screenshot, in.uv);
}
"#;

/// UI scaling configuration
#[derive(Clone, Copy, Debug)]
pub struct UiScale {
    pub base: f32,       // Base scale factor (1.0 = 100%)
    pub dpi_scale: f32,  // DPI-based scale
}

impl Default for UiScale {
    fn default() -> Self {
        Self {
            base: 1.0,
            dpi_scale: 1.0,
        }
    }
}

impl UiScale {
    pub fn effective(&self) -> f32 {
        self.base * self.dpi_scale
    }

    /// Scale a pixel value
    pub fn px(&self, value: f32) -> f32 {
        value * self.effective()
    }
}

/// Grid layout configuration for cards
struct CardGrid {
    card_width: f32,
    card_height: f32,
    gap: f32,
    columns: usize,
    margin_x: f32,
    margin_y: f32,
    /// Offset for virtual viewport (content area position)
    offset_x: f32,
    offset_y: f32,
}

impl CardGrid {
    fn new(screen_width: f32, _screen_height: f32, scale: &UiScale) -> Self {
        // Default grid for project chooser - original sizing
        let base_card_width = 320.0;
        let base_card_height = 180.0;
        let base_gap = 24.0;
        let base_margin = 60.0;

        let card_width = scale.px(base_card_width);
        let card_height = scale.px(base_card_height);
        let gap = scale.px(base_gap);
        let margin_x = scale.px(base_margin);
        let margin_y = scale.px(base_margin + 80.0); // Extra for title

        // Calculate columns based on available width
        let available_width = screen_width - margin_x * 2.0;
        let columns = ((available_width + gap) / (card_width + gap)).floor() as usize;
        let columns = columns.max(1);

        Self {
            card_width,
            card_height,
            gap,
            columns,
            margin_x,
            margin_y,
            offset_x: 0.0,
            offset_y: 0.0,
        }
    }

    /// Set offset for virtual viewport positioning
    fn with_offset(mut self, offset_x: f32, offset_y: f32) -> Self {
        self.offset_x = offset_x;
        self.offset_y = offset_y;
        self
    }

    /// Grid for PalaceLoop - dynamically calculates columns based on screen size
    ///
    /// Reference: At 1080p (1920px) with 1.5 scale on 5.5" display = 5 cards
    /// Cards scale proportionally to screen width and UI scale setting
    fn for_palace_loop(screen_width: f32, _screen_height: f32, scale: &UiScale) -> Self {
        let base_gap = 16.0;
        let base_margin = 40.0;

        let gap = scale.px(base_gap);
        let margin_x = scale.px(base_margin);
        let margin_y = scale.px(base_margin + 80.0); // Extra space for title + subtitle

        // Reference: 5 cards at 1920px with 1.5 scale = card width ~237 base
        // We want cards to feel consistent regardless of screen size
        // So we use a fixed base card width and calculate columns from that
        let base_card_width = 237.0;
        let card_width = scale.px(base_card_width);
        let card_height = card_width * 0.6; // Maintain aspect ratio

        // Calculate columns based on available width
        let available_width = screen_width - margin_x * 2.0;
        let columns = ((available_width + gap) / (card_width + gap)).floor() as usize;
        let columns = columns.max(1); // At least 1 column

        Self {
            card_width,
            card_height,
            gap,
            columns,
            margin_x,
            margin_y,
            offset_x: 0.0,
            offset_y: 0.0,
        }
    }

    fn card_position(&self, index: usize) -> (f32, f32) {
        let row = index / self.columns;
        let col = index % self.columns;

        let x = self.offset_x + self.margin_x + col as f32 * (self.card_width + self.gap);
        let y = self.offset_y + self.margin_y + row as f32 * (self.card_height + self.gap);

        (x, y)
    }
}

pub struct Renderer {
    surface: wgpu::Surface<'static>,
    device: std::sync::Arc<wgpu::Device>,
    queue: std::sync::Arc<wgpu::Queue>,
    config: wgpu::SurfaceConfiguration,
    size: PhysicalSize<u32>,
    // Glyphon text rendering
    font_system: FontSystem,
    swash_cache: SwashCache,
    text_cache: Cache,
    viewport: Viewport,
    text_atlas: TextAtlas,
    text_renderer: TextRenderer,
    text_queue: TextQueue,
    card_renderer: CardRenderer,
    sprite_renderer: SpriteRenderer,
    ui_scale: UiScale,
    gamepad_connected: bool,
    /// Texture for screenshot capture (created on-demand)
    screenshot_texture: Option<wgpu::Texture>,
    /// Bind group layout for blitting screenshot texture to surface
    blit_bind_group_layout: wgpu::BindGroupLayout,
    /// Pipeline for blitting screenshot texture to surface
    blit_pipeline: wgpu::RenderPipeline,
    /// Sampler for blit operations
    blit_sampler: wgpu::Sampler,
    /// Blur pipeline for modal background effect
    blur_pipeline: wgpu::RenderPipeline,
    /// Bind group layout for blur shader
    blur_bind_group_layout: wgpu::BindGroupLayout,
    /// Uniform buffer for blur direction/size
    blur_uniform_buffer: wgpu::Buffer,
    /// Scene texture for multi-pass rendering (base scene before blur)
    scene_texture: Option<wgpu::Texture>,
    /// Intermediate texture for blur ping-pong passes
    blur_texture: Option<wgpu::Texture>,
    /// Dark mode enabled (true = dark, false = light)
    dark_mode: bool,
    /// Whether UI scale is auto-detected (true) or user-set (false)
    is_auto_scale: bool,
    /// Layout mode based on display aspect ratio
    layout_mode: crate::state::LayoutMode,
    /// Animation start time for KITT scanner and other effects
    animation_start: std::time::Instant,
    /// Virtual viewport settings (for testing different aspect ratios)
    virtual_viewport: crate::VirtualViewport,
    /// Cached virtual viewport calculations (virtual_width, virtual_height, x_offset, y_offset)
    virtual_viewport_cache: (u32, u32, u32, u32),
    /// Monitor name this renderer is on (for display identification)
    monitor_name: String,
    /// Edit mode overlay cards (grid, handles, etc.)
    edit_mode_cards: Vec<CardInstance>,
}

impl Renderer {
    pub async fn new(window: &Window) -> Result<Self> {
        let size = window.inner_size();

        // Create WGPU instance with all backends
        let instance = wgpu::Instance::new(&wgpu::InstanceDescriptor {
            backends: wgpu::Backends::all(),
            ..Default::default()
        });

        // Create surface - we need 'static lifetime for the surface
        // SAFETY: The surface is created from a valid window handle
        let surface = unsafe {
            instance.create_surface_unsafe(
                wgpu::SurfaceTargetUnsafe::from_window(&window)
                    .context("Failed to create surface target")?,
            )
        }
        .context("Failed to create surface")?;

        // Request high-performance adapter
        let adapter = instance
            .request_adapter(&wgpu::RequestAdapterOptions {
                power_preference: wgpu::PowerPreference::HighPerformance,
                compatible_surface: Some(&surface),
                force_fallback_adapter: false,
            })
            .await
            .context("Failed to find suitable GPU adapter")?;

        tracing::info!("Using GPU: {}", adapter.get_info().name);
        tracing::info!("Backend: {:?}", adapter.get_info().backend);

        // Request device with features we need
        let (device, queue) = adapter
            .request_device(&wgpu::DeviceDescriptor {
                label: Some("Palace GPU Device"),
                required_features: wgpu::Features::empty(),
                required_limits: wgpu::Limits::default(),
                memory_hints: Default::default(),
                experimental_features: Default::default(),
                trace: Default::default(),
            })
            .await
            .context("Failed to create device")?;

        // Configure surface
        let surface_caps = surface.get_capabilities(&adapter);
        let surface_format = surface_caps
            .formats
            .iter()
            .find(|f| f.is_srgb())
            .copied()
            .unwrap_or(surface_caps.formats[0]);

        tracing::info!("Surface format: {:?}", surface_format);

        let config = wgpu::SurfaceConfiguration {
            usage: wgpu::TextureUsages::RENDER_ATTACHMENT,
            format: surface_format,
            width: size.width,
            height: size.height,
            present_mode: wgpu::PresentMode::AutoVsync,
            alpha_mode: surface_caps.alpha_modes[0],
            view_formats: vec![],
            desired_maximum_frame_latency: 2,
        };

        surface.configure(&device, &config);

        // Initialize glyphon text rendering with color emoji support
        let mut font_system = FontSystem::new();

        // Load fonts into the font system
        font_system.db_mut().load_font_data(FONT_JETBRAINS.to_vec());
        font_system.db_mut().load_font_data(FONT_EMOJI.to_vec());

        tracing::info!("Loaded {} fonts (with emoji support)", font_system.db().faces().count());

        let swash_cache = SwashCache::new();
        let text_cache = Cache::new(&device);
        let mut viewport = Viewport::new(&device, &text_cache);
        viewport.update(
            &queue,
            Resolution {
                width: size.width,
                height: size.height,
            },
        );

        // Use Accurate color mode for proper emoji rendering
        let mut text_atlas = TextAtlas::with_color_mode(
            &device,
            &queue,
            &text_cache,
            surface_format,
            glyphon::ColorMode::Accurate,
        );

        let text_renderer = TextRenderer::new(
            &mut text_atlas,
            &device,
            wgpu::MultisampleState::default(),
            None,
        );

        // Create card renderer
        let card_renderer = CardRenderer::new(&device, surface_format, size.width, size.height);

        // Create sprite renderer for Xbox button glyphs
        let sprite_renderer =
            SpriteRenderer::new(&device, &queue, surface_format, size.width, size.height);

        // Create blit pipeline for screenshot capture
        let blit_bind_group_layout =
            device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
                label: Some("Blit Bind Group Layout"),
                entries: &[
                    wgpu::BindGroupLayoutEntry {
                        binding: 0,
                        visibility: wgpu::ShaderStages::FRAGMENT,
                        ty: wgpu::BindingType::Texture {
                            sample_type: wgpu::TextureSampleType::Float { filterable: true },
                            view_dimension: wgpu::TextureViewDimension::D2,
                            multisampled: false,
                        },
                        count: None,
                    },
                    wgpu::BindGroupLayoutEntry {
                        binding: 1,
                        visibility: wgpu::ShaderStages::FRAGMENT,
                        ty: wgpu::BindingType::Sampler(wgpu::SamplerBindingType::Filtering),
                        count: None,
                    },
                ],
            });

        let blit_shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("Blit Shader"),
            source: wgpu::ShaderSource::Wgsl(BLIT_SHADER.into()),
        });

        let blit_pipeline_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("Blit Pipeline Layout"),
            bind_group_layouts: &[&blit_bind_group_layout],
            immediate_size: 0,
        });

        let blit_pipeline = device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
            label: Some("Blit Pipeline"),
            layout: Some(&blit_pipeline_layout),
            vertex: wgpu::VertexState {
                module: &blit_shader,
                entry_point: Some("vs_main"),
                buffers: &[],
                compilation_options: Default::default(),
            },
            fragment: Some(wgpu::FragmentState {
                module: &blit_shader,
                entry_point: Some("fs_main"),
                targets: &[Some(wgpu::ColorTargetState {
                    format: surface_format,
                    blend: None,
                    write_mask: wgpu::ColorWrites::ALL,
                })],
                compilation_options: Default::default(),
            }),
            primitive: wgpu::PrimitiveState {
                topology: wgpu::PrimitiveTopology::TriangleList,
                ..Default::default()
            },
            depth_stencil: None,
            multisample: wgpu::MultisampleState::default(),
            multiview_mask: None,
            cache: None,
        });

        let blit_sampler = device.create_sampler(&wgpu::SamplerDescriptor {
            label: Some("Blit Sampler"),
            mag_filter: wgpu::FilterMode::Linear,
            min_filter: wgpu::FilterMode::Linear,
            ..Default::default()
        });

        // Create blur pipeline for modal background effect
        let blur_bind_group_layout =
            device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
                label: Some("Blur Bind Group Layout"),
                entries: &[
                    wgpu::BindGroupLayoutEntry {
                        binding: 0,
                        visibility: wgpu::ShaderStages::FRAGMENT,
                        ty: wgpu::BindingType::Texture {
                            sample_type: wgpu::TextureSampleType::Float { filterable: true },
                            view_dimension: wgpu::TextureViewDimension::D2,
                            multisampled: false,
                        },
                        count: None,
                    },
                    wgpu::BindGroupLayoutEntry {
                        binding: 1,
                        visibility: wgpu::ShaderStages::FRAGMENT,
                        ty: wgpu::BindingType::Sampler(wgpu::SamplerBindingType::Filtering),
                        count: None,
                    },
                    wgpu::BindGroupLayoutEntry {
                        binding: 2,
                        visibility: wgpu::ShaderStages::FRAGMENT,
                        ty: wgpu::BindingType::Buffer {
                            ty: wgpu::BufferBindingType::Uniform,
                            has_dynamic_offset: false,
                            min_binding_size: None,
                        },
                        count: None,
                    },
                ],
            });

        let blur_shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("Blur Shader"),
            source: wgpu::ShaderSource::Wgsl(include_str!("shaders/blur.wgsl").into()),
        });

        let blur_pipeline_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("Blur Pipeline Layout"),
            bind_group_layouts: &[&blur_bind_group_layout],
            immediate_size: 0,
        });

        let blur_pipeline = device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
            label: Some("Blur Pipeline"),
            layout: Some(&blur_pipeline_layout),
            vertex: wgpu::VertexState {
                module: &blur_shader,
                entry_point: Some("vs_main"),
                buffers: &[],
                compilation_options: Default::default(),
            },
            fragment: Some(wgpu::FragmentState {
                module: &blur_shader,
                entry_point: Some("fs_main"),
                targets: &[Some(wgpu::ColorTargetState {
                    format: surface_format,
                    blend: None,
                    write_mask: wgpu::ColorWrites::ALL,
                })],
                compilation_options: Default::default(),
            }),
            primitive: wgpu::PrimitiveState {
                topology: wgpu::PrimitiveTopology::TriangleList,
                ..Default::default()
            },
            depth_stencil: None,
            multisample: wgpu::MultisampleState::default(),
            multiview_mask: None,
            cache: None,
        });

        // Uniform buffer for blur direction and texture size (2x vec2 = 16 bytes)
        let blur_uniform_buffer = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("Blur Uniform Buffer"),
            size: 16,
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        // Initialize with default scale - will be updated by App once display info is available
        let ui_scale = UiScale {
            base: 1.0,
            dpi_scale: 1.0,
        };

        Ok(Self {
            surface,
            device: std::sync::Arc::new(device),
            queue: std::sync::Arc::new(queue),
            config,
            size,
            font_system,
            swash_cache,
            text_cache,
            viewport,
            text_atlas,
            text_renderer,
            text_queue: TextQueue::new(),
            card_renderer,
            sprite_renderer,
            ui_scale,
            gamepad_connected: false,
            screenshot_texture: None,
            blit_bind_group_layout,
            blit_pipeline,
            blit_sampler,
            blur_pipeline,
            blur_bind_group_layout,
            blur_uniform_buffer,
            scene_texture: None,
            blur_texture: None,
            dark_mode: true,
            is_auto_scale: true,
            layout_mode: crate::state::LayoutMode::from_dimensions(size.width, size.height),
            animation_start: std::time::Instant::now(),
            virtual_viewport: crate::VirtualViewport::default(),
            virtual_viewport_cache: (size.width, size.height, 0, 0),
            monitor_name: String::new(),
            edit_mode_cards: Vec::new(),
        })
    }

    /// Create a new Renderer using shared GPU resources
    ///
    /// This allows multiple windows to share the same device and queue,
    /// which is more efficient than creating separate GPU contexts per window.
    pub fn new_with_shared(
        window: &Window,
        shared: &super::shared_gpu::SharedGpuResources,
    ) -> Result<Self> {
        let size = window.inner_size();

        // Create surface for this window using shared instance
        let surface = shared.create_surface(window)?;

        // Configure surface
        let config = shared.configure_surface(&surface, size.width, size.height);

        // Clone the Arc references to device and queue
        let device = shared.device.clone();
        let queue = shared.queue.clone();
        let surface_format = shared.surface_format;

        // Initialize glyphon text rendering with color emoji support
        let mut font_system = FontSystem::new();
        font_system.db_mut().load_font_data(FONT_JETBRAINS.to_vec());
        font_system.db_mut().load_font_data(FONT_EMOJI.to_vec());

        tracing::info!("Loaded {} fonts (with emoji support)", font_system.db().faces().count());

        let swash_cache = SwashCache::new();
        let text_cache = Cache::new(&device);
        let mut viewport = Viewport::new(&device, &text_cache);
        viewport.update(
            &queue,
            Resolution {
                width: size.width,
                height: size.height,
            },
        );

        let mut text_atlas = TextAtlas::with_color_mode(
            &device,
            &queue,
            &text_cache,
            surface_format,
            glyphon::ColorMode::Accurate,
        );

        let text_renderer = TextRenderer::new(
            &mut text_atlas,
            &device,
            wgpu::MultisampleState::default(),
            None,
        );

        // Create card renderer
        let card_renderer = CardRenderer::new(&device, surface_format, size.width, size.height);

        // Create sprite renderer for Xbox button glyphs
        let sprite_renderer =
            SpriteRenderer::new(&device, &queue, surface_format, size.width, size.height);

        // Create blit pipeline for screenshot capture
        let blit_bind_group_layout =
            device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
                label: Some("Blit Bind Group Layout"),
                entries: &[
                    wgpu::BindGroupLayoutEntry {
                        binding: 0,
                        visibility: wgpu::ShaderStages::FRAGMENT,
                        ty: wgpu::BindingType::Texture {
                            sample_type: wgpu::TextureSampleType::Float { filterable: true },
                            view_dimension: wgpu::TextureViewDimension::D2,
                            multisampled: false,
                        },
                        count: None,
                    },
                    wgpu::BindGroupLayoutEntry {
                        binding: 1,
                        visibility: wgpu::ShaderStages::FRAGMENT,
                        ty: wgpu::BindingType::Sampler(wgpu::SamplerBindingType::Filtering),
                        count: None,
                    },
                ],
            });

        let blit_shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("Blit Shader"),
            source: wgpu::ShaderSource::Wgsl(BLIT_SHADER.into()),
        });

        let blit_pipeline_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("Blit Pipeline Layout"),
            bind_group_layouts: &[&blit_bind_group_layout],
            immediate_size: 0,
        });

        let blit_pipeline = device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
            label: Some("Blit Pipeline"),
            layout: Some(&blit_pipeline_layout),
            vertex: wgpu::VertexState {
                module: &blit_shader,
                entry_point: Some("vs_main"),
                buffers: &[],
                compilation_options: Default::default(),
            },
            fragment: Some(wgpu::FragmentState {
                module: &blit_shader,
                entry_point: Some("fs_main"),
                targets: &[Some(wgpu::ColorTargetState {
                    format: surface_format,
                    blend: None,
                    write_mask: wgpu::ColorWrites::ALL,
                })],
                compilation_options: Default::default(),
            }),
            primitive: wgpu::PrimitiveState {
                topology: wgpu::PrimitiveTopology::TriangleList,
                ..Default::default()
            },
            depth_stencil: None,
            multisample: wgpu::MultisampleState::default(),
            multiview_mask: None,
            cache: None,
        });

        let blit_sampler = device.create_sampler(&wgpu::SamplerDescriptor {
            label: Some("Blit Sampler"),
            mag_filter: wgpu::FilterMode::Linear,
            min_filter: wgpu::FilterMode::Linear,
            ..Default::default()
        });

        // Create blur pipeline for modal background effect
        let blur_bind_group_layout =
            device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
                label: Some("Blur Bind Group Layout"),
                entries: &[
                    wgpu::BindGroupLayoutEntry {
                        binding: 0,
                        visibility: wgpu::ShaderStages::FRAGMENT,
                        ty: wgpu::BindingType::Texture {
                            sample_type: wgpu::TextureSampleType::Float { filterable: true },
                            view_dimension: wgpu::TextureViewDimension::D2,
                            multisampled: false,
                        },
                        count: None,
                    },
                    wgpu::BindGroupLayoutEntry {
                        binding: 1,
                        visibility: wgpu::ShaderStages::FRAGMENT,
                        ty: wgpu::BindingType::Sampler(wgpu::SamplerBindingType::Filtering),
                        count: None,
                    },
                    wgpu::BindGroupLayoutEntry {
                        binding: 2,
                        visibility: wgpu::ShaderStages::FRAGMENT,
                        ty: wgpu::BindingType::Buffer {
                            ty: wgpu::BufferBindingType::Uniform,
                            has_dynamic_offset: false,
                            min_binding_size: None,
                        },
                        count: None,
                    },
                ],
            });

        let blur_shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("Blur Shader"),
            source: wgpu::ShaderSource::Wgsl(include_str!("shaders/blur.wgsl").into()),
        });

        let blur_pipeline_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("Blur Pipeline Layout"),
            bind_group_layouts: &[&blur_bind_group_layout],
            immediate_size: 0,
        });

        let blur_pipeline = device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
            label: Some("Blur Pipeline"),
            layout: Some(&blur_pipeline_layout),
            vertex: wgpu::VertexState {
                module: &blur_shader,
                entry_point: Some("vs_main"),
                buffers: &[],
                compilation_options: Default::default(),
            },
            fragment: Some(wgpu::FragmentState {
                module: &blur_shader,
                entry_point: Some("fs_main"),
                targets: &[Some(wgpu::ColorTargetState {
                    format: surface_format,
                    blend: None,
                    write_mask: wgpu::ColorWrites::ALL,
                })],
                compilation_options: Default::default(),
            }),
            primitive: wgpu::PrimitiveState {
                topology: wgpu::PrimitiveTopology::TriangleList,
                ..Default::default()
            },
            depth_stencil: None,
            multisample: wgpu::MultisampleState::default(),
            multiview_mask: None,
            cache: None,
        });

        // Uniform buffer for blur direction and texture size
        let blur_uniform_buffer = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("Blur Uniform Buffer"),
            size: 16,
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        let ui_scale = UiScale {
            base: 1.0,
            dpi_scale: 1.0,
        };

        Ok(Self {
            surface,
            device,
            queue,
            config,
            size,
            font_system,
            swash_cache,
            text_cache,
            viewport,
            text_atlas,
            text_renderer,
            text_queue: TextQueue::new(),
            card_renderer,
            sprite_renderer,
            ui_scale,
            gamepad_connected: false,
            screenshot_texture: None,
            blit_bind_group_layout,
            blit_pipeline,
            blit_sampler,
            blur_pipeline,
            blur_bind_group_layout,
            blur_uniform_buffer,
            scene_texture: None,
            blur_texture: None,
            dark_mode: true,
            is_auto_scale: true,
            layout_mode: crate::state::LayoutMode::from_dimensions(size.width, size.height),
            animation_start: std::time::Instant::now(),
            virtual_viewport: crate::VirtualViewport::default(),
            virtual_viewport_cache: (size.width, size.height, 0, 0),
            monitor_name: String::new(),
            edit_mode_cards: Vec::new(),
        })
    }

    /// Set the monitor name for this renderer (used for display identification)
    pub fn set_monitor_name(&mut self, name: String) {
        self.monitor_name = name;
    }

    /// Set the virtual viewport settings and recalculate cache
    pub fn set_virtual_viewport(&mut self, viewport: crate::VirtualViewport) {
        self.virtual_viewport = viewport;
        self.update_virtual_viewport_cache();
    }

    /// Update the virtual viewport cache based on current native size
    fn update_virtual_viewport_cache(&mut self) {
        self.virtual_viewport_cache = self.virtual_viewport.calculate(self.size.width, self.size.height);
    }

    /// Get the effective rendering size (virtual or native)
    pub fn effective_size(&self) -> PhysicalSize<u32> {
        if self.virtual_viewport.is_active() {
            let (w, h, _, _) = self.virtual_viewport_cache;
            PhysicalSize::new(w, h)
        } else {
            self.size
        }
    }

    /// Get the virtual viewport offset (x, y)
    pub fn virtual_offset(&self) -> (u32, u32) {
        let (_, _, x, y) = self.virtual_viewport_cache;
        (x, y)
    }

    pub fn set_dark_mode(&mut self, dark_mode: bool) {
        self.dark_mode = dark_mode;
    }

    pub fn dark_mode(&self) -> bool {
        self.dark_mode
    }

    /// Set UI scale factor (e.g., 1.5 for 150% scaling)
    /// is_auto: true if auto-detected, false if user-set
    pub fn set_ui_scale(&mut self, scale: f32, is_auto: bool) {
        self.ui_scale.dpi_scale = scale;
        self.is_auto_scale = is_auto;
        tracing::info!("UI scale set to {} ({})", scale, if is_auto { "auto" } else { "manual" });
    }

    /// Get current UI scale factor
    pub fn ui_scale(&self) -> f32 {
        self.ui_scale.dpi_scale
    }

    /// Get animation time in seconds (for KITT scanner and other effects)
    fn animation_time(&self) -> f32 {
        self.animation_start.elapsed().as_secs_f32()
    }

    /// Calculate max scroll for text content
    /// Returns 0.0 if content fits within visible height
    pub fn calculate_max_scroll(&mut self, text: &str, font_size: f32, width: f32, visible_height: f32) -> f32 {
        let (_, content_height, _) = crate::renderer::text::measure_text(
            &mut self.font_system,
            text,
            font_size,
            width,
        );
        (content_height - visible_height).max(0.0)
    }

    /// Calculate max scroll for a suggestion card's description
    /// Uses the same layout parameters as rendering
    pub fn calculate_card_max_scroll(&mut self, card: &crate::state::SuggestionCard) -> f32 {
        let (content_w, content_h) = self.content_size();
        let grid = CardGrid::for_palace_loop(
            content_w,
            content_h,
            &self.ui_scale,
        );

        let text_margin = self.ui_scale.px(12.0);
        let desc_scale = self.ui_scale.px(13.0);
        let desc_width = grid.card_width - text_margin * 2.0;
        let visible_height = grid.card_height - text_margin * 2.0;

        // Build same text as render: description + command
        // Fall back to title if description is empty or whitespace-only
        let description = if card.description.trim().is_empty() {
            if card.streaming { "Loading..." } else { &card.title }
        } else {
            &card.description
        };

        let full_markdown = if let Some(ref cmd) = card.command {
            if cmd.trim().is_empty() {
                // No command to display
                description.to_string()
            } else {
                let display_cmd = if cmd.len() > 40 {
                    let truncated: String = cmd.chars().take(37).collect();
                    format!("$ {}...", truncated)
                } else {
                    format!("$ {}", cmd)
                };
                format!("{}\n\n{}", description, display_cmd)
            }
        } else {
            description.to_string()
        };

        // Parse markdown and measure the resulting plain text
        // This matches what the renderer does when it parses and displays
        let spans = crate::renderer::text::parse_markdown(&full_markdown);
        let plain_text = crate::renderer::text::spans_to_plain_text(&spans);

        self.calculate_max_scroll(&plain_text, desc_scale, desc_width, visible_height)
    }

    pub fn set_gamepad_connected(&mut self, connected: bool) {
        self.gamepad_connected = connected;
    }

    /// Get background color based on dark mode
    fn background_color(&self) -> wgpu::Color {
        if self.dark_mode {
            wgpu::Color::BLACK
        } else {
            wgpu::Color { r: 0.95, g: 0.95, b: 0.96, a: 1.0 } // Light gray-blue
        }
    }

    /// Get the content area dimensions - this is what all layout calculations should use
    /// Returns (width, height) of the area where content should be rendered
    fn content_size(&self) -> (f32, f32) {
        let (vw, vh, _, _) = self.virtual_viewport_cache;
        (vw as f32, vh as f32)
    }

    /// Get the content area offset - this is added to all positions for virtual viewport
    /// Returns (x_offset, y_offset) where content rendering should start
    fn content_offset(&self) -> (f32, f32) {
        let (_, _, x, y) = self.virtual_viewport_cache;
        (x as f32, y as f32)
    }

    /// Offset a position from content space to screen space
    /// All layout calculations use content_size(), then this offsets for virtual viewport
    #[inline]
    fn offset_pos(&self, x: f32, y: f32) -> (f32, f32) {
        let (ox, oy) = self.content_offset();
        (x + ox, y + oy)
    }

    /// Create a CardInstance with position offset applied for virtual viewport
    #[inline]
    fn card_at(&self, x: f32, y: f32, width: f32, height: f32, color: [f32; 4]) -> CardInstance {
        let (ox, oy) = self.content_offset();
        CardInstance::new(x + ox, y + oy, width, height, color)
    }

    /// Queue text with position offset applied for virtual viewport
    #[inline]
    fn text_at(&mut self, text: impl Into<String>, x: f32, y: f32, scale: f32, color: [f32; 4]) {
        let (ox, oy) = self.content_offset();
        self.text_queue.push(text, x + ox, y + oy, scale, color);
    }

    /// Queue bounded text with position offset applied for virtual viewport
    #[inline]
    fn text_bounded_at(
        &mut self,
        text: impl Into<String>,
        x: f32,
        y: f32,
        scale: f32,
        color: [f32; 4],
        bounds_width: f32,
        bounds_height: f32,
    ) {
        let (ox, oy) = self.content_offset();
        self.text_queue.push_bounded(text, x + ox, y + oy, scale, color, bounds_width, bounds_height);
    }

    /// Queue bounded text with scroll and position offset applied for virtual viewport
    #[inline]
    fn text_bounded_scroll_at(
        &mut self,
        text: impl Into<String>,
        x: f32,
        y: f32,
        scale: f32,
        color: [f32; 4],
        bounds_width: f32,
        bounds_height: f32,
        scroll_offset: f32,
    ) {
        let (ox, oy) = self.content_offset();
        self.text_queue.push_bounded_scroll(text, x + ox, y + oy, scale, color, bounds_width, bounds_height, scroll_offset);
    }

    /// Queue markdown text with position offset applied for virtual viewport
    #[inline]
    fn markdown_bounded_at(
        &mut self,
        markdown: &str,
        x: f32,
        y: f32,
        scale: f32,
        color: [f32; 4],
        bounds_width: f32,
        bounds_height: f32,
    ) {
        let (ox, oy) = self.content_offset();
        self.text_queue.push_markdown_bounded(markdown, x + ox, y + oy, scale, color, bounds_width, bounds_height);
    }

    /// Queue markdown text with scroll and position offset applied for virtual viewport
    #[inline]
    fn markdown_scroll_at(
        &mut self,
        markdown: &str,
        x: f32,
        y: f32,
        scale: f32,
        color: [f32; 4],
        bounds_width: f32,
        bounds_height: f32,
        scroll_offset: f32,
    ) {
        let (ox, oy) = self.content_offset();
        self.text_queue.push_markdown_scroll(markdown, x + ox, y + oy, scale, color, bounds_width, bounds_height, scroll_offset);
    }

    /// Build letterbox/pillarbox cards for virtual viewport
    /// Returns filled black cards that cover the areas outside the virtual viewport
    fn build_letterbox_cards(&self) -> Vec<CardInstance> {
        if !self.virtual_viewport.is_active() {
            return Vec::new();
        }

        let (content_w, content_h) = self.content_size();
        let (offset_x, offset_y) = self.content_offset();
        let native_w = self.size.width as f32;
        let native_h = self.size.height as f32;

        let mut cards = Vec::new();
        let black = [0.0, 0.0, 0.0, 1.0];

        // Pillarbox (left and right bars) - for narrower virtual viewport
        if offset_x > 0.5 {
            // Left bar
            cards.push(
                CardInstance::new(0.0, 0.0, offset_x, native_h, black)
                    .with_border_width(0.0)
                    .with_corner_radius(0.0)
                    .filled()
            );
            // Right bar
            let right_x = offset_x + content_w;
            let right_w = native_w - right_x;
            if right_w > 0.5 {
                cards.push(
                    CardInstance::new(right_x, 0.0, right_w, native_h, black)
                        .with_border_width(0.0)
                        .with_corner_radius(0.0)
                        .filled()
                );
            }
        }

        // Letterbox (top and bottom bars) - for shorter virtual viewport
        if offset_y > 0.5 {
            // Top bar
            cards.push(
                CardInstance::new(0.0, 0.0, native_w, offset_y, black)
                    .with_border_width(0.0)
                    .with_corner_radius(0.0)
                    .filled()
            );
            // Bottom bar
            let bottom_y = offset_y + content_h;
            let bottom_h = native_h - bottom_y;
            if bottom_h > 0.5 {
                cards.push(
                    CardInstance::new(0.0, bottom_y, native_w, bottom_h, black)
                        .with_border_width(0.0)
                        .with_corner_radius(0.0)
                        .filled()
                );
            }
        }

        cards
    }

    /// Get primary text color based on dark mode
    fn text_color(&self) -> [f32; 4] {
        if self.dark_mode {
            [1.0, 1.0, 1.0, 1.0] // White
        } else {
            [0.1, 0.1, 0.12, 1.0] // Near black
        }
    }

    /// Get secondary/dimmed text color based on dark mode
    fn text_color_dim(&self) -> [f32; 4] {
        if self.dark_mode {
            [0.6, 0.6, 0.65, 1.0] // Gray
        } else {
            [0.4, 0.4, 0.45, 1.0] // Darker gray
        }
    }

    /// Format a number with thousand separators (e.g., 1234567 -> "1,234,567")
    fn format_number(n: u64) -> String {
        let s = n.to_string();
        let chars: Vec<char> = s.chars().collect();
        let mut result = String::with_capacity(s.len() + s.len() / 3);
        for (i, c) in chars.iter().enumerate() {
            if i > 0 && (chars.len() - i) % 3 == 0 {
                result.push(',');
            }
            result.push(*c);
        }
        result
    }

    /// Get temporal rainbow color based on timestamp in log entry
    /// Entries with similar timestamps get similar hues - helps visually match left/right columns
    fn temporal_rainbow_color(&self, entry: &str, alpha: f32) -> [f32; 4] {
        // Try to parse timestamp [HH:MM:SS] at start
        if entry.starts_with('[') && entry.len() > 10 && entry.chars().nth(9) == Some(']') {
            // Extract seconds from timestamp for hue cycling
            // Format: [HH:MM:SS]
            if let Ok(seconds) = entry[7..9].parse::<u32>() {
                // Map seconds (0-59) to hue (0-360), cycle every minute
                let hue = (seconds as f32 / 60.0) * 360.0;
                let (r, g, b) = hsl_to_rgb(hue, 0.4, 0.65); // Soft saturation, moderate lightness
                return [r, g, b, alpha];
            }
        }
        // Default to blue-ish gray if no timestamp
        [0.5, 0.6, 0.8, alpha]
    }

    /// Get border color based on dark mode
    fn border_color(&self) -> [f32; 4] {
        if self.dark_mode {
            [0.35, 0.35, 0.4, 1.0] // Light gray
        } else {
            [0.3, 0.3, 0.35, 1.0] // Darker gray for light mode
        }
    }

    /// Get selected/accent border color based on dark mode
    fn accent_color(&self) -> [f32; 4] {
        if self.dark_mode {
            [0.4, 0.6, 1.0, 1.0] // Blue
        } else {
            [0.2, 0.4, 0.9, 1.0] // Darker blue
        }
    }

    /// Word wrap text to fit within a given width
    fn wrap_text(&self, text: &str, max_width: f32, font_size: f32) -> Vec<String> {
        let approx_char_width = font_size * 0.5;
        let max_chars = (max_width / approx_char_width) as usize;
        if max_chars == 0 {
            return vec![text.to_string()];
        }

        let mut lines = Vec::new();
        for paragraph in text.split('\n') {
            let words: Vec<&str> = paragraph.split_whitespace().collect();
            if words.is_empty() {
                lines.push(String::new());
                continue;
            }

            let mut current_line = String::new();
            let mut current_len = 0usize;
            for word in words {
                let word_len = word.chars().count();
                if current_line.is_empty() {
                    if word_len > max_chars {
                        // Word too long, split it by chars
                        let mut chars = word.chars().peekable();
                        while chars.peek().is_some() {
                            let chunk: String = chars.by_ref().take(max_chars).collect();
                            if chars.peek().is_some() {
                                lines.push(chunk);
                            } else {
                                current_line = chunk;
                                current_len = current_line.chars().count();
                            }
                        }
                    } else {
                        current_line = word.to_string();
                        current_len = word_len;
                    }
                } else if current_len + 1 + word_len <= max_chars {
                    current_line.push(' ');
                    current_line.push_str(word);
                    current_len += 1 + word_len;
                } else {
                    lines.push(current_line);
                    current_line = word.to_string();
                    current_len = word_len;
                }
            }
            if !current_line.is_empty() {
                lines.push(current_line);
            }
        }

        if lines.is_empty() {
            lines.push(String::new());
        }
        lines
    }

    pub fn resize(&mut self, new_size: PhysicalSize<u32>) {
        if new_size.width > 0 && new_size.height > 0 {
            self.size = new_size;
            self.config.width = new_size.width;
            self.config.height = new_size.height;
            self.surface.configure(&self.device, &self.config);
            self.viewport.update(
                &self.queue,
                Resolution {
                    width: new_size.width,
                    height: new_size.height,
                },
            );
            self.card_renderer
                .resize(&self.queue, new_size.width, new_size.height);
            self.sprite_renderer
                .resize(&self.queue, new_size.width, new_size.height);
            // Clear textures so they get recreated at new size
            self.screenshot_texture = None;
            self.scene_texture = None;
            self.blur_texture = None;
            // Update virtual viewport cache
            self.update_virtual_viewport_cache();
            // Update layout mode based on virtual dimensions (or native if no virtual viewport)
            let effective = self.effective_size();
            self.layout_mode = crate::state::LayoutMode::from_dimensions(effective.width, effective.height);
            tracing::debug!(
                "Resized to {}x{} (virtual: {}x{}, offset: {:?}, layout: {:?})",
                new_size.width, new_size.height,
                effective.width, effective.height,
                self.virtual_offset(),
                self.layout_mode
            );
        }
    }

    /// Take queued text, prepare buffers, and upload to GPU
    /// Returns PreparedText that must be kept alive until after render
    fn prepare_text(&mut self) -> PreparedText {
        let requests = self.text_queue.take();
        let markdown_requests = self.text_queue.take_markdown();
        let screen_width = self.size.width as f32;
        let screen_height = self.size.height as f32;

        // Update viewport for current screen size
        self.viewport.update(
            &self.queue,
            Resolution {
                width: self.size.width,
                height: self.size.height,
            },
        );

        let prepared = PreparedText::prepare(requests, markdown_requests, &mut self.font_system, screen_width);

        if let Err(e) = prepared.upload(
            &self.device,
            &self.queue,
            &mut self.font_system,
            &mut self.swash_cache,
            &mut self.text_atlas,
            &self.viewport,
            &mut self.text_renderer,
            screen_width,
            screen_height,
        ) {
            tracing::error!("Failed to prepare text: {:?}", e);
        }

        prepared
    }

    /// Render text in the current render pass
    /// PreparedText must be the result of the most recent prepare_text() call
    fn render_text<'a>(&'a self, render_pass: &mut wgpu::RenderPass<'a>) {
        if let Err(e) = self.text_renderer.render(&self.text_atlas, &self.viewport, render_pass) {
            tracing::error!("Failed to render text: {:?}", e);
        }
    }

    pub fn render(
        &mut self,
        state: &AppState,
        projects: &ProjectsConfig,
    ) -> Result<(), wgpu::SurfaceError> {
        let empty_reflow = std::collections::HashMap::new();
        self.render_with_screenshot(state, projects, None, &mut crate::debug::ScreenshotCapture::new(), false, None, None, &empty_reflow, None)
    }

    pub fn render_with_screenshot(
        &mut self,
        state: &AppState,
        projects: &ProjectsConfig,
        screenshot_path: Option<&std::path::PathBuf>,
        screenshot_capture: &mut crate::debug::ScreenshotCapture,
        edit_mode_active: bool,
        selected_panel: Option<&str>,
        resize_preview: Option<(&str, f32, f32)>, // (panel_id, delta_x, delta_y)
        reflow_positions: &std::collections::HashMap<&'static str, (f32, f32, f32, f32)>, // Panels displaced by reflow
        panel_chooser: Option<((f32, f32), usize)>, // (spawn_target, selection) if chooser is open
    ) -> Result<(), wgpu::SurfaceError> {
        // Clear edit mode cards from previous frame
        self.edit_mode_cards.clear();

        let output = self.surface.get_current_texture()?;
        let surface_view = output
            .texture
            .create_view(&wgpu::TextureViewDescriptor::default());

        // Determine if we're rendering a modal menu
        enum ModalType {
            MainMenu(usize),
            Settings(usize),
            UiScale { selected: usize, user_scale_override: Option<f32> },
            Permission { selected: usize, command: String, prefix: String },
            Execute(usize),
            Survey {
                question: String,
                header: String,
                options: Vec<crate::state::SurveyOption>,
                focused: usize,
                custom_input: String,
                custom_active: bool,
                multi_select: bool,
                selected_indices: Vec<usize>,
                use_quick_select: bool,
                scroll_offset: usize,
            },
            MultiDisplay {
                focus_index: usize,
                options: Vec<crate::state::DisplayOption>,
                remember_choice: bool,
                focused_row: usize,
                primary_pill_drag: Option<(usize, f32, f32)>,
            },
            AddCard(usize),
            CustomTask {
                name: String,
                description: String,
                active_field: usize,
                cursor: usize,
            },
            ProjectContext {
                project_index: usize,
                selected: usize,
                project_name: String,
                is_archived: bool,
            },
            LanguageSelect {
                project_index: usize,
                selected: usize,
                languages: Vec<String>,
            },
        }

        let (base_state, modal_type) = match state {
            AppState::MainMenu { previous_state, selected_item } => {
                (previous_state.as_ref(), Some(ModalType::MainMenu(*selected_item)))
            }
            AppState::SettingsMenu { previous_state, selected_item } => {
                // Get the base state from MainMenu's previous_state
                let actual_base = match previous_state.as_ref() {
                    AppState::MainMenu { previous_state, .. } => previous_state.as_ref(),
                    other => other,
                };
                (actual_base, Some(ModalType::Settings(*selected_item)))
            }
            AppState::UiScaleMenu { previous_state, selected_item, user_scale_override } => {
                // Navigate back to find the actual base state
                let actual_base = match previous_state.as_ref() {
                    AppState::SettingsMenu { previous_state, .. } => {
                        match previous_state.as_ref() {
                            AppState::MainMenu { previous_state, .. } => previous_state.as_ref(),
                            other => other,
                        }
                    }
                    other => other,
                };
                (actual_base, Some(ModalType::UiScale { selected: *selected_item, user_scale_override: *user_scale_override }))
            }
            AppState::PermissionModal { previous_state, selected_choice, command, command_prefix, .. } => {
                (previous_state.as_ref(), Some(ModalType::Permission {
                    selected: *selected_choice,
                    command: command.clone(),
                    prefix: command_prefix.clone(),
                }))
            }
            AppState::ExecuteModal { previous_state, selected_option } => {
                (previous_state.as_ref(), Some(ModalType::Execute(*selected_option)))
            }
            AppState::Survey { previous_state, question, header, options, focused_index, custom_input, custom_active, multi_select, selected_indices, use_quick_select, scroll_offset, .. } => {
                (previous_state.as_ref(), Some(ModalType::Survey {
                    question: question.clone(),
                    header: header.clone(),
                    options: options.clone(),
                    focused: *focused_index,
                    custom_input: custom_input.clone(),
                    custom_active: *custom_active,
                    multi_select: *multi_select,
                    selected_indices: selected_indices.clone(),
                    use_quick_select: *use_quick_select,
                    scroll_offset: *scroll_offset,
                }))
            }
            AppState::MultiDisplayDialog { previous_state, focus_index, options, remember_choice, focused_row, primary_pill_drag } => {
                (previous_state.as_ref(), Some(ModalType::MultiDisplay {
                    focus_index: *focus_index,
                    options: options.clone(),
                    remember_choice: *remember_choice,
                    focused_row: *focused_row,
                    primary_pill_drag: *primary_pill_drag,
                }))
            }
            AppState::AddCardMenu { previous_state, selected_option } => {
                (previous_state.as_ref(), Some(ModalType::AddCard(*selected_option)))
            }
            AppState::CustomTaskInput { previous_state, name, description, active_field, cursor } => {
                // Navigate through the AddCardMenu to find PalaceLoop
                let actual_base = match previous_state.as_ref() {
                    AppState::AddCardMenu { previous_state, .. } => previous_state.as_ref(),
                    other => other,
                };
                (actual_base, Some(ModalType::CustomTask {
                    name: name.clone(),
                    description: description.clone(),
                    active_field: *active_field,
                    cursor: *cursor
                }))
            }
            AppState::ProjectContextMenu { project_index, selected_option, previous_state } => {
                let project_name = projects.projects.get(*project_index)
                    .map(|p| p.name.clone())
                    .unwrap_or_else(|| "Unknown".to_string());
                let is_archived = projects.projects.get(*project_index)
                    .map(|p| p.archived)
                    .unwrap_or(false);
                (previous_state.as_ref(), Some(ModalType::ProjectContext {
                    project_index: *project_index,
                    selected: *selected_option,
                    project_name,
                    is_archived,
                }))
            }
            AppState::LanguageSelector { project_index, selected_index, languages, previous_state } => {
                (previous_state.as_ref(), Some(ModalType::LanguageSelect {
                    project_index: *project_index,
                    selected: *selected_index,
                    languages: languages.clone(),
                }))
            }
            _ => (state, None),
        };

        // Build cards for base state
        let base_cards = match base_state {
            AppState::ProjectChooser { selected_index, .. } => {
                self.build_project_cards(projects, *selected_index)
            }
            AppState::ProjectView { selected_action, .. } => {
                // Look up action_menu bounds from reflow_positions for real-time resize
                let menu_bounds = reflow_positions.get("action_menu").copied();
                self.build_action_cards(*selected_action, menu_bounds)
            }
            AppState::PalaceLoop { cards, focused_index, hovered_index, card_scroll_offset, generating, .. } => {
                // Show "+" card once we have cards (even while generating more) or when generation is done
                let show_add_card = !cards.is_empty() || !generating;
                self.build_suggestion_cards(cards, *focused_index, *hovered_index, None, *card_scroll_offset, show_add_card)
            }
            AppState::Executing { quest_log_visible: true, all_cards, quest_log_focus, executing_cards, task_statuses, .. } => {
                // Show all cards when quest log is visible, with status badges (no scroll, no + card in quest log)
                let exec_ids: Vec<usize> = executing_cards.iter().map(|c| c.id).collect();
                self.build_suggestion_cards(all_cards, *quest_log_focus, None, Some((&exec_ids, task_statuses)), 0.0, false)
            }
            AppState::MainMenu { .. } | AppState::SettingsMenu { .. } | AppState::UiScaleMenu { .. } | AppState::PermissionModal { .. } | AppState::ExecuteModal { .. } | AppState::AddCardMenu { .. } | AppState::CustomTaskInput { .. } | AppState::Survey { .. } | AppState::Executing { .. } | AppState::MultiDisplayDialog { .. } | AppState::ProjectContextMenu { .. } | AppState::LanguageSelector { .. } | AppState::NewMonitorDialog { .. } => Vec::new(),
            AppState::ScenarioDiffViewer { viewer, feedback_mode, feedback_options, selected_feedback, custom_feedback, custom_cursor, .. } => {
                self.build_diff_viewer_cards(viewer, *feedback_mode, feedback_options, *selected_feedback, custom_feedback.as_deref(), *custom_cursor)
            }
            AppState::ScenarioGenerator { generator, .. } => {
                self.build_generator_cards(generator)
            }
            AppState::Recording { inner_state, .. } => {
                // Delegate to inner state for card rendering
                match inner_state.as_ref() {
                    AppState::PalaceLoop { cards, focused_index, hovered_index, card_scroll_offset, generating, .. } => {
                        let show_add_card = !cards.is_empty() || !generating;
                        self.build_suggestion_cards(cards, *focused_index, *hovered_index, None, *card_scroll_offset, show_add_card)
                    }
                    _ => Vec::new(),
                }
            }
        };

        // When modal is open, render base scene then overlay + modal
        if let Some(ref modal) = modal_type {
            // For screenshots, render to intermediate texture
            let render_to_screenshot = screenshot_path.is_some();
            if render_to_screenshot && self.screenshot_texture.is_none() {
                self.screenshot_texture = Some(self.device.create_texture(&wgpu::TextureDescriptor {
                    label: Some("Screenshot Texture"),
                    size: wgpu::Extent3d {
                        width: self.size.width,
                        height: self.size.height,
                        depth_or_array_layers: 1,
                    },
                    mip_level_count: 1,
                    sample_count: 1,
                    dimension: wgpu::TextureDimension::D2,
                    format: self.config.format,
                    usage: wgpu::TextureUsages::RENDER_ATTACHMENT
                        | wgpu::TextureUsages::COPY_SRC
                        | wgpu::TextureUsages::TEXTURE_BINDING,
                    view_formats: &[],
                }));
            }

            let target_view = if render_to_screenshot {
                self.screenshot_texture.as_ref().unwrap()
                    .create_view(&wgpu::TextureViewDescriptor::default())
            } else {
                surface_view.clone()
            };

            // Queue base state text
            match base_state {
                AppState::ProjectChooser { selected_index, .. } => {
                    self.queue_project_chooser_text(projects, *selected_index);
                }
                AppState::ProjectView { project_path, selected_action } => {
                    let context_bounds = reflow_positions.get("context_widget").copied();
                    let menu_bounds = reflow_positions.get("action_menu").copied();
                    self.queue_project_view_text(project_path, *selected_action, context_bounds, menu_bounds);
                }
                AppState::PalaceLoop { cards, current_tool, tool_log, thought_log, log_scroll_offset, focused_index, hovered_index, detail_scroll_offset, card_scroll_offset, generating, .. } => {
                    let show_add_card = !cards.is_empty() || !generating;
                    self.queue_palace_loop_text(cards, current_tool.as_deref(), tool_log, thought_log, *log_scroll_offset, *focused_index, *hovered_index, *detail_scroll_offset, None, *card_scroll_offset, show_add_card);
                }
                AppState::Executing { tool_log, thought_log, log_scroll_offset, status, executor, quest_log_visible, all_cards, quest_log_focus, executing_cards, task_statuses, tokens_used, request_active, .. } => {
                    if *quest_log_visible {
                        let exec_ids: Vec<usize> = executing_cards.iter().map(|c| c.id).collect();
                        self.queue_palace_loop_text(all_cards, None, &[], &[], 0, *quest_log_focus, None, 0.0, Some((&exec_ids, task_statuses)), 0.0, false);
                    } else {
                        self.queue_executing_text(tool_log, thought_log, *log_scroll_offset, status, *executor, *tokens_used, *request_active);
                    }
                }
                AppState::MainMenu { .. } | AppState::SettingsMenu { .. } | AppState::UiScaleMenu { .. } | AppState::PermissionModal { .. } | AppState::ExecuteModal { .. } | AppState::AddCardMenu { .. } | AppState::CustomTaskInput { .. } | AppState::Survey { .. } | AppState::MultiDisplayDialog { .. } | AppState::ProjectContextMenu { .. } | AppState::LanguageSelector { .. } | AppState::NewMonitorDialog { .. } | AppState::ScenarioDiffViewer { .. } | AppState::ScenarioGenerator { .. } | AppState::Recording { .. } => {}
            }

            // Edit mode indicator (overlay)
            if edit_mode_active {
                let panels = self.compute_ui_panels(base_state);
                self.queue_edit_mode_indicator(&panels, selected_panel, resize_preview, reflow_positions);

                // Panel chooser dialog (if open)
                if let Some((spawn_pos, selection)) = panel_chooser {
                    self.queue_panel_chooser(spawn_pos, selection);
                }
            }

            // Prepare base text for Pass 1
            let _base_text = self.prepare_text();

            let mut encoder = self
                .device
                .create_command_encoder(&wgpu::CommandEncoderDescriptor {
                    label: Some("Modal Render Encoder"),
                });

            // Pass 1: Render base scene
            {
                let mut render_pass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
                    label: Some("Base Scene Pass"),
                    color_attachments: &[Some(wgpu::RenderPassColorAttachment {
                        view: &target_view,
                        resolve_target: None,
                        ops: wgpu::Operations {
                            load: wgpu::LoadOp::Clear(self.background_color()),
                            store: wgpu::StoreOp::Store,
                        },
                        depth_slice: None,
                    })],
                    depth_stencil_attachment: None,
                    timestamp_writes: None,
                    occlusion_query_set: None,
                    multiview_mask: None,
                });

                // Combine base cards and edit mode cards into one draw call
                let mut all_cards = base_cards.clone();
                all_cards.extend(self.edit_mode_cards.iter().cloned());
                self.card_renderer
                    .draw(&mut render_pass, &self.queue, &all_cards);

                if self.gamepad_connected {
                    let sprites = self.build_help_sprites(base_state);
                    self.sprite_renderer
                        .draw(&mut render_pass, &self.queue, &sprites);
                }

                self.render_text(&mut render_pass);
            }

            // Pass 2: Draw dark overlay + modal
            // Queue modal text based on type
            match modal {
                ModalType::MainMenu(selected) => self.queue_main_menu_text(*selected),
                ModalType::Settings(selected) => self.queue_settings_modal_text(*selected),
                ModalType::UiScale { selected, user_scale_override } => self.queue_ui_scale_modal_text(*selected, *user_scale_override),
                ModalType::Permission { selected, command, prefix } => self.queue_permission_modal_text(*selected, command, prefix),
                ModalType::Execute(selected) => self.queue_execute_modal_text(*selected),
                ModalType::Survey { question, header, options, focused, custom_input, custom_active, multi_select, selected_indices, use_quick_select, scroll_offset } => {
                    self.queue_survey_modal_text(question, header, options, *focused, custom_input, *custom_active, *multi_select, selected_indices, *use_quick_select, *scroll_offset);
                }
                ModalType::MultiDisplay { focus_index, options, remember_choice, focused_row, .. } => {
                    self.queue_multi_display_modal_text(*focus_index, options, *remember_choice, *focused_row);
                }
                ModalType::AddCard(selected) => {
                    self.queue_add_card_modal_text(*selected);
                }
                ModalType::CustomTask { name, description, active_field, cursor } => {
                    self.queue_custom_task_modal_text(name, description, *active_field, *cursor);
                }
                ModalType::ProjectContext { project_name, selected, is_archived, .. } => {
                    self.queue_project_context_modal_text(project_name, *selected, *is_archived);
                }
                ModalType::LanguageSelect { selected, languages, .. } => {
                    self.queue_language_selector_modal_text(*selected, languages);
                }
            }

            // Prepare modal text for Pass 2
            let _modal_text = self.prepare_text();

            // Fullscreen dark overlay card + modal cards
            let mut modal_cards = vec![
                // Dark semi-transparent overlay covering the whole screen (80% transparent = 0.2 alpha)
                CardInstance::new(0.0, 0.0, self.size.width as f32, self.size.height as f32, [0.02, 0.02, 0.03, 0.2])
                    .with_border_width(0.0)
                    .with_corner_radius(0.0),
            ];
            match modal {
                ModalType::MainMenu(selected) => modal_cards.extend(self.build_main_menu_cards(*selected)),
                ModalType::Settings(selected) => modal_cards.extend(self.build_settings_modal_cards(*selected)),
                ModalType::UiScale { selected, .. } => modal_cards.extend(self.build_ui_scale_modal_cards(*selected)),
                ModalType::Permission { selected, command, prefix: _ } => modal_cards.extend(self.build_permission_modal_cards(*selected, command)),
                ModalType::Execute(selected) => modal_cards.extend(self.build_execute_modal_cards(*selected)),
                ModalType::Survey { question: _, header: _, options, focused, custom_input: _, custom_active: _, multi_select: _, selected_indices, scroll_offset, .. } => {
                    modal_cards.extend(self.build_survey_modal_cards(options, *focused, selected_indices, *scroll_offset));
                }
                ModalType::MultiDisplay { focus_index, options, remember_choice, focused_row, primary_pill_drag } => {
                    modal_cards.extend(self.build_multi_display_modal_cards(*focus_index, options, *remember_choice, *focused_row, *primary_pill_drag));
                }
                ModalType::AddCard(selected) => {
                    modal_cards.extend(self.build_add_card_modal_cards(*selected));
                }
                ModalType::CustomTask { active_field, .. } => {
                    modal_cards.extend(self.build_custom_task_modal_cards(*active_field));
                }
                ModalType::ProjectContext { selected, is_archived, .. } => {
                    modal_cards.extend(self.build_project_context_modal_cards(*selected, *is_archived));
                }
                ModalType::LanguageSelect { selected, languages, .. } => {
                    modal_cards.extend(self.build_language_selector_modal_cards(*selected, languages));
                }
            }

            {
                let mut render_pass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
                    label: Some("Modal Pass"),
                    color_attachments: &[Some(wgpu::RenderPassColorAttachment {
                        view: &target_view,
                        resolve_target: None,
                        ops: wgpu::Operations {
                            load: wgpu::LoadOp::Load, // Preserve base scene
                            store: wgpu::StoreOp::Store,
                        },
                        depth_slice: None,
                    })],
                    depth_stencil_attachment: None,
                    timestamp_writes: None,
                    occlusion_query_set: None,
                    multiview_mask: None,
                });

                self.card_renderer
                    .draw(&mut render_pass, &self.queue, &modal_cards);

                // Build sprites: help sprites + modal-specific sprites
                let mut sprites = Vec::new();
                if self.gamepad_connected {
                    sprites.extend(self.build_help_sprites(state));
                }
                // Add permission modal button glyphs (always show, not just gamepad)
                if let ModalType::Permission { command, .. } = modal {
                    sprites.extend(self.build_permission_modal_sprites(command));
                }
                // Add survey modal button glyphs (only when quick-select is active)
                if let ModalType::Survey { options, use_quick_select, .. } = modal {
                    if *use_quick_select {
                        sprites.extend(self.build_survey_modal_sprites(options.len()));
                    }
                }
                if !sprites.is_empty() {
                    self.sprite_renderer
                        .draw(&mut render_pass, &self.queue, &sprites);
                }

                self.render_text(&mut render_pass);
            }

            // If screenshot, blit to surface and capture
            if render_to_screenshot {
                let screenshot_view = self.screenshot_texture.as_ref().unwrap()
                    .create_view(&wgpu::TextureViewDescriptor::default());

                let blit_bind_group = self.device.create_bind_group(&wgpu::BindGroupDescriptor {
                    label: Some("Blit Bind Group"),
                    layout: &self.blit_bind_group_layout,
                    entries: &[
                        wgpu::BindGroupEntry {
                            binding: 0,
                            resource: wgpu::BindingResource::TextureView(&screenshot_view),
                        },
                        wgpu::BindGroupEntry {
                            binding: 1,
                            resource: wgpu::BindingResource::Sampler(&self.blit_sampler),
                        },
                    ],
                });

                {
                    let mut blit_pass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
                        label: Some("Blit Pass"),
                        color_attachments: &[Some(wgpu::RenderPassColorAttachment {
                            view: &surface_view,
                            resolve_target: None,
                            ops: wgpu::Operations {
                                load: wgpu::LoadOp::Clear(self.background_color()),
                                store: wgpu::StoreOp::Store,
                            },
                            depth_slice: None,
                        })],
                        depth_stencil_attachment: None,
                        timestamp_writes: None,
                        occlusion_query_set: None,
                        multiview_mask: None,
                    });

                    blit_pass.set_pipeline(&self.blit_pipeline);
                    blit_pass.set_bind_group(0, &blit_bind_group, &[]);
                    blit_pass.draw(0..3, 0..1);
                }

                self.queue.submit(std::iter::once(encoder.finish()));

                // Capture screenshot
                if let Some(path) = screenshot_path {
                    if let Err(e) = screenshot_capture.start_capture(
                        &self.device,
                        &self.queue,
                        self.screenshot_texture.as_ref().unwrap(),
                        self.size.width,
                        self.size.height,
                        path.clone(),
                    ) {
                        tracing::error!("Failed to start screenshot capture: {}", e);
                    }
                }
            } else {
                self.queue.submit(std::iter::once(encoder.finish()));
            }
        } else if let Some(path) = screenshot_path {
            // Screenshot path (no modal)
            if self.screenshot_texture.is_none() {
                self.screenshot_texture = Some(self.device.create_texture(&wgpu::TextureDescriptor {
                    label: Some("Screenshot Texture"),
                    size: wgpu::Extent3d {
                        width: self.size.width,
                        height: self.size.height,
                        depth_or_array_layers: 1,
                    },
                    mip_level_count: 1,
                    sample_count: 1,
                    dimension: wgpu::TextureDimension::D2,
                    format: self.config.format,
                    usage: wgpu::TextureUsages::RENDER_ATTACHMENT
                        | wgpu::TextureUsages::COPY_SRC
                        | wgpu::TextureUsages::TEXTURE_BINDING,
                    view_formats: &[],
                }));
            }

            // Queue text
            match base_state {
                AppState::ProjectChooser { selected_index, .. } => {
                    self.queue_project_chooser_text(projects, *selected_index);
                }
                AppState::ProjectView { project_path, selected_action } => {
                    let context_bounds = reflow_positions.get("context_widget").copied();
                    let menu_bounds = reflow_positions.get("action_menu").copied();
                    self.queue_project_view_text(project_path, *selected_action, context_bounds, menu_bounds);
                }
                AppState::PalaceLoop { cards, current_tool, tool_log, thought_log, log_scroll_offset, focused_index, hovered_index, detail_scroll_offset, card_scroll_offset, generating, .. } => {
                    let show_add_card = !cards.is_empty() || !generating;
                    self.queue_palace_loop_text(cards, current_tool.as_deref(), tool_log, thought_log, *log_scroll_offset, *focused_index, *hovered_index, *detail_scroll_offset, None, *card_scroll_offset, show_add_card);
                }
                AppState::Executing { tool_log, thought_log, log_scroll_offset, status, executor, quest_log_visible, all_cards, quest_log_focus, executing_cards, task_statuses, tokens_used, request_active, .. } => {
                    if *quest_log_visible {
                        let exec_ids: Vec<usize> = executing_cards.iter().map(|c| c.id).collect();
                        self.queue_palace_loop_text(all_cards, None, &[], &[], 0, *quest_log_focus, None, 0.0, Some((&exec_ids, task_statuses)), 0.0, false);
                    } else {
                        self.queue_executing_text(tool_log, thought_log, *log_scroll_offset, status, *executor, *tokens_used, *request_active);
                    }
                }
                AppState::MainMenu { .. } => {}
                AppState::SettingsMenu { .. } => {}
                AppState::UiScaleMenu { .. } => {}
                AppState::PermissionModal { .. } => {}
                AppState::ExecuteModal { .. } => {}
                AppState::AddCardMenu { .. } => {}
                AppState::CustomTaskInput { .. } => {}
                AppState::Survey { .. } => {}
                AppState::MultiDisplayDialog { .. } => {}
                AppState::ProjectContextMenu { .. } => {}
                AppState::LanguageSelector { .. } => {}
                AppState::NewMonitorDialog { .. } => {}
                AppState::ScenarioDiffViewer { viewer, feedback_mode, feedback_options, selected_feedback, custom_feedback, custom_cursor, .. } => {
                    self.queue_diff_viewer_text(viewer, *feedback_mode, feedback_options, *selected_feedback, custom_feedback.as_deref(), *custom_cursor);
                }
                AppState::ScenarioGenerator { generator, .. } => {
                    self.queue_generator_text(generator);
                }
                AppState::Recording { inner_state, recorder, paused, .. } => {
                    // Queue inner state's text
                    match inner_state.as_ref() {
                        AppState::PalaceLoop { cards, current_tool, tool_log, thought_log, log_scroll_offset, focused_index, hovered_index, detail_scroll_offset, card_scroll_offset, generating, .. } => {
                            let show_add_card = !cards.is_empty() || !generating;
                            self.queue_palace_loop_text(cards, current_tool.as_deref(), tool_log, thought_log, *log_scroll_offset, *focused_index, *hovered_index, *detail_scroll_offset, None, *card_scroll_offset, show_add_card);
                        }
                        AppState::ProjectChooser { selected_index, .. } => {
                            self.queue_project_chooser_text(projects, *selected_index);
                        }
                        _ => {}
                    }
                    // Queue recording indicator overlay
                    self.queue_recording_indicator_text(recorder.elapsed(), *paused);
                }
            }

            // Edit mode indicator (overlay)
            if edit_mode_active {
                let panels = self.compute_ui_panels(base_state);
                self.queue_edit_mode_indicator(&panels, selected_panel, resize_preview, reflow_positions);

                // Panel chooser dialog (if open)
                if let Some((spawn_pos, selection)) = panel_chooser {
                    self.queue_panel_chooser(spawn_pos, selection);
                }
            }

            // Recording indicator cards (overlay)
            let mut recording_cards = Vec::new();
            if let AppState::Recording { paused, .. } = base_state {
                recording_cards = self.build_recording_indicator_cards(*paused);
            }

            // Prepare text
            let _prepared = self.prepare_text();

            let screenshot_texture = self.screenshot_texture.as_ref().unwrap();
            let screenshot_view = screenshot_texture.create_view(&wgpu::TextureViewDescriptor::default());

            let mut encoder = self
                .device
                .create_command_encoder(&wgpu::CommandEncoderDescriptor {
                    label: Some("Screenshot Render Encoder"),
                });

            {
                let mut render_pass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
                    label: Some("Screenshot Pass"),
                    color_attachments: &[Some(wgpu::RenderPassColorAttachment {
                        view: &screenshot_view,
                        resolve_target: None,
                        ops: wgpu::Operations {
                            load: wgpu::LoadOp::Clear(self.background_color()),
                            store: wgpu::StoreOp::Store,
                        },
                        depth_slice: None,
                    })],
                    depth_stencil_attachment: None,
                    timestamp_writes: None,
                    occlusion_query_set: None,
                    multiview_mask: None,
                });

                // Combine base cards, edit mode cards, and recording indicator into one draw call
                let mut all_cards = base_cards.clone();
                all_cards.extend(self.edit_mode_cards.iter().cloned());
                all_cards.extend(recording_cards.iter().cloned());
                self.card_renderer
                    .draw(&mut render_pass, &self.queue, &all_cards);

                if self.gamepad_connected {
                    let sprites = self.build_help_sprites(state);
                    self.sprite_renderer
                        .draw(&mut render_pass, &self.queue, &sprites);
                }

                self.render_text(&mut render_pass);
            }

            // Blit to surface
            let blit_bind_group = self.device.create_bind_group(&wgpu::BindGroupDescriptor {
                label: Some("Blit Bind Group"),
                layout: &self.blit_bind_group_layout,
                entries: &[
                    wgpu::BindGroupEntry {
                        binding: 0,
                        resource: wgpu::BindingResource::TextureView(&screenshot_view),
                    },
                    wgpu::BindGroupEntry {
                        binding: 1,
                        resource: wgpu::BindingResource::Sampler(&self.blit_sampler),
                    },
                ],
            });

            {
                let mut blit_pass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
                    label: Some("Blit Pass"),
                    color_attachments: &[Some(wgpu::RenderPassColorAttachment {
                        view: &surface_view,
                        resolve_target: None,
                        ops: wgpu::Operations {
                            load: wgpu::LoadOp::Clear(self.background_color()),
                            store: wgpu::StoreOp::Store,
                        },
                        depth_slice: None,
                    })],
                    depth_stencil_attachment: None,
                    timestamp_writes: None,
                    occlusion_query_set: None,
                    multiview_mask: None,
                });

                blit_pass.set_pipeline(&self.blit_pipeline);
                blit_pass.set_bind_group(0, &blit_bind_group, &[]);
                blit_pass.draw(0..3, 0..1);
            }

            self.queue.submit(std::iter::once(encoder.finish()));

            if let Err(e) = screenshot_capture.start_capture(
                &self.device,
                &self.queue,
                screenshot_texture,
                self.size.width,
                self.size.height,
                path.clone(),
            ) {
                tracing::error!("Failed to start screenshot capture: {}", e);
            }
        } else {
            // Normal render directly to surface (no modal, no screenshot)
            match base_state {
                AppState::ProjectChooser { selected_index, .. } => {
                    self.queue_project_chooser_text(projects, *selected_index);
                }
                AppState::ProjectView { project_path, selected_action } => {
                    let context_bounds = reflow_positions.get("context_widget").copied();
                    let menu_bounds = reflow_positions.get("action_menu").copied();
                    self.queue_project_view_text(project_path, *selected_action, context_bounds, menu_bounds);
                }
                AppState::PalaceLoop { cards, current_tool, tool_log, thought_log, log_scroll_offset, focused_index, hovered_index, detail_scroll_offset, card_scroll_offset, generating, .. } => {
                    let show_add_card = !cards.is_empty() || !generating;
                    self.queue_palace_loop_text(cards, current_tool.as_deref(), tool_log, thought_log, *log_scroll_offset, *focused_index, *hovered_index, *detail_scroll_offset, None, *card_scroll_offset, show_add_card);
                }
                AppState::Executing { tool_log, thought_log, log_scroll_offset, status, executor, quest_log_visible, all_cards, quest_log_focus, executing_cards, task_statuses, tokens_used, request_active, .. } => {
                    if *quest_log_visible {
                        // Show card deck view with execution status (no scroll in this view)
                        let exec_ids: Vec<usize> = executing_cards.iter().map(|c| c.id).collect();
                        self.queue_palace_loop_text(all_cards, None, &[], &[], 0, *quest_log_focus, None, 0.0, Some((&exec_ids, task_statuses)), 0.0, false);
                    } else {
                        // Show executor log view
                        self.queue_executing_text(tool_log, thought_log, *log_scroll_offset, status, *executor, *tokens_used, *request_active);
                    }
                }
                AppState::MainMenu { .. } => {}
                AppState::SettingsMenu { .. } => {}
                AppState::UiScaleMenu { .. } => {}
                AppState::PermissionModal { .. } => {}
                AppState::ExecuteModal { .. } => {}
                AppState::AddCardMenu { .. } => {}
                AppState::CustomTaskInput { .. } => {}
                AppState::Survey { .. } => {}
                AppState::MultiDisplayDialog { .. } => {}
                AppState::ProjectContextMenu { .. } => {}
                AppState::LanguageSelector { .. } => {}
                AppState::NewMonitorDialog { .. } => {}
                AppState::ScenarioDiffViewer { viewer, feedback_mode, feedback_options, selected_feedback, custom_feedback, custom_cursor, .. } => {
                    self.queue_diff_viewer_text(viewer, *feedback_mode, feedback_options, *selected_feedback, custom_feedback.as_deref(), *custom_cursor);
                }
                AppState::ScenarioGenerator { generator, .. } => {
                    self.queue_generator_text(generator);
                }
                AppState::Recording { inner_state, recorder, paused, .. } => {
                    // Queue inner state's text
                    match inner_state.as_ref() {
                        AppState::PalaceLoop { cards, current_tool, tool_log, thought_log, log_scroll_offset, focused_index, hovered_index, detail_scroll_offset, card_scroll_offset, generating, .. } => {
                            let show_add_card = !cards.is_empty() || !generating;
                            self.queue_palace_loop_text(cards, current_tool.as_deref(), tool_log, thought_log, *log_scroll_offset, *focused_index, *hovered_index, *detail_scroll_offset, None, *card_scroll_offset, show_add_card);
                        }
                        AppState::ProjectChooser { selected_index, .. } => {
                            self.queue_project_chooser_text(projects, *selected_index);
                        }
                        _ => {}
                    }
                    // Queue recording indicator overlay
                    self.queue_recording_indicator_text(recorder.elapsed(), *paused);
                }
            }

            // Edit mode indicator (overlay)
            if edit_mode_active {
                let panels = self.compute_ui_panels(base_state);
                self.queue_edit_mode_indicator(&panels, selected_panel, resize_preview, reflow_positions);

                // Panel chooser dialog (if open)
                if let Some((spawn_pos, selection)) = panel_chooser {
                    self.queue_panel_chooser(spawn_pos, selection);
                }
            }

            // Recording indicator cards (overlay)
            let mut recording_cards = Vec::new();
            if let AppState::Recording { paused, .. } = base_state {
                recording_cards = self.build_recording_indicator_cards(*paused);
            }

            // Prepare text
            let _prepared = self.prepare_text();

            let mut encoder = self
                .device
                .create_command_encoder(&wgpu::CommandEncoderDescriptor {
                    label: Some("Render Encoder"),
                });

            {
                let mut render_pass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
                    label: Some("Main Pass"),
                    color_attachments: &[Some(wgpu::RenderPassColorAttachment {
                        view: &surface_view,
                        resolve_target: None,
                        ops: wgpu::Operations {
                            load: wgpu::LoadOp::Clear(self.background_color()),
                            store: wgpu::StoreOp::Store,
                        },
                        depth_slice: None,
                    })],
                    depth_stencil_attachment: None,
                    timestamp_writes: None,
                    occlusion_query_set: None,
                    multiview_mask: None,
                });

                // Combine base cards, edit mode cards, and recording indicator into one draw call
                let mut all_cards = base_cards;
                all_cards.extend(self.edit_mode_cards.iter().cloned());
                all_cards.extend(recording_cards.iter().cloned());
                self.card_renderer
                    .draw(&mut render_pass, &self.queue, &all_cards);

                if self.gamepad_connected {
                    let sprites = self.build_help_sprites(state);
                    self.sprite_renderer
                        .draw(&mut render_pass, &self.queue, &sprites);
                }

                self.render_text(&mut render_pass);
            }

            self.queue.submit(std::iter::once(encoder.finish()));
        }

        output.present();

        Ok(())
    }

    /// Ensure scene and blur textures exist for multi-pass rendering
    fn ensure_blur_textures(&mut self) {
        let texture_desc = wgpu::TextureDescriptor {
            label: Some("Scene Texture"),
            size: wgpu::Extent3d {
                width: self.size.width,
                height: self.size.height,
                depth_or_array_layers: 1,
            },
            mip_level_count: 1,
            sample_count: 1,
            dimension: wgpu::TextureDimension::D2,
            format: self.config.format,
            usage: wgpu::TextureUsages::RENDER_ATTACHMENT | wgpu::TextureUsages::TEXTURE_BINDING,
            view_formats: &[],
        };

        if self.scene_texture.is_none() {
            self.scene_texture = Some(self.device.create_texture(&texture_desc));
        }

        if self.blur_texture.is_none() {
            let mut blur_desc = texture_desc.clone();
            blur_desc.label = Some("Blur Texture");
            self.blur_texture = Some(self.device.create_texture(&blur_desc));
        }
    }

    fn build_project_cards(
        &self,
        projects: &ProjectsConfig,
        selected: usize,
    ) -> Vec<CardInstance> {
        let (content_w, content_h) = self.content_size();
        let (offset_x, offset_y) = self.content_offset();
        let grid = CardGrid::new(
            content_w,
            content_h,
            &self.ui_scale,
        ).with_offset(offset_x, offset_y);

        projects
            .projects
            .iter()
            .enumerate()
            .map(|(i, project)| {
                let (x, y) = grid.card_position(i);
                let is_selected = i == selected;

                let mut card =
                    CardInstance::new(x, y, grid.card_width, grid.card_height, project.status.color())
                        .with_border_width(self.ui_scale.px(if is_selected { 4.0 } else { 2.5 }))
                        .with_corner_radius(self.ui_scale.px(16.0));

                if is_selected {
                    card = card.selected();
                }

                card
            })
            .collect()
    }

    fn build_action_cards(&self, selected: usize, panel_bounds: Option<(f32, f32, f32, f32)>) -> Vec<CardInstance> {
        use crate::state::ProjectAction;

        let (content_w, _content_h) = self.content_size();
        let (offset_x, offset_y) = self.content_offset();

        let actions = ProjectAction::all();
        let action_count = actions.len() as f32;

        // Use panel bounds if provided, otherwise calculate defaults
        let (menu_x, menu_y, menu_w, menu_h) = if let Some((px, py, pw, ph)) = panel_bounds {
            (px, py, pw, ph)
        } else {
            let left_margin = self.ui_scale.px(60.0);
            let title_scale = self.ui_scale.px(40.0);
            let top_margin = self.ui_scale.px(40.0);
            let menu_start_y = top_margin + title_scale + self.ui_scale.px(60.0);
            let card_width = content_w - left_margin * 2.0;
            let card_height = self.ui_scale.px(70.0);
            let card_gap = self.ui_scale.px(16.0);
            let menu_height = action_count * (card_height + card_gap) - card_gap;
            (offset_x + left_margin, offset_y + menu_start_y, card_width, menu_height)
        };

        // Calculate card dimensions dynamically based on panel bounds
        // Gap is proportional to panel height (roughly 15% of card+gap goes to gap)
        let padding = (menu_w * 0.01).max(2.0).min(self.ui_scale.px(5.0));
        let total_gap_space = menu_h * 0.12; // 12% of height for gaps
        let card_gap = total_gap_space / (action_count - 1.0).max(1.0);
        let card_width = menu_w - padding * 2.0;
        let card_height = (menu_h - padding * 2.0 - card_gap * (action_count - 1.0)) / action_count;

        // Action color - purple accent
        let action_color = [0.5, 0.3, 0.8, 1.0];

        // Scale border and corner radius based on card size
        let size_factor = (card_height / self.ui_scale.px(70.0)).clamp(0.3, 2.0);

        actions
            .iter()
            .enumerate()
            .map(|(i, _action)| {
                let x = menu_x + padding;
                let y = menu_y + padding + i as f32 * (card_height + card_gap);
                let is_selected = i == selected;

                let mut card = CardInstance::new(x, y, card_width, card_height, action_color)
                    .with_border_width(self.ui_scale.px(if is_selected { 3.0 } else { 1.5 }) * size_factor)
                    .with_corner_radius(self.ui_scale.px(12.0) * size_factor);

                if is_selected {
                    card = card.selected();
                }

                card
            })
            .collect()
    }

    /// Build suggestion cards with optional status badges for quest log
    /// exec_status: (executing_card_ids, task_statuses) for badge rendering
    /// scroll_offset: Vertical scroll offset in pixels (cards above this are clipped)
    /// show_add_card: Whether to show the "+" card at the end (only in PalaceLoop, not quest log)
    fn build_suggestion_cards(&self, cards: &[SuggestionCard], focused: usize, hovered: Option<usize>, exec_status: Option<(&[usize], &[TaskStatus])>, scroll_offset: f32, show_add_card: bool) -> Vec<CardInstance> {
        let (content_w, content_h) = self.content_size();
        let (offset_x, offset_y) = self.content_offset();

        let grid = CardGrid::for_palace_loop(
            content_w,
            content_h,
            &self.ui_scale,
        ).with_offset(offset_x, offset_y);

        let text_margin = self.ui_scale.px(12.0);

        let mut result: Vec<CardInstance> = cards
            .iter()
            .enumerate()
            .flat_map(|(i, card)| {
                let (x, base_y) = grid.card_position(i);
                let card_y = base_y - scroll_offset;

                // Skip cards that are completely off-screen (optimization)
                if card_y + grid.card_height < offset_y || card_y > offset_y + content_h {
                    return vec![];
                }

                // Card is "flipped" when focused via keyboard/gamepad OR hovered via mouse
                let is_flipped = i == focused || hovered == Some(i);

                // Use card's category color for BORDER only (alpha < 1.0 = OLED mode = black fill)
                let mut color = card.color();
                color[3] = 0.95; // OLED mode: colored border, black background

                let mut instance = CardInstance::new(x, card_y, grid.card_width, grid.card_height, color)
                    .with_border_width(self.ui_scale.px(if is_flipped { 4.0 } else { 2.0 }))
                    .with_corner_radius(self.ui_scale.px(12.0));

                if is_flipped {
                    instance = instance.selected();
                }

                // If selected for execution, use green border
                if card.selected {
                    instance = instance.with_border_color([0.2, 1.0, 0.4, 0.95]);
                }

                let mut cards_result = vec![instance];

                // Add status badge card for quest log mode
                if let Some((exec_ids, task_statuses)) = exec_status {
                    if let Some(exec_pos) = exec_ids.iter().position(|&id| id == card.id) {
                        // Get status from task_statuses array
                        let status = task_statuses.get(exec_pos).copied().unwrap_or(TaskStatus::Pending);
                        let badge_color = status.badge_color();
                        let badge_text = status.badge_text();

                        // Badge dimensions - positioned bottom-right
                        let badge_height = self.ui_scale.px(14.0);
                        // Width depends on text length
                        let badge_width = self.ui_scale.px(match badge_text.len() {
                            0..=4 => 40.0,
                            5..=7 => 55.0,
                            _ => 70.0,
                        });
                        let badge_x = x + grid.card_width - text_margin - badge_width;
                        let badge_y = card_y + grid.card_height - text_margin - badge_height;

                        let badge = CardInstance::new(badge_x, badge_y, badge_width, badge_height, badge_color)
                            .with_border_width(self.ui_scale.px(1.0))
                            .with_corner_radius(self.ui_scale.px(3.0));
                        cards_result.push(badge);
                    }
                }

                cards_result
            })
            .collect();

        // Add "+" card at the end (only in PalaceLoop, not quest log)
        if show_add_card {
            let add_card_index = cards.len();
            let (x, base_y) = grid.card_position(add_card_index);
            let card_y = base_y - scroll_offset;

            // Only show if visible on screen (positions already include offset)
            if card_y + grid.card_height >= offset_y && card_y <= offset_y + content_h {
                let is_focused = focused == add_card_index;
                let is_hovered = hovered == Some(add_card_index);
                let is_active = is_focused || is_hovered;

                // White border normally, green when active (no fill - hollow card)
                let border_color = if is_active {
                    [0.3, 0.9, 0.4, 1.0] // Green when selected
                } else {
                    [0.7, 0.7, 0.7, 0.6] // Dim white/gray when not
                };

                let add_card = CardInstance::new(x, card_y, grid.card_width, grid.card_height, border_color)
                    .with_border_width(self.ui_scale.px(if is_active { 3.0 } else { 2.0 }))
                    .with_corner_radius(self.ui_scale.px(12.0));
                // Don't call .selected() - keep it hollow (no fill)

                result.push(add_card);
            }
        }

        result
    }

    fn build_main_menu_cards(&self, selected: usize) -> Vec<CardInstance> {
        use crate::state::MainMenuItem;

        let (content_w, content_h) = self.content_size();
        let (offset_x, offset_y) = self.content_offset();

        let items = MainMenuItem::all();

        // Modal dimensions - centered on content area
        let modal_width = self.ui_scale.px(400.0).min(content_w - 40.0);
        let modal_height = self.ui_scale.px(220.0);
        let modal_x = offset_x + (content_w - modal_width) / 2.0;
        let modal_y = offset_y + (content_h - modal_height) / 2.0;

        let card_height = self.ui_scale.px(50.0);
        let card_gap = self.ui_scale.px(10.0);
        let inner_padding = self.ui_scale.px(20.0);
        let title_height = self.ui_scale.px(10.0); // No title, just small top padding

        let mut cards = Vec::new();

        // Modal background card - OLED black with subtle border
        cards.push(
            CardInstance::new(modal_x, modal_y, modal_width, modal_height, [0.0, 0.0, 0.0, 1.0])
                .with_border_width(self.ui_scale.px(1.5))
                .with_corner_radius(self.ui_scale.px(16.0))
        );

        // Menu item cards
        let menu_color = [0.3, 0.4, 0.6, 1.0];
        let card_start_y = modal_y + title_height;
        let card_width = modal_width - inner_padding * 2.0;

        for (i, _item) in items.iter().enumerate() {
            let y = card_start_y + i as f32 * (card_height + card_gap);
            let is_selected = i == selected;

            let mut card = CardInstance::new(
                modal_x + inner_padding,
                y,
                card_width,
                card_height,
                menu_color
            )
                .with_border_width(self.ui_scale.px(if is_selected { 3.0 } else { 1.5 }))
                .with_corner_radius(self.ui_scale.px(10.0));

            if is_selected {
                card = card.selected();
            }

            cards.push(card);
        }

        cards
    }

    fn queue_main_menu_text(&mut self, selected: usize) {
        use crate::state::MainMenuItem;

        let (content_w, content_h) = self.content_size();
        let (offset_x, offset_y) = self.content_offset();

        let modal_width = self.ui_scale.px(400.0).min(content_w - 40.0);
        let modal_x = offset_x + (content_w - modal_width) / 2.0;
        let modal_y = offset_y + (content_h - self.ui_scale.px(220.0)) / 2.0;

        let scale = self.ui_scale.px(26.0);
        let inner_padding = self.ui_scale.px(20.0);
        let title_height = self.ui_scale.px(10.0);
        let card_height = self.ui_scale.px(50.0);
        let card_gap = self.ui_scale.px(10.0);
        let text_padding = self.ui_scale.px(12.0);

        let items = MainMenuItem::all();
        let card_start_y = modal_y + title_height;

        for (i, item) in items.iter().enumerate() {
            let y = card_start_y + i as f32 * (card_height + card_gap) + text_padding;
            let label_color = if i == selected {
                [1.0, 1.0, 1.0, 1.0]
            } else {
                [0.8, 0.8, 0.9, 1.0]
            };

            self.text_queue.push(
                item.label(),
                modal_x + inner_padding + self.ui_scale.px(15.0),
                y,
                scale,
                label_color,
            );
        }

        let help_state = AppState::MainMenu {
            selected_item: selected,
            previous_state: Box::new(AppState::project_chooser()),
        };
        self.queue_help_legend(&help_state);
    }

    fn build_settings_modal_cards(&self, selected: usize) -> Vec<CardInstance> {
        use crate::state::SettingsItem;

        let (content_w, content_h) = self.content_size();
        let (offset_x, offset_y) = self.content_offset();

        let items = SettingsItem::all();

        // Modal dimensions - centered on content area
        let modal_width = self.ui_scale.px(500.0).min(content_w - 40.0);
        let modal_height = self.ui_scale.px(250.0);
        let modal_x = offset_x + (content_w - modal_width) / 2.0;
        let modal_y = offset_y + (content_h - modal_height) / 2.0;

        let card_height = self.ui_scale.px(60.0);
        let card_gap = self.ui_scale.px(12.0);
        let inner_padding = self.ui_scale.px(20.0);
        let title_height = self.ui_scale.px(50.0);

        let mut cards = Vec::new();

        // Modal background card - OLED black with subtle border
        cards.push(
            CardInstance::new(modal_x, modal_y, modal_width, modal_height, [0.0, 0.0, 0.0, 1.0])
                .with_border_width(self.ui_scale.px(1.5))
                .with_corner_radius(self.ui_scale.px(16.0))
        );

        // Settings item cards
        let settings_color = [0.2, 0.6, 0.7, 1.0];
        let card_start_y = modal_y + title_height;
        let card_width = modal_width - inner_padding * 2.0;

        for (i, _item) in items.iter().enumerate() {
            let y = card_start_y + i as f32 * (card_height + card_gap);
            let is_selected = i == selected;

            let mut card = CardInstance::new(
                modal_x + inner_padding,
                y,
                card_width,
                card_height,
                settings_color
            )
                .with_border_width(self.ui_scale.px(if is_selected { 3.0 } else { 1.5 }))
                .with_corner_radius(self.ui_scale.px(10.0));

            if is_selected {
                card = card.selected();
            }

            cards.push(card);
        }

        cards
    }

    fn queue_settings_modal_text(&mut self, selected: usize) {
        use crate::state::SettingsItem;

        let (content_w, content_h) = self.content_size();
        let (offset_x, offset_y) = self.content_offset();

        let modal_width = self.ui_scale.px(500.0).min(content_w - 40.0);
        let modal_height = self.ui_scale.px(250.0);
        let modal_x = offset_x + (content_w - modal_width) / 2.0;
        let modal_y = offset_y + (content_h - modal_height) / 2.0;

        let scale = self.ui_scale.px(24.0);
        let title_scale = self.ui_scale.px(32.0);
        let inner_padding = self.ui_scale.px(20.0);
        let title_height = self.ui_scale.px(50.0);
        let card_height = self.ui_scale.px(60.0);
        let card_gap = self.ui_scale.px(12.0);
        let text_padding = self.ui_scale.px(18.0);

        // Modal title
        self.text_queue.push("Settings", modal_x + inner_padding, modal_y + self.ui_scale.px(12.0), title_scale, [0.8, 0.9, 1.0, 1.0]);

        // Settings items
        let items = SettingsItem::all();
        let card_start_y = modal_y + title_height;

        for (i, item) in items.iter().enumerate() {
            let y = card_start_y + i as f32 * (card_height + card_gap) + text_padding;
            let label_color = if i == selected {
                [1.0, 1.0, 1.0, 1.0]
            } else {
                [0.8, 0.8, 0.9, 1.0]
            };

            self.text_queue.push(item.label(), modal_x + inner_padding + self.ui_scale.px(15.0), y, scale, label_color);

            // Value indicator on right side
            let value_text: String = match item {
                SettingsItem::Display => "→".to_string(),  // Arrow to indicate submenu
                SettingsItem::DarkMode => if self.dark_mode { "ON".to_string() } else { "OFF".to_string() },
                SettingsItem::UiScale => format!("{}%", (self.ui_scale.dpi_scale * 100.0) as i32),
            };
            let value_x = modal_x + modal_width - inner_padding - self.ui_scale.px(60.0);

            // Show "(auto)" indicator to the left of the value for auto-detected UI scale
            if matches!(item, SettingsItem::UiScale) && self.is_auto_scale {
                self.text_queue.push(
                    "(auto)",
                    value_x - self.ui_scale.px(55.0),
                    y + self.ui_scale.px(2.0),
                    scale * 0.6,
                    [0.5, 0.8, 0.9, 0.7],
                );
            }

            self.text_queue.push(
                &value_text,
                value_x,
                y,
                scale * 0.85,
                [0.5, 0.8, 0.9, 1.0],
            );
        }

        let help_state = AppState::SettingsMenu {
            selected_item: selected,
            previous_state: Box::new(AppState::MainMenu {
                selected_item: 0,
                previous_state: Box::new(AppState::project_chooser()),
            }),
        };
        self.queue_help_legend(&help_state);
    }

    fn build_ui_scale_modal_cards(&self, selected: usize) -> Vec<CardInstance> {
        use crate::state::UiScaleOption;
        let items = UiScaleOption::all();

        let (content_w, content_h) = self.content_size();
        let (offset_x, offset_y) = self.content_offset();

        // Modal dimensions - centered on content area
        let modal_width = self.ui_scale.px(400.0).min(content_w - 40.0);
        let item_count = items.len() as f32;
        let card_height = self.ui_scale.px(50.0);
        let card_gap = self.ui_scale.px(8.0);
        let inner_padding = self.ui_scale.px(20.0);
        let title_height = self.ui_scale.px(50.0);
        let modal_height = title_height + inner_padding + item_count * (card_height + card_gap);
        let modal_x = offset_x + (content_w - modal_width) / 2.0;
        let modal_y = offset_y + (content_h - modal_height) / 2.0;

        let mut cards = Vec::new();

        // Modal background card - OLED black with subtle border
        cards.push(
            CardInstance::new(modal_x, modal_y, modal_width, modal_height, [0.0, 0.0, 0.0, 1.0])
                .with_border_width(self.ui_scale.px(1.5))
                .with_corner_radius(self.ui_scale.px(16.0))
        );

        // Scale option cards
        let scale_color = [0.5, 0.3, 0.7, 1.0]; // Purple for scale options
        let card_start_y = modal_y + title_height;
        let card_width = modal_width - inner_padding * 2.0;

        for (i, _item) in items.iter().enumerate() {
            let y = card_start_y + i as f32 * (card_height + card_gap);
            let is_selected = i == selected;

            let mut card = CardInstance::new(
                modal_x + inner_padding,
                y,
                card_width,
                card_height,
                scale_color
            )
                .with_border_width(self.ui_scale.px(if is_selected { 3.0 } else { 1.5 }))
                .with_corner_radius(self.ui_scale.px(10.0));

            if is_selected {
                card = card.selected();
            }

            cards.push(card);
        }

        cards
    }

    fn build_permission_modal_cards(&self, selected: usize, command: &str) -> Vec<CardInstance> {
        use crate::state::PermissionChoice;
        let items = PermissionChoice::all();

        let (content_w, content_h) = self.content_size();
        let (offset_x, offset_y) = self.content_offset();

        // Dynamic modal dimensions based on command length
        let max_width = self.ui_scale.px(800.0).min(content_w * 0.9);
        let min_width = self.ui_scale.px(400.0);
        let inner_padding = self.ui_scale.px(20.0);
        let command_scale = self.ui_scale.px(16.0);

        // Estimate chars per line (monospace ~0.6 width ratio)
        let char_width = command_scale * 0.55;
        let usable_width = max_width - inner_padding * 2.0;
        let chars_per_line = (usable_width / char_width).floor() as usize;
        let command_lines = ((command.len() as f32) / chars_per_line as f32).ceil() as usize;
        let command_lines = command_lines.max(1).min(10); // Cap at 10 lines

        // Calculate width based on command (but capped)
        let command_text_width = (command.len().min(chars_per_line) as f32 * char_width) + inner_padding * 2.0;
        let modal_width = command_text_width.max(min_width).min(max_width);

        let item_count = items.len() as f32;
        let card_height = self.ui_scale.px(50.0);
        let card_gap = self.ui_scale.px(8.0);
        let title_height = self.ui_scale.px(50.0);
        let line_height = command_scale * 1.4;
        let command_height = line_height * command_lines as f32 + self.ui_scale.px(20.0);
        let modal_height = title_height + command_height + inner_padding + item_count * (card_height + card_gap);
        let modal_x = offset_x + (content_w - modal_width) / 2.0;
        let modal_y = offset_y + (content_h - modal_height) / 2.0;

        let mut cards = Vec::new();

        // Modal background card - OLED black with warning border
        cards.push(
            CardInstance::new(modal_x, modal_y, modal_width, modal_height, [1.0, 0.6, 0.2, 1.0]) // Orange warning border
                .with_border_width(self.ui_scale.px(2.0))
                .with_corner_radius(self.ui_scale.px(16.0))
        );

        // Choice option cards
        let card_start_y = modal_y + title_height + command_height;
        let card_width = modal_width - inner_padding * 2.0;

        // Leave space for glyph on left side of each card
        let glyph_space = self.ui_scale.px(50.0);
        let card_content_width = card_width - glyph_space;

        for (i, _item) in items.iter().enumerate() {
            let y = card_start_y + i as f32 * (card_height + card_gap);
            let is_selected = i == selected;

            // Color based on choice type (matches Xbox button colors)
            let choice_color = match i {
                0 => [0.25, 0.55, 0.25, 1.0], // A = Green (Yes once)
                1 => [0.2, 0.35, 0.6, 1.0],   // X = Blue (Yes always)
                2 => [0.6, 0.2, 0.2, 1.0],    // B = Red (No)
                _ => [0.55, 0.5, 0.15, 1.0],  // Y = Yellow (Suggest else)
            };

            let mut card = CardInstance::new(
                modal_x + inner_padding + glyph_space,
                y,
                card_content_width,
                card_height,
                choice_color
            )
                .with_border_width(self.ui_scale.px(if is_selected { 3.0 } else { 1.5 }))
                .with_corner_radius(self.ui_scale.px(10.0));

            if is_selected {
                card = card.selected();
            }

            cards.push(card);
        }

        cards
    }

    /// Build sprites for permission modal (button glyphs next to each option)
    fn build_permission_modal_sprites(&self, command: &str) -> Vec<SpriteInstance> {
        use crate::state::PermissionChoice;
        let items = PermissionChoice::all();

        let (content_w, content_h) = self.content_size();
        let (offset_x, offset_y) = self.content_offset();

        // Same sizing as build_permission_modal_cards
        let max_width = self.ui_scale.px(800.0).min(content_w * 0.9);
        let min_width = self.ui_scale.px(400.0);
        let inner_padding = self.ui_scale.px(20.0);
        let command_scale = self.ui_scale.px(16.0);

        let char_width = command_scale * 0.55;
        let usable_width = max_width - inner_padding * 2.0;
        let chars_per_line = (usable_width / char_width).floor() as usize;
        let command_lines = ((command.len() as f32) / chars_per_line as f32).ceil() as usize;
        let command_lines = command_lines.max(1).min(10);

        let command_text_width = (command.len().min(chars_per_line) as f32 * char_width) + inner_padding * 2.0;
        let modal_width = command_text_width.max(min_width).min(max_width);

        let card_height = self.ui_scale.px(50.0);
        let card_gap = self.ui_scale.px(8.0);
        let title_height = self.ui_scale.px(50.0);
        let line_height = command_scale * 1.4;
        let command_height = line_height * command_lines as f32 + self.ui_scale.px(20.0);
        let modal_height = title_height + command_height + inner_padding + items.len() as f32 * (card_height + card_gap);
        let modal_x = offset_x + (content_w - modal_width) / 2.0;
        let modal_y = offset_y + (content_h - modal_height) / 2.0;

        let card_start_y = modal_y + title_height + command_height;
        let glyph_size = self.ui_scale.px(36.0);

        let mut sprites = Vec::new();

        // Button glyph for each option
        let buttons = [XboxButton::A, XboxButton::X, XboxButton::B, XboxButton::Y];
        for (i, button) in buttons.iter().enumerate() {
            if i >= items.len() { break; }
            let y = card_start_y + i as f32 * (card_height + card_gap);
            let glyph_x = modal_x + inner_padding + self.ui_scale.px(6.0);
            let glyph_y = y + (card_height - glyph_size) / 2.0;

            sprites.push(SpriteInstance::new(glyph_x, glyph_y, glyph_size, *button));
        }

        sprites
    }

    /// Build sprites for survey modal (button glyphs to the LEFT of first 4 options)
    fn build_survey_modal_sprites(&self, option_count: usize) -> Vec<SpriteInstance> {
        let (_content_w, content_h) = self.content_size();
        let (offset_x, offset_y) = self.content_offset();

        // Must match queue_survey_modal_text dimensions - FULL WIDTH layout
        let margin = self.ui_scale.px(48.0);
        let card_height = self.ui_scale.px(60.0);
        let card_gap = self.ui_scale.px(10.0);
        let inner_padding = self.ui_scale.px(20.0);
        let title_height = self.ui_scale.px(80.0);

        // Calculate visible options (same as text rendering)
        let available_height = content_h - margin * 2.0 - title_height - self.ui_scale.px(60.0);
        let max_visible = (available_height / (card_height + card_gap)).floor() as usize;
        let total_options = option_count + 1; // +1 for "Other"

        let modal_height = title_height + inner_padding + max_visible.min(total_options) as f32 * (card_height + card_gap);
        let modal_x = offset_x + margin;
        let modal_y = offset_y + (content_h - modal_height) / 2.0;
        let card_start_y = modal_y + title_height;
        let glyph_size = self.ui_scale.px(32.0);

        let mut sprites = Vec::new();
        // Same order as permission modal: A=0, X=1, B=2, Y=3
        let buttons = [XboxButton::A, XboxButton::X, XboxButton::B, XboxButton::Y];

        for (i, button) in buttons.iter().enumerate() {
            if i >= total_options.min(max_visible) { break; }
            let y = card_start_y + i as f32 * (card_height + card_gap);
            // Position glyphs to the LEFT of the card (before inner_padding)
            let glyph_x = modal_x + (inner_padding - glyph_size) / 2.0;
            let glyph_y = y + (card_height - glyph_size) / 2.0;
            sprites.push(SpriteInstance::new(glyph_x, glyph_y, glyph_size, *button));
        }

        sprites
    }

    fn queue_ui_scale_modal_text(&mut self, selected: usize, user_scale_override: Option<f32>) {
        use crate::state::UiScaleOption;

        let (content_w, content_h) = self.content_size();
        let (offset_x, offset_y) = self.content_offset();

        let items = UiScaleOption::all();

        // Must match modal dimensions from build_ui_scale_modal_cards
        let modal_width = self.ui_scale.px(400.0).min(content_w - 40.0);
        let item_count = items.len() as f32;
        let card_height = self.ui_scale.px(50.0);
        let card_gap = self.ui_scale.px(8.0);
        let inner_padding = self.ui_scale.px(20.0);
        let title_height = self.ui_scale.px(50.0);
        let modal_height = title_height + inner_padding + item_count * (card_height + card_gap);
        let modal_x = offset_x + (content_w - modal_width) / 2.0;
        let modal_y = offset_y + (content_h - modal_height) / 2.0;

        let scale = self.ui_scale.px(22.0);
        let title_scale = self.ui_scale.px(28.0);
        let text_padding = self.ui_scale.px(14.0);
        let card_start_y = modal_y + title_height;

        // Modal title
        self.text_queue.push(
            "UI Scale",
            modal_x + inner_padding,
            modal_y + self.ui_scale.px(12.0),
            title_scale,
            [0.8, 0.9, 1.0, 1.0],
        );

        // Scale options
        for (i, item) in items.iter().enumerate() {
            let y = card_start_y + i as f32 * (card_height + card_gap) + text_padding;
            let is_selected = i == selected;

            let label_color = if is_selected {
                [1.0, 1.0, 1.0, 1.0]
            } else {
                [0.8, 0.8, 0.9, 1.0]
            };

            // Option label
            self.text_queue.push(
                item.label(),
                modal_x + inner_padding + self.ui_scale.px(15.0),
                y,
                scale,
                label_color,
            );

            // Show current indicator if this matches user's setting
            let is_current = item.value() == user_scale_override;
            if is_current {
                self.text_queue.push(
                    "(current)",
                    modal_x + modal_width - inner_padding - self.ui_scale.px(80.0),
                    y,
                    scale * 0.7,
                    [0.5, 0.7, 0.5, 1.0],
                );
            }
        }

        // Add help legend for modal state
        let help_state = AppState::UiScaleMenu {
            selected_item: selected,
            user_scale_override,
            previous_state: Box::new(AppState::SettingsMenu {
                selected_item: 0,
                previous_state: Box::new(AppState::MainMenu {
                    selected_item: 0,
                    previous_state: Box::new(AppState::project_chooser()),
                }),
            }),
        };
        self.queue_help_legend(&help_state);
    }

    fn queue_permission_modal_text(&mut self, selected: usize, command: &str, prefix: &str) {
        use crate::state::PermissionChoice;

        let (content_w, content_h) = self.content_size();
        let (offset_x, offset_y) = self.content_offset();

        let items = PermissionChoice::all();

        // Dynamic modal dimensions - must match build_permission_modal_cards
        let max_width = self.ui_scale.px(800.0).min(content_w * 0.9);
        let min_width = self.ui_scale.px(400.0);
        let inner_padding = self.ui_scale.px(20.0);
        let command_scale = self.ui_scale.px(16.0);

        // Estimate chars per line (monospace ~0.6 width ratio)
        let char_width = command_scale * 0.55;
        let usable_width = max_width - inner_padding * 2.0;
        let chars_per_line = (usable_width / char_width).floor() as usize;
        let command_lines = ((command.len() as f32) / chars_per_line as f32).ceil() as usize;
        let command_lines = command_lines.max(1).min(10); // Cap at 10 lines

        // Calculate width based on command (but capped)
        let command_text_width = (command.len().min(chars_per_line) as f32 * char_width) + inner_padding * 2.0;
        let modal_width = command_text_width.max(min_width).min(max_width);

        let item_count = items.len() as f32;
        let card_height = self.ui_scale.px(50.0);
        let card_gap = self.ui_scale.px(8.0);
        let title_height = self.ui_scale.px(50.0);
        let line_height = command_scale * 1.4;
        let command_height = line_height * command_lines as f32 + self.ui_scale.px(20.0);
        let modal_height = title_height + command_height + inner_padding + item_count * (card_height + card_gap);
        let modal_x = offset_x + (content_w - modal_width) / 2.0;
        let modal_y = offset_y + (content_h - modal_height) / 2.0;

        let scale = self.ui_scale.px(22.0);
        let title_scale = self.ui_scale.px(28.0);
        let text_padding = self.ui_scale.px(14.0);
        let card_start_y = modal_y + title_height + command_height;

        // Modal title
        self.text_queue.push(
            "⚠️ Permission Required",
            modal_x + inner_padding,
            modal_y + self.ui_scale.px(12.0),
            title_scale,
            [1.0, 0.8, 0.3, 1.0], // Warning yellow
        );

        // Command display (full text with wrapping)
        let text_area_width = modal_width - inner_padding * 2.0;
        self.text_queue.push_bounded(
            command,
            modal_x + inner_padding,
            modal_y + title_height + self.ui_scale.px(5.0),
            command_scale,
            [0.7, 0.9, 1.0, 1.0], // Command in light blue
            text_area_width,
            command_height,
        );

        // Pre-collect labels to avoid lifetime issues
        let labels: Vec<String> = items.iter().map(|item| item.label(prefix)).collect();

        // Space for glyph on left side (must match build_permission_modal_cards)
        let glyph_space = self.ui_scale.px(50.0);

        // Choice options (text positioned after glyph)
        for (i, label) in labels.iter().enumerate() {
            let y = card_start_y + i as f32 * (card_height + card_gap) + text_padding;
            let is_selected = i == selected;

            let label_color = if is_selected {
                [1.0, 1.0, 1.0, 1.0]
            } else {
                [0.8, 0.8, 0.9, 1.0]
            };

            // Option label (positioned after glyph space)
            self.text_queue.push(
                label,
                modal_x + inner_padding + glyph_space + self.ui_scale.px(10.0),
                y,
                scale,
                label_color,
            );
        }

        // Add help legend for modal state
        let help_state = AppState::PermissionModal {
            command: command.to_string(),
            command_prefix: prefix.to_string(),
            selected_choice: selected,
            previous_state: Box::new(AppState::PalaceLoop {
                project_path: std::path::PathBuf::new(),
                cards: Vec::new(),
                focused_index: 0,
                hovered_index: None,
                generating: false,
                current_tool: None,
                tool_log: Vec::new(),
                thought_log: Vec::new(),
                log_scroll_offset: 0,
                detail_scroll_offset: 0.0,
                detail_max_scroll: 0.0,
                card_scroll_offset: 0.0,
            }),
        };
        self.queue_help_legend(&help_state);
    }

    fn build_execute_modal_cards(&self, selected: usize) -> Vec<CardInstance> {
        use crate::state::ExecuteOption;
        let items = ExecuteOption::all();

        let (content_w, content_h) = self.content_size();
        let (offset_x, offset_y) = self.content_offset();

        // Modal dimensions - centered on content area
        let modal_width = self.ui_scale.px(400.0).min(content_w - 40.0);
        let item_count = items.len() as f32;
        let card_height = self.ui_scale.px(50.0);
        let card_gap = self.ui_scale.px(8.0);
        let inner_padding = self.ui_scale.px(20.0);
        let title_height = self.ui_scale.px(50.0);
        let modal_height = title_height + inner_padding + item_count * (card_height + card_gap);
        let modal_x = offset_x + (content_w - modal_width) / 2.0;
        let modal_y = offset_y + (content_h - modal_height) / 2.0;

        let mut cards = Vec::new();

        // Modal background card - OLED black with green border (execute = go)
        cards.push(
            CardInstance::new(modal_x, modal_y, modal_width, modal_height, [0.3, 0.8, 0.4, 1.0])
                .with_border_width(self.ui_scale.px(2.0))
                .with_corner_radius(self.ui_scale.px(16.0))
        );

        // Execute option cards
        let card_start_y = modal_y + title_height;
        let card_width = modal_width - inner_padding * 2.0;

        for (i, item) in items.iter().enumerate() {
            let y = card_start_y + i as f32 * (card_height + card_gap);
            let is_selected = i == selected;

            // Color based on option type
            let option_color = match item {
                ExecuteOption::Claude => [0.4, 0.5, 0.7, 1.0],    // Blue for Claude
                ExecuteOption::ZAi => [0.5, 0.3, 0.7, 1.0],       // Purple for Z.ai
                ExecuteOption::ZAiTurbo => [0.7, 0.3, 0.5, 1.0],  // Magenta for Turbo
            };

            let mut card = CardInstance::new(
                modal_x + inner_padding,
                y,
                card_width,
                card_height,
                option_color
            )
                .with_border_width(self.ui_scale.px(if is_selected { 3.0 } else { 1.5 }))
                .with_corner_radius(self.ui_scale.px(10.0));

            if is_selected {
                card = card.selected();
            }

            cards.push(card);
        }

        cards
    }

    fn queue_execute_modal_text(&mut self, selected: usize) {
        use crate::state::ExecuteOption;

        let (content_w, content_h) = self.content_size();
        let (offset_x, offset_y) = self.content_offset();

        let items = ExecuteOption::all();

        // Modal dimensions - match build_execute_modal_cards
        let modal_width = self.ui_scale.px(400.0).min(content_w - 40.0);
        let item_count = items.len() as f32;
        let card_height = self.ui_scale.px(50.0);
        let card_gap = self.ui_scale.px(8.0);
        let inner_padding = self.ui_scale.px(20.0);
        let title_height = self.ui_scale.px(50.0);
        let modal_height = title_height + inner_padding + item_count * (card_height + card_gap);
        let modal_x = offset_x + (content_w - modal_width) / 2.0;
        let modal_y = offset_y + (content_h - modal_height) / 2.0;

        let scale = self.ui_scale.px(22.0);
        let title_scale = self.ui_scale.px(28.0);
        let text_padding = self.ui_scale.px(14.0);
        let card_start_y = modal_y + title_height;

        // Modal title
        self.text_queue.push(
            "Execute Selected",
            modal_x + inner_padding,
            modal_y + self.ui_scale.px(12.0),
            title_scale,
            [0.8, 1.0, 0.9, 1.0],
        );

        // Execute options
        for (i, item) in items.iter().enumerate() {
            let y = card_start_y + i as f32 * (card_height + card_gap) + text_padding;
            let is_selected = i == selected;

            let label_color = if is_selected {
                [1.0, 1.0, 1.0, 1.0]
            } else {
                [0.8, 0.8, 0.9, 1.0]
            };

            // Option label
            self.text_queue.push(
                item.label(),
                modal_x + inner_padding + self.ui_scale.px(15.0),
                y,
                scale,
                label_color,
            );
        }

        // Add help legend for modal state
        let help_state = AppState::ExecuteModal {
            selected_option: selected,
            previous_state: Box::new(AppState::PalaceLoop {
                project_path: std::path::PathBuf::new(),
                cards: Vec::new(),
                focused_index: 0,
                hovered_index: None,
                generating: false,
                current_tool: None,
                tool_log: Vec::new(),
                thought_log: Vec::new(),
                log_scroll_offset: 0,
                detail_scroll_offset: 0.0,
                detail_max_scroll: 0.0,
                card_scroll_offset: 0.0,
            }),
        };
        self.queue_help_legend(&help_state);
    }

    fn queue_survey_modal_text(
        &mut self,
        question: &str,
        header: &str,
        options: &[crate::state::SurveyOption],
        focused: usize,
        custom_input: &str,
        custom_active: bool,
        multi_select: bool,
        selected_indices: &[usize],
        _use_quick_select: bool,
        scroll_offset: usize,
    ) {
        let (content_w, content_h) = self.content_size();
        let (offset_x, offset_y) = self.content_offset();

        // Modal dimensions - full width minus margins
        let margin = self.ui_scale.px(48.0);
        let modal_width = content_w - margin * 2.0;
        let card_height = self.ui_scale.px(60.0);
        let card_gap = self.ui_scale.px(10.0);
        let inner_padding = self.ui_scale.px(20.0);
        let title_height = self.ui_scale.px(80.0);

        // Calculate how many options fit on screen
        let available_height = content_h - margin * 2.0 - title_height - self.ui_scale.px(60.0);
        let max_visible = (available_height / (card_height + card_gap)).floor() as usize;
        let total_options = options.len() + 1; // +1 for "Other"

        let modal_height = title_height + inner_padding + max_visible.min(total_options) as f32 * (card_height + card_gap);
        let modal_x = offset_x + margin;
        let modal_y = offset_y + (content_h - modal_height) / 2.0;

        let scale = self.ui_scale.px(18.0);
        let title_scale = self.ui_scale.px(24.0);
        let small_scale = self.ui_scale.px(12.0);
        let text_padding = self.ui_scale.px(14.0);
        let card_start_y = modal_y + title_height;

        // Header badge
        if !header.is_empty() {
            self.text_queue.push(
                header,
                modal_x + inner_padding,
                modal_y + self.ui_scale.px(10.0),
                small_scale,
                [0.6, 0.8, 1.0, 1.0],
            );
        }

        // Question text
        self.text_queue.push_bounded(
            question,
            modal_x + inner_padding,
            modal_y + self.ui_scale.px(28.0),
            title_scale,
            [0.9, 0.9, 1.0, 1.0],
            modal_width - inner_padding * 2.0,
            self.ui_scale.px(50.0),
        );

        // Scroll indicator if needed
        if scroll_offset > 0 {
            self.text_queue.push(
                "▲ more above",
                modal_x + modal_width / 2.0 - self.ui_scale.px(40.0),
                card_start_y - self.ui_scale.px(18.0),
                small_scale,
                [0.5, 0.5, 0.6, 0.7],
            );
        }

        // Options (with scroll)
        let mut visible_idx = 0;
        for (i, opt) in options.iter().enumerate() {
            if i < scroll_offset { continue; }
            if visible_idx >= max_visible { break; }

            let y = card_start_y + visible_idx as f32 * (card_height + card_gap) + text_padding;
            let is_focused = i == focused;
            let is_selected = selected_indices.contains(&i);
            visible_idx += 1;

            // Checkbox for multi-select
            if multi_select {
                let check = if is_selected { "☑" } else { "☐" };
                self.text_queue.push(
                    check,
                    modal_x + inner_padding + self.ui_scale.px(10.0),
                    y,
                    scale,
                    if is_selected { [0.3, 1.0, 0.5, 1.0] } else { [0.5, 0.5, 0.6, 1.0] },
                );
            }

            // Label
            let label_x = modal_x + inner_padding + self.ui_scale.px(if multi_select { 40.0 } else { 10.0 });
            let label_color = if is_focused {
                [1.0, 1.0, 1.0, 1.0]
            } else {
                [0.8, 0.8, 0.9, 1.0]
            };
            self.text_queue.push(&opt.label, label_x, y, scale, label_color);

            // Description (below label)
            if !opt.description.is_empty() {
                let desc_color = if is_focused {
                    [0.7, 0.7, 0.8, 0.9]
                } else {
                    [0.5, 0.5, 0.6, 0.7]
                };
                self.text_queue.push_bounded(
                    &opt.description,
                    label_x,
                    y + scale + self.ui_scale.px(4.0),
                    small_scale,
                    desc_color,
                    modal_width - label_x + modal_x - inner_padding,
                    self.ui_scale.px(30.0),
                );
            }
        }

        // "Other" option (if visible)
        let other_idx = options.len();
        if other_idx >= scroll_offset && visible_idx < max_visible {
            let y = card_start_y + visible_idx as f32 * (card_height + card_gap) + text_padding;
            let is_other_focused = focused == other_idx;

            self.text_queue.push(
                "▶",
                modal_x + inner_padding + self.ui_scale.px(10.0),
                y,
                scale,
                if is_other_focused { [0.8, 0.5, 1.0, 1.0] } else { [0.5, 0.5, 0.6, 1.0] },
            );

            let label_x = modal_x + inner_padding + self.ui_scale.px(40.0);
            self.text_queue.push(
                "Other (type custom answer)",
                label_x,
                y,
                scale,
                if is_other_focused { [1.0, 1.0, 1.0, 1.0] } else { [0.6, 0.6, 0.7, 1.0] },
            );

            // Custom input field
            if custom_active {
                let input_y = y + scale + self.ui_scale.px(8.0);
                let display_text = if custom_input.is_empty() {
                    "Type here...".to_string()
                } else {
                    format!("{}▌", custom_input)
                };
                let input_color = if custom_input.is_empty() {
                    [0.4, 0.4, 0.5, 0.7]
                } else {
                    [0.9, 0.9, 1.0, 1.0]
                };
                self.text_queue.push(&display_text, label_x, input_y, scale, input_color);
            }
        }

        // Scroll indicator if more below
        if scroll_offset + max_visible < total_options {
            let y = card_start_y + max_visible as f32 * (card_height + card_gap);
            self.text_queue.push(
                "▼ more below",
                modal_x + modal_width / 2.0 - self.ui_scale.px(40.0),
                y,
                small_scale,
                [0.5, 0.5, 0.6, 0.7],
            );
        }

        // Help legend
        let help_state = AppState::Survey {
            question: String::new(),
            header: String::new(),
            options: Vec::new(),
            focused_index: 0,
            custom_input: String::new(),
            custom_active: false,
            multi_select,
            selected_indices: Vec::new(),
            use_quick_select: false,
            scroll_offset: 0,
            previous_state: Box::new(AppState::PalaceLoop {
                project_path: std::path::PathBuf::new(),
                cards: Vec::new(),
                focused_index: 0,
                hovered_index: None,
                generating: false,
                current_tool: None,
                tool_log: Vec::new(),
                thought_log: Vec::new(),
                log_scroll_offset: 0,
                detail_scroll_offset: 0.0,
                detail_max_scroll: 0.0,
                card_scroll_offset: 0.0,
            }),
            response_tx: None,
        };
        self.queue_help_legend(&help_state);
    }

    fn build_survey_modal_cards(
        &self,
        options: &[crate::state::SurveyOption],
        focused: usize,
        selected_indices: &[usize],
        scroll_offset: usize,
    ) -> Vec<CardInstance> {
        let (content_w, content_h) = self.content_size();
        let (offset_x, offset_y) = self.content_offset();

        // Full-width layout matching queue_survey_modal_text
        let margin = self.ui_scale.px(48.0);
        let modal_width = content_w - margin * 2.0;
        let card_height = self.ui_scale.px(60.0);
        let card_gap = self.ui_scale.px(10.0);
        let inner_padding = self.ui_scale.px(20.0);
        let title_height = self.ui_scale.px(80.0);

        // Calculate how many options fit on screen
        let available_height = content_h - margin * 2.0 - title_height - self.ui_scale.px(60.0);
        let max_visible = (available_height / (card_height + card_gap)).floor() as usize;
        let total_options = options.len() + 1; // +1 for "Other"

        let modal_height = title_height + inner_padding + max_visible.min(total_options) as f32 * (card_height + card_gap);
        let modal_x = offset_x + margin;
        let modal_y = offset_y + (content_h - modal_height) / 2.0;

        let mut cards = Vec::new();

        // Modal background card - purple border for questions
        cards.push(
            CardInstance::new(modal_x, modal_y, modal_width, modal_height, [0.6, 0.4, 0.9, 1.0])
                .with_border_width(self.ui_scale.px(2.0))
                .with_corner_radius(self.ui_scale.px(16.0))
        );

        // Option cards (with scroll)
        let card_start_y = modal_y + title_height;
        let card_width = modal_width - inner_padding * 2.0;

        let mut visible_idx = 0;
        for i in 0..total_options {
            if i < scroll_offset { continue; }
            if visible_idx >= max_visible { break; }

            let is_focused = i == focused;
            let is_selected = selected_indices.contains(&i);
            let y = card_start_y + visible_idx as f32 * (card_height + card_gap);
            visible_idx += 1;

            let border_color = if is_selected {
                [0.3, 1.0, 0.5, 0.95] // Green for selected
            } else if is_focused {
                [0.8, 0.6, 1.0, 0.95] // Purple for focused
            } else {
                [0.4, 0.4, 0.5, 0.85] // Gray for unfocused
            };

            let mut instance = CardInstance::new(
                modal_x + inner_padding,
                y,
                card_width,
                card_height,
                border_color,
            )
            .with_border_width(self.ui_scale.px(if is_focused { 3.0 } else { 1.5 }))
            .with_corner_radius(self.ui_scale.px(8.0));

            if is_focused {
                instance = instance.selected();
            }

            cards.push(instance);
        }

        cards
    }

    fn queue_multi_display_modal_text(&mut self, focus_index: usize, options: &[crate::state::DisplayOption], remember_choice: bool, focused_row: usize) {
        let (content_w, content_h) = self.content_size();
        let (offset_x, offset_y) = self.content_offset();

        let inner_padding = self.ui_scale.px(24.0);
        let card_height = self.ui_scale.px(60.0);
        let card_gap = self.ui_scale.px(12.0);
        let title_height = self.ui_scale.px(60.0);
        let toggle_height = self.ui_scale.px(40.0);
        let toggle_gap = self.ui_scale.px(8.0);
        let button_height = self.ui_scale.px(44.0);

        // Modal dimensions - monitors + remember toggle + apply button (no extend toggle)
        let modal_width = self.ui_scale.px(500.0).min(content_w * 0.8);
        let item_count = options.len() as f32;
        let modal_height = title_height + inner_padding + item_count * (card_height + card_gap) + toggle_height + toggle_gap * 2.0 + button_height;
        let modal_x = offset_x + (content_w - modal_width) / 2.0;
        let modal_y = offset_y + (content_h - modal_height) / 2.0;

        let scale = self.ui_scale.px(20.0);
        let title_scale = self.ui_scale.px(26.0);
        let small_scale = self.ui_scale.px(14.0);
        let text_padding = self.ui_scale.px(16.0);
        let card_start_y = modal_y + title_height;
        let card_width = modal_width - inner_padding * 2.0;

        // Big monitor identification number - find which monitor this renderer is on
        tracing::debug!("Rendering MultiDisplayDialog on monitor: '{}', options: {:?}",
            self.monitor_name,
            options.iter().map(|o| &o.id).collect::<Vec<_>>());
        let monitor_index = options.iter().position(|opt| opt.id == self.monitor_name);
        let big_number = match monitor_index {
            Some(idx) => format!("{}", idx + 1),
            None => format!("?:{}", &self.monitor_name), // Show monitor name if we can't find it
        };
        let big_scale = self.ui_scale.px(300.0);
        // Position centered on screen
        let num_x = offset_x + (content_w - self.ui_scale.px(150.0)) / 2.0;
        let num_y = offset_y + (content_h - self.ui_scale.px(300.0)) / 2.0;
        self.text_queue.push(
            &big_number,
            num_x,
            num_y,
            big_scale,
            [0.4, 0.6, 1.0, 0.6], // Semi-transparent blue, more visible
        );

        // Modal title
        self.text_queue.push(
            "🖥️ Display Settings",
            modal_x + inner_padding,
            modal_y + self.ui_scale.px(14.0),
            title_scale,
            [0.9, 0.9, 1.0, 1.0],
        );

        // Section header
        self.text_queue.push(
            "Monitors (Space=toggle, P=set primary)",
            modal_x + inner_padding,
            modal_y + title_height - self.ui_scale.px(20.0),
            small_scale,
            [0.6, 0.6, 0.7, 0.8],
        );

        // Display options - each with number badge, enable toggle, and Primary pill
        let option_count = options.len();
        for (i, opt) in options.iter().enumerate() {
            let y = card_start_y + i as f32 * (card_height + card_gap) + text_padding;
            let is_focused = focused_row == i; // Each monitor is its own row
            let is_enabled = opt.enabled;
            let is_primary = opt.is_primary;

            // Number badge (leftmost) - shows 1, 2, 3...
            let number_str = format!("{}", i + 1);
            let number_color = if is_focused {
                [0.9, 0.9, 1.0, 1.0]
            } else {
                [0.6, 0.6, 0.7, 0.8]
            };
            let number_scale = self.ui_scale.px(18.0);
            self.text_queue.push(
                &number_str,
                modal_x + inner_padding + self.ui_scale.px(4.0),
                y + self.ui_scale.px(10.0),
                number_scale,
                number_color,
            );

            // Enable toggle checkbox (after number)
            let toggle_check = if is_enabled { "☑" } else { "☐" };
            let toggle_color = if is_enabled {
                [0.3, 1.0, 0.5, 1.0]
            } else if is_focused {
                [0.8, 0.8, 0.9, 1.0]
            } else {
                [0.5, 0.5, 0.6, 0.7]
            };
            self.text_queue.push(
                toggle_check,
                modal_x + inner_padding + self.ui_scale.px(28.0),
                y + self.ui_scale.px(8.0),
                scale,
                toggle_color,
            );

            // Display name (shifted right to make room for number + toggle)
            let label_color = if is_focused {
                [1.0, 1.0, 1.0, 1.0]
            } else if is_enabled {
                [0.9, 0.9, 1.0, 1.0]
            } else {
                [0.6, 0.6, 0.7, 0.8]
            };
            self.text_queue.push(
                &opt.name,
                modal_x + inner_padding + self.ui_scale.px(68.0),
                y,
                scale,
                label_color,
            );

            // Resolution (aligned with name)
            let info_color = if is_focused {
                [0.7, 0.7, 0.8, 0.9]
            } else {
                [0.5, 0.5, 0.6, 0.7]
            };
            self.text_queue.push(
                &opt.resolution,
                modal_x + inner_padding + self.ui_scale.px(68.0),
                y + scale + self.ui_scale.px(4.0),
                small_scale,
                info_color,
            );

            // Primary pill (right side) - only shown on primary monitor
            if is_primary {
                let pill_x = modal_x + inner_padding + card_width - self.ui_scale.px(88.0);
                self.text_queue.push(
                    "Primary",
                    pill_x + self.ui_scale.px(8.0),
                    y + self.ui_scale.px(10.0),
                    small_scale,
                    [1.0, 1.0, 1.0, 1.0],
                );
            }
        }

        // Toggle row: "Remember this choice" (after all monitors)
        let remember_y = card_start_y + item_count * (card_height + card_gap) + toggle_gap;
        let remember_focused = focused_row == option_count;
        let remember_check = if remember_choice { "☑" } else { "☐" };
        let remember_color = if remember_choice {
            [0.3, 1.0, 0.5, 1.0]
        } else if remember_focused {
            [0.8, 0.8, 0.9, 1.0]
        } else {
            [0.6, 0.6, 0.7, 1.0]
        };
        self.text_queue.push(
            remember_check,
            modal_x + inner_padding + self.ui_scale.px(10.0),
            remember_y,
            scale,
            remember_color,
        );
        self.text_queue.push(
            "Remember this choice",
            modal_x + inner_padding + self.ui_scale.px(40.0),
            remember_y + self.ui_scale.px(2.0),
            small_scale,
            if remember_focused { [1.0, 1.0, 1.0, 1.0] } else { [0.7, 0.7, 0.8, 0.9] },
        );

        // Apply button (after remember)
        let apply_y = remember_y + toggle_height + toggle_gap;
        let apply_focused = focused_row == option_count + 1;
        self.text_queue.push(
            "Apply",
            modal_x + (modal_width / 2.0) - self.ui_scale.px(25.0),
            apply_y + self.ui_scale.px(6.0),
            scale,
            if apply_focused { [1.0, 1.0, 1.0, 1.0] } else { [0.8, 0.8, 0.9, 0.9] },
        );

        // Add help legend
        let help_state = AppState::MultiDisplayDialog {
            focus_index,
            options: options.to_vec(),
            remember_choice,
            focused_row,
            primary_pill_drag: None,
            previous_state: Box::new(AppState::project_chooser()),
        };
        self.queue_help_legend(&help_state);
    }

    fn build_multi_display_modal_cards(&self, _focus_index: usize, options: &[crate::state::DisplayOption], remember_choice: bool, focused_row: usize, primary_pill_drag: Option<(usize, f32, f32)>) -> Vec<CardInstance> {
        let (content_w, content_h) = self.content_size();
        let (offset_x, offset_y) = self.content_offset();

        let inner_padding = self.ui_scale.px(24.0);
        let card_height = self.ui_scale.px(60.0);
        let card_gap = self.ui_scale.px(12.0);
        let title_height = self.ui_scale.px(60.0);
        let toggle_height = self.ui_scale.px(40.0);
        let toggle_gap = self.ui_scale.px(8.0);
        let button_height = self.ui_scale.px(44.0);

        // Modal dimensions - monitors + remember toggle + apply button (no extend toggle)
        let modal_width = self.ui_scale.px(500.0).min(content_w * 0.8);
        let item_count = options.len() as f32;
        let modal_height = title_height + inner_padding + item_count * (card_height + card_gap) + toggle_height + toggle_gap * 2.0 + button_height;
        let modal_x = offset_x + (content_w - modal_width) / 2.0;
        let modal_y = offset_y + (content_h - modal_height) / 2.0;

        let card_start_y = modal_y + title_height;

        let mut cards = Vec::new();

        // Modal background - blue border for display selection
        cards.push(
            CardInstance::new(modal_x, modal_y, modal_width, modal_height, [0.3, 0.5, 0.9, 1.0])
                .with_border_width(self.ui_scale.px(2.0))
                .with_corner_radius(self.ui_scale.px(16.0))
        );

        // Display option cards - each with enable toggle and Primary pill
        let card_width = modal_width - inner_padding * 2.0;
        let option_count = options.len();
        for (i, opt) in options.iter().enumerate() {
            let is_focused = focused_row == i; // Each monitor is its own row
            let is_enabled = opt.enabled;
            let is_primary = opt.is_primary;
            let y = card_start_y + i as f32 * (card_height + card_gap);

            let border_color = if is_focused {
                [0.4, 0.7, 1.0, 0.95] // Blue for focused
            } else if is_enabled {
                [0.3, 0.6, 0.5, 0.8] // Green tint for enabled
            } else {
                [0.35, 0.35, 0.4, 0.7] // Dim gray for disabled
            };

            let mut instance = CardInstance::new(
                modal_x + inner_padding,
                y,
                card_width,
                card_height,
                border_color,
            )
            .with_border_width(self.ui_scale.px(if is_focused { 3.0 } else if is_enabled { 2.0 } else { 1.5 }))
            .with_corner_radius(self.ui_scale.px(8.0));

            if is_focused {
                instance = instance.selected();
            }

            cards.push(instance);

            // Primary pill card (if this is the primary monitor)
            if is_primary {
                let pill_width = self.ui_scale.px(80.0);
                let pill_height = self.ui_scale.px(28.0);
                let pill_x = modal_x + inner_padding + card_width - pill_width - self.ui_scale.px(8.0);
                let pill_y = y + (card_height - pill_height) / 2.0;

                cards.push(
                    CardInstance::new(pill_x, pill_y, pill_width, pill_height, [0.2, 0.6, 0.9, 0.95])
                        .with_border_width(self.ui_scale.px(1.5))
                        .with_corner_radius(self.ui_scale.px(14.0))
                );
            }
        }

        // Toggle row: Remember choice (after all monitors)
        let remember_y = card_start_y + item_count * (card_height + card_gap) + toggle_gap;
        let remember_focused = focused_row == option_count;
        let remember_color = if remember_focused {
            if remember_choice { [0.3, 0.7, 0.5, 0.8] } else { [0.4, 0.5, 0.7, 0.7] }
        } else {
            if remember_choice { [0.2, 0.5, 0.4, 0.5] } else { [0.3, 0.3, 0.4, 0.4] }
        };
        let mut remember_card = CardInstance::new(
            modal_x + inner_padding,
            remember_y - self.ui_scale.px(4.0),
            card_width,
            self.ui_scale.px(32.0),
            remember_color,
        )
        .with_border_width(self.ui_scale.px(if remember_focused { 2.0 } else { 1.0 }))
        .with_corner_radius(self.ui_scale.px(6.0));
        if remember_focused {
            remember_card = remember_card.selected();
        }
        cards.push(remember_card);

        // Apply button (after remember)
        let apply_y = remember_y + toggle_height + toggle_gap;
        let apply_focused = focused_row == option_count + 1;
        let apply_color = if apply_focused {
            [0.3, 0.6, 0.9, 0.9] // Bright blue when focused
        } else {
            [0.25, 0.45, 0.7, 0.7] // Dimmer blue
        };
        let button_width = self.ui_scale.px(120.0);
        let mut apply_card = CardInstance::new(
            modal_x + (modal_width - button_width) / 2.0,
            apply_y - self.ui_scale.px(4.0),
            button_width,
            self.ui_scale.px(36.0),
            apply_color,
        )
        .with_border_width(self.ui_scale.px(if apply_focused { 2.5 } else { 1.5 }))
        .with_corner_radius(self.ui_scale.px(8.0));
        if apply_focused {
            apply_card = apply_card.selected();
        }
        cards.push(apply_card);

        // Floating Primary pill when dragging
        if let Some((_source_idx, cursor_x, cursor_y)) = primary_pill_drag {
            let pill_width = self.ui_scale.px(80.0);
            let pill_height = self.ui_scale.px(28.0);
            // Center pill on cursor
            let pill_x = cursor_x - pill_width / 2.0;
            let pill_y = cursor_y - pill_height / 2.0;

            cards.push(
                CardInstance::new(pill_x, pill_y, pill_width, pill_height, [0.3, 0.7, 1.0, 0.95])
                    .with_border_width(self.ui_scale.px(2.0))
                    .with_corner_radius(self.ui_scale.px(14.0))
                    .selected()
            );
        }

        cards
    }

    fn queue_add_card_modal_text(&mut self, selected: usize) {
        use crate::state::AddCardOption;

        let (content_w, content_h) = self.content_size();
        let (offset_x, offset_y) = self.content_offset();

        let modal_width = self.ui_scale.px(400.0).min(content_w * 0.8);
        let modal_x = offset_x + (content_w - modal_width) / 2.0;
        let modal_y = offset_y + (content_h - self.ui_scale.px(250.0)) / 2.0;
        let inner_padding = self.ui_scale.px(24.0);
        let title_height = self.ui_scale.px(50.0);
        let card_height = self.ui_scale.px(50.0);
        let card_gap = self.ui_scale.px(12.0);

        // Title
        let title = "Add task";
        self.text_queue.push(
            title,
            modal_x + inner_padding,
            modal_y + self.ui_scale.px(16.0),
            self.ui_scale.px(24.0),
            [0.3, 0.9, 1.0, 1.0],
        );

        // Options
        let card_start_y = modal_y + title_height;
        for (i, option) in AddCardOption::all().iter().enumerate() {
            let is_selected = i == selected;
            let y = card_start_y + i as f32 * (card_height + card_gap);

            let text_color = if is_selected {
                self.text_color()
            } else {
                self.text_color_dim()
            };

            self.text_queue.push(
                option.label(),
                modal_x + inner_padding + self.ui_scale.px(12.0),
                y + self.ui_scale.px(16.0),
                self.ui_scale.px(16.0),
                text_color,
            );
        }

        // Help legend
        let help_state = AppState::AddCardMenu {
            selected_option: selected,
            previous_state: Box::new(AppState::project_chooser()),
        };
        self.queue_help_legend(&help_state);
    }

    fn build_add_card_modal_cards(&self, selected: usize) -> Vec<CardInstance> {
        use crate::state::AddCardOption;

        let (content_w, content_h) = self.content_size();
        let (offset_x, offset_y) = self.content_offset();

        let inner_padding = self.ui_scale.px(24.0);
        let card_height = self.ui_scale.px(50.0);
        let card_gap = self.ui_scale.px(12.0);
        let title_height = self.ui_scale.px(50.0);
        let item_count = AddCardOption::all().len() as f32;

        // Modal dimensions
        let modal_width = self.ui_scale.px(400.0).min(content_w * 0.8);
        let modal_height = title_height + inner_padding + item_count * (card_height + card_gap);
        let modal_x = offset_x + (content_w - modal_width) / 2.0;
        let modal_y = offset_y + (content_h - modal_height) / 2.0;

        let card_start_y = modal_y + title_height;

        let mut cards = Vec::new();

        // Modal background - cyan border for add card
        cards.push(
            CardInstance::new(modal_x, modal_y, modal_width, modal_height, [0.3, 0.8, 0.9, 1.0])
                .with_border_width(self.ui_scale.px(2.0))
                .with_corner_radius(self.ui_scale.px(16.0))
        );

        // Option cards
        let card_width = modal_width - inner_padding * 2.0;
        for (i, _opt) in AddCardOption::all().iter().enumerate() {
            let is_selected = i == selected;
            let y = card_start_y + i as f32 * (card_height + card_gap);

            let border_color = if is_selected {
                [0.3, 0.9, 1.0, 0.95] // Cyan for selected
            } else {
                [0.4, 0.4, 0.5, 0.85] // Gray for unselected
            };

            let mut instance = CardInstance::new(
                modal_x + inner_padding,
                y,
                card_width,
                card_height,
                border_color,
            )
            .with_border_width(self.ui_scale.px(if is_selected { 3.0 } else { 1.5 }))
            .with_corner_radius(self.ui_scale.px(8.0));

            if is_selected {
                instance = instance.selected();
            }

            cards.push(instance);
        }

        cards
    }

    fn queue_custom_task_modal_text(&mut self, name: &str, description: &str, active_field: usize, cursor: usize) {
        let (content_w, content_h) = self.content_size();
        let (offset_x, offset_y) = self.content_offset();

        let modal_width = self.ui_scale.px(500.0).min(content_w * 0.9);
        let modal_height = self.ui_scale.px(280.0);
        let modal_x = offset_x + (content_w - modal_width) / 2.0;
        let modal_y = offset_y + (content_h - modal_height) / 2.0;
        let inner_padding = self.ui_scale.px(24.0);

        // Title
        self.text_queue.push(
            "Custom Task",
            modal_x + inner_padding,
            modal_y + self.ui_scale.px(16.0),
            self.ui_scale.px(20.0),
            [0.3, 0.9, 1.0, 1.0],
        );

        // Name label
        let name_label_y = modal_y + self.ui_scale.px(50.0);
        let name_color = if active_field == 0 { [0.5, 0.9, 0.6, 1.0] } else { self.text_color_dim() };
        self.text_queue.push(
            "Name:",
            modal_x + inner_padding,
            name_label_y,
            self.ui_scale.px(14.0),
            name_color,
        );

        // Name input content
        let name_input_y = modal_y + self.ui_scale.px(68.0);
        let name_display = if name.is_empty() { "Task title (optional)" } else { name };
        let name_text_color = if name.is_empty() { self.text_color_dim() } else { self.text_color() };
        self.text_queue.push(
            name_display,
            modal_x + inner_padding + self.ui_scale.px(12.0),
            name_input_y + self.ui_scale.px(14.0),
            self.ui_scale.px(16.0),
            name_text_color,
        );

        // Description label
        let desc_label_y = modal_y + self.ui_scale.px(130.0);
        let desc_color = if active_field == 1 { [0.5, 0.9, 0.6, 1.0] } else { self.text_color_dim() };
        self.text_queue.push(
            "Description:",
            modal_x + inner_padding,
            desc_label_y,
            self.ui_scale.px(14.0),
            desc_color,
        );

        // Description input content
        let desc_input_y = modal_y + self.ui_scale.px(148.0);
        let desc_display = if description.is_empty() { "What should be done? (optional)" } else { description };
        let desc_text_color = if description.is_empty() { self.text_color_dim() } else { self.text_color() };
        self.text_queue.push(
            desc_display,
            modal_x + inner_padding + self.ui_scale.px(12.0),
            desc_input_y + self.ui_scale.px(14.0),
            self.ui_scale.px(16.0),
            desc_text_color,
        );

        // Blinking cursor in active field
        let cursor_visible = (self.animation_time() * 2.0).fract() < 0.5;
        if cursor_visible {
            let (active_text, input_y) = if active_field == 0 {
                (name, name_input_y)
            } else {
                (description, desc_input_y)
            };
            // Only show cursor if field has content (otherwise placeholder is shown)
            let char_width = self.ui_scale.px(8.5);
            let cursor_x = modal_x + inner_padding + self.ui_scale.px(12.0) + cursor as f32 * char_width;
            if !active_text.is_empty() || cursor == 0 {
                self.text_queue.push(
                    "|",
                    cursor_x,
                    input_y + self.ui_scale.px(12.0),
                    self.ui_scale.px(18.0),
                    [0.9, 0.9, 0.9, 1.0],
                );
            }
        }

        // Hint
        self.text_queue.push(
            "Tab to switch fields • Enter to confirm • Esc to cancel",
            modal_x + inner_padding,
            modal_y + self.ui_scale.px(220.0),
            self.ui_scale.px(12.0),
            self.text_color_dim(),
        );

        // Note about optional fields
        self.text_queue.push(
            "Fill either field - the other will be suggested",
            modal_x + inner_padding,
            modal_y + self.ui_scale.px(240.0),
            self.ui_scale.px(11.0),
            [0.5, 0.5, 0.5, 0.8],
        );

        // Help legend
        let help_state = AppState::CustomTaskInput {
            name: name.to_string(),
            description: description.to_string(),
            active_field,
            cursor,
            previous_state: Box::new(AppState::project_chooser()),
        };
        self.queue_help_legend(&help_state);
    }

    fn build_custom_task_modal_cards(&self, active_field: usize) -> Vec<CardInstance> {
        let (content_w, content_h) = self.content_size();
        let (offset_x, offset_y) = self.content_offset();

        let inner_padding = self.ui_scale.px(24.0);

        // Modal dimensions - taller for two fields
        let modal_width = self.ui_scale.px(500.0).min(content_w * 0.9);
        let modal_height = self.ui_scale.px(280.0);
        let modal_x = offset_x + (content_w - modal_width) / 2.0;
        let modal_y = offset_y + (content_h - modal_height) / 2.0;

        let mut cards = Vec::new();

        // Modal background - cyan border
        cards.push(
            CardInstance::new(modal_x, modal_y, modal_width, modal_height, [0.3, 0.8, 0.9, 1.0])
                .with_border_width(self.ui_scale.px(2.0))
                .with_corner_radius(self.ui_scale.px(16.0))
        );

        let input_height = self.ui_scale.px(44.0);
        let input_width = modal_width - inner_padding * 2.0;

        // Name input field background
        let name_input_y = modal_y + self.ui_scale.px(68.0);
        let name_border_color = if active_field == 0 {
            [0.4, 0.9, 0.5, 1.0] // Green when active
        } else {
            [0.3, 0.3, 0.35, 0.95] // Dim when inactive
        };
        cards.push(
            CardInstance::new(
                modal_x + inner_padding,
                name_input_y,
                input_width,
                input_height,
                name_border_color,
            )
            .with_border_width(self.ui_scale.px(if active_field == 0 { 2.0 } else { 1.5 }))
            .with_corner_radius(self.ui_scale.px(8.0))
        );

        // Description input field background
        let desc_input_y = modal_y + self.ui_scale.px(148.0);
        let desc_border_color = if active_field == 1 {
            [0.4, 0.9, 0.5, 1.0] // Green when active
        } else {
            [0.3, 0.3, 0.35, 0.95] // Dim when inactive
        };
        cards.push(
            CardInstance::new(
                modal_x + inner_padding,
                desc_input_y,
                input_width,
                input_height,
                desc_border_color,
            )
            .with_border_width(self.ui_scale.px(if active_field == 1 { 2.0 } else { 1.5 }))
            .with_corner_radius(self.ui_scale.px(8.0))
        );

        cards
    }

    fn queue_project_context_modal_text(&mut self, project_name: &str, selected: usize, is_archived: bool) {
        use crate::state::ProjectContextOption;

        let (content_w, content_h) = self.content_size();
        let (offset_x, offset_y) = self.content_offset();

        let inner_padding = self.ui_scale.px(32.0);
        let title_height = self.ui_scale.px(80.0);
        let card_height = self.ui_scale.px(72.0);
        let card_gap = self.ui_scale.px(16.0);
        let item_count = ProjectContextOption::all().len() as f32;

        // Modal dimensions - must match build_project_context_modal_cards exactly
        // Use 16:9 center column width (roughly 960px at 1080p) for better proportions
        let target_width = self.ui_scale.px(600.0);
        let modal_width = target_width.min(content_w * 0.9);
        let modal_height = title_height + inner_padding + item_count * (card_height + card_gap);
        let modal_x = offset_x + (content_w - modal_width) / 2.0;
        let modal_y = offset_y + (content_h - modal_height) / 2.0;

        // Title - larger and more prominent
        self.text_queue.push(
            project_name,
            modal_x + inner_padding,
            modal_y + self.ui_scale.px(20.0),
            self.ui_scale.px(32.0),
            [0.9, 0.6, 0.2, 1.0], // Orange for project name
        );

        // Options - text centered vertically in each card
        let card_start_y = modal_y + title_height;
        for (i, option) in ProjectContextOption::all().iter().enumerate() {
            let is_selected = i == selected;
            let y = card_start_y + i as f32 * (card_height + card_gap);

            let text_color = if is_selected {
                self.text_color()
            } else {
                self.text_color_dim()
            };

            // Center text vertically: card_y + (card_height - text_height) / 2
            let text_size = self.ui_scale.px(26.0);
            let text_y = y + (card_height - text_size) / 2.0;

            self.text_queue.push(
                option.label(is_archived),
                modal_x + inner_padding + self.ui_scale.px(16.0),
                text_y,
                text_size,
                text_color,
            );
        }
    }

    fn build_project_context_modal_cards(&self, selected: usize, is_archived: bool) -> Vec<CardInstance> {
        use crate::state::ProjectContextOption;

        let (content_w, content_h) = self.content_size();
        let (offset_x, offset_y) = self.content_offset();

        let inner_padding = self.ui_scale.px(32.0);
        let card_height = self.ui_scale.px(72.0);
        let card_gap = self.ui_scale.px(16.0);
        let title_height = self.ui_scale.px(80.0);
        let item_count = ProjectContextOption::all().len() as f32;

        // Modal dimensions - must match queue_project_context_modal_text exactly
        // Use 16:9 center column width (roughly 960px at 1080p) for better proportions
        let target_width = self.ui_scale.px(600.0);
        let modal_width = target_width.min(content_w * 0.9);
        let modal_height = title_height + inner_padding + item_count * (card_height + card_gap);
        let modal_x = offset_x + (content_w - modal_width) / 2.0;
        let modal_y = offset_y + (content_h - modal_height) / 2.0;

        let card_start_y = modal_y + title_height;

        let mut cards = Vec::new();

        // Modal background - orange border for project context (OLED style: border only)
        cards.push(
            CardInstance::new(modal_x, modal_y, modal_width, modal_height, [0.9, 0.6, 0.2, 1.0])
                .with_border_width(self.ui_scale.px(3.0))
                .with_corner_radius(self.ui_scale.px(20.0))
        );

        // Option cards
        let card_width = modal_width - inner_padding * 2.0;
        for (i, option) in ProjectContextOption::all().iter().enumerate() {
            let is_selected_card = i == selected;
            let y = card_start_y + i as f32 * (card_height + card_gap);

            // Color based on option type
            let border_color = if is_selected_card {
                match option {
                    ProjectContextOption::Remove => [0.9, 0.3, 0.3, 1.0], // Red for remove
                    ProjectContextOption::ChangeLanguage => [0.3, 0.7, 0.9, 1.0], // Blue for language
                    ProjectContextOption::ToggleArchive => {
                        if is_archived {
                            [0.4, 0.9, 0.4, 1.0] // Green for unarchive
                        } else {
                            [0.8, 0.7, 0.3, 1.0] // Yellow for archive
                        }
                    }
                    ProjectContextOption::Cancel => [0.5, 0.5, 0.6, 1.0], // Gray for cancel
                }
            } else {
                [0.4, 0.4, 0.5, 0.85] // Gray for unselected
            };

            let mut instance = CardInstance::new(
                modal_x + inner_padding,
                y,
                card_width,
                card_height,
                border_color,
            )
            .with_border_width(self.ui_scale.px(if is_selected_card { 3.5 } else { 2.0 }))
            .with_corner_radius(self.ui_scale.px(12.0));

            if is_selected_card {
                instance = instance.selected();
            }

            cards.push(instance);
        }

        cards
    }

    fn queue_language_selector_modal_text(&mut self, selected: usize, languages: &[String]) {
        let (content_w, content_h) = self.content_size();
        let (offset_x, offset_y) = self.content_offset();

        let modal_width = self.ui_scale.px(350.0).min(content_w * 0.7);
        let modal_x = offset_x + (content_w - modal_width) / 2.0;
        let modal_y = offset_y + (content_h - self.ui_scale.px(300.0)) / 2.0;
        let inner_padding = self.ui_scale.px(24.0);
        let title_height = self.ui_scale.px(50.0);
        let card_height = self.ui_scale.px(40.0);
        let card_gap = self.ui_scale.px(8.0);

        // Title
        self.text_queue.push(
            "Select Language",
            modal_x + inner_padding,
            modal_y + self.ui_scale.px(14.0),
            self.ui_scale.px(20.0),
            [0.3, 0.7, 0.9, 1.0], // Blue for language selector
        );

        // Language options
        let card_start_y = modal_y + title_height;
        for (i, lang) in languages.iter().enumerate() {
            let is_selected = i == selected;
            let y = card_start_y + i as f32 * (card_height + card_gap);

            let text_color = if is_selected {
                self.text_color()
            } else {
                self.text_color_dim()
            };

            self.text_queue.push(
                lang,
                modal_x + inner_padding + self.ui_scale.px(12.0),
                y + self.ui_scale.px(12.0),
                self.ui_scale.px(16.0),
                text_color,
            );
        }
    }

    fn build_language_selector_modal_cards(&self, selected: usize, languages: &[String]) -> Vec<CardInstance> {
        let (content_w, content_h) = self.content_size();
        let (offset_x, offset_y) = self.content_offset();

        let inner_padding = self.ui_scale.px(24.0);
        let card_height = self.ui_scale.px(40.0);
        let card_gap = self.ui_scale.px(8.0);
        let title_height = self.ui_scale.px(50.0);
        let item_count = languages.len() as f32;

        // Modal dimensions
        let modal_width = self.ui_scale.px(350.0).min(content_w * 0.7);
        let modal_height = title_height + inner_padding + item_count * (card_height + card_gap);
        let modal_x = offset_x + (content_w - modal_width) / 2.0;
        let modal_y = offset_y + (content_h - modal_height) / 2.0;

        let card_start_y = modal_y + title_height;

        let mut cards = Vec::new();

        // Modal background - blue border for language selector (OLED style)
        cards.push(
            CardInstance::new(modal_x, modal_y, modal_width, modal_height, [0.3, 0.7, 0.9, 1.0])
                .with_border_width(self.ui_scale.px(2.0))
                .with_corner_radius(self.ui_scale.px(16.0))
        );

        // Language option cards
        let card_width = modal_width - inner_padding * 2.0;
        for (i, _lang) in languages.iter().enumerate() {
            let is_selected_card = i == selected;
            let y = card_start_y + i as f32 * (card_height + card_gap);

            let border_color = if is_selected_card {
                [0.3, 0.8, 0.9, 1.0] // Cyan when selected
            } else {
                [0.4, 0.4, 0.5, 0.85] // Gray for unselected
            };

            let mut instance = CardInstance::new(
                modal_x + inner_padding,
                y,
                card_width,
                card_height,
                border_color,
            )
            .with_border_width(self.ui_scale.px(if is_selected_card { 2.5 } else { 1.5 }))
            .with_corner_radius(self.ui_scale.px(8.0));

            if is_selected_card {
                instance = instance.selected();
            }

            cards.push(instance);
        }

        cards
    }

    fn queue_project_chooser_text(&mut self, projects: &ProjectsConfig, selected: usize) {
        let (content_w, content_h) = self.content_size();
        let (offset_x, offset_y) = self.content_offset();

        let scale = self.ui_scale.px(32.0);
        let title_scale = self.ui_scale.px(42.0);
        let left_margin = self.ui_scale.px(60.0);
        let top_margin = self.ui_scale.px(40.0);

        // Title
        self.text_queue.push("PALACE", offset_x + left_margin, offset_y + top_margin, title_scale, [0.8, 0.6, 1.0, 1.0]);

        // Subtitle
        let subtitle = if projects.projects.is_empty() {
            "No projects yet"
        } else {
            "Projects"
        };
        let dim_color = self.text_color_dim();
        self.text_queue.push(subtitle, offset_x + left_margin, offset_y + top_margin + title_scale + 8.0, scale * 0.6, dim_color);

        // Grid for cards
        let grid = CardGrid::new(
            content_w,
            content_h,
            &self.ui_scale,
        ).with_offset(offset_x, offset_y);

        for (i, p) in projects.projects.iter().enumerate() {
            let (x, y) = grid.card_position(i);
            let is_selected = i == selected;

            let text_margin = self.ui_scale.px(16.0);
            let name_scale = self.ui_scale.px(if is_selected { 22.0 } else { 20.0 });

            // Project name
            let name_color = if is_selected {
                self.text_color()
            } else if self.dark_mode {
                [0.85, 0.85, 0.9, 1.0]
            } else {
                [0.3, 0.3, 0.35, 1.0]
            };

            self.text_queue.push_bounded(
                &p.name,
                x + text_margin,
                y + text_margin,
                name_scale,
                name_color,
                grid.card_width - text_margin * 2.0,
                grid.card_height,
            );

            // Description
            let description = if p.description.is_empty() {
                p.path.to_string_lossy().to_string()
            } else {
                p.description.clone()
            };
            let desc_y = y + text_margin + name_scale + self.ui_scale.px(8.0);
            let dim_color = self.text_color_dim();
            self.text_queue.push_bounded(
                &description,
                x + text_margin,
                desc_y,
                self.ui_scale.px(12.0),
                dim_color,
                grid.card_width - text_margin * 2.0,
                self.ui_scale.px(40.0),
            );

            // Languages at bottom of card
            let lang_y = y + grid.card_height - text_margin - self.ui_scale.px(16.0);
            let mut lang_x = x + text_margin;

            for (li, lang) in p.languages.iter().enumerate() {
                if li > 0 {
                    self.text_queue.push(" + ", lang_x, lang_y, self.ui_scale.px(12.0), [0.4, 0.4, 0.45, 1.0]);
                    lang_x += self.ui_scale.px(24.0);
                }

                let lang_color = crate::projects::language_color(lang);
                self.text_queue.push(lang, lang_x, lang_y, self.ui_scale.px(13.0), lang_color);
                lang_x += self.ui_scale.px(lang.len() as f32 * 8.0 + 8.0);
            }

            // Status indicator (top right)
            let status_text = match p.status {
                crate::renderer::ProjectStatus::Unknown => "",
                crate::renderer::ProjectStatus::Building => "BUILDING",
                crate::renderer::ProjectStatus::Error => "ERROR",
                crate::renderer::ProjectStatus::Passing => "PASSING",
                crate::renderer::ProjectStatus::Active => "ACTIVE",
            };

            if !status_text.is_empty() {
                self.text_queue.push(
                    status_text,
                    x + grid.card_width - text_margin - self.ui_scale.px(60.0),
                    y + text_margin,
                    self.ui_scale.px(10.0),
                    p.status.color(),
                );
            }
        }

        // Add help legend
        self.queue_help_legend(&AppState::ProjectChooser { selected_index: selected, show_archived: false });

        // Empty state message
        if projects.projects.is_empty() {
            let center_y = offset_y + content_h / 2.0;
            self.text_queue.push(
                "Launch Palace from a project directory to add it",
                offset_x + left_margin,
                center_y,
                scale * 0.7,
                [0.5, 0.5, 0.6, 1.0],
            );
        }
    }

    // --- Scenario Generator Rendering ---

    /// Build cards for the scenario generator wizard
    pub fn build_generator_cards(&self, generator: &crate::scenario::ScenarioGenerator) -> Vec<CardInstance> {
        use crate::scenario::{GeneratorState, StartOption};

        let (content_w, content_h) = self.content_size();
        let (offset_x, offset_y) = self.content_offset();

        let mut cards = Vec::new();
        let padding = self.ui_scale.px(24.0);
        let card_height = self.ui_scale.px(60.0);
        let card_gap = self.ui_scale.px(12.0);
        let card_width = content_w.min(self.ui_scale.px(400.0)) - padding * 2.0;
        let start_x = offset_x + (content_w - card_width) / 2.0;
        let start_y = offset_y + self.ui_scale.px(150.0);

        match generator.current_state() {
            GeneratorState::Start { selected } => {
                for (i, opt) in StartOption::all().iter().enumerate() {
                    let is_selected = i == *selected;
                    let y = start_y + i as f32 * (card_height + card_gap);

                    let bg_color = if is_selected {
                        [0.15, 0.2, 0.3, 1.0]
                    } else {
                        [0.1, 0.1, 0.15, 0.8]
                    };

                    let card = CardInstance::new(start_x, y, card_width, card_height, bg_color)
                        .with_corner_radius(self.ui_scale.px(12.0))
                        .with_border_width(if is_selected { self.ui_scale.px(2.0) } else { 0.0 })
                        .with_border_color([0.4, 0.6, 1.0, 0.8]);

                    cards.push(if is_selected { card.selected() } else { card });
                }
            }
            GeneratorState::DescriptionInput { .. } | GeneratorState::Feedback { .. } => {
                // Text input card
                let input_height = self.ui_scale.px(120.0);
                cards.push(
                    CardInstance::new(start_x, start_y, card_width, input_height, [0.08, 0.08, 0.12, 1.0])
                        .with_corner_radius(self.ui_scale.px(8.0))
                        .with_border_width(self.ui_scale.px(2.0))
                        .with_border_color([0.4, 0.5, 0.7, 0.6])
                );
            }
            GeneratorState::Survey { current_question, selected_option, selected_options, .. } => {
                for (i, _opt) in current_question.options.iter().enumerate() {
                    let is_selected = i == *selected_option;
                    let is_checked = selected_options.contains(&i);
                    let y = start_y + i as f32 * (card_height + card_gap);

                    let bg_color = if is_selected {
                        [0.15, 0.2, 0.3, 1.0]
                    } else if is_checked {
                        [0.1, 0.15, 0.2, 1.0]
                    } else {
                        [0.1, 0.1, 0.15, 0.8]
                    };

                    let card = CardInstance::new(start_x, y, card_width, card_height, bg_color)
                        .with_corner_radius(self.ui_scale.px(12.0))
                        .with_border_width(if is_selected || is_checked { self.ui_scale.px(2.0) } else { 0.0 })
                        .with_border_color(if is_checked { [0.3, 0.7, 0.3, 0.8] } else { [0.4, 0.6, 1.0, 0.8] });

                    cards.push(if is_selected { card.selected() } else { card });
                }
            }
            GeneratorState::Preview { .. } => {
                // Preview card (full width)
                let preview_height = content_h - self.ui_scale.px(200.0);
                cards.push(
                    CardInstance::new(start_x, start_y, card_width, preview_height, [0.06, 0.06, 0.1, 1.0])
                        .with_corner_radius(self.ui_scale.px(8.0))
                        .with_border_width(self.ui_scale.px(1.0))
                        .with_border_color([0.3, 0.3, 0.4, 0.5])
                );
            }
            GeneratorState::Enriching { .. } | GeneratorState::Generating { .. } => {
                // Loading spinner area
                cards.push(
                    CardInstance::new(start_x + card_width / 3.0, start_y, card_width / 3.0, self.ui_scale.px(80.0), [0.1, 0.1, 0.15, 0.6])
                        .with_corner_radius(self.ui_scale.px(12.0))
                );
            }
            _ => {}
        }

        cards
    }

    /// Queue text for the scenario generator wizard
    pub fn queue_generator_text(&mut self, generator: &crate::scenario::ScenarioGenerator) {
        use crate::scenario::{GeneratorState, StartOption};

        let (content_w, content_h) = self.content_size();
        let (offset_x, offset_y) = self.content_offset();

        let padding = self.ui_scale.px(24.0);
        let card_height = self.ui_scale.px(60.0);
        let card_gap = self.ui_scale.px(12.0);
        let card_width = content_w.min(self.ui_scale.px(400.0)) - padding * 2.0;
        let start_x = offset_x + (content_w - card_width) / 2.0;
        let start_y = offset_y + self.ui_scale.px(150.0);

        let title_scale = self.ui_scale.px(32.0);
        let label_scale = self.ui_scale.px(18.0);
        let desc_scale = self.ui_scale.px(14.0);
        let text_color = self.text_color();
        let dim_color = self.text_color_dim();

        // Title
        self.text_queue.push(
            "SCENARIO GENERATOR",
            offset_x + (content_w - self.ui_scale.px(280.0)) / 2.0,
            offset_y + self.ui_scale.px(60.0),
            title_scale,
            [0.7, 0.5, 1.0, 1.0],
        );

        match generator.current_state() {
            GeneratorState::Start { selected } => {
                // Subtitle
                self.text_queue.push(
                    "How would you like to create a scenario?",
                    start_x,
                    offset_y + self.ui_scale.px(110.0),
                    desc_scale,
                    dim_color,
                );

                // Options
                for (i, opt) in StartOption::all().iter().enumerate() {
                    let is_selected = i == *selected;
                    let y = start_y + i as f32 * (card_height + card_gap);

                    self.text_queue.push(
                        opt.label(),
                        start_x + self.ui_scale.px(16.0),
                        y + self.ui_scale.px(12.0),
                        label_scale,
                        if is_selected { text_color } else { dim_color },
                    );
                    self.text_queue.push(
                        opt.description(),
                        start_x + self.ui_scale.px(16.0),
                        y + self.ui_scale.px(34.0),
                        desc_scale,
                        [0.5, 0.5, 0.6, 0.8],
                    );
                }
            }
            GeneratorState::DescriptionInput { text, cursor } => {
                // Instructions
                self.text_queue.push(
                    "Describe what you want to build:",
                    start_x,
                    offset_y + self.ui_scale.px(110.0),
                    desc_scale,
                    dim_color,
                );

                // Input text with cursor
                let display_text = if text.is_empty() {
                    "Type your description here..."
                } else {
                    text.as_str()
                };
                let input_color = if text.is_empty() { [0.4, 0.4, 0.5, 0.6] } else { text_color };
                self.text_queue.push_bounded(
                    display_text,
                    start_x + self.ui_scale.px(12.0),
                    start_y + self.ui_scale.px(12.0),
                    label_scale,
                    input_color,
                    card_width - self.ui_scale.px(24.0),
                    self.ui_scale.px(96.0),
                );

                // Hint
                self.text_queue.push(
                    "Press Enter to submit, Esc to go back",
                    start_x,
                    start_y + self.ui_scale.px(140.0),
                    desc_scale,
                    [0.5, 0.5, 0.6, 0.6],
                );
            }
            GeneratorState::Enriching { original } => {
                self.text_queue.push(
                    "Expanding your description...",
                    start_x,
                    start_y,
                    label_scale,
                    [0.6, 0.8, 1.0, 1.0],
                );
                self.text_queue.push_bounded(
                    original,
                    start_x,
                    start_y + self.ui_scale.px(40.0),
                    desc_scale,
                    dim_color,
                    card_width,
                    self.ui_scale.px(100.0),
                );
            }
            GeneratorState::Survey { current_question, selected_option, selected_options, .. } => {
                // Question
                self.text_queue.push_bounded(
                    &current_question.question,
                    start_x,
                    offset_y + self.ui_scale.px(100.0),
                    label_scale,
                    text_color,
                    card_width,
                    self.ui_scale.px(50.0),
                );

                // Context (if any)
                if let Some(context) = &current_question.context {
                    self.text_queue.push_bounded(
                        context,
                        start_x,
                        offset_y + self.ui_scale.px(130.0),
                        desc_scale,
                        [0.5, 0.6, 0.7, 0.8],
                        card_width,
                        self.ui_scale.px(40.0),
                    );
                }

                // Options
                for (i, opt) in current_question.options.iter().enumerate() {
                    let is_selected = i == *selected_option;
                    let is_checked = selected_options.contains(&i);
                    let y = start_y + i as f32 * (card_height + card_gap);

                    // Checkbox indicator for multi-select
                    let prefix = if current_question.multi_select {
                        if is_checked { "☑ " } else { "☐ " }
                    } else {
                        ""
                    };

                    self.text_queue.push(
                        &format!("{}{}", prefix, opt.label),
                        start_x + self.ui_scale.px(16.0),
                        y + self.ui_scale.px(12.0),
                        label_scale,
                        if is_selected || is_checked { text_color } else { dim_color },
                    );

                    if let Some(desc) = &opt.description {
                        self.text_queue.push(
                            desc,
                            start_x + self.ui_scale.px(16.0),
                            y + self.ui_scale.px(34.0),
                            desc_scale,
                            [0.5, 0.5, 0.6, 0.8],
                        );
                    }
                }
            }
            GeneratorState::Generating { .. } => {
                self.text_queue.push(
                    "Generating scenario...",
                    start_x + self.ui_scale.px(50.0),
                    start_y + self.ui_scale.px(30.0),
                    label_scale,
                    [0.6, 0.8, 1.0, 1.0],
                );
            }
            GeneratorState::Preview { scenario, scroll_offset } => {
                // Preview title
                self.text_queue.push(
                    "Generated Scenario Preview",
                    start_x,
                    offset_y + self.ui_scale.px(110.0),
                    desc_scale,
                    dim_color,
                );

                // YAML preview (simple version)
                let yaml = format!(
                    "scenario:\n  name: \"{}\"\n\ngoals:\n{}\n\nproject:\n  path: {}",
                    scenario.scenario.name,
                    scenario.goals.iter()
                        .map(|g| format!("  - \"{}\"", g))
                        .collect::<Vec<_>>()
                        .join("\n"),
                    scenario.project.path,
                );

                self.text_queue.push_bounded(
                    &yaml,
                    start_x + self.ui_scale.px(12.0),
                    start_y + self.ui_scale.px(12.0) - scroll_offset,
                    desc_scale,
                    text_color,
                    card_width - self.ui_scale.px(24.0),
                    content_h - self.ui_scale.px(250.0),
                );

                // Actions hint
                self.text_queue.push(
                    "[A] Accept  [B] Feedback  [Esc] Cancel",
                    start_x,
                    offset_y + content_h - self.ui_scale.px(50.0),
                    desc_scale,
                    [0.5, 0.5, 0.6, 0.7],
                );
            }
            GeneratorState::Feedback { feedback_text, .. } => {
                self.text_queue.push(
                    "What would you like to change?",
                    start_x,
                    offset_y + self.ui_scale.px(110.0),
                    desc_scale,
                    dim_color,
                );

                let display_text = if feedback_text.is_empty() {
                    "Enter your feedback..."
                } else {
                    feedback_text.as_str()
                };
                let input_color = if feedback_text.is_empty() { [0.4, 0.4, 0.5, 0.6] } else { text_color };
                self.text_queue.push_bounded(
                    display_text,
                    start_x + self.ui_scale.px(12.0),
                    start_y + self.ui_scale.px(12.0),
                    label_scale,
                    input_color,
                    card_width - self.ui_scale.px(24.0),
                    self.ui_scale.px(96.0),
                );
            }
            GeneratorState::ForkPicker { files, selected } => {
                self.text_queue.push(
                    "Select scenario to fork:",
                    start_x,
                    offset_y + self.ui_scale.px(110.0),
                    desc_scale,
                    dim_color,
                );

                if files.is_empty() {
                    self.text_queue.push(
                        "No scenario files found",
                        start_x,
                        start_y,
                        label_scale,
                        [0.6, 0.4, 0.4, 0.8],
                    );
                } else {
                    for (i, path) in files.iter().enumerate() {
                        let is_selected = i == *selected;
                        let y = start_y + i as f32 * (card_height + card_gap);
                        let name = path.file_name()
                            .and_then(|n| n.to_str())
                            .unwrap_or("unknown");

                        self.text_queue.push(
                            name,
                            start_x + self.ui_scale.px(16.0),
                            y + self.ui_scale.px(20.0),
                            label_scale,
                            if is_selected { text_color } else { dim_color },
                        );
                    }
                }
            }
            GeneratorState::Done { output_path } => {
                self.text_queue.push(
                    "✓ Scenario saved!",
                    start_x + self.ui_scale.px(50.0),
                    start_y,
                    label_scale,
                    [0.3, 0.8, 0.3, 1.0],
                );
                self.text_queue.push(
                    &*output_path.to_string_lossy(),
                    start_x,
                    start_y + self.ui_scale.px(40.0),
                    desc_scale,
                    dim_color,
                );
            }
        }
    }

    // --- Scenario Diff Viewer Rendering ---

    /// Build cards for the diff viewer (side-by-side diff with synchronized scrolling)
    pub fn build_diff_viewer_cards(
        &self,
        viewer: &crate::scenario::DiffViewer,
        feedback_mode: bool,
        feedback_options: &[crate::scenario::FeedbackOption],
        selected_feedback: usize,
        custom_feedback: Option<&str>,
        _custom_cursor: usize,
    ) -> Vec<CardInstance> {
        use crate::scenario::DiffKind;

        let (content_w, content_h) = self.content_size();
        let (offset_x, offset_y) = self.content_offset();

        let mut cards = Vec::new();

        let padding = self.ui_scale.px(16.0);
        let header_height = self.ui_scale.px(60.0);
        let col_width = (content_w - padding * 3.0) / 2.0;
        let line_height = viewer.line_height;

        // Left column background (original)
        cards.push(
            CardInstance::new(offset_x + padding, offset_y + header_height, col_width, content_h - header_height - self.ui_scale.px(50.0), [0.06, 0.06, 0.08, 1.0])
                .with_corner_radius(self.ui_scale.px(8.0))
        );

        // Right column background (corrected)
        cards.push(
            CardInstance::new(offset_x + padding * 2.0 + col_width, offset_y + header_height, col_width, content_h - header_height - self.ui_scale.px(50.0), [0.06, 0.06, 0.08, 1.0])
                .with_corner_radius(self.ui_scale.px(8.0))
        );

        // Render diff chunks
        let start_y = offset_y + header_height + padding;
        let visible_height = content_h - header_height - self.ui_scale.px(100.0);

        for (i, chunk) in viewer.chunks.iter().enumerate() {
            let chunk_y = viewer.chunk_y(i) - viewer.scroll_offset;

            // Skip if off-screen
            if chunk_y + chunk.height_lines() as f32 * line_height < 0.0
                || chunk_y > visible_height
            {
                continue;
            }

            let actual_y = start_y + chunk_y;
            let chunk_height = if viewer.collapsed.contains(&i) {
                viewer.collapsed_height()
            } else {
                chunk.height_lines() as f32 * line_height
            };

            // Skip if completely off-screen
            if actual_y > offset_y + content_h || actual_y + chunk_height < offset_y + header_height {
                continue;
            }

            let is_focused = i == viewer.cursor_chunk;

            // Background color based on diff kind
            let bg_color = match chunk.kind {
                DiffKind::Context => [0.0, 0.0, 0.0, 0.0],
                DiffKind::Added => [0.1, 0.3, 0.1, 0.4],
                DiffKind::Removed => [0.3, 0.1, 0.1, 0.4],
                DiffKind::Modified => [0.3, 0.3, 0.1, 0.4],
            };

            // Collapsed marker bar
            if viewer.collapsed.contains(&i) {
                let marker_color = chunk.kind.marker_color();
                // Left marker
                cards.push(
                    CardInstance::new(offset_x + padding, actual_y, col_width, viewer.collapsed_height(), marker_color)
                        .with_corner_radius(2.0)
                );
                // Right marker
                cards.push(
                    CardInstance::new(offset_x + padding * 2.0 + col_width, actual_y, col_width, viewer.collapsed_height(), marker_color)
                        .with_corner_radius(2.0)
                );
            } else {
                // Left side (original)
                if chunk.has_original() && bg_color[3] > 0.0 {
                    cards.push(
                        CardInstance::new(offset_x + padding, actual_y, col_width, chunk_height, bg_color)
                            .with_corner_radius(4.0)
                    );
                }

                // Right side (corrected)
                if chunk.has_corrected() && bg_color[3] > 0.0 {
                    cards.push(
                        CardInstance::new(offset_x + padding * 2.0 + col_width, actual_y, col_width, chunk_height, bg_color)
                            .with_corner_radius(4.0)
                    );
                }

                // Marker bar for added (right only has content)
                if chunk.kind == DiffKind::Added {
                    cards.push(
                        CardInstance::new(offset_x + padding + col_width / 2.0 - self.ui_scale.px(20.0), actual_y + chunk_height / 2.0 - 2.0, self.ui_scale.px(40.0), 4.0, [0.3, 0.15, 0.15, 0.8])
                            .with_corner_radius(2.0)
                    );
                }

                // Marker bar for removed (left only has content)
                if chunk.kind == DiffKind::Removed {
                    cards.push(
                        CardInstance::new(offset_x + padding * 2.0 + col_width + col_width / 2.0 - self.ui_scale.px(20.0), actual_y + chunk_height / 2.0 - 2.0, self.ui_scale.px(40.0), 4.0, [0.15, 0.3, 0.15, 0.8])
                            .with_corner_radius(2.0)
                    );
                }
            }

            // Focus indicator
            if is_focused {
                cards.push(
                    CardInstance::new(offset_x + padding - 4.0, actual_y - 2.0, col_width * 2.0 + padding + 8.0, chunk_height + 4.0, [0.3, 0.5, 0.9, 0.0])
                        .with_corner_radius(self.ui_scale.px(6.0))
                        .with_border_width(self.ui_scale.px(2.0))
                        .with_border_color([0.4, 0.6, 1.0, 0.8])
                        .selected()
                );
            }
        }

        // Feedback modal overlay
        if feedback_mode {
            // Dim background
            cards.push(
                CardInstance::new(offset_x, offset_y, content_w, content_h, [0.0, 0.0, 0.0, 0.6])
            );

            // Modal card
            let modal_width = content_w.min(self.ui_scale.px(500.0));
            let option_height = self.ui_scale.px(50.0);
            let option_gap = self.ui_scale.px(8.0);
            let modal_padding = self.ui_scale.px(24.0);
            let header_h = self.ui_scale.px(80.0);

            // Calculate modal height based on content
            let options_height = if custom_feedback.is_some() {
                self.ui_scale.px(120.0) // Custom input area
            } else {
                feedback_options.len() as f32 * (option_height + option_gap) - option_gap
            };
            let modal_height = header_h + options_height + modal_padding * 2.0;
            let modal_x = offset_x + (content_w - modal_width) / 2.0;
            let modal_y = offset_y + (content_h - modal_height) / 2.0;

            // Modal background
            cards.push(
                CardInstance::new(modal_x, modal_y, modal_width, modal_height, [0.1, 0.1, 0.15, 0.98])
                    .with_corner_radius(self.ui_scale.px(16.0))
                    .with_border_width(self.ui_scale.px(2.0))
                    .with_border_color([0.4, 0.5, 0.7, 0.6])
            );

            if custom_feedback.is_some() {
                // Custom feedback text input area
                cards.push(
                    CardInstance::new(
                        modal_x + modal_padding,
                        modal_y + header_h,
                        modal_width - modal_padding * 2.0,
                        self.ui_scale.px(100.0),
                        [0.08, 0.08, 0.12, 1.0],
                    )
                    .with_corner_radius(self.ui_scale.px(8.0))
                    .with_border_width(self.ui_scale.px(2.0))
                    .with_border_color([0.3, 0.5, 0.8, 0.8])
                );
            } else {
                // Feedback option cards
                let options_start_y = modal_y + header_h;
                for (i, _option) in feedback_options.iter().enumerate() {
                    let is_selected = i == selected_feedback;
                    let opt_y = options_start_y + i as f32 * (option_height + option_gap);

                    let bg_color = if is_selected {
                        [0.15, 0.2, 0.3, 1.0]
                    } else {
                        [0.08, 0.08, 0.12, 0.8]
                    };

                    let mut card = CardInstance::new(
                        modal_x + modal_padding,
                        opt_y,
                        modal_width - modal_padding * 2.0,
                        option_height,
                        bg_color,
                    )
                    .with_corner_radius(self.ui_scale.px(8.0));

                    if is_selected {
                        card = card
                            .with_border_width(self.ui_scale.px(2.0))
                            .with_border_color([0.4, 0.6, 1.0, 0.8])
                            .selected();
                    }

                    cards.push(card);
                }
            }
        }

        cards
    }

    /// Queue text for the diff viewer
    pub fn queue_diff_viewer_text(
        &mut self,
        viewer: &crate::scenario::DiffViewer,
        feedback_mode: bool,
        feedback_options: &[crate::scenario::FeedbackOption],
        selected_feedback: usize,
        custom_feedback: Option<&str>,
        custom_cursor: usize,
    ) {
        use crate::scenario::DiffKind;

        let (content_w, content_h) = self.content_size();
        let (offset_x, offset_y) = self.content_offset();

        let padding = self.ui_scale.px(16.0);
        let header_height = self.ui_scale.px(60.0);
        let col_width = (content_w - padding * 3.0) / 2.0;
        let title_scale = self.ui_scale.px(24.0);
        let code_scale = self.ui_scale.px(13.0);
        let desc_scale = self.ui_scale.px(12.0);
        let text_color = self.text_color();
        let dim_color = self.text_color_dim();
        let line_height = viewer.line_height;

        // Title
        self.text_queue.push(
            "SCENARIO CORRECTOR",
            offset_x + (content_w - self.ui_scale.px(240.0)) / 2.0,
            offset_y + self.ui_scale.px(16.0),
            title_scale,
            [0.8, 0.5, 0.5, 1.0],
        );

        // Column headers
        self.text_queue.push("ORIGINAL", offset_x + padding + self.ui_scale.px(8.0), offset_y + self.ui_scale.px(44.0), desc_scale, [0.6, 0.4, 0.4, 1.0]);
        self.text_queue.push("CORRECTED", offset_x + padding * 2.0 + col_width + self.ui_scale.px(8.0), offset_y + self.ui_scale.px(44.0), desc_scale, [0.4, 0.6, 0.4, 1.0]);

        // Change counter
        let change_info = if let Some(idx) = viewer.current_change_index() {
            format!("Change {} of {}", idx, viewer.change_count())
        } else {
            format!("{} changes", viewer.change_count())
        };
        self.text_queue.push(&change_info, offset_x + content_w - self.ui_scale.px(120.0), offset_y + self.ui_scale.px(20.0), desc_scale, dim_color);

        // Iteration badge
        self.text_queue.push(&format!("Iteration {}", viewer.iteration), offset_x + content_w - self.ui_scale.px(120.0), offset_y + self.ui_scale.px(36.0), desc_scale, [0.5, 0.5, 0.7, 0.8]);

        // Render chunk text
        let start_y = offset_y + header_height + padding;
        let visible_height = content_h - header_height - self.ui_scale.px(100.0);

        for (i, chunk) in viewer.chunks.iter().enumerate() {
            let chunk_y = viewer.chunk_y(i) - viewer.scroll_offset;

            // Skip if off-screen
            if chunk_y + chunk.height_lines() as f32 * line_height < 0.0
                || chunk_y > visible_height
            {
                continue;
            }

            let actual_y = start_y + chunk_y;

            // Skip collapsed chunks (just show marker)
            if viewer.collapsed.contains(&i) {
                let kind_label = match chunk.kind {
                    DiffKind::Context => "context",
                    DiffKind::Added => "added",
                    DiffKind::Removed => "removed",
                    DiffKind::Modified => "modified",
                };
                let lines_info = format!("{} ({} lines)", kind_label, chunk.height_lines());
                self.text_queue.push(&lines_info, offset_x + content_w / 2.0 - self.ui_scale.px(40.0), actual_y, desc_scale, [0.5, 0.5, 0.5, 0.7]);
                continue;
            }

            // Original lines (left column)
            for (j, line) in chunk.original_lines.iter().enumerate() {
                let line_y = actual_y + j as f32 * line_height;
                if line_y > offset_y + content_h - self.ui_scale.px(50.0) {
                    break;
                }
                self.text_queue.push_bounded(
                    line,
                    offset_x + padding + self.ui_scale.px(4.0),
                    line_y,
                    code_scale,
                    text_color,
                    col_width - self.ui_scale.px(8.0),
                    line_height,
                );
            }

            // Corrected lines (right column)
            for (j, line) in chunk.corrected_lines.iter().enumerate() {
                let line_y = actual_y + j as f32 * line_height;
                if line_y > offset_y + content_h - self.ui_scale.px(50.0) {
                    break;
                }
                self.text_queue.push_bounded(
                    line,
                    offset_x + padding * 2.0 + col_width + self.ui_scale.px(4.0),
                    line_y,
                    code_scale,
                    text_color,
                    col_width - self.ui_scale.px(8.0),
                    line_height,
                );
            }

            // Explanation tooltip for focused chunk
            if i == viewer.cursor_chunk {
                if let Some(explanation) = &chunk.explanation {
                    let tooltip_y = offset_y + content_h - self.ui_scale.px(80.0);
                    self.text_queue.push_bounded(
                        explanation,
                        offset_x + padding,
                        tooltip_y,
                        desc_scale,
                        [0.7, 0.7, 0.8, 0.9],
                        content_w - padding * 2.0,
                        self.ui_scale.px(40.0),
                    );
                }
            }
        }

        // Overall explanation
        if !viewer.explanation.is_empty() && !feedback_mode {
            self.text_queue.push_bounded(
                &viewer.explanation,
                offset_x + padding,
                offset_y + content_h - self.ui_scale.px(45.0),
                desc_scale,
                [0.6, 0.6, 0.7, 0.8],
                content_w - padding * 2.0 - self.ui_scale.px(200.0),
                self.ui_scale.px(20.0),
            );
        }

        // Controls hint (changes based on mode)
        if !feedback_mode {
            let controls = "[↑↓] Navigate  [N/P] Next/Prev Change  [Enter] Collapse  [A] Accept  [F] Feedback";
            self.text_queue.push(controls, offset_x + content_w - self.ui_scale.px(450.0), offset_y + content_h - self.ui_scale.px(25.0), desc_scale, [0.4, 0.4, 0.5, 0.7]);
        }

        // Feedback modal text
        if feedback_mode {
            let modal_width = content_w.min(self.ui_scale.px(500.0));
            let option_height = self.ui_scale.px(50.0);
            let option_gap = self.ui_scale.px(8.0);
            let modal_padding = self.ui_scale.px(24.0);
            let header_h = self.ui_scale.px(80.0);

            let options_height = if custom_feedback.is_some() {
                self.ui_scale.px(120.0)
            } else {
                feedback_options.len() as f32 * (option_height + option_gap) - option_gap
            };
            let modal_height = header_h + options_height + modal_padding * 2.0;
            let modal_x = offset_x + (content_w - modal_width) / 2.0;
            let modal_y = offset_y + (content_h - modal_height) / 2.0;

            let label_scale = self.ui_scale.px(16.0);

            // Modal title
            self.text_queue.push(
                "PROVIDE FEEDBACK",
                modal_x + modal_padding,
                modal_y + self.ui_scale.px(20.0),
                title_scale,
                [0.8, 0.6, 0.5, 1.0],
            );

            // Chunk info
            if let Some(chunk) = viewer.chunks.get(viewer.cursor_chunk) {
                let chunk_info = format!("Chunk {} of {}: {:?}", viewer.cursor_chunk + 1, viewer.chunks.len(), chunk.kind);
                self.text_queue.push(
                    &chunk_info,
                    modal_x + modal_padding,
                    modal_y + self.ui_scale.px(50.0),
                    desc_scale,
                    dim_color,
                );
            }

            if let Some(custom_text) = custom_feedback {
                // Custom feedback input mode
                self.text_queue.push(
                    "Enter your feedback:",
                    modal_x + modal_padding,
                    modal_y + header_h - self.ui_scale.px(20.0),
                    desc_scale,
                    dim_color,
                );

                // Show the custom text with cursor
                let display_text = if custom_text.is_empty() {
                    "Type your feedback here..."
                } else {
                    custom_text
                };
                let text_color = if custom_text.is_empty() {
                    [0.4, 0.4, 0.5, 0.5]
                } else {
                    text_color
                };
                self.text_queue.push_bounded(
                    display_text,
                    modal_x + modal_padding + self.ui_scale.px(8.0),
                    modal_y + header_h + self.ui_scale.px(8.0),
                    code_scale,
                    text_color,
                    modal_width - modal_padding * 2.0 - self.ui_scale.px(16.0),
                    self.ui_scale.px(80.0),
                );

                // Cursor indicator (simple block)
                if !custom_text.is_empty() && custom_cursor <= custom_text.len() {
                    // Approximate cursor position
                    let char_width = self.ui_scale.px(8.0);
                    let cursor_x = modal_x + modal_padding + self.ui_scale.px(8.0) + custom_cursor as f32 * char_width;
                    self.text_queue.push("│", cursor_x, modal_y + header_h + self.ui_scale.px(8.0), code_scale, [0.4, 0.6, 1.0, 1.0]);
                }

                // Controls
                self.text_queue.push(
                    "[Enter] Submit  [Esc] Cancel",
                    modal_x + modal_padding,
                    modal_y + modal_height - self.ui_scale.px(30.0),
                    desc_scale,
                    [0.4, 0.4, 0.5, 0.7],
                );
            } else {
                // Feedback options mode
                self.text_queue.push(
                    "What's wrong with this correction?",
                    modal_x + modal_padding,
                    modal_y + header_h - self.ui_scale.px(20.0),
                    desc_scale,
                    dim_color,
                );

                let options_start_y = modal_y + header_h;
                for (i, option) in feedback_options.iter().enumerate() {
                    let is_selected = i == selected_feedback;
                    let opt_y = options_start_y + i as f32 * (option_height + option_gap);

                    let color = if is_selected { text_color } else { dim_color };
                    self.text_queue.push(
                        &option.label,
                        modal_x + modal_padding + self.ui_scale.px(16.0),
                        opt_y + self.ui_scale.px(16.0),
                        label_scale,
                        color,
                    );
                }

                // Controls
                self.text_queue.push(
                    "[↑↓] Navigate  [A/Enter] Select  [B/Esc] Cancel",
                    modal_x + modal_padding,
                    modal_y + modal_height - self.ui_scale.px(30.0),
                    desc_scale,
                    [0.4, 0.4, 0.5, 0.7],
                );
            }
        }
    }

    fn queue_project_view_text(
        &mut self,
        project_path: &std::path::Path,
        selected_action: usize,
        context_bounds: Option<(f32, f32, f32, f32)>,
        menu_bounds: Option<(f32, f32, f32, f32)>,
    ) {
        use crate::state::ProjectAction;

        let (_content_w, _content_h) = self.content_size();
        let (offset_x, offset_y) = self.content_offset();

        let project_name = project_path
            .file_name()
            .and_then(|n| n.to_str())
            .unwrap_or("Unknown");
        let path_str = project_path.to_string_lossy();

        // Context widget bounds (title + path)
        let (ctx_x, ctx_y, ctx_w, ctx_h) = if let Some((px, py, pw, ph)) = context_bounds {
            (px, py, pw, ph)
        } else {
            let title_scale = self.ui_scale.px(36.0);
            let left_margin = self.ui_scale.px(60.0);
            let top_margin = self.ui_scale.px(40.0);
            (offset_x + left_margin, offset_y + top_margin, self.ui_scale.px(300.0), title_scale + self.ui_scale.px(30.0))
        };

        // Calculate title text size based on context panel height
        let base_title_scale = self.ui_scale.px(36.0);
        let title_scale = if context_bounds.is_some() {
            // Scale title to fit context panel - roughly 40% of height for title
            (ctx_h * 0.4).clamp(self.ui_scale.px(14.0), self.ui_scale.px(48.0))
        } else {
            base_title_scale
        };
        let subtitle_scale = (title_scale * 0.4).clamp(self.ui_scale.px(10.0), self.ui_scale.px(18.0));
        let padding = (ctx_w * 0.02).max(2.0).min(self.ui_scale.px(8.0));

        // Title
        self.text_queue.push(project_name, ctx_x + padding, ctx_y + padding, title_scale, [1.0, 1.0, 1.0, 1.0]);

        // Path subtitle
        self.text_queue.push(
            path_str.as_ref(),
            ctx_x + padding,
            ctx_y + padding + title_scale + padding,
            subtitle_scale,
            [0.4, 0.4, 0.5, 1.0],
        );

        // Menu bounds for action labels
        let actions = ProjectAction::all();
        let action_count = actions.len() as f32;
        let (menu_x, menu_y, menu_w, menu_h) = if let Some((px, py, pw, ph)) = menu_bounds {
            (px, py, pw, ph)
        } else {
            let left_margin = self.ui_scale.px(60.0);
            let top_margin = self.ui_scale.px(40.0);
            let menu_start_y = offset_y + top_margin + base_title_scale + self.ui_scale.px(60.0);
            let card_height = self.ui_scale.px(70.0);
            let card_gap = self.ui_scale.px(16.0);
            let menu_height = action_count * (card_height + card_gap) - card_gap;
            (offset_x + left_margin, menu_start_y, _content_w - left_margin * 2.0, menu_height)
        };

        // Calculate card dimensions dynamically (must match build_action_cards)
        let menu_padding = (menu_w * 0.01).max(2.0).min(self.ui_scale.px(5.0));
        let total_gap_space = menu_h * 0.12;
        let card_gap = total_gap_space / (action_count - 1.0).max(1.0);
        let card_height = (menu_h - menu_padding * 2.0 - card_gap * (action_count - 1.0)) / action_count;

        // Scale text sizes based on card height
        let size_factor = (card_height / self.ui_scale.px(70.0)).clamp(0.3, 2.0);
        let label_scale = self.ui_scale.px(22.0) * size_factor;
        let desc_scale = self.ui_scale.px(12.0) * size_factor;
        let text_margin = self.ui_scale.px(20.0) * size_factor;

        // Action labels and descriptions
        for (i, action) in actions.iter().enumerate() {
            let y = menu_y + menu_padding + i as f32 * (card_height + card_gap);
            let is_selected = i == selected_action;
            let label_color = if is_selected {
                [1.0, 1.0, 1.0, 1.0]
            } else {
                [0.75, 0.75, 0.8, 1.0]
            };

            self.text_queue.push(
                action.label(),
                menu_x + menu_padding + text_margin,
                y + text_margin,
                label_scale,
                label_color,
            );

            self.text_queue.push(
                action.description(),
                menu_x + menu_padding + text_margin,
                y + text_margin + label_scale * 1.2,
                desc_scale,
                [0.45, 0.45, 0.5, 1.0],
            );
        }

        // Add help legend
        let help_state = AppState::ProjectView {
            project_path: project_path.to_path_buf(),
            selected_action,
        };
        self.queue_help_legend(&help_state);
    }

    /// exec_status: Optional (executing_card_ids, task_statuses) for quest log status indicators
    /// card_scroll_offset: Vertical scroll offset for the card grid (in pixels)
    /// show_add_card: Whether to render the "+" add task card
    fn queue_palace_loop_text(&mut self, cards: &[SuggestionCard], current_tool: Option<&str>, tool_log: &[String], thought_log: &[String], log_scroll: usize, focused_index: usize, hovered_index: Option<usize>, detail_scroll: f32, exec_status: Option<(&[usize], &[TaskStatus])>, card_scroll_offset: f32, show_add_card: bool) {
        let (content_w, content_h) = self.content_size();
        let (offset_x, offset_y) = self.content_offset();

        let title_scale = self.ui_scale.px(42.0);
        let left_margin = self.ui_scale.px(60.0);
        let top_margin = self.ui_scale.px(40.0);

        let grid = CardGrid::for_palace_loop(
            content_w,
            content_h,
            &self.ui_scale,
        ).with_offset(offset_x, offset_y);

        // Title
        self.text_queue.push("PALACE LOOP", offset_x + left_margin, offset_y + top_margin, title_scale, [0.8, 0.6, 1.0, 1.0]);

        // Subtitle with current tool if any
        let subtitle_y = offset_y + top_margin + title_scale + 8.0;
        let subtitle = if let Some(tool) = current_tool {
            format!("Analyzing... {}", tool)
        } else if cards.is_empty() {
            "Gathering suggestions...".to_string()
        } else {
            format!("{} suggestions", cards.len())
        };
        let dim_color = self.text_color_dim();
        self.text_queue.push(&subtitle, offset_x + left_margin, subtitle_y, self.ui_scale.px(18.0), dim_color);

        // Two-column waterfall logs (tools left, thoughts right)
        // Only show logs when cards haven't appeared yet (during analysis phase)
        if cards.is_empty() {
            let _line_height = self.ui_scale.px(16.0);
            let log_scale = self.ui_scale.px(13.0);
            let log_y_start = subtitle_y + self.ui_scale.px(28.0);

            // Calculate max lines to fill available content height
            let available_height = offset_y + content_h - log_y_start - self.ui_scale.px(40.0);

            // Split content: left half for tools, right half for thoughts
            let content_mid = offset_x + content_w / 2.0;
            let right_margin = self.ui_scale.px(40.0);

            // Left column: Tool calls (with proper text measurement for wrapping)
            let left_col_width = content_w / 2.0 - left_margin - self.ui_scale.px(20.0);
            if !tool_log.is_empty() {
                let visible_start = log_scroll as usize;
                let visible_start = visible_start.min(tool_log.len().saturating_sub(1));
                let mut y_pos = 0.0;
                let mut entry_idx = 0;
                for entry in tool_log.iter().skip(visible_start) {
                    if y_pos >= available_height {
                        break;
                    }
                    // Measure how many lines this entry will take
                    let (_w, h, _lines) = crate::renderer::text::measure_text(
                        &mut self.font_system,
                        entry,
                        log_scale,
                        left_col_width,
                    );
                    let alpha = (0.7 - (entry_idx as f32 * 0.03)).max(0.25);
                    let y = log_y_start + y_pos;
                    self.text_queue.push_bounded(entry, offset_x + left_margin, y, log_scale, [0.5, 0.7, 0.9, alpha], left_col_width, h);
                    y_pos += h;
                    entry_idx += 1;
                }
            }

            // Right column: AI thoughts/commentary (with proper text measurement for wrapping)
            let right_col_width = content_w / 2.0 - right_margin;
            if !thought_log.is_empty() {
                let visible_start = log_scroll.min(thought_log.len().saturating_sub(1));
                let mut y_pos = 0.0;
                let mut entry_idx = 0;
                for entry in thought_log.iter().skip(visible_start) {
                    if y_pos >= available_height {
                        break;
                    }
                    // Measure how many lines this entry will take
                    let (_w, h, _lines) = crate::renderer::text::measure_text(
                        &mut self.font_system,
                        entry,
                        log_scale,
                        right_col_width,
                    );
                    let alpha = (0.7 - (entry_idx as f32 * 0.03)).max(0.25);
                    let y = log_y_start + y_pos;
                    self.text_queue.push_bounded(entry, content_mid + self.ui_scale.px(10.0), y, log_scale, [0.7, 0.6, 0.8, alpha], right_col_width, h);
                    y_pos += h;
                    entry_idx += 1;
                }
            }
        }

        // Card text - "flip" behavior: focused cards show description, others show title
        let text_margin = self.ui_scale.px(12.0);
        for (i, card) in cards.iter().enumerate() {
            let (x, base_y) = grid.card_position(i);
            let y = base_y - card_scroll_offset;

            // Skip cards that are completely off-screen
            if y + grid.card_height < offset_y || y > offset_y + content_h {
                continue;
            }

            // Determine if this card is "flipped" (showing back face with description)
            let is_flipped = i == focused_index || hovered_index == Some(i);

            if is_flipped {
                // BACK FACE: Description + command (all scrollable together)
                let desc_color = self.text_color();
                let desc_scale = self.ui_scale.px(13.0);
                let desc_width = grid.card_width - text_margin * 2.0;
                let visible_height = grid.card_height - text_margin * 2.0;

                // Calculate scroll offset for this card (only applies to focused card)
                let scroll_offset = if i == focused_index { detail_scroll } else { 0.0 };

                // Build combined content: description + command
                // Fall back to title if description is empty or whitespace-only
                let description = if card.description.trim().is_empty() {
                    if card.streaming { "Loading..." } else { &card.title }
                } else {
                    &card.description
                };

                let full_text = if let Some(ref cmd) = card.command {
                    if cmd.trim().is_empty() {
                        // No command to display
                        description.to_string()
                    } else {
                        let display_cmd = if cmd.len() > 40 {
                            let truncated: String = cmd.chars().take(37).collect();
                            format!("$ {}...", truncated)
                        } else {
                            format!("$ {}", cmd)
                        };
                        format!("{}\n\n{}", description, display_cmd)
                    }
                } else {
                    description.to_string()
                };

                // Render with scroll offset applied (markdown for styling)
                self.text_queue.push_markdown_scroll(
                    &full_text,
                    x + text_margin,
                    y + text_margin,
                    desc_scale,
                    desc_color,
                    desc_width,
                    visible_height,
                    scroll_offset,
                );
            } else {
                // FRONT FACE: Show title + category
                let title_color = if card.streaming {
                    [0.85, 0.85, 0.9, 1.0]
                } else {
                    self.text_color()
                };
                let display_title = if card.title.is_empty() {
                    if card.streaming { "..." } else { "Untitled" }
                } else {
                    &card.title
                };

                // Title fills most of the card
                let title_height = grid.card_height - text_margin * 2.0 - self.ui_scale.px(24.0);
                self.text_queue.push_bounded(
                    display_title,
                    x + text_margin,
                    y + text_margin,
                    self.ui_scale.px(16.0),
                    title_color,
                    grid.card_width - text_margin * 2.0,
                    title_height,
                );

                // Category badge (bottom left, FRONT FACE ONLY)
                if !card.category.is_empty() {
                    self.text_queue.push(
                        &card.category.to_uppercase(),
                        x + text_margin,
                        y + grid.card_height - text_margin - self.ui_scale.px(10.0),
                        self.ui_scale.px(9.0),
                        card.color(),
                    );
                }
            }

            // Status badge text in bottom-right corner (quest log mode only)
            // Badge rect is rendered by build_suggestion_cards, this adds the text
            if let Some((exec_ids, task_statuses)) = exec_status {
                if let Some(exec_pos) = exec_ids.iter().position(|&id| id == card.id) {
                    // Get status from task_statuses array
                    let status = task_statuses.get(exec_pos).copied().unwrap_or(TaskStatus::Pending);
                    let status_text = status.badge_text();

                    // Position text centered in badge (badge is rendered by card builder)
                    let badge_height = self.ui_scale.px(14.0);
                    let badge_width = self.ui_scale.px(match status_text.len() {
                        0..=4 => 40.0,
                        5..=7 => 55.0,
                        _ => 70.0,
                    });
                    let badge_x = x + grid.card_width - text_margin - badge_width;
                    let badge_y = y + grid.card_height - text_margin - badge_height;

                    // Center text in badge
                    let text_size = self.ui_scale.px(8.0);
                    let text_width = status_text.len() as f32 * self.ui_scale.px(4.5);
                    let text_x = badge_x + (badge_width - text_width) / 2.0;
                    let text_y = badge_y + (badge_height - text_size) / 2.0 - self.ui_scale.px(1.0);

                    // White text on colored badge
                    self.text_queue.push(status_text, text_x, text_y, text_size, [1.0, 1.0, 1.0, 1.0]);
                }
            } else if card.selected {
                // Normal PalaceLoop mode: show selection checkmark
                let indicator_x = x + grid.card_width - text_margin - self.ui_scale.px(16.0);
                let indicator_y = y + grid.card_height - text_margin - self.ui_scale.px(14.0);
                self.text_queue.push("✓", indicator_x, indicator_y, self.ui_scale.px(18.0), [0.2, 1.0, 0.4, 1.0]);
            }
        }

        // "+" card at the end (only when in card view, not during analysis)
        if show_add_card {
            let add_card_index = cards.len();
            let (add_x, add_base_y) = grid.card_position(add_card_index);
            let add_y = add_base_y - card_scroll_offset;

            // Only render text if card is visible
            if add_y + grid.card_height >= offset_y && add_y <= offset_y + content_h {
            let is_focused = focused_index == add_card_index;
            let is_hovered = hovered_index == Some(add_card_index);
            let is_active = is_focused || is_hovered;

            // Large "+" symbol centered in the card
            let plus_scale = self.ui_scale.px(48.0);
            let plus_color = if is_active {
                [0.3, 0.9, 0.4, 1.0] // Green when active
            } else {
                [0.7, 0.7, 0.7, 0.6] // Dim white/gray when not
            };

            // Center the "+"
            let plus_width = plus_scale * 0.6; // Approximate width
            let plus_x = add_x + (grid.card_width - plus_width) / 2.0;
            let plus_y = add_y + (grid.card_height - plus_scale) / 2.0 - self.ui_scale.px(8.0);
            self.text_queue.push("+", plus_x, plus_y, plus_scale, plus_color);

            // "Add task" label below the +
            let label_scale = self.ui_scale.px(12.0);
            let label = "Add task";
            let label_width = label.len() as f32 * self.ui_scale.px(6.0);
            let label_x = add_x + (grid.card_width - label_width) / 2.0;
            let label_y = plus_y + plus_scale + self.ui_scale.px(4.0);
            self.text_queue.push(label, label_x, label_y, label_scale, plus_color);
            }
        }

        // Add help legend
        let help_state = AppState::PalaceLoop {
            project_path: std::path::PathBuf::new(),
            cards: Vec::new(),
            focused_index: 0,
            hovered_index: None,
            generating: false,
            current_tool: None,
            tool_log: Vec::new(),
            thought_log: Vec::new(),
            log_scroll_offset: 0,
            detail_scroll_offset: 0.0,
            detail_max_scroll: 0.0,
            card_scroll_offset: 0.0,
        };
        self.queue_help_legend(&help_state);
    }

    /// Queue text for Executing state - two columns: tools (left), thoughts (right)
    /// Left: colorized timestamp + icon + action, plain summary, colored dot
    /// Right: colorized timestamp, plain commentary text
    fn queue_executing_text(
        &mut self,
        tool_log: &[String],
        thought_log: &[String],
        log_scroll: f32,
        status: &crate::state::ExecutionStatus,
        _executor: crate::state::ExecuteOption,
        tokens_used: u64,
        request_active: bool,
    ) {
        let (content_w, content_h) = self.content_size();
        let (offset_x, offset_y) = self.content_offset();

        let margin = self.ui_scale.px(48.0);
        let title_scale = self.ui_scale.px(28.0);
        let log_scale = self.ui_scale.px(13.0);

        // Title - "EXECUTING" in accent color
        self.text_queue.push(
            "EXECUTING",
            offset_x + margin,
            offset_y + margin,
            title_scale,
            [0.8, 0.6, 1.0, 1.0], // Purple accent like PalaceLoop
        );

        // KITT-style scanner display (top-right, inline with title)
        // 8-segment scanner that sweeps back and forth when active
        let scanner_y = offset_y + margin + self.ui_scale.px(4.0);
        let scanner_segment_w = self.ui_scale.px(12.0);
        let scanner_gap = self.ui_scale.px(2.0);
        let scanner_segments = 8;
        let scanner_total_w = scanner_segments as f32 * (scanner_segment_w + scanner_gap) - scanner_gap;
        let scanner_x = offset_x + content_w - margin - scanner_total_w - self.ui_scale.px(100.0);

        if request_active {
            // Animate: position oscillates from 0 to scanner_segments-1 and back
            let anim_time = self.animation_time();
            let cycle_duration = 0.8; // Full cycle (back and forth) in seconds
            let phase = (anim_time / cycle_duration).fract();
            // Triangle wave: 0->1->0 over phase 0->1
            let position = if phase < 0.5 {
                phase * 2.0 // 0 to 1
            } else {
                2.0 - phase * 2.0 // 1 to 0
            };
            let active_segment = (position * (scanner_segments - 1) as f32) as usize;

            for i in 0..scanner_segments {
                let seg_x = scanner_x + i as f32 * (scanner_segment_w + scanner_gap);
                let distance = (i as i32 - active_segment as i32).abs() as f32;

                // Brightness falls off from active segment (KITT red glow effect)
                let brightness = (1.0 - distance * 0.25).max(0.1);
                let alpha = (1.0 - distance * 0.2).max(0.3);

                // Red color with brightness falloff
                let color = [0.9 * brightness, 0.1 * brightness, 0.1 * brightness, alpha];
                let char = if distance < 0.5 { "█" } else if distance < 1.5 { "▓" } else if distance < 2.5 { "▒" } else { "░" };

                self.text_queue.push(
                    char,
                    seg_x,
                    scanner_y,
                    title_scale * 0.7,
                    color,
                );
            }
        } else {
            // Idle state: dim segments
            for i in 0..scanner_segments {
                let seg_x = scanner_x + i as f32 * (scanner_segment_w + scanner_gap);
                self.text_queue.push(
                    "░",
                    seg_x,
                    scanner_y,
                    title_scale * 0.7,
                    [0.3, 0.1, 0.1, 0.4],
                );
            }
        }

        // Token count next to scanner
        let small_text = self.ui_scale.px(14.0);
        let token_text = if tokens_used > 0 {
            format!("{} tokens", Self::format_number(tokens_used))
        } else {
            "– tokens".to_string()
        };
        let (token_w, _, _) = crate::renderer::text::measure_text(
            &mut self.font_system,
            &token_text,
            small_text,
            content_w,
        );
        self.text_queue.push(
            &token_text,
            offset_x + content_w - margin - token_w,
            scanner_y + self.ui_scale.px(4.0),
            small_text,
            if tokens_used > 0 { [0.6, 0.7, 0.8, 0.9] } else { self.text_color_dim() },
        );

        // Status line
        let status_text = match status {
            crate::state::ExecutionStatus::Pending => "⏳ Waiting to start...".to_string(),
            crate::state::ExecutionStatus::Running { current_card, total_cards } => {
                format!("Running task {}/{}", current_card + 1, total_cards)
            }
            crate::state::ExecutionStatus::Completed => "✅ Completed".to_string(),
            crate::state::ExecutionStatus::Failed(err) => format!("❌ Failed: {}", err),
            crate::state::ExecutionStatus::Cancelled => "⛔ Cancelled".to_string(),
        };

        let status_color = match status {
            crate::state::ExecutionStatus::Completed => [0.2, 0.8, 0.4, 1.0],
            crate::state::ExecutionStatus::Failed(_) => [0.9, 0.3, 0.3, 1.0],
            crate::state::ExecutionStatus::Cancelled => [0.8, 0.6, 0.2, 1.0],
            _ => self.text_color_dim(),
        };

        let subtitle_y = offset_y + margin + title_scale + self.ui_scale.px(8.0);
        self.text_queue.push(
            &status_text,
            offset_x + margin,
            subtitle_y,
            self.ui_scale.px(16.0),
            status_color,
        );

        // Column layout depends on aspect ratio
        // Standard (16:9, 21:9): 2 columns - tools left, thoughts right
        // Ultrawide (32:9): 3 columns - tools left, thoughts center, status/quest right
        let is_ultrawide = self.layout_mode.is_ultrawide();
        let log_start_y = subtitle_y + self.ui_scale.px(28.0);
        let available_height = offset_y + content_h - log_start_y - margin;
        let col_gap = self.ui_scale.px(20.0);
        let plain_color = self.text_color_dim();

        // Calculate column positions based on layout mode
        let (left_col_x, left_col_width, center_col_x, center_col_width, right_col_x, right_col_width) = if is_ultrawide {
            // Ultrawide: 3 equal columns with gaps
            let total_gap = col_gap * 2.0; // 2 gaps between 3 columns
            let col_w = (content_w - margin * 2.0 - total_gap) / 3.0;
            let left_x = offset_x + margin;
            let center_x = offset_x + margin + col_w + col_gap;
            let right_x = offset_x + margin + col_w * 2.0 + col_gap * 2.0;
            (left_x, col_w, center_x, col_w, right_x, col_w)
        } else {
            // Standard: 2 columns split at midpoint
            let content_mid = content_w / 2.0;
            let left_w = content_mid - margin - col_gap / 2.0;
            let right_w = content_mid - margin - col_gap / 2.0;
            let right_x = offset_x + content_mid + col_gap / 2.0;
            // No third column in standard mode (unused values)
            (offset_x + margin, left_w, right_x, right_w, 0.0, 0.0)
        };

        // Left column: Tool calls
        // Format: [HH:MM:SS] 💻 command  summary ●
        // Colorized: timestamp + icon + action | Plain: summary | Colored: dot
        // Uses pixel-based scrolling
        let left_col_width = left_col_width;
        if !tool_log.is_empty() {
            let mut y_offset = 0.0;

            for entry in tool_log.iter() {
                // Check for error prefix
                let (is_error, clean_entry) = if entry.starts_with("ERR:") {
                    (true, &entry[4..])
                } else {
                    (false, entry.as_str())
                };

                // Measure full text height for layout
                let (_w, h, _lines) = crate::renderer::text::measure_text(
                    &mut self.font_system,
                    clean_entry,
                    log_scale,
                    left_col_width,
                );

                // Apply scroll offset
                let y = log_start_y + y_offset - log_scroll;

                // Skip entries above viewport
                if y + h < log_start_y {
                    y_offset += h;
                    continue;
                }
                // Stop rendering entries below viewport
                if y >= log_start_y + available_height {
                    break;
                }

                let alpha = 0.9; // No fade - clarity over aesthetics

                // Parse entry: [HH:MM:SS] icon action  summary dot
                // Split at double-space to separate colored part from plain part
                if let Some(double_space_pos) = clean_entry.find("  ") {
                    let colored_part = &clean_entry[..double_space_pos];
                    let plain_part = &clean_entry[double_space_pos + 2..];

                    // Get rainbow color for timestamp
                    let rainbow = self.temporal_rainbow_color(clean_entry, alpha);

                    // Render colored part (timestamp + icon + action)
                    self.text_queue.push_bounded(
                        colored_part,
                        left_col_x,
                        y,
                        log_scale,
                        rainbow,
                        left_col_width,
                        h,
                    );

                    // Calculate x position after colored part
                    let (colored_w, _, _) = crate::renderer::text::measure_text(
                        &mut self.font_system,
                        &format!("{}  ", colored_part),
                        log_scale,
                        left_col_width,
                    );

                    // Check if there's a status dot at the end
                    // " ●" is 4 bytes: 1 (space) + 3 (● in UTF-8)
                    let (summary, maybe_dot) = if plain_part.ends_with(" ●") {
                        (&plain_part[..plain_part.len() - 4], Some("●"))
                    } else {
                        (plain_part, None)
                    };

                    // Render plain summary (only if there's room after the colored part)
                    let remaining_width = left_col_width - colored_w;
                    if remaining_width > 0.0 {
                        self.text_queue.push_bounded(
                            summary,
                            left_col_x + colored_w,
                            y,
                            log_scale,
                            [plain_color[0], plain_color[1], plain_color[2], alpha],
                            remaining_width,
                            h,
                        );

                        // Render colored dot if present
                        if let Some(dot) = maybe_dot {
                            let (summary_w, _, _) = crate::renderer::text::measure_text(
                                &mut self.font_system,
                                &format!("{} ", summary),
                                log_scale,
                                remaining_width,
                            );
                            let dot_color = if is_error {
                                [0.9, 0.3, 0.3, alpha] // Red
                            } else {
                                [0.3, 0.8, 0.4, alpha] // Green
                            };
                            self.text_queue.push(
                                dot,
                                left_col_x + colored_w + summary_w,
                                y,
                                log_scale,
                                dot_color,
                            );
                        }
                    }
                } else {
                    // No double-space separator, render whole thing with rainbow
                    let rainbow = self.temporal_rainbow_color(clean_entry, alpha);
                    self.text_queue.push_bounded(
                        clean_entry,
                        left_col_x,
                        y,
                        log_scale,
                        rainbow,
                        left_col_width,
                        h,
                    );
                }

                y_offset += h;
            }
        }

        // Center column (standard: right, ultrawide: center): Thoughts/commentary
        // Format: [HH:MM:SS] text
        // Colorized: timestamp | Markdown: commentary
        // Uses pixel-based scrolling - offset all entries by scroll amount
        let thought_col_width = center_col_width;
        if !thought_log.is_empty() {
            let mut y_offset = 0.0;
            let x = center_col_x;

            for entry in thought_log.iter() {
                let (_w, h, _lines) = crate::renderer::text::measure_text(
                    &mut self.font_system,
                    entry,
                    log_scale,
                    thought_col_width,
                );

                // Apply scroll offset
                let y = log_start_y + y_offset - log_scroll;

                // Skip entries above viewport
                if y + h < log_start_y {
                    y_offset += h;
                    continue;
                }
                // Stop rendering entries below viewport
                if y >= log_start_y + available_height {
                    break;
                }

                let alpha = 0.9; // No fade - clarity over aesthetics

                // Parse: [HH:MM:SS] rest
                // Colorized timestamp, markdown rest
                if entry.starts_with('[') && entry.len() > 10 && entry.chars().nth(9) == Some(']') {
                    let timestamp_part = &entry[..10]; // "[HH:MM:SS]"
                    let rest = entry[10..].trim_start();

                    // Rainbow color for timestamp
                    let rainbow = self.temporal_rainbow_color(entry, alpha);

                    // Render timestamp in color
                    self.text_queue.push(
                        timestamp_part,
                        x,
                        y,
                        log_scale,
                        rainbow,
                    );

                    // Calculate x after timestamp
                    let (ts_w, _, _) = crate::renderer::text::measure_text(
                        &mut self.font_system,
                        &format!("{} ", timestamp_part),
                        log_scale,
                        thought_col_width,
                    );

                    // Render rest as markdown for rich formatting (only if there's room)
                    let remaining_width = thought_col_width - ts_w;
                    if remaining_width > 0.0 {
                        self.text_queue.push_markdown_bounded(
                            rest,
                            x + ts_w,
                            y,
                            log_scale,
                            [plain_color[0], plain_color[1], plain_color[2], alpha],
                            remaining_width,
                            h,
                        );
                    }
                } else {
                    // No timestamp, render as markdown
                    self.text_queue.push_markdown_bounded(
                        entry,
                        x,
                        y,
                        log_scale,
                        [plain_color[0], plain_color[1], plain_color[2], alpha],
                        thought_col_width,
                        h,
                    );
                }

                y_offset += h;
            }
        }

        // Right column (ultrawide only): Status/quest info
        // In ultrawide mode, we have a third column for additional info
        if is_ultrawide && right_col_width > 0.0 {
            // Show enhanced status info in the right column
            let status_text = "Quest Status";
            self.text_queue.push(
                status_text,
                right_col_x,
                log_start_y,
                self.ui_scale.px(16.0),
                [0.6, 0.8, 1.0, 0.9],
            );

            // Show status indicator more prominently
            let status_y = log_start_y + self.ui_scale.px(30.0);
            let indicator_color = if request_active {
                [0.3, 0.9, 0.4, 1.0]
            } else {
                [0.5, 0.5, 0.5, 0.6]
            };
            let status_label = if request_active {
                "● Processing request..."
            } else {
                "○ Idle"
            };
            self.text_queue.push(
                status_label,
                right_col_x,
                status_y,
                self.ui_scale.px(14.0),
                indicator_color,
            );

            // Token count
            let token_y = status_y + self.ui_scale.px(24.0);
            let token_text = format!("{} tokens used", Self::format_number(tokens_used));
            self.text_queue.push(
                &token_text,
                right_col_x,
                token_y,
                self.ui_scale.px(12.0),
                self.text_color_dim(),
            );
        }

        // Build help state for this view
        let help_state = AppState::Executing {
            project_path: std::path::PathBuf::new(),
            executing_cards: Vec::new(),
            all_cards: Vec::new(),
            status: status.clone(),
            task_statuses: Vec::new(),
            tool_log: Vec::new(),
            thought_log: Vec::new(),
            log_scroll_offset: 0.0,
            executor: _executor,
            previous_state: Box::new(AppState::project_chooser()),
            quest_log_visible: false,
            quest_log_focus: 0,
            tokens_used: 0,
            request_active: false,
        };
        self.queue_help_legend(&help_state);
    }

    /// Build help legend items: returns (button, label) pairs for current state
    fn get_help_items(&self, state: &AppState) -> Vec<(XboxButton, &'static str)> {
        match state {
            AppState::ProjectChooser { .. } => vec![
                (XboxButton::LeftStick, "Navigate"),
                (XboxButton::A, "Select"),
                (XboxButton::B, "Back"),
            ],
            AppState::ProjectView { .. } => vec![
                (XboxButton::LeftStick, "Navigate"),
                (XboxButton::A, "Select"),
                (XboxButton::B, "Back"),
            ],
            AppState::MainMenu { .. } | AppState::SettingsMenu { .. } | AppState::UiScaleMenu { .. } => vec![
                (XboxButton::LeftStick, "Navigate"),
                (XboxButton::A, "Select"),
                (XboxButton::B, "Back"),
            ],
            AppState::PermissionModal { .. } => vec![
                (XboxButton::A, "Yes"),
                (XboxButton::B, "No"),
                (XboxButton::X, "Always"),
                (XboxButton::Y, "Suggest"),
            ],
            AppState::ExecuteModal { .. } => vec![
                (XboxButton::LeftStick, "Navigate"),
                (XboxButton::A, "Select"),
                (XboxButton::B, "Back"),
            ],
            AppState::PalaceLoop { .. } => vec![
                (XboxButton::LeftStick, "Navigate"),
                (XboxButton::A, "Select"),
                (XboxButton::B, "Back"),
                (XboxButton::X, "Run"),
                (XboxButton::Y, "Filter"),
            ],
            AppState::Survey { multi_select, .. } => {
                if *multi_select {
                    vec![
                        (XboxButton::LeftStick, "Navigate"),
                        (XboxButton::A, "Toggle"),
                        (XboxButton::B, "Cancel"),
                        (XboxButton::X, "Confirm"),
                    ]
                } else {
                    vec![
                        (XboxButton::LeftStick, "Navigate"),
                        (XboxButton::A, "Select"),
                        (XboxButton::B, "Cancel"),
                    ]
                }
            },
            AppState::Executing { status, quest_log_visible, .. } => {
                if *quest_log_visible {
                    vec![
                        (XboxButton::LeftStick, "Navigate"),
                        (XboxButton::B, "Back"),
                    ]
                } else if status.is_done() {
                    vec![
                        (XboxButton::B, "Back"),
                    ]
                } else {
                    vec![
                        (XboxButton::B, "Cancel"),
                        (XboxButton::RightStick, "Scroll"),
                    ]
                }
            },
            AppState::AddCardMenu { .. } => vec![
                (XboxButton::LeftStick, "Navigate"),
                (XboxButton::A, "Select"),
                (XboxButton::B, "Cancel"),
            ],
            AppState::CustomTaskInput { .. } => vec![
                (XboxButton::A, "Confirm"),
                (XboxButton::B, "Cancel"),
            ],
            AppState::MultiDisplayDialog { .. } => vec![
                (XboxButton::LeftStick, "Navigate"),
                (XboxButton::A, "Select"),
                (XboxButton::B, "Cancel"),
            ],
            AppState::ProjectContextMenu { .. } => vec![
                (XboxButton::LeftStick, "Navigate"),
                (XboxButton::A, "Select"),
                (XboxButton::B, "Cancel"),
            ],
            AppState::LanguageSelector { .. } => vec![
                (XboxButton::LeftStick, "Navigate"),
                (XboxButton::A, "Select"),
                (XboxButton::B, "Cancel"),
            ],
            AppState::NewMonitorDialog { .. } => vec![
                (XboxButton::LeftStick, "Navigate"),
                (XboxButton::A, "Select"),
                (XboxButton::B, "Cancel"),
            ],
            AppState::ScenarioDiffViewer { .. } => vec![
                (XboxButton::LeftStick, "Navigate"),
                (XboxButton::A, "Collapse"),
                (XboxButton::B, "Back"),
                (XboxButton::X, "Accept"),
            ],
            AppState::ScenarioGenerator { .. } => vec![
                (XboxButton::LeftStick, "Navigate"),
                (XboxButton::A, "Select"),
                (XboxButton::B, "Back"),
            ],
            AppState::Recording { .. } => vec![
                (XboxButton::Start, "Stop Recording"),
            ],
        }
    }

    fn build_help_sprites(&self, state: &AppState) -> Vec<SpriteInstance> {
        let items = self.get_help_items(state);
        let mut sprites = Vec::new();

        let (content_w, _content_h) = self.content_size();
        let (offset_x, offset_y) = self.content_offset();

        let glyph_size = self.ui_scale.px(24.0);
        let right_margin = self.ui_scale.px(20.0);
        let top_margin = self.ui_scale.px(20.0);
        let inner_gap = self.ui_scale.px(4.0); // Gap between glyph and label
        let item_gap = self.ui_scale.px(16.0); // Gap between items
        let label_width = self.ui_scale.px(44.0);

        // Each item: [glyph][inner_gap][label]
        // Between items: [item_gap]
        let item_width = glyph_size + inner_gap + label_width;
        let total_width = items.len() as f32 * item_width
            + (items.len().saturating_sub(1)) as f32 * item_gap;

        let start_x = offset_x + content_w - right_margin - total_width;
        let y = offset_y + top_margin;

        for (i, (button, _label)) in items.iter().enumerate() {
            let x = start_x + i as f32 * (item_width + item_gap);
            sprites.push(SpriteInstance::new(x, y, glyph_size, *button));
        }

        sprites
    }

    /// Build help legend text sections (labels after glyphs) - top right corner
    fn queue_help_legend(&mut self, state: &AppState) {
        let (content_w, _content_h) = self.content_size();
        let (offset_x, offset_y) = self.content_offset();

        if self.gamepad_connected {
            // Show gamepad hints
            let items = self.get_help_items(state);
            let glyph_size = self.ui_scale.px(24.0);
            let right_margin = self.ui_scale.px(20.0);
            let top_margin = self.ui_scale.px(20.0);
            let inner_gap = self.ui_scale.px(4.0);
            let item_gap = self.ui_scale.px(16.0);
            let label_width = self.ui_scale.px(44.0);
            let label_scale = self.ui_scale.px(14.0);

            let item_width = glyph_size + inner_gap + label_width;
            let total_width = items.len() as f32 * item_width
                + (items.len().saturating_sub(1)) as f32 * item_gap;
            let start_x = offset_x + content_w - right_margin - total_width;
            let y = offset_y + top_margin + (glyph_size - label_scale) / 2.0;

            for (i, (_, label)) in items.iter().enumerate() {
                let x = start_x + i as f32 * (item_width + item_gap) + glyph_size + inner_gap;
                self.text_queue.push(*label, x, y, label_scale, [0.6, 0.6, 0.65, 1.0]);
            }
        } else {
            // Show keyboard hints when no gamepad connected
            self.queue_keyboard_hints(state);
        }
    }

    /// Get keyboard hints for current state (shown when no gamepad connected)
    fn get_keyboard_hints(&self, state: &AppState) -> Vec<(&'static str, &'static str)> {
        match state {
            AppState::ProjectChooser { .. } => vec![
                ("WASD", "Navigate"),
                ("Enter", "Select"),
                ("Esc", "Menu"),
                ("Alt+F", "Fullscreen"),
            ],
            AppState::ProjectView { .. } => vec![
                ("WASD", "Navigate"),
                ("Enter", "Select"),
                ("Esc", "Back"),
            ],
            AppState::MainMenu { .. } | AppState::SettingsMenu { .. } | AppState::UiScaleMenu { .. } => vec![
                ("↑↓", "Navigate"),
                ("Enter", "Select"),
                ("Esc", "Back"),
            ],
            AppState::PermissionModal { .. } => vec![
                ("Y", "Yes"),
                ("N", "No"),
                ("A", "Always"),
            ],
            AppState::ExecuteModal { .. } => vec![
                ("↑↓", "Navigate"),
                ("Enter", "Select"),
                ("Esc", "Back"),
            ],
            AppState::PalaceLoop { generating, .. } => {
                if *generating {
                    vec![
                        ("WASD", "Navigate"),
                        ("Enter", "Select"),
                        ("ZXCV", "Quick 1-4"),
                        ("Esc", "Back"),
                    ]
                } else {
                    vec![
                        ("WASD", "Navigate"),
                        ("Enter", "Select"),
                        ("X", "Execute"),
                        ("ZXCV", "Quick 1-4"),
                    ]
                }
            },
            AppState::Survey { multi_select, .. } => {
                if *multi_select {
                    vec![
                        ("↑↓", "Navigate"),
                        ("Space", "Toggle"),
                        ("Enter", "Confirm"),
                        ("Esc", "Cancel"),
                    ]
                } else {
                    vec![
                        ("↑↓", "Navigate"),
                        ("Enter", "Select"),
                        ("Esc", "Cancel"),
                    ]
                }
            },
            AppState::Executing { status, quest_log_visible, .. } => {
                if *quest_log_visible {
                    vec![
                        ("WASD", "Navigate"),
                        ("Tab", "Hide Log"),
                        ("Esc", "Back"),
                    ]
                } else if status.is_done() {
                    vec![
                        ("Enter", "Back"),
                        ("Tab", "Quest Log"),
                    ]
                } else {
                    vec![
                        ("↑↓", "Scroll"),
                        ("Tab", "Quest Log"),
                        ("Esc", "Cancel"),
                    ]
                }
            },
            AppState::AddCardMenu { .. } => vec![
                ("↑↓", "Navigate"),
                ("Enter", "Select"),
                ("Esc", "Cancel"),
            ],
            AppState::CustomTaskInput { .. } => vec![
                ("Enter", "Confirm"),
                ("Esc", "Cancel"),
            ],
            AppState::MultiDisplayDialog { .. } => vec![
                ("↑↓", "Navigate"),
                ("Enter", "Select"),
                ("Esc", "Cancel"),
            ],
            AppState::ProjectContextMenu { .. } => vec![
                ("↑↓", "Navigate"),
                ("Enter", "Select"),
                ("Esc", "Cancel"),
            ],
            AppState::LanguageSelector { .. } => vec![
                ("↑↓", "Navigate"),
                ("Enter", "Select"),
                ("Esc", "Cancel"),
            ],
            AppState::NewMonitorDialog { .. } => vec![
                ("←→", "Navigate"),
                ("Enter", "Select"),
                ("Esc", "Cancel"),
            ],
            AppState::ScenarioDiffViewer { .. } => vec![
                ("WASD", "Navigate"),
                ("N/P", "Next/Prev Change"),
                ("Enter", "Collapse"),
                ("Esc", "Back"),
            ],
            AppState::ScenarioGenerator { .. } => vec![
                ("↑↓", "Navigate"),
                ("Enter", "Select"),
                ("Esc", "Back"),
            ],
            AppState::Recording { .. } => vec![
                ("F9", "Pause"),
                ("F10", "Stop"),
            ],
        }
    }

    /// Render keyboard hints in top-right corner (when no gamepad connected)
    fn queue_keyboard_hints(&mut self, state: &AppState) {
        let hints = self.get_keyboard_hints(state);
        if hints.is_empty() {
            return;
        }

        let (content_w, _content_h) = self.content_size();
        let (offset_x, offset_y) = self.content_offset();

        let right_margin = self.ui_scale.px(20.0);
        let top_margin = self.ui_scale.px(20.0);
        let item_gap = self.ui_scale.px(16.0);
        let key_scale = self.ui_scale.px(12.0);
        let label_scale = self.ui_scale.px(14.0);
        let bracket_color = [0.5, 0.5, 0.55, 1.0];
        let key_color = [0.8, 0.8, 0.85, 1.0];
        let label_color = [0.6, 0.6, 0.65, 1.0];

        // Calculate total width (estimate: key width varies)
        let avg_item_width = self.ui_scale.px(80.0);
        let total_width = hints.len() as f32 * avg_item_width;
        let mut x = offset_x + content_w - right_margin - total_width;
        let y = offset_y + top_margin;

        for (key, label) in hints {
            // Render "[KEY] Label" format
            self.text_queue.push("[", x, y, key_scale, bracket_color);
            x += self.ui_scale.px(6.0);
            self.text_queue.push(key, x, y, key_scale, key_color);
            x += self.ui_scale.px(key.len() as f32 * 7.0);
            self.text_queue.push("]", x, y, key_scale, bracket_color);
            x += self.ui_scale.px(8.0);
            self.text_queue.push(label, x, y + self.ui_scale.px(1.0), label_scale, label_color);
            x += self.ui_scale.px(label.len() as f32 * 7.0) + item_gap;
        }
    }

    #[allow(dead_code)]
    pub fn device(&self) -> &wgpu::Device {
        &self.device
    }

    #[allow(dead_code)]
    pub fn queue(&self) -> &wgpu::Queue {
        &self.queue
    }

    #[allow(dead_code)]
    pub fn size(&self) -> PhysicalSize<u32> {
        self.size
    }

    #[allow(dead_code)]
    /// Start a non-blocking screenshot capture of the last rendered frame
    /// The capture will complete asynchronously over the next few frames
    pub fn start_screenshot(
        &self,
        capture: &mut crate::debug::ScreenshotCapture,
        texture: &wgpu::Texture,
        path: std::path::PathBuf,
    ) -> Result<std::path::PathBuf, String> {
        capture.start_capture(
            &self.device,
            &self.queue,
            texture,
            self.size.width,
            self.size.height,
            path,
        )
    }

    /// Compute UI panel bounds for the current state
    ///
    /// Returns a vector of (panel_id, x, y, width, height) for each visible panel.
    /// These are the actual UI elements that can be selected/resized in edit mode.
    /// This is public so the App can call it to populate edit_mode.ui_panels for hit testing.
    pub fn compute_ui_panels(&self, state: &AppState) -> Vec<(&'static str, f32, f32, f32, f32)> {
        let (content_w, _content_h) = self.content_size();
        let (offset_x, offset_y) = self.content_offset();

        match state {
            AppState::ProjectView { .. } => {
                use crate::state::ProjectAction;

                let left_margin = self.ui_scale.px(60.0);
                let top_margin = self.ui_scale.px(40.0);
                let title_scale = self.ui_scale.px(36.0);

                // Panel 1: Project context widget (top-left info box)
                let context_x = offset_x + left_margin - self.ui_scale.px(10.0);
                let context_y = offset_y + top_margin - self.ui_scale.px(5.0);
                let context_w = self.ui_scale.px(300.0); // Reasonable width for project name
                let context_h = title_scale + self.ui_scale.px(30.0); // Title + path

                // Panel 2: Action menu (the 4 buttons)
                let menu_start_y = top_margin + title_scale + self.ui_scale.px(60.0);
                let card_height = self.ui_scale.px(70.0);
                let card_gap = self.ui_scale.px(16.0);
                let actions_count = ProjectAction::all().len();
                let menu_height = actions_count as f32 * (card_height + card_gap) - card_gap;
                let menu_x = offset_x + left_margin - self.ui_scale.px(5.0);
                let menu_y = offset_y + menu_start_y - self.ui_scale.px(5.0);
                let menu_w = content_w - left_margin * 2.0 + self.ui_scale.px(10.0);

                vec![
                    ("context_widget", context_x, context_y, context_w, context_h),
                    ("action_menu", menu_x, menu_y, menu_w, menu_height + self.ui_scale.px(10.0)),
                ]
            }
            AppState::ProjectChooser { .. } => {
                // For now, treat the whole project grid as one panel
                let margin = self.ui_scale.px(20.0);
                vec![(
                    "project_grid",
                    offset_x + margin,
                    offset_y + margin,
                    content_w - margin * 2.0,
                    _content_h - margin * 2.0,
                )]
            }
            AppState::PalaceLoop { .. } => {
                // Cards panel takes most of the screen
                let margin = self.ui_scale.px(20.0);
                vec![(
                    "quest_log",
                    offset_x + margin,
                    offset_y + margin,
                    content_w - margin * 2.0,
                    _content_h - margin * 2.0,
                )]
            }
            // Other states - no editable panels for now
            _ => vec![],
        }
    }

    /// Queue edit mode UI - selection borders and resize handles around actual UI panels
    ///
    /// Android ICS/Honeycomb style: each panel gets a blue selection border with
    /// draggable handles at corners and edges. No grid overlay - just handles on real content.
    ///
    /// Panels in `reflow_positions` are rendered at their new positions (ICS-style dynamic reflow).
    fn queue_edit_mode_indicator(
        &mut self,
        panels: &[(&'static str, f32, f32, f32, f32)],
        selected: Option<&str>,
        resize_preview: Option<(&str, f32, f32)>, // (panel_id, delta_x, delta_y)
        reflow_positions: &std::collections::HashMap<&'static str, (f32, f32, f32, f32)>,
    ) {
        let (content_w, content_h) = self.content_size();
        let (offset_x, offset_y) = self.content_offset();

        // Subtle glow animation for handles (slow, gentle)
        let anim_time = self.animation_time();
        let glow = 0.85 + 0.15 * (anim_time * 1.5 * std::f32::consts::PI).sin();

        // Colors - steady selection border, subtle glow on handles
        let selection_color = [0.2, 0.6, 1.0, 0.9]; // Blue selection border (no animation)
        let unselected_color = [0.4, 0.4, 0.5, 0.6]; // Dimmer for unselected (no animation)
        // Handle color: base blue with subtle brightness variation via glow
        let handle_color = [0.3 * glow, 0.6 * glow, 1.0 * glow, 0.95]; // Subtle internal glow
        let handle_size = self.ui_scale.px(16.0);
        let border_width = self.ui_scale.px(3.0);
        let unselected_border = self.ui_scale.px(2.0);

        // Draw selection + handles for each panel
        for &(panel_id, orig_x, orig_y, orig_w, orig_h) in panels {
            let is_selected = selected == Some(panel_id);
            let color = if is_selected { selection_color } else { unselected_color };
            let bw = if is_selected { border_width } else { unselected_border };

            // Use reflow position if this panel was displaced, otherwise use original
            let (x, y, w, h) = if let Some(&(rx, ry, rw, rh)) = reflow_positions.get(panel_id) {
                (rx, ry, rw, rh) // Panel displaced by reflow - use new position
            } else {
                (orig_x, orig_y, orig_w, orig_h) // Not displaced - use original
            };

            // Check if this panel is being resized - if so, draw at new size
            let (x, y, w, h) = if let Some((resize_id, delta_x, delta_y)) = resize_preview {
                if panel_id == resize_id {
                    // Apply resize delta to the panel being resized
                    let new_w = (w + delta_x).max(100.0);
                    let new_h = (h + delta_y).max(50.0);
                    (x, y, new_w, new_h)
                } else {
                    (x, y, w, h)
                }
            } else {
                (x, y, w, h)
            };

            // Selection border around this panel
            self.edit_mode_cards.push(
                CardInstance::new(x, y, w, h, color)
                    .with_border_width(bw)
                    .with_corner_radius(self.ui_scale.px(8.0))
            );

            // Only draw handles on selected panel (or all if none selected)
            if is_selected || selected.is_none() {
                // Corner resize handles
                let corners = [
                    (x - handle_size / 2.0, y - handle_size / 2.0),                 // Top-left
                    (x + w - handle_size / 2.0, y - handle_size / 2.0),             // Top-right
                    (x - handle_size / 2.0, y + h - handle_size / 2.0),             // Bottom-left
                    (x + w - handle_size / 2.0, y + h - handle_size / 2.0),         // Bottom-right
                ];

                for (hx, hy) in corners {
                    self.edit_mode_cards.push(
                        CardInstance::new(hx, hy, handle_size, handle_size, handle_color)
                            .with_border_width(self.ui_scale.px(2.0))
                            .with_corner_radius(self.ui_scale.px(4.0))
                            .selected()
                    );
                }

                // Edge handles (midpoint bars)
                let edge_width = handle_size * 1.5;
                let edge_height = handle_size * 0.7;

                // Top and bottom edges (horizontal bars)
                let h_edges = [
                    (x + w / 2.0 - edge_width / 2.0, y - edge_height / 2.0),        // Top
                    (x + w / 2.0 - edge_width / 2.0, y + h - edge_height / 2.0),    // Bottom
                ];
                for (hx, hy) in h_edges {
                    self.edit_mode_cards.push(
                        CardInstance::new(hx, hy, edge_width, edge_height, handle_color)
                            .with_border_width(self.ui_scale.px(2.0))
                            .with_corner_radius(self.ui_scale.px(3.0))
                            .selected()
                    );
                }

                // Left and right edges (vertical bars)
                let v_edges = [
                    (x - edge_height / 2.0, y + h / 2.0 - edge_width / 2.0),        // Left
                    (x + w - edge_height / 2.0, y + h / 2.0 - edge_width / 2.0),    // Right
                ];
                for (hx, hy) in v_edges {
                    self.edit_mode_cards.push(
                        CardInstance::new(hx, hy, edge_height, edge_width, handle_color)
                            .with_border_width(self.ui_scale.px(2.0))
                            .with_corner_radius(self.ui_scale.px(3.0))
                            .selected()
                    );
                }
            }
        }

        // Help text at bottom
        let help_text = "Click panel to select • Drag handles to resize • F2/ESC to exit";
        self.text_queue.push(
            help_text,
            offset_x + content_w / 2.0 - self.ui_scale.px(200.0),
            offset_y + content_h - self.ui_scale.px(30.0),
            self.ui_scale.px(14.0),
            [0.5, 0.7, 1.0, 0.9],
        );
    }

    /// Build recording indicator cards (red dot in top-right)
    pub fn build_recording_indicator_cards(&self, paused: bool) -> Vec<CardInstance> {
        let (content_w, _content_h) = self.content_size();
        let (offset_x, offset_y) = self.content_offset();

        let mut cards = Vec::new();

        // Red dot position (top-right)
        let dot_size = self.ui_scale.px(12.0);
        let margin = self.ui_scale.px(16.0);
        let dot_x = offset_x + content_w - margin - dot_size - self.ui_scale.px(50.0);
        let dot_y = offset_y + margin;

        // Red recording dot (pulsing when active, solid when paused)
        let alpha = if paused {
            0.5
        } else {
            let anim_time = self.animation_time();
            0.7 + 0.3 * (anim_time * 3.0 * std::f32::consts::PI).sin()
        };

        let dot_color = if paused {
            [0.7, 0.3, 0.1, alpha] // Orange when paused
        } else {
            [0.9, 0.1, 0.1, alpha] // Red when recording
        };

        cards.push(
            CardInstance::new(dot_x, dot_y, dot_size, dot_size, dot_color)
                .with_corner_radius(dot_size / 2.0) // Circle
                .with_border_width(0.0)
        );

        cards
    }

    /// Queue recording indicator text (REC + elapsed time)
    pub fn queue_recording_indicator_text(&mut self, elapsed: std::time::Duration, paused: bool) {
        let (content_w, content_h) = self.content_size();
        let (offset_x, offset_y) = self.content_offset();

        let margin = self.ui_scale.px(16.0);
        let dot_size = self.ui_scale.px(12.0);

        // "REC" text next to dot
        let rec_x = offset_x + content_w - margin - self.ui_scale.px(40.0);
        let rec_y = offset_y + margin;
        let rec_text = if paused { "PAUSED" } else { "REC" };
        let rec_color = if paused {
            [0.8, 0.6, 0.2, 1.0] // Orange
        } else {
            [0.9, 0.3, 0.3, 1.0] // Red
        };

        self.text_queue.push(
            rec_text,
            rec_x,
            rec_y,
            self.ui_scale.px(14.0),
            rec_color,
        );

        // Elapsed time in status bar (bottom-left)
        let minutes = elapsed.as_secs() / 60;
        let seconds = elapsed.as_secs() % 60;
        let time_text = format!("Recording {:02}:{:02}", minutes, seconds);

        self.text_queue.push(
            &time_text,
            offset_x + margin,
            offset_y + content_h - self.ui_scale.px(30.0),
            self.ui_scale.px(14.0),
            [0.7, 0.7, 0.7, 0.9],
        );

        // Recording controls hint
        let hint_text = "F9: Pause | F10: Stop & Save";
        self.text_queue.push(
            hint_text,
            offset_x + margin + self.ui_scale.px(140.0),
            offset_y + content_h - self.ui_scale.px(30.0),
            self.ui_scale.px(12.0),
            [0.5, 0.5, 0.5, 0.7],
        );
    }

    /// Render panel chooser dialog with ICS-style widget previews
    ///
    /// Shows actual mini-rendered previews of each panel type using fixture data,
    /// following the Android ICS widget picker pattern.
    fn queue_panel_chooser(&mut self, spawn_pos: (f32, f32), selection: usize) {
        use crate::fixtures::{get_panel_fixture, PanelFixture};
        use crate::panels::available_panel_types;

        let (spawn_x, spawn_y) = spawn_pos;
        let panel_types = available_panel_types();

        // Grid layout: 2 columns of preview cards
        let columns = 2;
        let preview_width = self.ui_scale.px(200.0);
        let preview_height = self.ui_scale.px(150.0);
        let gap = self.ui_scale.px(12.0);
        let padding = self.ui_scale.px(16.0);
        let title_height = self.ui_scale.px(40.0);
        let label_height = self.ui_scale.px(24.0);

        let rows = (panel_types.len() + columns - 1) / columns;
        let dialog_width = columns as f32 * preview_width + (columns - 1) as f32 * gap + padding * 2.0;
        let dialog_height = title_height + rows as f32 * (preview_height + label_height + gap) + padding;

        // Position dialog near spawn point, but keep on screen
        let (content_w, content_h) = self.content_size();
        let (offset_x, offset_y) = self.content_offset();

        let mut dialog_x = spawn_x - dialog_width / 2.0;
        let mut dialog_y = spawn_y - dialog_height / 2.0;

        // Clamp to screen bounds
        dialog_x = dialog_x.max(offset_x + padding).min(offset_x + content_w - dialog_width - padding);
        dialog_y = dialog_y.max(offset_y + padding).min(offset_y + content_h - dialog_height - padding);

        // Background card (darker, semi-transparent)
        let bg_color = [0.08, 0.08, 0.12, 0.98];
        let bg_card = CardInstance::new(dialog_x, dialog_y, dialog_width, dialog_height, bg_color)
            .with_corner_radius(self.ui_scale.px(20.0))
            .with_border_width(self.ui_scale.px(2.0))
            .with_border_color([0.3, 0.4, 0.6, 0.8]);
        self.edit_mode_cards.push(bg_card);

        // Title
        self.text_queue.push(
            "Add Panel",
            dialog_x + padding,
            dialog_y + padding,
            self.ui_scale.px(20.0),
            [0.9, 0.9, 1.0, 1.0],
        );

        // Render each panel type as a preview card
        let previews_start_y = dialog_y + title_height;

        for (i, panel_type) in panel_types.iter().enumerate() {
            let col = i % columns;
            let row = i / columns;

            let preview_x = dialog_x + padding + col as f32 * (preview_width + gap);
            let preview_y = previews_start_y + row as f32 * (preview_height + label_height + gap);
            let is_selected = i == selection;

            // Selection highlight (glow effect)
            if is_selected {
                let glow = CardInstance::new(
                    preview_x - self.ui_scale.px(4.0),
                    preview_y - self.ui_scale.px(4.0),
                    preview_width + self.ui_scale.px(8.0),
                    preview_height + label_height + self.ui_scale.px(8.0),
                    [0.3, 0.5, 0.9, 0.4],
                )
                .with_corner_radius(self.ui_scale.px(14.0));
                self.edit_mode_cards.push(glow);
            }

            // Preview card background
            let preview_bg_color = if is_selected {
                [0.15, 0.18, 0.25, 1.0]
            } else {
                [0.12, 0.12, 0.16, 1.0]
            };
            let preview_card = CardInstance::new(preview_x, preview_y, preview_width, preview_height, preview_bg_color)
                .with_corner_radius(self.ui_scale.px(10.0))
                .with_border_width(self.ui_scale.px(1.0))
                .with_border_color(if is_selected {
                    [0.5, 0.6, 0.9, 0.8]
                } else {
                    [0.25, 0.25, 0.3, 0.6]
                });
            self.edit_mode_cards.push(preview_card);

            // Render mini preview content based on panel type
            self.queue_panel_preview(
                panel_type.id,
                preview_x,
                preview_y,
                preview_width,
                preview_height,
            );

            // Panel name label below preview
            let label_y = preview_y + preview_height + self.ui_scale.px(6.0);
            let name_color = if is_selected {
                [1.0, 1.0, 1.0, 1.0]
            } else {
                [0.7, 0.7, 0.8, 1.0]
            };
            self.text_queue.push(
                panel_type.name,
                preview_x + preview_width / 2.0 - self.ui_scale.px(40.0), // Roughly centered
                label_y,
                self.ui_scale.px(13.0),
                name_color,
            );
        }

        // Hint text at bottom
        self.text_queue.push(
            "↑↓ Navigate  Enter Select  Esc Cancel",
            dialog_x + padding,
            dialog_y + dialog_height - self.ui_scale.px(20.0),
            self.ui_scale.px(10.0),
            [0.4, 0.4, 0.5, 0.8],
        );
    }

    /// Render a mini preview of a panel type using fixture data
    fn queue_panel_preview(
        &mut self,
        panel_type: &str,
        x: f32,
        y: f32,
        width: f32,
        height: f32,
    ) {
        use crate::fixtures::{get_panel_fixture, PanelFixture};

        let padding = self.ui_scale.px(6.0);
        let inner_x = x + padding;
        let inner_y = y + padding;
        let inner_w = width - padding * 2.0;
        let inner_h = height - padding * 2.0;

        match get_panel_fixture(panel_type) {
            Some(PanelFixture::QuestLog(cards)) => {
                self.queue_quest_log_preview(inner_x, inner_y, inner_w, inner_h, &cards);
            }
            Some(PanelFixture::Execution { tool_log, thought_log }) => {
                self.queue_execution_preview(inner_x, inner_y, inner_w, inner_h, &tool_log, &thought_log);
            }
            Some(PanelFixture::Analysis(log)) => {
                self.queue_analysis_preview(inner_x, inner_y, inner_w, inner_h, &log);
            }
            Some(PanelFixture::ProjectChooser(projects)) => {
                self.queue_project_chooser_preview(inner_x, inner_y, inner_w, inner_h, &projects);
            }
            None => {
                // Fallback: just show panel type name
                self.text_queue.push(
                    panel_type,
                    inner_x + inner_w / 2.0 - self.ui_scale.px(30.0),
                    inner_y + inner_h / 2.0,
                    self.ui_scale.px(12.0),
                    [0.5, 0.5, 0.6, 0.8],
                );
            }
        }
    }

    /// Mini preview of quest log panel (card grid)
    fn queue_quest_log_preview(
        &mut self,
        x: f32,
        y: f32,
        width: f32,
        height: f32,
        cards: &[crate::state::SuggestionCard],
    ) {
        let mini_card_w = self.ui_scale.px(50.0);
        let mini_card_h = self.ui_scale.px(35.0);
        let gap = self.ui_scale.px(4.0);
        let cols = ((width + gap) / (mini_card_w + gap)).floor() as usize;

        for (i, card) in cards.iter().take(6).enumerate() {
            let col = i % cols;
            let row = i / cols;

            let card_x = x + col as f32 * (mini_card_w + gap);
            let card_y = y + row as f32 * (mini_card_h + gap);

            if card_y + mini_card_h > y + height {
                break; // Don't overflow
            }

            // Card color based on category
            let card_color = match card.category.as_str() {
                "setup" => [0.2, 0.35, 0.5, 0.9],
                "gameplay" => [0.25, 0.4, 0.3, 0.9],
                "ui" => [0.4, 0.3, 0.45, 0.9],
                "polish" => [0.45, 0.35, 0.25, 0.9],
                _ => [0.25, 0.25, 0.3, 0.9],
            };

            let mini_card = CardInstance::new(card_x, card_y, mini_card_w, mini_card_h, card_color)
                .with_corner_radius(self.ui_scale.px(4.0));
            self.edit_mode_cards.push(mini_card);

            // Selection indicator (checkmark area)
            if card.selected {
                let check_size = self.ui_scale.px(8.0);
                let check = CardInstance::new(
                    card_x + mini_card_w - check_size - self.ui_scale.px(2.0),
                    card_y + self.ui_scale.px(2.0),
                    check_size,
                    check_size,
                    [0.3, 0.8, 0.4, 0.9],
                )
                .with_corner_radius(self.ui_scale.px(2.0));
                self.edit_mode_cards.push(check);
            }

            // Tiny title text (truncated)
            let title: String = card.title.chars().take(8).collect();
            self.text_queue.push(
                Box::leak(title.into_boxed_str()),
                card_x + self.ui_scale.px(3.0),
                card_y + self.ui_scale.px(4.0),
                self.ui_scale.px(7.0),
                [0.9, 0.9, 0.95, 0.9],
            );
        }
    }

    /// Mini preview of execution panel (two columns: tools | thoughts)
    fn queue_execution_preview(
        &mut self,
        x: f32,
        y: f32,
        width: f32,
        height: f32,
        tool_log: &[String],
        thought_log: &[String],
    ) {
        let col_width = width / 2.0 - self.ui_scale.px(2.0);
        let line_height = self.ui_scale.px(10.0);

        // Left column header
        self.text_queue.push(
            "Tools",
            x + self.ui_scale.px(2.0),
            y,
            self.ui_scale.px(8.0),
            [0.6, 0.7, 0.9, 0.9],
        );

        // Right column header
        self.text_queue.push(
            "Thoughts",
            x + col_width + self.ui_scale.px(4.0),
            y,
            self.ui_scale.px(8.0),
            [0.9, 0.7, 0.6, 0.9],
        );

        // Tool log entries (left column)
        let entries_y = y + self.ui_scale.px(12.0);
        for (i, entry) in tool_log.iter().take(8).enumerate() {
            let entry_y = entries_y + i as f32 * line_height;
            if entry_y + line_height > y + height {
                break;
            }
            // Extract just the icon and action (skip timestamp for mini view)
            let display: String = entry.chars().skip(11).take(12).collect();
            self.text_queue.push(
                Box::leak(display.into_boxed_str()),
                x + self.ui_scale.px(2.0),
                entry_y,
                self.ui_scale.px(6.0),
                [0.7, 0.75, 0.85, 0.8],
            );
        }

        // Thought log entries (right column)
        for (i, entry) in thought_log.iter().take(8).enumerate() {
            let entry_y = entries_y + i as f32 * line_height;
            if entry_y + line_height > y + height {
                break;
            }
            let display: String = entry.chars().take(14).collect();
            self.text_queue.push(
                Box::leak(display.into_boxed_str()),
                x + col_width + self.ui_scale.px(4.0),
                entry_y,
                self.ui_scale.px(6.0),
                [0.85, 0.75, 0.7, 0.8],
            );
        }
    }

    /// Mini preview of analysis panel (log list)
    fn queue_analysis_preview(
        &mut self,
        x: f32,
        y: f32,
        _width: f32,
        height: f32,
        log: &[String],
    ) {
        let line_height = self.ui_scale.px(12.0);

        for (i, entry) in log.iter().take(8).enumerate() {
            let entry_y = y + i as f32 * line_height;
            if entry_y + line_height > y + height {
                break;
            }
            let display: String = entry.chars().take(20).collect();
            self.text_queue.push(
                Box::leak(display.into_boxed_str()),
                x + self.ui_scale.px(2.0),
                entry_y,
                self.ui_scale.px(7.0),
                [0.7, 0.8, 0.7, 0.9],
            );
        }
    }

    /// Mini preview of project chooser (project grid)
    fn queue_project_chooser_preview(
        &mut self,
        x: f32,
        y: f32,
        width: f32,
        height: f32,
        projects: &[crate::fixtures::FixtureProject],
    ) {
        let card_w = self.ui_scale.px(45.0);
        let card_h = self.ui_scale.px(30.0);
        let gap = self.ui_scale.px(4.0);
        let cols = ((width + gap) / (card_w + gap)).floor() as usize;

        for (i, project) in projects.iter().take(4).enumerate() {
            let col = i % cols;
            let row = i / cols;

            let card_x = x + col as f32 * (card_w + gap);
            let card_y = y + row as f32 * (card_h + gap);

            if card_y + card_h > y + height {
                break;
            }

            // Project card
            let card_color = [0.2, 0.22, 0.28, 0.9];
            let card = CardInstance::new(card_x, card_y, card_w, card_h, card_color)
                .with_corner_radius(self.ui_scale.px(4.0));
            self.edit_mode_cards.push(card);

            // Project name (truncated)
            let name: String = project.name.chars().take(6).collect();
            self.text_queue.push(
                Box::leak(name.into_boxed_str()),
                card_x + self.ui_scale.px(3.0),
                card_y + self.ui_scale.px(4.0),
                self.ui_scale.px(7.0),
                [0.9, 0.9, 0.95, 0.9],
            );

            // Language indicator
            if let Some(lang) = project.languages.first() {
                let lang_short: String = lang.chars().take(4).collect();
                self.text_queue.push(
                    Box::leak(lang_short.into_boxed_str()),
                    card_x + self.ui_scale.px(3.0),
                    card_y + self.ui_scale.px(16.0),
                    self.ui_scale.px(6.0),
                    [0.5, 0.6, 0.7, 0.8],
                );
            }
        }
    }

    /// Poll pending screenshot captures (call each frame)
    pub fn poll_screenshots(&self, capture: &mut crate::debug::ScreenshotCapture) {
        capture.poll_pending(&self.device);
    }
}

/// Convert HSL to RGB (helper for temporal rainbow colors)
fn hsl_to_rgb(h: f32, s: f32, l: f32) -> (f32, f32, f32) {
    let c = (1.0 - (2.0 * l - 1.0).abs()) * s;
    let h_prime = h / 60.0;
    let x = c * (1.0 - (h_prime % 2.0 - 1.0).abs());
    let (r1, g1, b1) = match h_prime as i32 {
        0 => (c, x, 0.0),
        1 => (x, c, 0.0),
        2 => (0.0, c, x),
        3 => (0.0, x, c),
        4 => (x, 0.0, c),
        _ => (c, 0.0, x),
    };
    let m = l - c / 2.0;
    (r1 + m, g1 + m, b1 + m)
}
