use crate::projects::ProjectsConfig;
use crate::renderer::cards::{CardInstance, CardRenderer};
use crate::renderer::sprites::{SpriteInstance, SpriteRenderer, XboxButton};
use crate::state::{AppState, SuggestionCard};
use anyhow::{Context, Result};
use wgpu_text::glyph_brush::{ab_glyph::FontRef, Layout, Section, Text};
use wgpu_text::BrushBuilder;
use winit::dpi::PhysicalSize;
use winit::window::Window;

// Embedded font - using a monospace font for code display
const FONT_BYTES: &[u8] = include_bytes!("fonts/JetBrainsMono-Regular.ttf");

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
        }
    }

    /// Grid specifically for PalaceLoop - targets 5 columns
    fn for_palace_loop(screen_width: f32, _screen_height: f32, scale: &UiScale) -> Self {
        let target_columns = 5;
        let base_gap = 16.0;
        let base_margin = 40.0;

        let gap = scale.px(base_gap);
        let margin_x = scale.px(base_margin);
        let margin_y = scale.px(base_margin + 80.0); // Extra space for title + subtitle

        // Calculate card width to fit exactly 5 columns
        let available_width = screen_width - margin_x * 2.0;
        let card_width = (available_width - gap * (target_columns - 1) as f32) / target_columns as f32;
        let card_height = card_width * 0.6; // Maintain aspect ratio

        let columns = target_columns;

        Self {
            card_width,
            card_height,
            gap,
            columns,
            margin_x,
            margin_y,
        }
    }

    fn card_position(&self, index: usize) -> (f32, f32) {
        let row = index / self.columns;
        let col = index % self.columns;

        let x = self.margin_x + col as f32 * (self.card_width + self.gap);
        let y = self.margin_y + row as f32 * (self.card_height + self.gap);

        (x, y)
    }
}

pub struct Renderer {
    surface: wgpu::Surface<'static>,
    device: wgpu::Device,
    queue: wgpu::Queue,
    config: wgpu::SurfaceConfiguration,
    size: PhysicalSize<u32>,
    text_brush: wgpu_text::TextBrush<FontRef<'static>>,
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

        // Create text brush
        let font = FontRef::try_from_slice(FONT_BYTES).context("Failed to load font")?;
        let text_brush =
            BrushBuilder::using_font(font).build(&device, size.width, size.height, surface_format);

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
            device,
            queue,
            config,
            size,
            text_brush,
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
        })
    }

    pub fn set_dark_mode(&mut self, dark_mode: bool) {
        self.dark_mode = dark_mode;
    }

    pub fn dark_mode(&self) -> bool {
        self.dark_mode
    }

    /// Set UI scale factor (e.g., 1.5 for 150% scaling)
    pub fn set_ui_scale(&mut self, scale: f32) {
        self.ui_scale.dpi_scale = scale;
        tracing::info!("UI scale set to {}", scale);
    }

    /// Get current UI scale factor
    pub fn ui_scale(&self) -> f32 {
        self.ui_scale.dpi_scale
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

    pub fn resize(&mut self, new_size: PhysicalSize<u32>) {
        if new_size.width > 0 && new_size.height > 0 {
            self.size = new_size;
            self.config.width = new_size.width;
            self.config.height = new_size.height;
            self.surface.configure(&self.device, &self.config);
            self.text_brush
                .resize_view(new_size.width as f32, new_size.height as f32, &self.queue);
            self.card_renderer
                .resize(&self.queue, new_size.width, new_size.height);
            self.sprite_renderer
                .resize(&self.queue, new_size.width, new_size.height);
            // Clear textures so they get recreated at new size
            self.screenshot_texture = None;
            self.scene_texture = None;
            self.blur_texture = None;
            tracing::debug!("Resized to {}x{}", new_size.width, new_size.height);
        }
    }

    pub fn render(
        &mut self,
        state: &AppState,
        projects: &ProjectsConfig,
    ) -> Result<(), wgpu::SurfaceError> {
        self.render_with_screenshot(state, projects, None, &mut crate::debug::ScreenshotCapture::new())
    }

    pub fn render_with_screenshot(
        &mut self,
        state: &AppState,
        projects: &ProjectsConfig,
        screenshot_path: Option<&std::path::PathBuf>,
        screenshot_capture: &mut crate::debug::ScreenshotCapture,
    ) -> Result<(), wgpu::SurfaceError> {
        let output = self.surface.get_current_texture()?;
        let surface_view = output
            .texture
            .create_view(&wgpu::TextureViewDescriptor::default());

        // Determine if we're rendering a modal menu
        enum ModalType {
            MainMenu(usize),
            Settings(usize),
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
            _ => (state, None),
        };

        // Build cards for base state
        let base_cards = match base_state {
            AppState::ProjectChooser { selected_index } => {
                self.build_project_cards(projects, *selected_index)
            }
            AppState::ProjectView { selected_action, .. } => {
                self.build_action_cards(*selected_action)
            }
            AppState::PalaceLoop { cards, focused_index, .. } => {
                self.build_suggestion_cards(cards, *focused_index)
            }
            AppState::MainMenu { .. } | AppState::SettingsMenu { .. } => Vec::new(),
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
                AppState::ProjectChooser { selected_index } => {
                    self.queue_project_chooser_text(projects, *selected_index);
                }
                AppState::ProjectView { project_path, selected_action } => {
                    self.queue_project_view_text(project_path, *selected_action);
                }
                AppState::PalaceLoop { cards, current_tool, .. } => {
                    self.queue_palace_loop_text(cards, current_tool.as_deref());
                }
                AppState::MainMenu { .. } | AppState::SettingsMenu { .. } => {}
            }

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

                self.card_renderer
                    .draw(&mut render_pass, &self.queue, &base_cards);

                if self.gamepad_connected {
                    let sprites = self.build_help_sprites(base_state);
                    self.sprite_renderer
                        .draw(&mut render_pass, &self.queue, &sprites);
                }

                self.text_brush.draw(&mut render_pass);
            }

            // Pass 2: Draw dark overlay + modal
            // Queue modal text based on type
            match modal {
                ModalType::MainMenu(selected) => self.queue_main_menu_text(*selected),
                ModalType::Settings(selected) => self.queue_settings_modal_text(*selected),
            }

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

                if self.gamepad_connected {
                    let sprites = self.build_help_sprites(state);
                    self.sprite_renderer
                        .draw(&mut render_pass, &self.queue, &sprites);
                }

                self.text_brush.draw(&mut render_pass);
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
                AppState::ProjectChooser { selected_index } => {
                    self.queue_project_chooser_text(projects, *selected_index);
                }
                AppState::ProjectView { project_path, selected_action } => {
                    self.queue_project_view_text(project_path, *selected_action);
                }
                AppState::PalaceLoop { cards, current_tool, .. } => {
                    self.queue_palace_loop_text(cards, current_tool.as_deref());
                }
                AppState::MainMenu { .. } => {}
                AppState::SettingsMenu { .. } => {}
            }

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

                self.card_renderer
                    .draw(&mut render_pass, &self.queue, &base_cards);

                if self.gamepad_connected {
                    let sprites = self.build_help_sprites(state);
                    self.sprite_renderer
                        .draw(&mut render_pass, &self.queue, &sprites);
                }

                self.text_brush.draw(&mut render_pass);
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
                AppState::ProjectChooser { selected_index } => {
                    self.queue_project_chooser_text(projects, *selected_index);
                }
                AppState::ProjectView { project_path, selected_action } => {
                    self.queue_project_view_text(project_path, *selected_action);
                }
                AppState::PalaceLoop { cards, current_tool, .. } => {
                    self.queue_palace_loop_text(cards, current_tool.as_deref());
                }
                AppState::MainMenu { .. } => {}
                AppState::SettingsMenu { .. } => {}
            }

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

                self.card_renderer
                    .draw(&mut render_pass, &self.queue, &base_cards);

                if self.gamepad_connected {
                    let sprites = self.build_help_sprites(state);
                    self.sprite_renderer
                        .draw(&mut render_pass, &self.queue, &sprites);
                }

                self.text_brush.draw(&mut render_pass);
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
        let grid = CardGrid::new(
            self.size.width as f32,
            self.size.height as f32,
            &self.ui_scale,
        );

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

    fn build_action_cards(&self, selected: usize) -> Vec<CardInstance> {
        use crate::state::ProjectAction;

        let actions = ProjectAction::all();
        let left_margin = self.ui_scale.px(60.0);
        let title_scale = self.ui_scale.px(40.0);
        let top_margin = self.ui_scale.px(40.0);
        let menu_start_y = top_margin + title_scale + self.ui_scale.px(60.0);
        let card_width = self.size.width as f32 - left_margin * 2.0;
        let card_height = self.ui_scale.px(70.0);
        let card_gap = self.ui_scale.px(16.0);

        // Action color - purple accent
        let action_color = [0.5, 0.3, 0.8, 1.0];

        actions
            .iter()
            .enumerate()
            .map(|(i, _action)| {
                let y = menu_start_y + i as f32 * (card_height + card_gap);
                let is_selected = i == selected;

                let mut card = CardInstance::new(left_margin, y, card_width, card_height, action_color)
                    .with_border_width(self.ui_scale.px(if is_selected { 3.0 } else { 1.5 }))
                    .with_corner_radius(self.ui_scale.px(12.0));

                if is_selected {
                    card = card.selected();
                }

                card
            })
            .collect()
    }

    fn build_suggestion_cards(&self, cards: &[SuggestionCard], focused: usize) -> Vec<CardInstance> {
        let grid = CardGrid::for_palace_loop(
            self.size.width as f32,
            self.size.height as f32,
            &self.ui_scale,
        );

        cards
            .iter()
            .enumerate()
            .map(|(i, card)| {
                let (x, y) = grid.card_position(i);
                let is_focused = i == focused;

                // Use card's category color for BORDER only (alpha < 1.0 = OLED mode = black fill)
                let mut color = card.color();
                color[3] = 0.95; // OLED mode: colored border, black background

                let mut instance = CardInstance::new(x, y, grid.card_width, grid.card_height, color)
                    .with_border_width(self.ui_scale.px(if is_focused { 4.0 } else { 2.0 }))
                    .with_corner_radius(self.ui_scale.px(12.0));

                if is_focused {
                    instance = instance.selected();
                }

                // If selected for execution, use green border
                if card.selected {
                    instance = instance.with_border_color([0.2, 1.0, 0.4, 0.95]);
                }

                instance
            })
            .collect()
    }

    fn build_main_menu_cards(&self, selected: usize) -> Vec<CardInstance> {
        use crate::state::MainMenuItem;

        let items = MainMenuItem::all();

        // Modal dimensions - centered on screen
        let modal_width = self.ui_scale.px(400.0).min(self.size.width as f32 - 40.0);
        let modal_height = self.ui_scale.px(220.0);
        let modal_x = (self.size.width as f32 - modal_width) / 2.0;
        let modal_y = (self.size.height as f32 - modal_height) / 2.0;

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

        // Must match modal dimensions from build_main_menu_cards
        let modal_width = self.ui_scale.px(400.0).min(self.size.width as f32 - 40.0);
        let modal_height = self.ui_scale.px(220.0);
        let modal_x = (self.size.width as f32 - modal_width) / 2.0;
        let modal_y = (self.size.height as f32 - modal_height) / 2.0;

        let scale = self.ui_scale.px(26.0);
        let inner_padding = self.ui_scale.px(20.0);
        let title_height = self.ui_scale.px(10.0);
        let card_height = self.ui_scale.px(50.0);
        let card_gap = self.ui_scale.px(10.0);
        let text_padding = self.ui_scale.px(12.0);

        let mut sections = Vec::new();

        // Menu items (no title)
        let items = MainMenuItem::all();
        let card_start_y = modal_y + title_height;

        for (i, item) in items.iter().enumerate() {
            let y = card_start_y + i as f32 * (card_height + card_gap) + text_padding;
            let is_selected = i == selected;

            let label_color = if is_selected {
                [1.0, 1.0, 1.0, 1.0]
            } else {
                [0.8, 0.8, 0.9, 1.0]
            };

            // Center the text in the card
            sections.push(
                Section::default()
                    .add_text(
                        Text::new(item.label())
                            .with_scale(scale)
                            .with_color(label_color),
                    )
                    .with_screen_position((modal_x + inner_padding + self.ui_scale.px(15.0), y))
                    .with_layout(Layout::default()),
            );
        }

        let _ = self.text_brush.queue(&self.device, &self.queue, sections);
    }

    fn build_settings_modal_cards(&self, selected: usize) -> Vec<CardInstance> {
        use crate::state::SettingsItem;

        let items = SettingsItem::all();

        // Modal dimensions - centered on screen
        let modal_width = self.ui_scale.px(500.0).min(self.size.width as f32 - 40.0);
        let modal_height = self.ui_scale.px(250.0);
        let modal_x = (self.size.width as f32 - modal_width) / 2.0;
        let modal_y = (self.size.height as f32 - modal_height) / 2.0;

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

        // Must match modal dimensions from build_settings_modal_cards
        let modal_width = self.ui_scale.px(500.0).min(self.size.width as f32 - 40.0);
        let modal_height = self.ui_scale.px(250.0);
        let modal_x = (self.size.width as f32 - modal_width) / 2.0;
        let modal_y = (self.size.height as f32 - modal_height) / 2.0;

        let scale = self.ui_scale.px(24.0);
        let title_scale = self.ui_scale.px(32.0);
        let inner_padding = self.ui_scale.px(20.0);
        let title_height = self.ui_scale.px(50.0);
        let card_height = self.ui_scale.px(60.0);
        let card_gap = self.ui_scale.px(12.0);
        let text_padding = self.ui_scale.px(18.0);

        // NOTE: Modal text is ADDED to existing sections from base state
        // We queue these separately after base state text
        let mut sections = Vec::new();

        // Modal title
        sections.push(
            Section::default()
                .add_text(
                    Text::new("Settings")
                        .with_scale(title_scale)
                        .with_color([0.8, 0.9, 1.0, 1.0]),
                )
                .with_screen_position((modal_x + inner_padding, modal_y + self.ui_scale.px(12.0)))
                .with_layout(Layout::default()),
        );

        // Settings items
        let items = SettingsItem::all();
        let card_start_y = modal_y + title_height;

        for (i, item) in items.iter().enumerate() {
            let y = card_start_y + i as f32 * (card_height + card_gap) + text_padding;
            let is_selected = i == selected;

            let label_color = if is_selected {
                [1.0, 1.0, 1.0, 1.0]
            } else {
                [0.8, 0.8, 0.9, 1.0]
            };

            // Item label
            sections.push(
                Section::default()
                    .add_text(
                        Text::new(item.label())
                            .with_scale(scale)
                            .with_color(label_color),
                    )
                    .with_screen_position((modal_x + inner_padding + self.ui_scale.px(15.0), y))
                    .with_layout(Layout::default()),
            );

            // Value indicator on right side
            let value_text = match item {
                SettingsItem::DarkMode => if self.dark_mode { "ON" } else { "OFF" },
                SettingsItem::UiScale => "100%", // TODO: Get actual settings value
            };
            sections.push(
                Section::default()
                    .add_text(
                        Text::new(value_text)
                            .with_scale(scale * 0.85)
                            .with_color([0.5, 0.8, 0.9, 1.0]),
                    )
                    .with_screen_position((modal_x + modal_width - inner_padding - self.ui_scale.px(60.0), y))
                    .with_layout(Layout::default()),
            );
        }

        let _ = self.text_brush.queue(&self.device, &self.queue, sections);
    }

    fn queue_project_chooser_text(&mut self, projects: &ProjectsConfig, selected: usize) {
        let scale = self.ui_scale.px(32.0);
        let title_scale = self.ui_scale.px(42.0);
        let left_margin = self.ui_scale.px(60.0);
        let top_margin = self.ui_scale.px(40.0);
        let help_y = self.size.height as f32 - self.ui_scale.px(40.0);

        // Build all sections - must queue ALL at once!
        let mut sections = Vec::new();

        // Title
        sections.push(
            Section::default()
                .add_text(
                    Text::new("PALACE")
                        .with_scale(title_scale)
                        .with_color([0.8, 0.6, 1.0, 1.0]),
                )
                .with_screen_position((left_margin, top_margin))
                .with_layout(Layout::default()),
        );

        // Subtitle
        let subtitle = if projects.projects.is_empty() {
            "No projects yet"
        } else {
            "Projects"
        };
        sections.push(
            Section::default()
                .add_text(
                    Text::new(subtitle)
                        .with_scale(scale * 0.6)
                        .with_color(self.text_color_dim()),
                )
                .with_screen_position((left_margin, top_margin + title_scale + 8.0))
                .with_layout(Layout::default()),
        );

        // Grid for cards
        let grid = CardGrid::new(
            self.size.width as f32,
            self.size.height as f32,
            &self.ui_scale,
        );

        // Pre-collect all strings to ensure they live long enough
        let project_strings: Vec<_> = projects.projects.iter().map(|p| {
            let description = if p.description.is_empty() {
                p.path.to_string_lossy().to_string()
            } else {
                p.description.clone()
            };
            (p.name.clone(), description, p.languages.clone(), p.status)
        }).collect();

        for (i, (name, description, languages, status)) in project_strings.iter().enumerate() {
            let (x, y) = grid.card_position(i);
            let is_selected = i == selected;

            let text_margin = self.ui_scale.px(16.0);
            let name_scale = self.ui_scale.px(if is_selected { 22.0 } else { 20.0 });

            // Project name
            let name_color = if is_selected {
                self.text_color()
            } else {
                if self.dark_mode {
                    [0.85, 0.85, 0.9, 1.0]
                } else {
                    [0.3, 0.3, 0.35, 1.0]
                }
            };

            sections.push(
                Section::default()
                    .add_text(
                        Text::new(name)
                            .with_scale(name_scale)
                            .with_color(name_color),
                    )
                    .with_screen_position((x + text_margin, y + text_margin))
                    .with_bounds((grid.card_width - text_margin * 2.0, grid.card_height))
                    .with_layout(Layout::default()),
            );

            // Description
            let desc_y = y + text_margin + name_scale + self.ui_scale.px(8.0);
            sections.push(
                Section::default()
                    .add_text(
                        Text::new(description)
                            .with_scale(self.ui_scale.px(12.0))
                            .with_color(self.text_color_dim()),
                    )
                    .with_screen_position((x + text_margin, desc_y))
                    .with_bounds((grid.card_width - text_margin * 2.0, self.ui_scale.px(40.0)))
                    .with_layout(Layout::default()),
            );

            // Languages at bottom of card
            let lang_y = y + grid.card_height - text_margin - self.ui_scale.px(16.0);
            let mut lang_x = x + text_margin;

            for (li, lang) in languages.iter().enumerate() {
                if li > 0 {
                    sections.push(
                        Section::default()
                            .add_text(
                                Text::new(" + ")
                                    .with_scale(self.ui_scale.px(12.0))
                                    .with_color([0.4, 0.4, 0.45, 1.0]),
                            )
                            .with_screen_position((lang_x, lang_y))
                            .with_layout(Layout::default()),
                    );
                    lang_x += self.ui_scale.px(24.0);
                }

                let lang_color = crate::projects::language_color(lang);
                sections.push(
                    Section::default()
                        .add_text(
                            Text::new(lang)
                                .with_scale(self.ui_scale.px(13.0))
                                .with_color(lang_color),
                        )
                        .with_screen_position((lang_x, lang_y))
                        .with_layout(Layout::default()),
                );
                lang_x += self.ui_scale.px(lang.len() as f32 * 8.0 + 8.0);
            }

            // Status indicator (top right)
            let status_text = match status {
                crate::renderer::ProjectStatus::Unknown => "",
                crate::renderer::ProjectStatus::Building => "BUILDING",
                crate::renderer::ProjectStatus::Error => "ERROR",
                crate::renderer::ProjectStatus::Passing => "PASSING",
                crate::renderer::ProjectStatus::Active => "ACTIVE",
            };

            if !status_text.is_empty() {
                sections.push(
                    Section::default()
                        .add_text(
                            Text::new(status_text)
                                .with_scale(self.ui_scale.px(10.0))
                                .with_color(status.color()),
                        )
                        .with_screen_position((
                            x + grid.card_width - text_margin - self.ui_scale.px(60.0),
                            y + text_margin,
                        ))
                        .with_layout(Layout::default()),
                );
            }
        }

        // Help text
        let help_text = if self.gamepad_connected {
            "  Navigate        Select      Back"
        } else {
            "[Arrows/WASD] Navigate  [Enter] Select  [Esc] Exit"
        };
        sections.push(
            Section::default()
                .add_text(
                    Text::new(help_text)
                        .with_scale(scale * 0.5)
                        .with_color([0.35, 0.35, 0.4, 1.0]),
                )
                .with_screen_position((left_margin, help_y))
                .with_layout(Layout::default()),
        );

        // Empty state message
        if projects.projects.is_empty() {
            let center_y = self.size.height as f32 / 2.0;
            sections.push(
                Section::default()
                    .add_text(
                        Text::new("Launch Palace from a project directory to add it")
                            .with_scale(scale * 0.7)
                            .with_color([0.5, 0.5, 0.6, 1.0]),
                    )
                    .with_screen_position((left_margin, center_y))
                    .with_layout(Layout::default()),
            );
        }

        let _ = self.text_brush.queue(&self.device, &self.queue, sections);
    }

    fn queue_project_view_text(&mut self, project_path: &std::path::Path, selected_action: usize) {
        use crate::state::ProjectAction;

        let scale = self.ui_scale.px(32.0);
        let title_scale = self.ui_scale.px(36.0);
        let left_margin = self.ui_scale.px(60.0);
        let top_margin = self.ui_scale.px(40.0);

        let project_name = project_path
            .file_name()
            .and_then(|n| n.to_str())
            .unwrap_or("Unknown")
            .to_string();
        let path_str = project_path.to_string_lossy().to_string();

        let actions = ProjectAction::all();
        let card_height = self.ui_scale.px(70.0);
        let card_gap = self.ui_scale.px(16.0);
        let menu_start_y = top_margin + title_scale + self.ui_scale.px(60.0);
        let text_margin = self.ui_scale.px(20.0);
        let help_y = self.size.height as f32 - self.ui_scale.px(40.0);

        // Build all sections - must queue ALL at once, not separately!
        let mut sections = Vec::new();

        // Title
        sections.push(
            Section::default()
                .add_text(
                    Text::new(&project_name)
                        .with_scale(title_scale)
                        .with_color([1.0, 1.0, 1.0, 1.0]),
                )
                .with_screen_position((left_margin, top_margin))
                .with_layout(Layout::default()),
        );

        // Path subtitle
        sections.push(
            Section::default()
                .add_text(
                    Text::new(&path_str)
                        .with_scale(self.ui_scale.px(14.0))
                        .with_color([0.4, 0.4, 0.5, 1.0]),
                )
                .with_screen_position((left_margin, top_margin + title_scale + self.ui_scale.px(8.0)))
                .with_layout(Layout::default()),
        );

        // Action labels and descriptions
        for (i, action) in actions.iter().enumerate() {
            let y = menu_start_y + i as f32 * (card_height + card_gap);
            let is_selected = i == selected_action;
            let label_color = if is_selected {
                [1.0, 1.0, 1.0, 1.0]
            } else {
                [0.75, 0.75, 0.8, 1.0]
            };

            sections.push(
                Section::default()
                    .add_text(
                        Text::new(action.label())
                            .with_scale(self.ui_scale.px(22.0))
                            .with_color(label_color),
                    )
                    .with_screen_position((left_margin + text_margin, y + text_margin))
                    .with_layout(Layout::default()),
            );

            sections.push(
                Section::default()
                    .add_text(
                        Text::new(action.description())
                            .with_scale(self.ui_scale.px(12.0))
                            .with_color([0.45, 0.45, 0.5, 1.0]),
                    )
                    .with_screen_position((left_margin + text_margin, y + text_margin + self.ui_scale.px(28.0)))
                    .with_layout(Layout::default()),
            );
        }

        // Help text
        let help_text = if self.gamepad_connected {
            "  Navigate        Select      Back"
        } else {
            "[Arrows] Navigate  [Enter] Select  [Backspace] Back"
        };
        sections.push(
            Section::default()
                .add_text(
                    Text::new(help_text)
                        .with_scale(scale * 0.6)
                        .with_color([0.4, 0.4, 0.5, 1.0]),
                )
                .with_screen_position((left_margin, help_y))
                .with_layout(Layout::default()),
        );

        let _ = self.text_brush.queue(&self.device, &self.queue, sections);
    }

    fn queue_palace_loop_text(&mut self, cards: &[SuggestionCard], current_tool: Option<&str>) {
        let title_scale = self.ui_scale.px(42.0);
        let left_margin = self.ui_scale.px(60.0);
        let top_margin = self.ui_scale.px(40.0);
        let help_y = self.size.height as f32 - self.ui_scale.px(40.0);

        let grid = CardGrid::for_palace_loop(
            self.size.width as f32,
            self.size.height as f32,
            &self.ui_scale,
        );

        let mut sections = Vec::new();

        // Title
        sections.push(
            Section::default()
                .add_text(
                    Text::new("PALACE LOOP")
                        .with_scale(title_scale)
                        .with_color([0.8, 0.6, 1.0, 1.0]),
                )
                .with_screen_position((left_margin, top_margin))
                .with_layout(Layout::default()),
        );

        // Subtitle with current tool if any
        let subtitle = if let Some(tool) = current_tool {
            format!("Analyzing... {}", tool)
        } else if cards.is_empty() {
            "Gathering suggestions...".to_string()
        } else {
            format!("{} suggestions", cards.len())
        };
        sections.push(
            Section::default()
                .add_text(
                    Text::new(&subtitle)
                        .with_scale(self.ui_scale.px(18.0))
                        .with_color(self.text_color_dim()),
                )
                .with_screen_position((left_margin, top_margin + title_scale + 8.0))
                .with_layout(Layout::default()),
        );

        // Pre-collect strings that need to outlive the loop
        let card_strings: Vec<_> = cards.iter().map(|card| {
            let category_upper = card.category.to_uppercase();
            let display_cmd = card.command.as_ref().map(|cmd| {
                if cmd.len() > 40 {
                    format!("$ {}...", &cmd[..37])
                } else {
                    format!("$ {}", cmd)
                }
            });
            (category_upper, display_cmd)
        }).collect();

        // Card text
        let text_margin = self.ui_scale.px(16.0);
        for (i, card) in cards.iter().enumerate() {
            let (x, y) = grid.card_position(i);
            let (category_upper, display_cmd) = &card_strings[i];

            // Category badge (bottom right, near border)
            if !card.category.is_empty() {
                sections.push(
                    Section::default()
                        .add_text(
                            Text::new(category_upper)
                                .with_scale(self.ui_scale.px(10.0))
                                .with_color(card.color()),
                        )
                        .with_screen_position((
                            x + grid.card_width - text_margin - self.ui_scale.px(40.0),
                            y + grid.card_height - text_margin - self.ui_scale.px(12.0),
                        ))
                        .with_layout(Layout::default()),
                );
            }

            // Title
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
            sections.push(
                Section::default()
                    .add_text(
                        Text::new(display_title)
                            .with_scale(self.ui_scale.px(20.0))
                            .with_color(title_color),
                    )
                    .with_screen_position((x + text_margin, y + text_margin))
                    .with_bounds((grid.card_width - text_margin * 2.0, grid.card_height))
                    .with_layout(Layout::default()),
            );

            // Description
            if !card.description.is_empty() {
                let desc_y = y + text_margin + self.ui_scale.px(28.0);
                sections.push(
                    Section::default()
                        .add_text(
                            Text::new(&card.description)
                                .with_scale(self.ui_scale.px(12.0))
                                .with_color(self.text_color_dim()),
                        )
                        .with_screen_position((x + text_margin, desc_y))
                        .with_bounds((grid.card_width - text_margin * 2.0, self.ui_scale.px(60.0)))
                        .with_layout(Layout::default()),
                );
            }

            // Command at bottom if present
            if let Some(ref cmd) = display_cmd {
                let cmd_y = y + grid.card_height - text_margin - self.ui_scale.px(16.0);
                sections.push(
                    Section::default()
                        .add_text(
                            Text::new(cmd)
                                .with_scale(self.ui_scale.px(11.0))
                                .with_color([0.4, 0.8, 0.5, 1.0]), // Green for commands
                        )
                        .with_screen_position((x + text_margin, cmd_y))
                        .with_layout(Layout::default()),
                );
            }

            // Selected indicator (checkmark)
            if card.selected {
                sections.push(
                    Section::default()
                        .add_text(
                            Text::new("✓")
                                .with_scale(self.ui_scale.px(24.0))
                                .with_color([0.2, 1.0, 0.4, 1.0]),
                        )
                        .with_screen_position((
                            x + grid.card_width - text_margin - self.ui_scale.px(24.0),
                            y + grid.card_height - text_margin - self.ui_scale.px(24.0),
                        ))
                        .with_layout(Layout::default()),
                );
            }
        }

        // Help text
        let help_text = if self.gamepad_connected {
            "  Navigate      Toggle       Execute      Back"
        } else {
            "[Arrows] Navigate  [Space] Toggle  [Enter] Execute  [Esc] Back"
        };
        sections.push(
            Section::default()
                .add_text(
                    Text::new(help_text)
                        .with_scale(self.ui_scale.px(16.0))
                        .with_color([0.35, 0.35, 0.4, 1.0]),
                )
                .with_screen_position((left_margin, help_y))
                .with_layout(Layout::default()),
        );

        let _ = self.text_brush.queue(&self.device, &self.queue, sections);
    }

    fn build_help_sprites(&self, state: &AppState) -> Vec<SpriteInstance> {
        let mut sprites = Vec::new();
        let glyph_size = self.ui_scale.px(24.0);
        let left_margin = self.ui_scale.px(60.0);

        match state {
            AppState::ProjectChooser { .. } => {
                let help_y = self.size.height as f32 - self.ui_scale.px(40.0) - glyph_size * 0.25;

                // Navigation icon
                sprites.push(SpriteInstance::new(left_margin, help_y, glyph_size, XboxButton::DPad));

                // A button for select (positioned after "Navigate" text)
                sprites.push(SpriteInstance::new(
                    left_margin + self.ui_scale.px(140.0),
                    help_y,
                    glyph_size,
                    XboxButton::A,
                ));

                // B button for back (positioned after "Select" text)
                sprites.push(SpriteInstance::new(
                    left_margin + self.ui_scale.px(230.0),
                    help_y,
                    glyph_size,
                    XboxButton::B,
                ));
            }
            AppState::ProjectView { .. } => {
                let help_y = self.size.height as f32 - self.ui_scale.px(40.0) - glyph_size * 0.25;

                // D-Pad for navigation
                sprites.push(SpriteInstance::new(left_margin, help_y, glyph_size, XboxButton::DPad));

                // A button for select
                sprites.push(SpriteInstance::new(
                    left_margin + self.ui_scale.px(140.0),
                    help_y,
                    glyph_size,
                    XboxButton::A,
                ));

                // B button for back
                sprites.push(SpriteInstance::new(
                    left_margin + self.ui_scale.px(230.0),
                    help_y,
                    glyph_size,
                    XboxButton::B,
                ));
            }
            AppState::MainMenu { .. } | AppState::SettingsMenu { .. } => {
                let help_y = self.size.height as f32 - self.ui_scale.px(40.0) - glyph_size * 0.25;

                // D-Pad for navigation
                sprites.push(SpriteInstance::new(left_margin, help_y, glyph_size, XboxButton::DPad));

                // A button for select/toggle
                sprites.push(SpriteInstance::new(
                    left_margin + self.ui_scale.px(140.0),
                    help_y,
                    glyph_size,
                    XboxButton::A,
                ));

                // B button for back
                sprites.push(SpriteInstance::new(
                    left_margin + self.ui_scale.px(230.0),
                    help_y,
                    glyph_size,
                    XboxButton::B,
                ));
            }
            AppState::PalaceLoop { .. } => {
                let help_y = self.size.height as f32 - self.ui_scale.px(40.0) - glyph_size * 0.25;

                // D-Pad for navigation
                sprites.push(SpriteInstance::new(left_margin, help_y, glyph_size, XboxButton::DPad));

                // A button for toggle selection
                sprites.push(SpriteInstance::new(
                    left_margin + self.ui_scale.px(130.0),
                    help_y,
                    glyph_size,
                    XboxButton::A,
                ));

                // X button for execute
                sprites.push(SpriteInstance::new(
                    left_margin + self.ui_scale.px(220.0),
                    help_y,
                    glyph_size,
                    XboxButton::X,
                ));

                // B button for back
                sprites.push(SpriteInstance::new(
                    left_margin + self.ui_scale.px(320.0),
                    help_y,
                    glyph_size,
                    XboxButton::B,
                ));
            }
        }

        sprites
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

    /// Poll pending screenshot captures (call each frame)
    pub fn poll_screenshots(&self, capture: &mut crate::debug::ScreenshotCapture) {
        capture.poll_pending(&self.device);
    }
}
