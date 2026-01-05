use crate::projects::ProjectsConfig;
use crate::renderer::cards::{CardInstance, CardRenderer};
use crate::renderer::sprites::{SpriteInstance, SpriteRenderer, XboxButton};
use crate::renderer::text::{PreparedText, TextQueue, TextRequest};
use crate::state::{AppState, SuggestionCard};
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
            tracing::debug!("Resized to {}x{}", new_size.width, new_size.height);
        }
    }

    /// Take queued text, prepare buffers, and upload to GPU
    /// Returns PreparedText that must be kept alive until after render
    fn prepare_text(&mut self) -> PreparedText {
        let requests = self.text_queue.take();
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

        let prepared = PreparedText::prepare(requests, &mut self.font_system, screen_width);

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
            UiScale(usize),
            Permission { selected: usize, command: String, prefix: String },
            Execute(usize),
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
            AppState::UiScaleMenu { previous_state, selected_item } => {
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
                (actual_base, Some(ModalType::UiScale(*selected_item)))
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
            AppState::MainMenu { .. } | AppState::SettingsMenu { .. } | AppState::UiScaleMenu { .. } | AppState::PermissionModal { .. } | AppState::ExecuteModal { .. } => Vec::new(),
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
                AppState::PalaceLoop { cards, current_tool, tool_log, thought_log, log_scroll_offset, focused_index, detail_scroll_offset, .. } => {
                    self.queue_palace_loop_text(cards, current_tool.as_deref(), tool_log, thought_log, *log_scroll_offset, *focused_index, *detail_scroll_offset);
                }
                AppState::MainMenu { .. } | AppState::SettingsMenu { .. } | AppState::UiScaleMenu { .. } | AppState::PermissionModal { .. } | AppState::ExecuteModal { .. } => {}
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

                self.card_renderer
                    .draw(&mut render_pass, &self.queue, &base_cards);

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
                ModalType::UiScale(selected) => self.queue_ui_scale_modal_text(*selected),
                ModalType::Permission { selected, command, prefix } => self.queue_permission_modal_text(*selected, command, prefix),
                ModalType::Execute(selected) => self.queue_execute_modal_text(*selected),
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
                ModalType::UiScale(selected) => modal_cards.extend(self.build_ui_scale_modal_cards(*selected)),
                ModalType::Permission { selected, command, prefix: _ } => modal_cards.extend(self.build_permission_modal_cards(*selected, command)),
                ModalType::Execute(selected) => modal_cards.extend(self.build_execute_modal_cards(*selected)),
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
                AppState::ProjectChooser { selected_index } => {
                    self.queue_project_chooser_text(projects, *selected_index);
                }
                AppState::ProjectView { project_path, selected_action } => {
                    self.queue_project_view_text(project_path, *selected_action);
                }
                AppState::PalaceLoop { cards, current_tool, tool_log, thought_log, log_scroll_offset, focused_index, detail_scroll_offset, .. } => {
                    self.queue_palace_loop_text(cards, current_tool.as_deref(), tool_log, thought_log, *log_scroll_offset, *focused_index, *detail_scroll_offset);
                }
                AppState::MainMenu { .. } => {}
                AppState::SettingsMenu { .. } => {}
                AppState::UiScaleMenu { .. } => {}
                AppState::PermissionModal { .. } => {}
                AppState::ExecuteModal { .. } => {}
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

                self.card_renderer
                    .draw(&mut render_pass, &self.queue, &base_cards);

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
                AppState::ProjectChooser { selected_index } => {
                    self.queue_project_chooser_text(projects, *selected_index);
                }
                AppState::ProjectView { project_path, selected_action } => {
                    self.queue_project_view_text(project_path, *selected_action);
                }
                AppState::PalaceLoop { cards, current_tool, tool_log, thought_log, log_scroll_offset, focused_index, detail_scroll_offset, .. } => {
                    self.queue_palace_loop_text(cards, current_tool.as_deref(), tool_log, thought_log, *log_scroll_offset, *focused_index, *detail_scroll_offset);
                }
                AppState::MainMenu { .. } => {}
                AppState::SettingsMenu { .. } => {}
                AppState::UiScaleMenu { .. } => {}
                AppState::PermissionModal { .. } => {}
                AppState::ExecuteModal { .. } => {}
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

                self.card_renderer
                    .draw(&mut render_pass, &self.queue, &base_cards);

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

        let modal_width = self.ui_scale.px(400.0).min(self.size.width as f32 - 40.0);
        let modal_x = (self.size.width as f32 - modal_width) / 2.0;
        let modal_y = (self.size.height as f32 - self.ui_scale.px(220.0)) / 2.0;

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
            let value_text = match item {
                SettingsItem::DarkMode => if self.dark_mode { "ON" } else { "OFF" },
                SettingsItem::UiScale => "100%",
            };
            self.text_queue.push(
                value_text,
                modal_x + modal_width - inner_padding - self.ui_scale.px(60.0),
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

        // Modal dimensions - centered on screen
        let modal_width = self.ui_scale.px(400.0).min(self.size.width as f32 - 40.0);
        let item_count = items.len() as f32;
        let card_height = self.ui_scale.px(50.0);
        let card_gap = self.ui_scale.px(8.0);
        let inner_padding = self.ui_scale.px(20.0);
        let title_height = self.ui_scale.px(50.0);
        let modal_height = title_height + inner_padding + item_count * (card_height + card_gap);
        let modal_x = (self.size.width as f32 - modal_width) / 2.0;
        let modal_y = (self.size.height as f32 - modal_height) / 2.0;

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

        // Dynamic modal dimensions based on command length
        let max_width = self.ui_scale.px(800.0).min(self.size.width as f32 * 0.9);
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
        let modal_x = (self.size.width as f32 - modal_width) / 2.0;
        let modal_y = (self.size.height as f32 - modal_height) / 2.0;

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

        // Same sizing as build_permission_modal_cards
        let max_width = self.ui_scale.px(800.0).min(self.size.width as f32 * 0.9);
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
        let modal_x = (self.size.width as f32 - modal_width) / 2.0;
        let modal_y = (self.size.height as f32 - modal_height) / 2.0;

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

    fn queue_ui_scale_modal_text(&mut self, selected: usize) {
        use crate::state::UiScaleOption;

        let items = UiScaleOption::all();

        // Must match modal dimensions from build_ui_scale_modal_cards
        let modal_width = self.ui_scale.px(400.0).min(self.size.width as f32 - 40.0);
        let item_count = items.len() as f32;
        let card_height = self.ui_scale.px(50.0);
        let card_gap = self.ui_scale.px(8.0);
        let inner_padding = self.ui_scale.px(20.0);
        let title_height = self.ui_scale.px(50.0);
        let modal_height = title_height + inner_padding + item_count * (card_height + card_gap);
        let modal_x = (self.size.width as f32 - modal_width) / 2.0;
        let modal_y = (self.size.height as f32 - modal_height) / 2.0;

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

            // Show current indicator if this is the current scale
            if item.value() == self.ui_scale.dpi_scale {
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

        let items = PermissionChoice::all();

        // Dynamic modal dimensions - must match build_permission_modal_cards
        let max_width = self.ui_scale.px(800.0).min(self.size.width as f32 * 0.9);
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
        let modal_x = (self.size.width as f32 - modal_width) / 2.0;
        let modal_y = (self.size.height as f32 - modal_height) / 2.0;

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
                generating: false,
                current_tool: None,
                tool_log: Vec::new(),
                thought_log: Vec::new(),
                log_scroll_offset: 0,
                detail_scroll_offset: 0.0,
            }),
        };
        self.queue_help_legend(&help_state);
    }

    fn build_execute_modal_cards(&self, selected: usize) -> Vec<CardInstance> {
        use crate::state::ExecuteOption;
        let items = ExecuteOption::all();

        // Modal dimensions - centered on screen
        let modal_width = self.ui_scale.px(400.0).min(self.size.width as f32 - 40.0);
        let item_count = items.len() as f32;
        let card_height = self.ui_scale.px(50.0);
        let card_gap = self.ui_scale.px(8.0);
        let inner_padding = self.ui_scale.px(20.0);
        let title_height = self.ui_scale.px(50.0);
        let modal_height = title_height + inner_padding + item_count * (card_height + card_gap);
        let modal_x = (self.size.width as f32 - modal_width) / 2.0;
        let modal_y = (self.size.height as f32 - modal_height) / 2.0;

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

        let items = ExecuteOption::all();

        // Modal dimensions - match build_execute_modal_cards
        let modal_width = self.ui_scale.px(400.0).min(self.size.width as f32 - 40.0);
        let item_count = items.len() as f32;
        let card_height = self.ui_scale.px(50.0);
        let card_gap = self.ui_scale.px(8.0);
        let inner_padding = self.ui_scale.px(20.0);
        let title_height = self.ui_scale.px(50.0);
        let modal_height = title_height + inner_padding + item_count * (card_height + card_gap);
        let modal_x = (self.size.width as f32 - modal_width) / 2.0;
        let modal_y = (self.size.height as f32 - modal_height) / 2.0;

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
                generating: false,
                current_tool: None,
                tool_log: Vec::new(),
                thought_log: Vec::new(),
                log_scroll_offset: 0,
                detail_scroll_offset: 0.0,
            }),
        };
        self.queue_help_legend(&help_state);
    }

    fn queue_project_chooser_text(&mut self, projects: &ProjectsConfig, selected: usize) {
        let scale = self.ui_scale.px(32.0);
        let title_scale = self.ui_scale.px(42.0);
        let left_margin = self.ui_scale.px(60.0);
        let top_margin = self.ui_scale.px(40.0);

        // Title
        self.text_queue.push("PALACE", left_margin, top_margin, title_scale, [0.8, 0.6, 1.0, 1.0]);

        // Subtitle
        let subtitle = if projects.projects.is_empty() {
            "No projects yet"
        } else {
            "Projects"
        };
        let dim_color = self.text_color_dim();
        self.text_queue.push(subtitle, left_margin, top_margin + title_scale + 8.0, scale * 0.6, dim_color);

        // Grid for cards
        let grid = CardGrid::new(
            self.size.width as f32,
            self.size.height as f32,
            &self.ui_scale,
        );

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
        self.queue_help_legend(&AppState::ProjectChooser { selected_index: selected });

        // Empty state message
        if projects.projects.is_empty() {
            let center_y = self.size.height as f32 / 2.0;
            self.text_queue.push(
                "Launch Palace from a project directory to add it",
                left_margin,
                center_y,
                scale * 0.7,
                [0.5, 0.5, 0.6, 1.0],
            );
        }
    }

    fn queue_project_view_text(&mut self, project_path: &std::path::Path, selected_action: usize) {
        use crate::state::ProjectAction;

        let title_scale = self.ui_scale.px(36.0);
        let left_margin = self.ui_scale.px(60.0);
        let top_margin = self.ui_scale.px(40.0);

        let project_name = project_path
            .file_name()
            .and_then(|n| n.to_str())
            .unwrap_or("Unknown");
        let path_str = project_path.to_string_lossy();

        let actions = ProjectAction::all();
        let card_height = self.ui_scale.px(70.0);
        let card_gap = self.ui_scale.px(16.0);
        let menu_start_y = top_margin + title_scale + self.ui_scale.px(60.0);
        let text_margin = self.ui_scale.px(20.0);

        // Title
        self.text_queue.push(project_name, left_margin, top_margin, title_scale, [1.0, 1.0, 1.0, 1.0]);

        // Path subtitle
        self.text_queue.push(
            path_str.as_ref(),
            left_margin,
            top_margin + title_scale + self.ui_scale.px(8.0),
            self.ui_scale.px(14.0),
            [0.4, 0.4, 0.5, 1.0],
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

            self.text_queue.push(
                action.label(),
                left_margin + text_margin,
                y + text_margin,
                self.ui_scale.px(22.0),
                label_color,
            );

            self.text_queue.push(
                action.description(),
                left_margin + text_margin,
                y + text_margin + self.ui_scale.px(28.0),
                self.ui_scale.px(12.0),
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

    fn queue_palace_loop_text(&mut self, cards: &[SuggestionCard], current_tool: Option<&str>, tool_log: &[String], thought_log: &[String], log_scroll: usize, _focused_index: usize, _detail_scroll: f32) {
        let title_scale = self.ui_scale.px(42.0);
        let left_margin = self.ui_scale.px(60.0);
        let top_margin = self.ui_scale.px(40.0);

        let grid = CardGrid::for_palace_loop(
            self.size.width as f32,
            self.size.height as f32,
            &self.ui_scale,
        );

        // Title
        self.text_queue.push("PALACE LOOP", left_margin, top_margin, title_scale, [0.8, 0.6, 1.0, 1.0]);

        // Subtitle with current tool if any
        let subtitle_y = top_margin + title_scale + 8.0;
        let subtitle = if let Some(tool) = current_tool {
            format!("Analyzing... {}", tool)
        } else if cards.is_empty() {
            "Gathering suggestions...".to_string()
        } else {
            format!("{} suggestions", cards.len())
        };
        let dim_color = self.text_color_dim();
        self.text_queue.push(&subtitle, left_margin, subtitle_y, self.ui_scale.px(18.0), dim_color);

        // Two-column waterfall logs (tools left, thoughts right)
        let line_height = self.ui_scale.px(16.0);
        let log_scale = self.ui_scale.px(13.0);
        let log_y_start = subtitle_y + self.ui_scale.px(28.0);

        // Calculate max lines to fill available screen height
        let available_height = self.size.height as f32 - log_y_start - self.ui_scale.px(40.0);
        let max_lines = (available_height / line_height).floor() as usize;

        // Split screen: left half for tools, right half for thoughts
        let screen_mid = self.size.width as f32 / 2.0;
        let right_margin = self.ui_scale.px(40.0);
        let approx_char_width = log_scale * 0.5;
        let left_max_chars = ((screen_mid - left_margin - self.ui_scale.px(20.0)) / approx_char_width) as usize;
        let right_max_chars = ((screen_mid - right_margin) / approx_char_width) as usize;

        // Left column: Tool calls (with proper text measurement for wrapping)
        let left_col_width = screen_mid - left_margin - self.ui_scale.px(20.0);
        if !tool_log.is_empty() {
            let visible_start = log_scroll.min(tool_log.len().saturating_sub(1));
            let mut y_offset = 0.0;
            let mut entry_idx = 0;
            for entry in tool_log.iter().skip(visible_start) {
                if y_offset >= available_height {
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
                let y = log_y_start + y_offset;
                self.text_queue.push_bounded(entry, left_margin, y, log_scale, [0.5, 0.7, 0.9, alpha], left_col_width, h);
                y_offset += h;
                entry_idx += 1;
            }
        }

        // Right column: AI thoughts/commentary (with proper text measurement for wrapping)
        let right_col_width = screen_mid - right_margin;
        if !thought_log.is_empty() {
            let visible_start = log_scroll.min(thought_log.len().saturating_sub(1));
            let mut y_offset = 0.0;
            let mut entry_idx = 0;
            for entry in thought_log.iter().skip(visible_start) {
                if y_offset >= available_height {
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
                let y = log_y_start + y_offset;
                self.text_queue.push_bounded(entry, screen_mid + self.ui_scale.px(10.0), y, log_scale, [0.7, 0.6, 0.8, alpha], right_col_width, h);
                y_offset += h;
                entry_idx += 1;
            }
        }

        // Card text
        let text_margin = self.ui_scale.px(16.0);
        for (i, card) in cards.iter().enumerate() {
            let (x, y) = grid.card_position(i);

            // Category badge (bottom right, near border)
            if !card.category.is_empty() {
                self.text_queue.push(
                    &card.category.to_uppercase(),
                    x + grid.card_width - text_margin - self.ui_scale.px(40.0),
                    y + grid.card_height - text_margin - self.ui_scale.px(12.0),
                    self.ui_scale.px(10.0),
                    card.color(),
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
            self.text_queue.push_bounded(
                display_title,
                x + text_margin,
                y + text_margin,
                self.ui_scale.px(20.0),
                title_color,
                grid.card_width - text_margin * 2.0,
                grid.card_height,
            );

            // Description
            if !card.description.is_empty() {
                let desc_y = y + text_margin + self.ui_scale.px(28.0);
                let dim_color = self.text_color_dim();
                self.text_queue.push_bounded(
                    &card.description,
                    x + text_margin,
                    desc_y,
                    self.ui_scale.px(12.0),
                    dim_color,
                    grid.card_width - text_margin * 2.0,
                    self.ui_scale.px(60.0),
                );
            }

            // Command at bottom if present
            if let Some(ref cmd) = card.command {
                let display_cmd = if cmd.len() > 40 {
                    format!("$ {}...", &cmd[..37])
                } else {
                    format!("$ {}", cmd)
                };
                let cmd_y = y + grid.card_height - text_margin - self.ui_scale.px(16.0);
                self.text_queue.push(&display_cmd, x + text_margin, cmd_y, self.ui_scale.px(11.0), [0.4, 0.8, 0.5, 1.0]);
            }

            // Selected indicator (checkmark)
            if card.selected {
                self.text_queue.push(
                    "✓",
                    x + grid.card_width - text_margin - self.ui_scale.px(24.0),
                    y + grid.card_height - text_margin - self.ui_scale.px(24.0),
                    self.ui_scale.px(24.0),
                    [0.2, 1.0, 0.4, 1.0],
                );
            }
        }

        // Add help legend
        let help_state = AppState::PalaceLoop {
            project_path: std::path::PathBuf::new(),
            cards: Vec::new(),
            focused_index: 0,
            generating: false,
            current_tool: None,
            tool_log: Vec::new(),
            thought_log: Vec::new(),
            log_scroll_offset: 0,
            detail_scroll_offset: 0.0,
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
        }
    }

    fn build_help_sprites(&self, state: &AppState) -> Vec<SpriteInstance> {
        let items = self.get_help_items(state);
        let mut sprites = Vec::new();

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

        let start_x = self.size.width as f32 - right_margin - total_width;
        let y = top_margin;

        for (i, (button, _label)) in items.iter().enumerate() {
            let x = start_x + i as f32 * (item_width + item_gap);
            sprites.push(SpriteInstance::new(x, y, glyph_size, *button));
        }

        sprites
    }

    /// Build help legend text sections (labels after glyphs) - top right corner
    fn queue_help_legend(&mut self, state: &AppState) {
        if !self.gamepad_connected {
            return;
        }

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
        let start_x = self.size.width as f32 - right_margin - total_width;
        let y = top_margin + (glyph_size - label_scale) / 2.0;

        for (i, (_, label)) in items.iter().enumerate() {
            let x = start_x + i as f32 * (item_width + item_gap) + glyph_size + inner_gap;
            self.text_queue.push(*label, x, y, label_scale, [0.6, 0.6, 0.65, 1.0]);
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

    /// Poll pending screenshot captures (call each frame)
    pub fn poll_screenshots(&self, capture: &mut crate::debug::ScreenshotCapture) {
        capture.poll_pending(&self.device);
    }
}
