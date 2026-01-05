use bytemuck::{Pod, Zeroable};
use wgpu::util::DeviceExt;
use std::path::Path;

// Xbox button assets by Arks: https://arks.itch.io/xbox-buttons
const ASSETS_DIR: &str = "/home/wings/assets/XBOX BUTTONS - Premium Assets/XBOX BUTTONS - Premium Assets/Svg";

/// Xbox controller button types
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum XboxButton {
    A,
    B,
    X,
    Y,
    LB,
    RB,
    LT,
    RT,
    DPad,
    DPadUp,
    DPadDown,
    DPadLeft,
    DPadRight,
    LeftStick,
    RightStick,
    Start,
    Back,
}

impl XboxButton {
    /// Get the SVG filename for this button
    fn svg_filename(&self) -> &'static str {
        match self {
            XboxButton::A => "button_xbox_digital_a_1.svg",
            XboxButton::B => "button_xbox_digital_b_1.svg",
            XboxButton::X => "button_xbox_digital_x_1.svg",
            XboxButton::Y => "button_xbox_digital_y_1.svg",
            XboxButton::LB => "button_xbox_digital_bumper_dark_1.svg",
            XboxButton::RB => "button_xbox_digital_bumper_dark_2.svg",
            XboxButton::LT => "button_xbox_analog_trigger_dark_1.svg",
            XboxButton::RT => "button_xbox_analog_trigger_dark_2.svg",
            XboxButton::DPad => "button_xbox_dpad_dark_1.svg",
            XboxButton::DPadUp => "button_xbox_dpad_dark_3.svg",
            XboxButton::DPadDown => "button_xbox_dpad_dark_5.svg",
            XboxButton::DPadLeft => "button_xbox_dpad_dark_7.svg",
            XboxButton::DPadRight => "button_xbox_dpad_dark_4.svg",
            XboxButton::LeftStick => "button_xbox_analog_l.svg",
            XboxButton::RightStick => "button_xbox_analog_r.svg",
            XboxButton::Start => "button_xbox_digital_start_1.svg",
            XboxButton::Back => "button_xbox_digital_back_1.svg",
        }
    }

    /// Get the UV coordinates for this button in the sprite atlas
    /// Atlas layout: 4x4 grid of 64x64 icons
    pub fn uv_rect(&self) -> [f32; 4] {
        let index = match self {
            XboxButton::A => 0,
            XboxButton::B => 1,
            XboxButton::X => 2,
            XboxButton::Y => 3,
            XboxButton::LB => 4,
            XboxButton::RB => 5,
            XboxButton::LT => 6,
            XboxButton::RT => 7,
            XboxButton::DPad => 8,
            XboxButton::DPadUp => 9,
            XboxButton::DPadDown => 10,
            XboxButton::DPadLeft => 11,
            XboxButton::DPadRight => 12,
            XboxButton::LeftStick => 13,
            XboxButton::RightStick => 14,
            XboxButton::Start => 15,
            XboxButton::Back => 15, // Same as start
        };

        let col = index % 4;
        let row = index / 4;
        let u = col as f32 / 4.0;
        let v = row as f32 / 4.0;
        let size = 1.0 / 4.0;

        [u, v, u + size, v + size]
    }

    /// Get color tint for this button (white = no tint, use original colors)
    pub fn color(&self) -> [f32; 4] {
        [1.0, 1.0, 1.0, 1.0] // No tint - use SVG colors
    }

    /// All buttons in atlas order
    fn all_in_order() -> [XboxButton; 16] {
        [
            XboxButton::A,
            XboxButton::B,
            XboxButton::X,
            XboxButton::Y,
            XboxButton::LB,
            XboxButton::RB,
            XboxButton::LT,
            XboxButton::RT,
            XboxButton::DPad,
            XboxButton::DPadUp,
            XboxButton::DPadDown,
            XboxButton::DPadLeft,
            XboxButton::DPadRight,
            XboxButton::LeftStick,
            XboxButton::RightStick,
            XboxButton::Start,
        ]
    }
}

/// Sprite instance data
#[repr(C)]
#[derive(Clone, Copy, Debug, Pod, Zeroable)]
pub struct SpriteInstance {
    pub position: [f32; 2],  // Screen position (pixels)
    pub size: [f32; 2],      // Size in pixels
    pub uv_rect: [f32; 4],   // u_min, v_min, u_max, v_max
    pub color: [f32; 4],     // Tint color
}

impl SpriteInstance {
    pub fn new(x: f32, y: f32, size: f32, button: XboxButton) -> Self {
        Self {
            position: [x, y],
            size: [size, size],
            uv_rect: button.uv_rect(),
            color: button.color(),
        }
    }
}

#[repr(C)]
#[derive(Clone, Copy, Debug, Pod, Zeroable)]
struct Uniforms {
    screen_size: [f32; 2],
    _padding: [f32; 2],
}

#[repr(C)]
#[derive(Clone, Copy, Debug, Pod, Zeroable)]
struct Vertex {
    position: [f32; 2],
    uv: [f32; 2],
}

const QUAD_VERTICES: &[Vertex] = &[
    Vertex { position: [0.0, 0.0], uv: [0.0, 0.0] },
    Vertex { position: [1.0, 0.0], uv: [1.0, 0.0] },
    Vertex { position: [1.0, 1.0], uv: [1.0, 1.0] },
    Vertex { position: [0.0, 1.0], uv: [0.0, 1.0] },
];

const QUAD_INDICES: &[u16] = &[0, 1, 2, 0, 2, 3];

pub struct SpriteRenderer {
    pipeline: wgpu::RenderPipeline,
    vertex_buffer: wgpu::Buffer,
    index_buffer: wgpu::Buffer,
    instance_buffer: wgpu::Buffer,
    uniform_buffer: wgpu::Buffer,
    bind_group: wgpu::BindGroup,
    #[allow(dead_code)]
    texture: wgpu::Texture,
    max_instances: usize,
    screen_size: [f32; 2],
}

impl SpriteRenderer {
    pub fn new(
        device: &wgpu::Device,
        queue: &wgpu::Queue,
        format: wgpu::TextureFormat,
        width: u32,
        height: u32,
    ) -> Self {
        // Create texture atlas from SVG files
        let cell_size = 64u32;
        let texture_size = cell_size * 4; // 4x4 grid = 256x256

        let texture = device.create_texture(&wgpu::TextureDescriptor {
            label: Some("Xbox Glyph Atlas"),
            size: wgpu::Extent3d {
                width: texture_size,
                height: texture_size,
                depth_or_array_layers: 1,
            },
            mip_level_count: 1,
            sample_count: 1,
            dimension: wgpu::TextureDimension::D2,
            format: wgpu::TextureFormat::Rgba8UnormSrgb,
            usage: wgpu::TextureUsages::TEXTURE_BINDING | wgpu::TextureUsages::COPY_DST,
            view_formats: &[],
        });

        // Load SVG assets into texture atlas
        let atlas_data = Self::build_atlas_from_svgs(texture_size as usize, cell_size as usize);
        queue.write_texture(
            wgpu::TexelCopyTextureInfo {
                texture: &texture,
                mip_level: 0,
                origin: wgpu::Origin3d::ZERO,
                aspect: wgpu::TextureAspect::All,
            },
            &atlas_data,
            wgpu::TexelCopyBufferLayout {
                offset: 0,
                bytes_per_row: Some(texture_size * 4),
                rows_per_image: Some(texture_size),
            },
            wgpu::Extent3d {
                width: texture_size,
                height: texture_size,
                depth_or_array_layers: 1,
            },
        );

        let texture_view = texture.create_view(&wgpu::TextureViewDescriptor::default());
        let sampler = device.create_sampler(&wgpu::SamplerDescriptor {
            address_mode_u: wgpu::AddressMode::ClampToEdge,
            address_mode_v: wgpu::AddressMode::ClampToEdge,
            mag_filter: wgpu::FilterMode::Linear,
            min_filter: wgpu::FilterMode::Linear,
            ..Default::default()
        });

        // Shader
        let shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("Sprite Shader"),
            source: wgpu::ShaderSource::Wgsl(include_str!("shaders/sprite.wgsl").into()),
        });

        // Uniform buffer
        let screen_size = [width as f32, height as f32];
        let uniforms = Uniforms {
            screen_size,
            _padding: [0.0, 0.0],
        };

        let uniform_buffer = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("Sprite Uniform Buffer"),
            contents: bytemuck::cast_slice(&[uniforms]),
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
        });

        // Bind group layout
        let bind_group_layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("Sprite Bind Group Layout"),
            entries: &[
                wgpu::BindGroupLayoutEntry {
                    binding: 0,
                    visibility: wgpu::ShaderStages::VERTEX,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Uniform,
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
                wgpu::BindGroupLayoutEntry {
                    binding: 1,
                    visibility: wgpu::ShaderStages::FRAGMENT,
                    ty: wgpu::BindingType::Texture {
                        sample_type: wgpu::TextureSampleType::Float { filterable: true },
                        view_dimension: wgpu::TextureViewDimension::D2,
                        multisampled: false,
                    },
                    count: None,
                },
                wgpu::BindGroupLayoutEntry {
                    binding: 2,
                    visibility: wgpu::ShaderStages::FRAGMENT,
                    ty: wgpu::BindingType::Sampler(wgpu::SamplerBindingType::Filtering),
                    count: None,
                },
            ],
        });

        let bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("Sprite Bind Group"),
            layout: &bind_group_layout,
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: uniform_buffer.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: wgpu::BindingResource::TextureView(&texture_view),
                },
                wgpu::BindGroupEntry {
                    binding: 2,
                    resource: wgpu::BindingResource::Sampler(&sampler),
                },
            ],
        });

        // Pipeline layout
        let pipeline_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("Sprite Pipeline Layout"),
            bind_group_layouts: &[&bind_group_layout],
            immediate_size: 0,
        });

        // Vertex buffer layouts
        let vertex_buffer_layout = wgpu::VertexBufferLayout {
            array_stride: std::mem::size_of::<Vertex>() as wgpu::BufferAddress,
            step_mode: wgpu::VertexStepMode::Vertex,
            attributes: &[
                wgpu::VertexAttribute {
                    offset: 0,
                    shader_location: 0,
                    format: wgpu::VertexFormat::Float32x2,
                },
                wgpu::VertexAttribute {
                    offset: 8,
                    shader_location: 1,
                    format: wgpu::VertexFormat::Float32x2,
                },
            ],
        };

        let instance_buffer_layout = wgpu::VertexBufferLayout {
            array_stride: std::mem::size_of::<SpriteInstance>() as wgpu::BufferAddress,
            step_mode: wgpu::VertexStepMode::Instance,
            attributes: &[
                wgpu::VertexAttribute {
                    offset: 0,
                    shader_location: 2,
                    format: wgpu::VertexFormat::Float32x2, // position
                },
                wgpu::VertexAttribute {
                    offset: 8,
                    shader_location: 3,
                    format: wgpu::VertexFormat::Float32x2, // size
                },
                wgpu::VertexAttribute {
                    offset: 16,
                    shader_location: 4,
                    format: wgpu::VertexFormat::Float32x4, // uv_rect
                },
                wgpu::VertexAttribute {
                    offset: 32,
                    shader_location: 5,
                    format: wgpu::VertexFormat::Float32x4, // color
                },
            ],
        };

        // Create pipeline
        let pipeline = device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
            label: Some("Sprite Pipeline"),
            layout: Some(&pipeline_layout),
            vertex: wgpu::VertexState {
                module: &shader,
                entry_point: Some("vs_main"),
                buffers: &[vertex_buffer_layout, instance_buffer_layout],
                compilation_options: Default::default(),
            },
            fragment: Some(wgpu::FragmentState {
                module: &shader,
                entry_point: Some("fs_main"),
                targets: &[Some(wgpu::ColorTargetState {
                    format,
                    blend: Some(wgpu::BlendState::ALPHA_BLENDING),
                    write_mask: wgpu::ColorWrites::ALL,
                })],
                compilation_options: Default::default(),
            }),
            primitive: wgpu::PrimitiveState::default(),
            depth_stencil: None,
            multisample: wgpu::MultisampleState::default(),
            multiview_mask: None,
            cache: None,
        });

        // Create buffers
        let vertex_buffer = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("Sprite Vertex Buffer"),
            contents: bytemuck::cast_slice(QUAD_VERTICES),
            usage: wgpu::BufferUsages::VERTEX,
        });

        let index_buffer = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("Sprite Index Buffer"),
            contents: bytemuck::cast_slice(QUAD_INDICES),
            usage: wgpu::BufferUsages::INDEX,
        });

        let max_instances = 64;
        let instance_buffer = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("Sprite Instance Buffer"),
            size: (std::mem::size_of::<SpriteInstance>() * max_instances) as wgpu::BufferAddress,
            usage: wgpu::BufferUsages::VERTEX | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        Self {
            pipeline,
            vertex_buffer,
            index_buffer,
            instance_buffer,
            uniform_buffer,
            bind_group,
            texture,
            max_instances,
            screen_size,
        }
    }

    /// Build texture atlas from SVG files
    fn build_atlas_from_svgs(atlas_size: usize, cell_size: usize) -> Vec<u8> {
        let mut atlas = vec![0u8; atlas_size * atlas_size * 4];

        for (index, button) in XboxButton::all_in_order().iter().enumerate() {
            let col = index % 4;
            let row = index / 4;
            let x_offset = col * cell_size;
            let y_offset = row * cell_size;

            // Try to load and render SVG
            let svg_path = Path::new(ASSETS_DIR).join(button.svg_filename());
            if let Some(pixels) = Self::render_svg(&svg_path, cell_size as u32) {
                // Copy rendered pixels into atlas
                for y in 0..cell_size {
                    for x in 0..cell_size {
                        let src_idx = (y * cell_size + x) * 4;
                        let dst_x = x_offset + x;
                        let dst_y = y_offset + y;
                        let dst_idx = (dst_y * atlas_size + dst_x) * 4;

                        atlas[dst_idx] = pixels[src_idx];
                        atlas[dst_idx + 1] = pixels[src_idx + 1];
                        atlas[dst_idx + 2] = pixels[src_idx + 2];
                        atlas[dst_idx + 3] = pixels[src_idx + 3];
                    }
                }
            } else {
                // Fallback: draw colored circle placeholder
                Self::draw_placeholder_circle(&mut atlas, atlas_size, x_offset, y_offset, cell_size, button);
            }
        }

        atlas
    }

    /// Render an SVG file to RGBA pixels
    fn render_svg(path: &Path, size: u32) -> Option<Vec<u8>> {
        // Load SVG
        let svg_data = std::fs::read(path).ok()?;
        let tree = resvg::usvg::Tree::from_data(&svg_data, &resvg::usvg::Options::default()).ok()?;

        // Create pixmap for rendering - explicitly transparent
        let mut pixmap = resvg::tiny_skia::Pixmap::new(size, size)?;
        // Pixmap::new() should be transparent, but let's be explicit
        pixmap.fill(resvg::tiny_skia::Color::TRANSPARENT);

        // Calculate scale to fit SVG in cell
        let svg_size = tree.size();
        let scale = (size as f32 / svg_size.width()).min(size as f32 / svg_size.height());

        // Center the SVG in the cell
        let x_offset = (size as f32 - svg_size.width() * scale) / 2.0;
        let y_offset = (size as f32 - svg_size.height() * scale) / 2.0;

        let transform = resvg::tiny_skia::Transform::from_scale(scale, scale)
            .post_translate(x_offset, y_offset);

        // Render SVG to pixmap
        resvg::render(&tree, transform, &mut pixmap.as_mut());

        Some(pixmap.take())
    }

    /// Draw a placeholder circle when SVG loading fails
    fn draw_placeholder_circle(
        atlas: &mut [u8],
        atlas_size: usize,
        x_offset: usize,
        y_offset: usize,
        cell_size: usize,
        button: &XboxButton,
    ) {
        let color = match button {
            XboxButton::A => [100u8, 230, 100, 255],
            XboxButton::B => [230, 80, 80, 255],
            XboxButton::X => [80, 130, 230, 255],
            XboxButton::Y => [230, 200, 50, 255],
            _ => [180, 180, 180, 255],
        };

        let cx = cell_size / 2;
        let cy = cell_size / 2;
        let radius = (cell_size / 2 - 4) as i32;

        for y in 0..cell_size {
            for x in 0..cell_size {
                let dx = (x as i32) - (cx as i32);
                let dy = (y as i32) - (cy as i32);
                let dist = ((dx * dx + dy * dy) as f32).sqrt();

                let dst_x = x_offset + x;
                let dst_y = y_offset + y;
                let dst_idx = (dst_y * atlas_size + dst_x) * 4;

                if dist < radius as f32 {
                    atlas[dst_idx] = color[0];
                    atlas[dst_idx + 1] = color[1];
                    atlas[dst_idx + 2] = color[2];
                    atlas[dst_idx + 3] = color[3];
                } else if dist < (radius + 2) as f32 {
                    let alpha = ((radius + 2) as f32 - dist) / 2.0;
                    atlas[dst_idx] = color[0];
                    atlas[dst_idx + 1] = color[1];
                    atlas[dst_idx + 2] = color[2];
                    atlas[dst_idx + 3] = (alpha * 255.0) as u8;
                }
            }
        }
    }

    pub fn resize(&mut self, queue: &wgpu::Queue, width: u32, height: u32) {
        self.screen_size = [width as f32, height as f32];
        let uniforms = Uniforms {
            screen_size: self.screen_size,
            _padding: [0.0, 0.0],
        };
        queue.write_buffer(&self.uniform_buffer, 0, bytemuck::cast_slice(&[uniforms]));
    }

    pub fn draw<'a>(
        &'a self,
        render_pass: &mut wgpu::RenderPass<'a>,
        queue: &wgpu::Queue,
        sprites: &[SpriteInstance],
    ) {
        if sprites.is_empty() || sprites.len() > self.max_instances {
            return;
        }

        queue.write_buffer(&self.instance_buffer, 0, bytemuck::cast_slice(sprites));

        render_pass.set_pipeline(&self.pipeline);
        render_pass.set_bind_group(0, &self.bind_group, &[]);
        render_pass.set_vertex_buffer(0, self.vertex_buffer.slice(..));
        render_pass.set_vertex_buffer(1, self.instance_buffer.slice(..));
        render_pass.set_index_buffer(self.index_buffer.slice(..), wgpu::IndexFormat::Uint16);
        render_pass.draw_indexed(0..6, 0, 0..sprites.len() as u32);
    }
}
