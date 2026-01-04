use bytemuck::{Pod, Zeroable};
use wgpu::util::DeviceExt;

/// Card instance data - matches shader InstanceInput
#[repr(C)]
#[derive(Clone, Copy, Debug, Pod, Zeroable)]
pub struct CardInstance {
    pub rect: [f32; 4],         // x, y, width, height
    pub border_color: [f32; 4], // RGBA
    pub border_width: f32,
    pub corner_radius: f32,
    pub selected: f32,
    pub _padding: f32,
}

impl CardInstance {
    pub fn new(x: f32, y: f32, width: f32, height: f32, color: [f32; 4]) -> Self {
        Self {
            rect: [x, y, width, height],
            border_color: color,
            border_width: 3.0,
            corner_radius: 12.0,
            selected: 0.0,
            _padding: 0.0,
        }
    }

    pub fn selected(mut self) -> Self {
        self.selected = 1.0;
        self.border_width = 4.0;
        self
    }

    pub fn with_border_width(mut self, width: f32) -> Self {
        self.border_width = width;
        self
    }

    pub fn with_corner_radius(mut self, radius: f32) -> Self {
        self.corner_radius = radius;
        self
    }
}

/// Project status determines card border color
#[derive(Clone, Copy, Debug, Default)]
pub enum ProjectStatus {
    #[default]
    Unknown,   // Gray border
    Building,  // Yellow/orange pulsing
    Error,     // Red border
    Passing,   // Green border
    Active,    // Blue border (currently working)
}

impl ProjectStatus {
    pub fn color(&self) -> [f32; 4] {
        match self {
            ProjectStatus::Unknown => [0.4, 0.4, 0.5, 1.0],
            ProjectStatus::Building => [1.0, 0.7, 0.2, 1.0],
            ProjectStatus::Error => [1.0, 0.3, 0.3, 1.0],
            ProjectStatus::Passing => [0.3, 0.9, 0.4, 1.0],
            ProjectStatus::Active => [0.4, 0.6, 1.0, 1.0],
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

pub struct CardRenderer {
    pipeline: wgpu::RenderPipeline,
    vertex_buffer: wgpu::Buffer,
    index_buffer: wgpu::Buffer,
    instance_buffer: wgpu::Buffer,
    uniform_buffer: wgpu::Buffer,
    uniform_bind_group: wgpu::BindGroup,
    max_instances: usize,
    screen_size: [f32; 2],
}

impl CardRenderer {
    pub fn new(
        device: &wgpu::Device,
        format: wgpu::TextureFormat,
        width: u32,
        height: u32,
    ) -> Self {
        let shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("Card Shader"),
            source: wgpu::ShaderSource::Wgsl(include_str!("shaders/card.wgsl").into()),
        });

        // Uniform buffer
        let screen_size = [width as f32, height as f32];
        let uniforms = Uniforms {
            screen_size,
            _padding: [0.0, 0.0],
        };

        let uniform_buffer = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("Card Uniform Buffer"),
            contents: bytemuck::cast_slice(&[uniforms]),
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
        });

        let uniform_bind_group_layout =
            device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
                label: Some("Card Uniform Bind Group Layout"),
                entries: &[wgpu::BindGroupLayoutEntry {
                    binding: 0,
                    visibility: wgpu::ShaderStages::VERTEX | wgpu::ShaderStages::FRAGMENT,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Uniform,
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                }],
            });

        let uniform_bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("Card Uniform Bind Group"),
            layout: &uniform_bind_group_layout,
            entries: &[wgpu::BindGroupEntry {
                binding: 0,
                resource: uniform_buffer.as_entire_binding(),
            }],
        });

        // Pipeline layout
        let pipeline_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("Card Pipeline Layout"),
            bind_group_layouts: &[&uniform_bind_group_layout],
            immediate_size: 0,
        });

        // Vertex buffer layout
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

        // Instance buffer layout
        let instance_buffer_layout = wgpu::VertexBufferLayout {
            array_stride: std::mem::size_of::<CardInstance>() as wgpu::BufferAddress,
            step_mode: wgpu::VertexStepMode::Instance,
            attributes: &[
                wgpu::VertexAttribute {
                    offset: 0,
                    shader_location: 2,
                    format: wgpu::VertexFormat::Float32x4, // rect
                },
                wgpu::VertexAttribute {
                    offset: 16,
                    shader_location: 3,
                    format: wgpu::VertexFormat::Float32x4, // border_color
                },
                wgpu::VertexAttribute {
                    offset: 32,
                    shader_location: 4,
                    format: wgpu::VertexFormat::Float32, // border_width
                },
                wgpu::VertexAttribute {
                    offset: 36,
                    shader_location: 5,
                    format: wgpu::VertexFormat::Float32, // corner_radius
                },
                wgpu::VertexAttribute {
                    offset: 40,
                    shader_location: 6,
                    format: wgpu::VertexFormat::Float32, // selected
                },
                wgpu::VertexAttribute {
                    offset: 44,
                    shader_location: 7,
                    format: wgpu::VertexFormat::Float32, // padding
                },
            ],
        };

        // Create pipeline
        let pipeline = device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
            label: Some("Card Pipeline"),
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
            primitive: wgpu::PrimitiveState {
                topology: wgpu::PrimitiveTopology::TriangleList,
                strip_index_format: None,
                front_face: wgpu::FrontFace::Ccw,
                cull_mode: None,
                polygon_mode: wgpu::PolygonMode::Fill,
                unclipped_depth: false,
                conservative: false,
            },
            depth_stencil: None,
            multisample: wgpu::MultisampleState::default(),
            multiview_mask: None,
            cache: None,
        });

        // Create vertex and index buffers
        let vertex_buffer = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("Card Vertex Buffer"),
            contents: bytemuck::cast_slice(QUAD_VERTICES),
            usage: wgpu::BufferUsages::VERTEX,
        });

        let index_buffer = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("Card Index Buffer"),
            contents: bytemuck::cast_slice(QUAD_INDICES),
            usage: wgpu::BufferUsages::INDEX,
        });

        // Pre-allocate instance buffer for up to 64 cards
        let max_instances = 64;
        let instance_buffer = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("Card Instance Buffer"),
            size: (std::mem::size_of::<CardInstance>() * max_instances) as wgpu::BufferAddress,
            usage: wgpu::BufferUsages::VERTEX | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        Self {
            pipeline,
            vertex_buffer,
            index_buffer,
            instance_buffer,
            uniform_buffer,
            uniform_bind_group,
            max_instances,
            screen_size,
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
        cards: &[CardInstance],
    ) {
        if cards.is_empty() || cards.len() > self.max_instances {
            return;
        }

        // Update instance buffer
        queue.write_buffer(&self.instance_buffer, 0, bytemuck::cast_slice(cards));

        render_pass.set_pipeline(&self.pipeline);
        render_pass.set_bind_group(0, &self.uniform_bind_group, &[]);
        render_pass.set_vertex_buffer(0, self.vertex_buffer.slice(..));
        render_pass.set_vertex_buffer(1, self.instance_buffer.slice(..));
        render_pass.set_index_buffer(self.index_buffer.slice(..), wgpu::IndexFormat::Uint16);
        render_pass.draw_indexed(0..6, 0, 0..cards.len() as u32);
    }

    pub fn screen_size(&self) -> [f32; 2] {
        self.screen_size
    }
}
