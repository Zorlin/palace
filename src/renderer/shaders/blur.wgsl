// Gaussian blur shader - two-pass (horizontal then vertical)
// This shader does a single pass, call twice with different directions

struct VertexOutput {
    @builtin(position) position: vec4<f32>,
    @location(0) uv: vec2<f32>,
};

@group(0) @binding(0) var input_texture: texture_2d<f32>;
@group(0) @binding(1) var input_sampler: sampler;

struct BlurUniforms {
    direction: vec2<f32>,  // (1,0) for horizontal, (0,1) for vertical
    texture_size: vec2<f32>,
};

@group(0) @binding(2) var<uniform> uniforms: BlurUniforms;

// Fullscreen triangle vertex shader
@vertex
fn vs_main(@builtin(vertex_index) vertex_index: u32) -> VertexOutput {
    var out: VertexOutput;
    // Generate fullscreen triangle
    let x = f32(i32(vertex_index & 1u) * 4 - 1);
    let y = f32(i32(vertex_index >> 1u) * 4 - 1);
    out.position = vec4<f32>(x, y, 0.0, 1.0);
    out.uv = vec2<f32>((x + 1.0) * 0.5, (1.0 - y) * 0.5);
    return out;
}

// 9-tap Gaussian blur kernel
// Weights for sigma ~= 2.0
const KERNEL_SIZE: i32 = 9;
const WEIGHTS: array<f32, 5> = array<f32, 5>(
    0.227027,  // center
    0.1945946, // +/- 1
    0.1216216, // +/- 2
    0.054054,  // +/- 3
    0.016216   // +/- 4
);

@fragment
fn fs_main(in: VertexOutput) -> @location(0) vec4<f32> {
    let pixel_size = 1.0 / uniforms.texture_size;
    let dir = uniforms.direction * pixel_size;

    var result = textureSample(input_texture, input_sampler, in.uv) * WEIGHTS[0];

    for (var i: i32 = 1; i < 5; i++) {
        let offset = dir * f32(i);
        result += textureSample(input_texture, input_sampler, in.uv + offset) * WEIGHTS[i];
        result += textureSample(input_texture, input_sampler, in.uv - offset) * WEIGHTS[i];
    }

    return result;
}
