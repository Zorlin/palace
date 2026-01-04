// Sprite shader - renders textured quads from a sprite atlas
// Used for Xbox controller button glyphs
//
// Xbox button assets by Arks: https://arks.itch.io/xbox-buttons

struct VertexInput {
    @location(0) position: vec2<f32>,
    @location(1) uv: vec2<f32>,
};

struct InstanceInput {
    @location(2) pos: vec2<f32>,       // Screen position in pixels
    @location(3) size: vec2<f32>,      // Size in pixels
    @location(4) uv_rect: vec4<f32>,   // u_min, v_min, u_max, v_max
    @location(5) color: vec4<f32>,     // Tint color
};

struct VertexOutput {
    @builtin(position) clip_position: vec4<f32>,
    @location(0) uv: vec2<f32>,
    @location(1) color: vec4<f32>,
};

struct Uniforms {
    screen_size: vec2<f32>,
    _padding: vec2<f32>,
};

@group(0) @binding(0)
var<uniform> uniforms: Uniforms;

@group(0) @binding(1)
var sprite_texture: texture_2d<f32>;

@group(0) @binding(2)
var sprite_sampler: sampler;

@vertex
fn vs_main(vertex: VertexInput, instance: InstanceInput) -> VertexOutput {
    var out: VertexOutput;

    // Calculate vertex position in pixels
    let pixel_pos = instance.pos + vertex.position * instance.size;

    // Convert to clip space (-1 to 1)
    let clip_pos = (pixel_pos / uniforms.screen_size) * 2.0 - 1.0;
    out.clip_position = vec4<f32>(clip_pos.x, -clip_pos.y, 0.0, 1.0);

    // Interpolate UV within the sprite's region in the atlas
    let uv_size = instance.uv_rect.zw - instance.uv_rect.xy;
    out.uv = instance.uv_rect.xy + vertex.uv * uv_size;
    out.color = instance.color;

    return out;
}

@fragment
fn fs_main(in: VertexOutput) -> @location(0) vec4<f32> {
    let tex_color = textureSample(sprite_texture, sprite_sampler, in.uv);
    return tex_color * in.color;
}
