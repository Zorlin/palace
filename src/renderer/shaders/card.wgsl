// Card shader - renders rectangles with rounded borders
// OLED optimized: pure black background, colored borders only

struct VertexInput {
    @location(0) position: vec2<f32>,
    @location(1) uv: vec2<f32>,
};

struct InstanceInput {
    @location(2) rect: vec4<f32>,      // x, y, width, height (in pixels)
    @location(3) border_color: vec4<f32>,
    @location(4) border_width: f32,
    @location(5) corner_radius: f32,
    @location(6) selected: f32,        // 0.0 or 1.0
    @location(7) _padding: f32,
};

struct VertexOutput {
    @builtin(position) clip_position: vec4<f32>,
    @location(0) uv: vec2<f32>,
    @location(1) rect_size: vec2<f32>,
    @location(2) border_color: vec4<f32>,
    @location(3) border_width: f32,
    @location(4) corner_radius: f32,
    @location(5) selected: f32,
};

struct Uniforms {
    screen_size: vec2<f32>,
    _padding: vec2<f32>,
};

@group(0) @binding(0)
var<uniform> uniforms: Uniforms;

@vertex
fn vs_main(vertex: VertexInput, instance: InstanceInput) -> VertexOutput {
    var out: VertexOutput;

    // Calculate vertex position in pixels
    let pixel_pos = instance.rect.xy + vertex.position * instance.rect.zw;

    // Convert to clip space (-1 to 1)
    let clip_pos = (pixel_pos / uniforms.screen_size) * 2.0 - 1.0;
    out.clip_position = vec4<f32>(clip_pos.x, -clip_pos.y, 0.0, 1.0);

    // Pass UV coords scaled to rect size
    out.uv = vertex.uv * instance.rect.zw;
    out.rect_size = instance.rect.zw;
    out.border_color = instance.border_color;
    out.border_width = instance.border_width;
    out.corner_radius = instance.corner_radius;
    out.selected = instance.selected;

    return out;
}

// Signed distance function for rounded rectangle
fn sd_rounded_rect(p: vec2<f32>, size: vec2<f32>, radius: f32) -> f32 {
    let q = abs(p) - size + vec2<f32>(radius);
    return length(max(q, vec2<f32>(0.0))) + min(max(q.x, q.y), 0.0) - radius;
}

@fragment
fn fs_main(in: VertexOutput) -> @location(0) vec4<f32> {
    // Center the coordinate system
    let p = in.uv - in.rect_size * 0.5;
    let half_size = in.rect_size * 0.5;

    // Calculate distance from rounded rectangle edge
    let d = sd_rounded_rect(p, half_size, in.corner_radius);

    // Anti-aliased border - draw border on the INSIDE of the edge
    let aa = 1.0; // Anti-aliasing width in pixels

    // Fill: inside the card (d < 0)
    let fill_alpha = 1.0 - smoothstep(-aa, aa, d);

    // Border is drawn from (edge - border_width) to edge
    let inner_edge = -in.border_width;
    let outer_edge = 0.0;

    // Smooth transitions for crisp anti-aliased borders
    let alpha_outer = 1.0 - smoothstep(outer_edge - aa, outer_edge + aa, d);
    let alpha_inner = smoothstep(inner_edge - aa, inner_edge + aa, d);
    let border_alpha = alpha_outer * alpha_inner;

    // Glow effect for selected cards (ONLY outside the card edge)
    var glow_alpha = 0.0;
    if (in.selected > 0.5 && d > 0.0) {
        // Only glow outside the card (d > 0)
        glow_alpha = exp(-d * 0.06) * 0.35;
    }

    // Semi-transparent overlay mode: no border, just a filled rectangle with alpha
    // Used for modal background overlays
    if (in.border_width < 0.5) {
        // Simple filled rectangle with the specified alpha
        if (fill_alpha < 0.01) {
            discard;
        }
        return vec4<f32>(in.border_color.rgb, fill_alpha * in.border_color.a);
    }

    // If border_color alpha is 1.0, render as filled card (not OLED border-only)
    // This allows modal backgrounds to be fully opaque
    if (in.border_color.a > 0.99) {
        // Filled card mode: fill + border
        let fill_color = vec3<f32>(in.border_color.rgb * 0.15); // Darker fill
        let combined_alpha = max(fill_alpha, max(border_alpha, glow_alpha));

        if (combined_alpha < 0.01) {
            discard;
        }

        // Blend fill and border
        if (border_alpha > 0.01) {
            return vec4<f32>(in.border_color.rgb, combined_alpha);
        } else {
            return vec4<f32>(fill_color, fill_alpha);
        }
    }

    // OLED mode: border-only (transparent fill)
    let final_alpha = max(border_alpha, glow_alpha);

    // Discard fully transparent pixels for OLED optimization
    if (final_alpha < 0.01) {
        discard;
    }

    return vec4<f32>(in.border_color.rgb, final_alpha);
}
