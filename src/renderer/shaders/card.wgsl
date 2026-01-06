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
    @location(7) filled: f32,          // 0.0 = OLED (border only), 1.0 = filled background
};

struct VertexOutput {
    @builtin(position) clip_position: vec4<f32>,
    @location(0) uv: vec2<f32>,
    @location(1) rect_size: vec2<f32>,
    @location(2) border_color: vec4<f32>,
    @location(3) border_width: f32,
    @location(4) corner_radius: f32,
    @location(5) selected: f32,
    @location(6) filled: f32,
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
    out.filled = instance.filled;

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

    // Anti-aliasing width in pixels
    let aa = 1.0;

    // Fill: inside the card (d < 0)
    let fill_alpha = 1.0 - smoothstep(-aa, aa, d);

    // Border is drawn from (edge - border_width) to edge
    let inner_edge = -in.border_width;
    let outer_edge = 0.0;

    // Smooth transitions for crisp anti-aliased borders
    let alpha_outer = 1.0 - smoothstep(outer_edge - aa, outer_edge + aa, d);
    let alpha_inner = smoothstep(inner_edge - aa, inner_edge + aa, d);
    let border_alpha = alpha_outer * alpha_inner;

    // Selection effects
    var glow_alpha = 0.0;
    var selection_fill_alpha = 0.0;
    if (in.selected > 0.5) {
        // Subtle rounded glow outside the card edge
        if (d > 0.0) {
            glow_alpha = exp(-d * 0.08) * 0.3;
        }
        // Subtle inner fill for contrast (inside the card)
        if (d < -in.border_width) {
            let inner_dist = abs(d + in.border_width);
            selection_fill_alpha = smoothstep(0.0, in.border_width * 2.0, inner_dist) * 0.12;
        }
    }

    // Semi-transparent overlay mode: no border, just a filled rectangle with alpha
    // Used for modal background overlays (border_width == 0)
    if (in.border_width < 0.5) {
        if (fill_alpha < 0.01) {
            discard;
        }
        return vec4<f32>(in.border_color.rgb, fill_alpha * in.border_color.a);
    }

    // Filled mode: render with dark fill background
    if (in.filled > 0.5) {
        let fill_color = vec3<f32>(in.border_color.rgb * 0.15); // Darker fill
        let combined_alpha = max(fill_alpha, max(border_alpha, glow_alpha));

        if (combined_alpha < 0.01) {
            discard;
        }

        // Border draws on top of fill
        if (border_alpha > 0.01) {
            return vec4<f32>(in.border_color.rgb, combined_alpha * in.border_color.a);
        } else {
            return vec4<f32>(fill_color, fill_alpha * in.border_color.a);
        }
    }

    // OLED mode (default): border-only with transparent fill
    let final_alpha = max(max(border_alpha, glow_alpha), selection_fill_alpha);

    // Discard fully transparent pixels for OLED optimization
    if (final_alpha < 0.01) {
        discard;
    }

    // Selection fill uses a slightly brighter version of border color
    if (selection_fill_alpha > border_alpha && selection_fill_alpha > glow_alpha) {
        let bright_color = in.border_color.rgb * 1.3;
        return vec4<f32>(bright_color, selection_fill_alpha * in.border_color.a);
    }

    return vec4<f32>(in.border_color.rgb, final_alpha * in.border_color.a);
}
