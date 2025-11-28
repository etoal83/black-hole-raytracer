// フルスクリーンクアッドでテクスチャを表示するシェーダ

struct VertexInput {
    @location(0) position: vec2<f32>,
}

struct VertexOutput {
    @builtin(position) clip_position: vec4<f32>,
    @location(0) tex_coords: vec2<f32>,
}

@vertex
fn vs_main(in: VertexInput) -> VertexOutput {
    var out: VertexOutput;
    out.clip_position = vec4<f32>(in.position, 0.0, 1.0);
    // 頂点位置からテクスチャ座標に変換 [-1, 1] -> [0, 1]
    out.tex_coords = in.position * 0.5 + 0.5;
    // Y軸を反転（テクスチャ座標系に合わせる）
    out.tex_coords.y = 1.0 - out.tex_coords.y;
    return out;
}

@group(0) @binding(0) var output_texture: texture_2d<f32>;
@group(0) @binding(1) var texture_sampler: sampler;

@fragment
fn fs_main(in: VertexOutput) -> @location(0) vec4<f32> {
    return textureSample(output_texture, texture_sampler, in.tex_coords);
}
