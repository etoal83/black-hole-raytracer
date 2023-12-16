// struct VertexInput {
//     @location(0) position: vec3<f32>,
//     @location(1) color: vec3<f32>,
// };

struct VertexOutput {
    @builtin(position) clip_position: vec4<f32>,
    @location(0) color: vec4f,
};

// @stage(vertex)
@vertex
fn vs_main(
    @builtin(vertex_index) vidx: u32
) -> VertexOutput {
    var vout: VertexOutput;
    var pos = array(
        vec2f( 0.0,  0.5),
        vec2f(-0.5, -0.5),
        vec2f( 0.5, -0.5)
    );
    var color = array<vec4f, 3>(
        vec4f(1., 0., 0., 1.),
        vec4f(0., 1., 0., 1.),
        vec4f(0., 0., 1., 1.),
    );
    vout.clip_position = vec4f(pos[vidx], 0.0, 1.0);
    vout.color = color[vidx];
    return vout;
}

// @stage(fragment)
@fragment
fn fs_main(fin: VertexOutput) -> @location(0) vec4<f32> {
    return fin.color;
}
