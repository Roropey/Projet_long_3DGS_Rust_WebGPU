// Vertex shader
//struct CameraUniform {
//    view_proj: mat4x4<f32>,
//};

//struct Splat {
//    rotation: vec4<f32>,
//    center: vec3<f32>,
//    paddingA: f32,
//    scale: vec3<f32>,
//    alpha: f32,
//    colorSH: array<f32, 48>,
//}

struct Splat {
    center : vec3<f32>,
    paddingA: f32,
    cov : mat2x2<f32>,
    color : vec4<f32>
}


//@group(0) @binding(0) var<uniform> camera: CameraUniform;
@group(0) @binding(0) var<storage> splats: array<Splat>;


struct VertexInput {
    @location(0) position: vec3<f32>,
    @location(1) color: vec3<f32>,
}

struct VertexOutput {
    @builtin(position) gl_Position: vec4<f32>,
    @location(0) @interpolate(flat) color: vec4<f32>,
    @location(1) @interpolate(linear) gl_TexCoord: vec2<f32>,
}

@vertex
fn vs_main(
    @builtin(instance_index) gl_InstanceID: u32,
    @builtin(vertex_index) gl_VertexID: u32,
) -> VertexOutput {
    var stage_out: VertexOutput;
    var discard_quad: bool;
    var splat : Splat =  splats[gl_InstanceID];
    //let world_position = splats[splat_index].center;
    stage_out.color = splat.color;
    var quad_vertices = array<vec2<f32>, 4>(
        vec2<f32>(-1.0, -1.0),
        vec2<f32>(-1.0,  1.0),
        vec2<f32>( 1.0, -1.0),
        vec2<f32>( 1.0,  1.0),
    );
    stage_out.gl_TexCoord = quad_vertices[gl_VertexID]*4.0;
    stage_out.gl_Position = vec4<f32>(stage_out.gl_TexCoord*splat.cov + splat.center.xy, 0.0, 1.0);

    return stage_out;
}



// Fragment shader

@fragment
fn fs_main(in: VertexOutput) -> @location(0) vec4<f32> {
    let power = dot(in.gl_TexCoord, in.gl_TexCoord);
    let alpha = in.color.a * exp(-0.5 * power);
    
    return vec4<f32>(in.color.rgb * alpha, alpha);
}