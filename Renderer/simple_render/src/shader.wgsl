
struct Splat {
    center : vec3<f32>,
    padding : f32,
    cov : mat2x2<f32>,
    color : vec4<f32>
}


@group(0) @binding(0) var<storage> splats: array<Splat>;
@group(0) @binding(1) var<storage, read_write> radii: array<f32>;



struct VertexInput {
    @location(0) position: vec3<f32>,
    @location(1) color: vec3<f32>,
}

struct VertexOutput {
    @builtin(position) position: vec4<f32>,
    @location(0)  color: vec4<f32>,
    @location(1)  texCoord: vec2<f32>,
}

@compute @workgroup_size(64,1,1)
fn computeRadii(@builtin(global_invocation_id) global_id: vec3<u32>) {
    
    let id = global_id.x;

    let world_position = splats[id].center;
    let covariance = splats[id].cov;
    let mid: f32 = 0.5 * (covariance[0][0] + covariance[1][1]); // Assurez-vous que c'est covariance[1][1] pour une matrice 2D
    let det: f32 = covariance[0][0] * covariance[1][1] - covariance[0][1] * covariance[1][0];
    let lambda1: f32 = mid + sqrt(max(0.1, mid * mid - det));
    let lambda2: f32 = mid - sqrt(max(0.1, mid * mid - det));

    // Calcul du "rayon" basé sur les valeurs propres de la matrice de covariance
    let radius: f32 = ceil(3.0 * sqrt(max(lambda1, lambda2)));

    // Stockage du rayon calculé dans le buffer des radii
    radii[id] = radius; //radius; // Stocker le rayon à l'indice original, si nécessaire
}

// Vertex shader
@vertex
fn vs_main(
    @builtin(instance_index) instanceID: u32,
    @builtin(vertex_index) vertexID: u32,
) -> VertexOutput {
    var stage_out: VertexOutput;
    var quad_vertices = array<vec2<f32>, 4>(vec2<f32>(-1.0, -1.0), vec2<f32>(-1.0,  1.0),
                                            vec2<f32>( 1.0, -1.0), vec2<f32>( 1.0,  1.0),);
    stage_out.texCoord = quad_vertices[vertexID]*4.0;

    var splat : Splat =  splats[instanceID];
    stage_out.position = vec4<f32>(stage_out.texCoord*splat.cov + splat.center.xy, 0.0, 1.0);
    stage_out.color = splat.color;
    return stage_out;
}

// Fragment shader
@fragment
fn fs_main(in: VertexOutput) -> @location(0) vec4<f32> {
    let alpha = in.color.a * exp(-0.5 * dot(in.texCoord, in.texCoord));
    return vec4<f32>(in.color.rgb*alpha, alpha);
}