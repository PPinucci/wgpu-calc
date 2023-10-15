struct Mat2 {
    mtx: array<array<f32,3>,3>,
}
@group(0) @binding(0)
var<storage,read_write>  a: Mat2;
@group(0) @binding(1)
var<storage,read_write>  b: Mat2;

@compute @workgroup_size(1,1,1)
fn add_1 (@builtin(global_invocation_id) id: vec3<u32>) {
        a.mtx[id.x][id.y] = a.mtx[id.x][id.y] + 1.0;
}

@compute @workgroup_size(1,1,1)
fn add_matrices (@builtin(global_invocation_id) id: vec3<u32>) {
        a.mtx[id.x][id.y] = a.mtx[id.x][id.y] + b.mtx[id.x][id.y];
}


