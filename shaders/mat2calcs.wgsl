struct Mat2 {
    mtx: array<array<f32,3>,3>,
}
@group(0) @binding(0)
var<storage,read_write>  a: Mat2;
@group(0) @binding(1)
var<storage,read_write>  b: Mat2;
@group(0) @binding(2)
var<storage,read_write>  r: Mat2;

@compute @workgroup_size(1)
fn result_to_a() {
    a = r;
}

@compute @workgroup_size(1)
fn result_to_b() {
    b = r;
}

@compute @workgroup_size(1,1,1)
fn add (@builtin(global_invocation_id) id: vec3<u32>) {
        r.mtx[id.x][id.y] = a.mtx[id.x][id.y] + b.mtx[id.x][id.y];
}
