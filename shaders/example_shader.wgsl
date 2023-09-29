 struct Mat2 {
    elements: array<f32>,
    }

@group(0) @binding(0)
var<storage,read_write>  a: Mat2;
@group(0) @binding(1)
var<storage,read_write>  b: Mat2; 

@compute @workgroup_size(3,3)
fn add (@builtin(global_invocation_id) id: vec3<u32>) {
    a.elements[id.x] = a.elements[id.x] + b.elements[id.x];
} 