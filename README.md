# wgpu-calc
Library to use wgpu as a calculation tool

This crate aim is to build an infrastructure to leverage the GPU for calculations

The project is based on the [`wgpu`] crate as base, and it builds on top of calc shader creations
a layer which makes hopefully easier to build and run calculations on the GPU. just by implementing the
[`variable::Variable`] trait.
Please keep in mind this is is an amateur project, and thus not fully tested nor complete.

A simple example of the crate usage is the following:
```
extern crate wgpu_calc;
use std::sync::{Arc, Mutex};

use bytemuck;
use ndarray::{array, Array2};
use wgpu_calc::algorithm::{Algorithm, Function, VariableBind};
use wgpu_calc::coding::Shader;
use wgpu_calc::variable::Variable;

// we create a struct which will implement the [`Variable`] trait
#[derive(Debug, PartialEq)]
struct GpuArray2<'a> {
    data: Vec<f32>,
    n_rows: u64,
    n_cols: u64,
    name: &'a str,
}
 // here are simple implementations to have the [`Variable`] implementation be easier
impl<'a> GpuArray2<'a> {
    fn new(array: Array2<f32>, name: &'a str) -> GpuArray2<'a> {
        let (n_cols, n_rows) = array.dim();
        let data = array.as_slice().unwrap().to_owned();
        Self {
            data,
            n_rows: n_rows as u64,
            n_cols: n_cols as u64,
            name,
        }
    }

    fn get_dims(&self) -> (usize, usize) {
        (self.n_rows as usize, self.n_cols as usize)
    }

    fn to_array(&self) -> Array2<f32> {
        return Array2::from_shape_vec(
            (self.n_cols as usize, self.n_rows as usize),
            self.data.clone(),
        )
        .unwrap();
    }
}

// implementing the [`Variable`] trait is pretty simple for this struct
impl Variable for GpuArray2<'_> {
    // the byte size of the array is simply the dimensions by the size of an f32
    // keep in mind that building a more complex size could be complicated due to the
    // necessity of arranging the memory correclty in the GPU
    fn byte_size(&self) -> u64 {
        let base_size: u64 = std::mem::size_of::<f32>() as u64;
        base_size * self.n_cols * self.n_rows
    }

    // casting the data to an array of u8 is easy having the data as an array
    // and using bytemuck
    fn byte_data(&self) -> &[u8] {
        bytemuck::cast_slice(&self.data)
    }

    // this is needed to get the size of the workgroups calculating in the GPU
    // (basically the block size)
    fn dimension_sizes(&self) -> [u32; 3] {
        [self.n_rows as u32, self.n_cols as u32, 1]
    }

    // this is mostly for debugging purposes, so the GPU can throw errors with more specs
    fn get_name(&self) -> Option<&str> {
        Some(self.name)
    }

    // This is the opposite of the [`byte_data`] method, used to read variabled back from
    // an array of u8
    fn read_data(&mut self, slice: &[u8]) {
        let vec: Vec<f32> = bytemuck::cast_slice(slice).to_owned();
        self.data = vec;
    }
}

#[tokio::main]
async fn main() {
    // we create a ndarray
    let array = array![[0., 0., 0.], [1., 1., 1.], [2., 2., 2.]];

    // we start the Algorithm (this will also get the GPU device from the machine and set the
    // correct compiler)
    let mut algorithm = Algorithm::new(Some("Test algorithm")).await.unwrap();

    // we create a Variable and put inside a Arc<Mutex>. This is needed to be able to
    // have a shared reference of it during the instantiation of the calculi
    let var = Arc::new(Mutex::new(GpuArray2::new(array, "test array")));

    // this is a simple shader writte in WGSL
    let shader = Shader::from_content("
         struct Mat2 {
             elements: array<array<f32,3>,3>,
             }

         @group(0) @binding(0)
         var<storage,read_write>  a: Mat2;

         @compute @workgroup_size(1,1)
         fn add_1 (@builtin(global_invocation_id) id: vec3<u32>) {
             a.elements[id.x][id.y] = a.elements[id.x][id.y] + 1.0;
         }"
    );
    // we clone the variable to be able to pass it to the bindig
    let bind1 = Arc::clone(&var);
    // we create a new variable bind, which links `var` with the bind group 0
    // defined in the shader here above. This method cosumes the bind
    let bindings = vec![VariableBind::new(bind1, 0)];

    // we create a new function with the shader written here above using
    // 'add_1' as the entry point (working function)
    let function = Function::new(&shader, "add_1", bindings);

    // we add the function to the algorithm. Notice this will not execute anything, and
    // we could add more of them to be executed sequentially. In this step the variable is
    // written in the GPU buffer
    algorithm.add_fun(function);

    // this phisically executes all the added functions on the GPU
    algorithm.run().await.unwrap();

    // we need to use this method to extract a variable. This
    // operation is expensive since it copies data from the GPU to the CPU
    // this is why it's not done automatically for all the variables
    algorithm.get_output_unmap(&var).await.unwrap();

    // here we put a lock on the variable to use it to extract the result
    let var_lock = var.lock().unwrap();

    // we convert back the variable to the original format (ndarray::Array in this case)
    let result = var_lock.to_array();

    // we verify that the calculus is infact correct and the result is the original array
    // with 1 added to each element
    let check = array![[1., 1., 1.], [2., 2., 2.], [3., 3., 3.]];
    assert_eq!(result, check)
}

```

Although the crate works for this simple situations (which is not a lot, but still enough to implement and execute any
kind of linear algebra with 1D to 3D matrices) more work is still needed to make some internal features of wgpu accessible
to the API user, like automatic padding for structs or more default implementations of the Variable trait

To notice also that the [`algorithm::Algorithm`] currently executes the [`algorithm::Function`]s only serially, i.e. each function is submitted to be run
in parallel in the GPU in the order it's added to the [`algorithm::Algorithm`].
The infrastructure of the crate is already in place to optimize and run in parallel functions which don't act on the same [`variable::Variable`].

Another improvement to be done is the parallelisation of the buffers write, which is always the worst bottleneck of the CPU-GPU interface.
