extern crate wgpu_calc;
use std::sync::{Arc, Mutex};

use bytemuck;
use ndarray::{array, Array2};
use wgpu_calc::algorithm::{Algorithm, Function, VariableBind};
use wgpu_calc::coding::Shader;
use wgpu_calc::variable::Variable;

#[derive(Debug, PartialEq)]
struct GpuArray2<'a> {
    data: Vec<f32>,
    n_rows: u64,
    n_cols: u64,
    name: &'a str,
}

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
}

impl Variable for GpuArray2<'_> {
    fn byte_size(&self) -> u64 {
        let base_size: u64 = std::mem::size_of::<f32>() as u64;
        base_size * self.n_cols * self.n_rows
    }

    fn byte_data(&self) -> &[u8] {
        bytemuck::cast_slice(&self.data)
    }

    fn dimension_sizes(&self) -> [u32; 3] {
        [self.n_rows as u32, self.n_cols as u32, 1]
    }

    fn get_name(&self) -> Option<&str> {
        Some(self.name)
    }

    fn read_data(&mut self, slice: &[u8]) {
        todo!()
    }

    
}

#[tokio::test]
async fn add_1_test() {
    let array = array![[0., 0., 0.], [1., 1., 1.], [2., 2., 2.]];

    let mut algorithm = Algorithm::new(Some("Test algorithm")).await.unwrap();

    let var = Arc::new(Mutex::new(GpuArray2::new(array, "test array")));
    let (nrows, ncols) = var.lock().unwrap().get_dims();

    let mut shader = Shader::from_file_path("./tests/shaders/mat2calcs.pwgsl").unwrap();
    shader.replace("€cols", ncols.to_string().as_str());
    shader.replace("€nrow", nrows.to_string().as_str());

    let bind1 = Arc::clone(&var);

    let bindings = vec![VariableBind::new(bind1, 0)];

    let function = Function::new(&shader, "add_1", bindings);

    algorithm.add_function(function);

    // print!("{:?}", algorithm.get_operations())

    algorithm.finish().await.unwrap();


    let output = Arc::clone(&var);

    algorithm
        .get_output_unmap(&output)
        .await
        .unwrap();

    print!("{:?}", var.lock().unwrap())
}
