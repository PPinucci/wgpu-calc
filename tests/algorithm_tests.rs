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

    fn to_array(&self)->Array2<f32> {
        return Array2::from_shape_vec((self.n_cols as usize,self.n_rows as usize), self.data.clone()).unwrap();
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
        let vec: Vec<f32> = bytemuck::cast_slice(slice).to_owned();
        self.data = vec;
    }

    
}

#[tokio::test]
async fn add_1_test() {
    let array = array![[0., 0., 0.], [1., 1., 1.], [2., 2., 2.]];

    let mut algorithm = Algorithm::new(Some("Test algorithm")).await.unwrap();

    let var = Arc::new(Mutex::new(GpuArray2::new(array, "test array")));

    let shader = Shader::from_file_path("./tests/shaders/mat2calcs.wgsl").unwrap();
    
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

    let var_lock = var.lock().unwrap();
    let result = var_lock.to_array();
    print!("{:?}", result);
    let check = array![[1., 1., 1.], [2., 2., 2.], [3., 3., 3.]];
    assert_eq!(result, check)
}

#[tokio::test]
async fn add_1_large() {
    let array = Array2::zeros((500,500));

    let mut algorithm = Algorithm::new(Some("Test algorithm")).await.unwrap();

    let var = Arc::new(Mutex::new(GpuArray2::new(array, "test array")));
    let (nrows, ncols) = var.lock().unwrap().get_dims();

    let mut shader = Shader::from_file_path("./tests/shaders/mat2calcs.pwgsl").unwrap();
    shader.replace("€ncol", ncols.to_string().as_str());
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

    let var_lock = var.lock().unwrap();
    let result = var_lock.to_array();
    let check = Array2::ones((500,500));
    assert_eq!(result, check)
}

#[tokio::test]
async fn add_1_two_buffers() {
    let array_1 = Array2::zeros((500,500));
    let array_2 = Array2::zeros((500,500));

    let mut algorithm = Algorithm::new(Some("Test algorithm")).await.unwrap();

    let var_1 = Arc::new(Mutex::new(GpuArray2::new(array_1, "array_1")));
    let var_2 = Arc::new(Mutex::new(GpuArray2::new(array_2, "array_1")));

    
    let (nrows, ncols) = var_1.lock().unwrap().get_dims();

    let mut shader = Shader::from_file_path("./tests/shaders/mat2calcs.pwgsl").unwrap();
    shader.replace("€ncol", ncols.to_string().as_str());
    shader.replace("€nrow", nrows.to_string().as_str());

    let bind1 = Arc::clone(&var_1);
    let bind2 = Arc::clone(&var_2);

    let bindings_1 = vec![VariableBind::new(bind1, 0)];
    let bindings_2 = vec![VariableBind::new(bind2, 0)];

    let function1 = Function::new(&shader, "add_1", bindings_1);
    let function2 = Function::new(&shader, "add_1", bindings_2);

    algorithm.add_function(function1);
    algorithm.add_function(function2);

    // print!("{:?}", algorithm.get_operations())

    algorithm.finish().await.unwrap();


    let output_1 = Arc::clone(&var_1);
    let output_2 = Arc::clone(&var_2);

    algorithm
        .get_output_unmap(&output_1)
        .await
        .unwrap();
    algorithm
        .get_output_unmap(&output_2)
        .await
        .unwrap();

    let var_lock_1 = var_1.lock().unwrap();
    let var_lock_2  = var_2.lock().unwrap();
    
    let result_1 = var_lock_1.to_array();
    let result_2 = var_lock_2.to_array();


    let check = Array2::ones((500,500));
    assert_eq!(result_1, check);
    assert_eq!(result_2, check);
}

#[tokio::test]
async fn add_1_two_binds_same_var() {
    let array_1 = Array2::zeros((500,500));

    let mut algorithm = Algorithm::new(Some("Test algorithm")).await.unwrap();

    let var_1 = Arc::new(Mutex::new(GpuArray2::new(array_1, "array_1")));

    
    let (nrows, ncols) = var_1.lock().unwrap().get_dims();

    let mut shader = Shader::from_file_path("./tests/shaders/mat2calcs.pwgsl").unwrap();
    shader.replace("€ncol", ncols.to_string().as_str());
    shader.replace("€nrow", nrows.to_string().as_str());

    let bind1 = Arc::clone(&var_1);
    let bind2 = Arc::clone(&var_1);

    let bindings_1 = vec![VariableBind::new(bind1, 0)];
    let bindings_2 = vec![VariableBind::new(bind2, 0)];

    let function1 = Function::new(&shader, "add_1", bindings_1);
    let function2 = Function::new(&shader, "add_1", bindings_2);

    algorithm.add_function(function1);
    algorithm.add_function(function2);

    print!("{:?}", algorithm.get_operations());

    algorithm.finish().await.unwrap();


    let output_1 = Arc::clone(&var_1);

    algorithm
        .get_output_unmap(&output_1)
        .await
        .unwrap();

    let var_lock_1 = var_1.lock().unwrap();
    
    let result_1 = var_lock_1.to_array();

    let check = Array2::ones((500,500)) + 1.0;
    assert_eq!(result_1, check);
}

#[tokio::test]
async fn add_matrices() {
    let array_1 = Array2::ones((500,500));
    let array_2 = Array2::ones((500,500));

    let mut algorithm = Algorithm::new(Some("Test algorithm")).await.unwrap();

    let var_1 = Arc::new(Mutex::new(GpuArray2::new(array_1, "array_1")));
    let var_2 = Arc::new(Mutex::new(GpuArray2::new(array_2, "array_1")));

    
    let (nrows, ncols) = var_1.lock().unwrap().get_dims();

    let mut shader = Shader::from_file_path("./tests/shaders/mat2calcs.pwgsl").unwrap();
    shader.replace("€ncol", ncols.to_string().as_str());
    shader.replace("€nrow", nrows.to_string().as_str());

    let bind1 = Arc::clone(&var_1);
    let bind2 = Arc::clone(&var_2);

    let bindings_1 = vec![VariableBind::new(bind1, 0),VariableBind::new(bind2, 1)];
    

    let function1 = Function::new(&shader, "add_matrices", bindings_1);

    algorithm.add_function(function1);

    // print!("{:?}", algorithm.get_operations())

    algorithm.finish().await.unwrap();


    // let output_1 = Arc::clone(&var_1);
    // let output_2 = Arc::clone(&var_2);

    algorithm
        .get_output_unmap(&var_1)
        .await
        .unwrap();
    algorithm
        .get_output_unmap(&var_2)
        .await
        .unwrap();

    let var_lock_1 = var_1.lock().unwrap();
    let var_lock_2  = var_2.lock().unwrap();
    
    let result_1 = var_lock_1.to_array();
    let result_2 = var_lock_2.to_array();


    let check_2 = Array2::ones((500,500));
    let check_1 = Array2::ones((500,500)) +1.0;
    assert_eq!(result_1, check_1);
    assert_eq!(result_2, check_2);
}
