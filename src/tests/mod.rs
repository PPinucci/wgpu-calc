use crate::algorithm::{Algorithm, Function, Variable, VariableBind};
use crate::coding::Shader;
use bytemuck;
use ndarray::{array, Array2};

#[derive(Debug, PartialEq)]
struct GpuArray2<'a> {
    data: &'a [f32],
    n_rows: u64,
    n_cols: u64,
    name: &'a str,
}

impl<'a> GpuArray2<'a> {
    fn new(array: &'a Array2<f32>, name: &'a str) -> GpuArray2<'a> {
        let (n_cols, n_rows) = array.dim();
        let data = array.as_slice().unwrap();
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
}

#[test]
fn add_test() {
    let array = array![[0., 0., 0.], [1., 1., 1.], [2., 2., 2.]];

    let var = GpuArray2::new(&array, "test array");
    let (nrows, ncols) = var.get_dims();

    let mut shader = Shader::from_file_path("./shaders/mat2calcs.pwgsl").unwrap();
    shader.replace("€cols", ncols.to_string().as_str());
    shader.replace("€nrow", nrows.to_string().as_str());

    let bindings = [VariableBind::new(&var, 0), VariableBind::new(&var, 1)];

    let function = Function::new(&shader, "add", &bindings);

    let mut algorithm = Algorithm::new(Some("Test algorithm"));
    algorithm.add_function(&function);

    // let solver = algorithm.finish();

    // algorithm.run()
    todo!()
}
