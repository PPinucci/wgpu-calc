//! This lib contains the interface to the user defined algorithms
//!
//! Its job is to translate an easier to write code into code which will get to the
//! GPU doing the job itself.
//!
#![allow(dead_code)]
use wgpu::BufferDescriptor;

use crate::errors::VariableError;
use std::error::Error;
use std::fmt::Debug;

use crate::coding::Shader;
pub struct Algorithm<'a, V: Variable> {
    // are we sure? Probably better to keep a sequence of operations, which can be translated to 
    variables: Vec<V>,
    functions: Vec<Function<'a, V>>,
}

pub struct Function<'a, V: Variable> {
    shader: &'a Shader,
    entry_point: &'a str,
    variables: Vec<VariableBind<'a, V>>,
}

#[derive(Debug, PartialEq, Eq)]
/// This struct holds a pair of (variable, bind_no)
/// 
/// It's useful for the Function to know which bind group to associate this
/// variable to. 
pub struct VariableBind<'a, V>
where
    V: Variable,
{
    variable: &'a V,
    bind_group: u32,
}

impl<'a, V> Algorithm<'a, V> 
where
    V: Variable
{
    pub fn new()->Self{
        Algorithm { variables:Vec::new(), functions:Vec::new() }
    }

    pub fn add_function(&'a mut self, _function:Function<'a,V>) {
        match self.functions.len() {
            0 => {
                todo!()
                // self.variables.push(value)
            }
            _=> {todo!()}
        }
    }
}

impl<V> Function<'_, V>
where
    V: Variable
{
    /// Creates a new function from the base entities
    /// 
    /// The aim of this object is basically to organize everything in one place, allowing to better organize the procedures
    /// to pass the function to the GPU.
    /// Note this struct only keeps a reference to the shader code, and indirectly to the Variable object
    /// This is to avoid memory allocation, to speed up the initialisation process and to allow for modificatio of shaders and 
    /// variables up to the submission of the [`Algorithm`] to the GPU.
    /// 
    /// # Arguments
    /// * - `shader` - a reference to a [`coding::Shader`] element, which contains the shader which will perform the operation
    /// * - `entry_point` - the name of the function inside the ['shader`] which will execute the code
    /// *- `vars` - an array reference of [`VariableBind`] which will be the variables passed to the GPU (with the relative bind number)
    pub fn new<'a>(
        shader: &'a Shader,
        entry_point: &'a str,
        vars: &'a [VariableBind<V>],
    ) -> Function<'a, V> {
        let mut variables: Vec<VariableBind<'_, V>> = Vec::new();
        for variable in vars {
            variables.push(variable.clone())
        }
        Function {
            shader,
            entry_point,
            variables,
        }
    }
}

impl<'a, V> Clone for VariableBind<'a, V>
where
    V: Variable,
{
    fn clone(&self) -> Self {
        Self {
            variable: self.variable.clone(),
            bind_group: self.bind_group.clone(),
        }
    }
}

impl<V> VariableBind<'_, V>
where
    V: Variable,
{
    /// Creates a new [`VariableBind`] from the variable and the binding group number
    /// 
    /// This associated the variable, and thus will associate the correct buffer, to the
    /// bind group which has `bind_group` value inside the shader code.
    pub fn new<'a>(variable: &'a V, bind_group: u32) -> VariableBind<'a, V> {
        VariableBind {
            variable,
            bind_group,
        }
    }

    /// gets the [`Variable`] from the [`VariableBind`]
    /// 
    /// This is useful to perform actions on the variables, in particular for the [`Algorithm`] to 
    /// create the associated [`wgpu::BufferDescriptor`] and/or optimize the solution of the 
    pub fn get_variable(&self)-> &V {
        self.variable
    }
}
pub trait Variable
where
    Self: Eq +Debug,
{
    /// This gets a buffer descriptor from the [`Variable`] itself
    /// 
    /// It is useful to create the buffer, the bind group layouts and the ipelines which will be executed
    /// on the GPU
    fn to_buffer_descriptor<'a>(&'a self, label: Option<&'a str>) -> BufferDescriptor {
        return BufferDescriptor {
            label,
            mapped_at_creation: false,
            size: self.byte_size(),
            usage: wgpu::BufferUsages::STORAGE
                | wgpu::BufferUsages::COPY_DST
                | wgpu::BufferUsages::COPY_SRC,
        };
    }

    /// This function calculates the byte size of the object
    /// 
    /// The size needs to be valid and true, as it will be used to calculate the dimension
    /// of the buffer, other than the workgroup needed to be used.
    /// To implement the function consider using [`std::mem::size_of`], while transforming the
    /// the object in simple ones.
    /// Plase read the WGLS sizing standards to better format your object before getting down the 
    /// path of executing calculations. 
    /// 
    fn byte_size(&self) -> u64;

    /// This method is needed to pass the data to the GPU
    /// 
    /// The GPU needs the data as an ordered stream of bit, which is stored in 
    /// the buffer and than distributed to the thread.
    /// Condier using [`bytemuk`] to perform this operation.
    fn byte_data(&self) -> &[u8];

    /// This method is needed to better distribute the workload for the [`Variable`] calculation
    /// 
    /// It returns the size in number of byte for each dimension of the [`Variable`], with its
    /// primary intent being the usage with matrices of maximum 3 dimensions ( see [`Variable::chek_dimensions`])
    /// Each dimension will be associated with a workgroup id in the GPU allowing the parallel execution of the calculus
    fn dimension_sizes(&self) -> &[u32];

    /// This method checks the dimension of the object
    /// 
    /// The obect needs to have maximum 3 dimensions, since this is the maximum number of workgroups available
    /// Consider stacking in one of the dimension higher dimension object so that they will fit in 3 dimensions.
    /// For how the workgroup are limited in memory size, it's better not to stack in the 3rd dimension (which has the lowest maximum workgroup dimension), 
    /// rotate the object instead so it's stacked in one of the other 2 dimensions.
    /// 
    /// # Ouptut
    /// The function returns a [`Result`] which is either a [`VariableError::DimensionError`] variant or,
    /// if Ok(), the dimensions of the object.
    /// 
    /// # Tip
    /// Use this to retrieve the dimension of the object, since it's the easiest and safest way to avoid incompatibility with the GPU
    fn check_dimensions(&self) -> Result<&[u32], Box<dyn Error + '_>>
    {
        let dimension = self.dimension_sizes();
        if dimension.len() > 3 {
            return Err(Box::new(VariableError::DimensionError(self)));
        }
        Ok(dimension)
    }

    /// This method defines the workgroup count for the object
    /// 
    /// It takes the dimension of the object and 
    fn get_workgroup(&self) -> Result<[u32; 3], Box<dyn Error + '_>>
    where
        Self: Debug,
    {
        let dimensions = self.check_dimensions()?;
        // let _size = self.byte_size();

        let mut workgroup = [1u32; 3];
        for id in 0..dimensions.len() {
            match (dimensions[id], id) {
                (0..=65535, _) => workgroup[id] = dimensions[id],
                (65536..=4194240, _) => {
                    // error to convey workgoup number for i, convey also the dimension which gave the error
                    return Err(Box::new(VariableError::<u32>::WorkgroupDimensionError(
                        id as u32,
                    )));
                }
                (4194241..=16776960, 1 | 2) => {
                    // same as above
                    return Err(Box::new(VariableError::<u32>::WorkgroupDimensionError(
                        id as u32,
                    )));
                }
                _ => {
                    // fatal error not possible to instantiate element, too big
                    panic!("Variable dimension is too big, please decrease size in order to fit to the allowed calculation dimension")
                }
            }
        }
        Ok(workgroup)
    }
}