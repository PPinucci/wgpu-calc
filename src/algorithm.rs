//! This lib contains the interface to the user defined algorithms
//!
//! Its job is to translate an easier to write code into code which will get to the
//! GPU doing the job itself.
//!
#![allow(dead_code)]
use wgpu::BufferDescriptor;
use anyhow::{self, Ok};

use crate::errors::VariableError;
use crate::solver::Solver;
use std::error::Error;
use std::fmt::Debug;

use crate::coding::Shader;
pub struct Algorithm<'a,V:Variable> {
    variables: Option<&'a V>, // are we sure it's useful? 
    operations: Vec<Operation<'a,V>>
}

pub struct Function<'a, V: Variable> {
    shader: &'a Shader,
    entry_point: &'a str,
    variables: Vec<VariableBind<'a, V>>,
}

#[derive(Debug, PartialEq)]
/// This struct holds a pair of (variable, bind_no)
/// 
/// It's useful for the Function to know which bind group to associate this
/// variable to. 
pub struct VariableBind<'a, V>
where
    V: Variable,
{
    pub variable: &'a V,
    pub bind_group: u32,
}

enum Operation<'a,V:Variable>{
    BufferWrite{buffers:Vec<wgpu::BufferDescriptor<'a>>},
    Bind{descriptor: Vec<VariableBind<'a,V>>},
    Execute{shader:Shader,entry:&'a str},
    Parallel(Vec<Operation<'a,V>>)
}

impl<V:Variable> Algorithm<'_,V>
where
{
    /// Creates a new empty [`Algorithm`]
    /// 
    /// The [`Algorithm`] will not be instantiated with any [`Operation`], will just be empty
    pub fn new()->Self{
        Algorithm { operations: Vec::new(),variables: None}
    }

    /// Adds a [`Function`] to the [`Algorithm`]
    /// 
    /// This function takes the [`Function`] and translate it into [`Operation`], at the same time optimizing
    /// the calculation pipeline
    pub fn add_function(&mut self, function:Function<V>) 
    where
        V: Variable
    {
       let mut b_write: Operation<'_, V> = Operation::BufferWrite { buffers: Vec::new() };
       let f_var = function.variables;

       for var in f_var {
        let var_pos = self.variables
        .iter()
        .position(|&existing_var| existing_var == var.get_variable());
    
        if var_pos.is_none() {
            b_write.add_buffer(var.get_variable()).unwrap()
        }



       }
    }

    /// Consumes the [`Algorithm`] and gives back a [`Solver`]
    /// 
    /// 
    pub fn finish(self) -> Result<Solver<'static>,Box<dyn Error>> {
        todo!()
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

impl<'a,V:Variable> Operation<'a,V> {

    fn add_buffer(&mut self, var:&'a V)->Result<(), anyhow::Error> {
        match self {
            Operation::BufferWrite { buffers } => {
                let descriptor = wgpu::BufferDescriptor{ 
                    label: var.get_name(), 
                    size: var.byte_size(), 
                    usage: wgpu::BufferUsages::STORAGE
                    | wgpu::BufferUsages::COPY_DST
                    | wgpu::BufferUsages::COPY_SRC, 
                    mapped_at_creation: false
                };

                buffers.push(descriptor);
                return Ok(());
            },
            _ => { Err(anyhow::anyhow!("Trying to add a buffer to any Operation other than BufferWrite"))}
        }
        
    }
    
}

pub trait Variable
where
    Self: PartialEq +Debug,
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

    fn get_name(&self)->Option<&str>;

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
    fn dimension_sizes(&self) -> [u32;3];

    /// This method defines the workgroup count for the object
    /// 
    /// It takes the dimension of the object and 
    fn get_workgroup(&self) -> Result<[u32; 3], Box<dyn Error + '_>>
    where
        Self: Debug,
    {
        let dimensions = self.dimension_sizes();

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