//! This lib contains the interface to the user defined algorithms
//!
//! Its job is to translate an easier to write code into code which will get to the
//! GPU doing the job itself.
//!
#![allow(dead_code)]
use anyhow::{self, Ok};
use wgpu::BufferDescriptor;

use crate::errors::VariableError;
use crate::solver::Solver;
// use std::error::Error;
use std::fmt::Debug;

use crate::coding::Shader;
pub struct Algorithm<'a, V: Variable> {
    variables: Vec<&'a VariableBind<'a, V>>, // are we sure it's useful?
    operations: Vec<Operation<'a, V>>,
}

pub struct Function<'a, V: Variable> {
    shader: &'a Shader,
    entry_point: &'a str,
    variables: Vec<VariableBind<'a, V>>,
}

pub struct Mutable;
pub struct Immutable;

#[derive(Debug, PartialEq)]
/// This struct holds a pair of (variable, bind_no)
///
/// It's useful for the Function to know which bind group to associate this
/// variable to.
pub struct VariableBind<'a, V, Type = Mutable>
where
    V: Variable,
{
    variable: &'a V,
    bind_group: u32,
    mutable: std::marker::PhantomData<Type>,
}

enum Operation<'a, V: Variable> {
    BufferWrite {
        buffers: Vec<wgpu::BufferDescriptor<'a>>,
    },
    Bind {
        descriptor: Vec<&'a VariableBind<'a, V>>,
    },
    Execute {
        shader: &'a Shader,
        entry: &'a str,
    },
    Parallel(Vec<Operation<'a, V>>),
}

impl<'a, V: Variable> Algorithm<'a, V> {
    /// Creates a new empty [`Algorithm`]
    ///
    /// The [`Algorithm`] will not be instantiated with any [`Operation`], will just be empty
    pub fn new() -> Self {
        Algorithm {
            operations: Vec::new(),
            variables: Vec::new(),
        }
    }

    /// Adds a [`Function`] to the [`Algorithm`]
    ///
    /// This function takes the [`Function`] and translate it into [`Operation`], at the same time optimizing
    /// the calculation pipeline
    /// 
    /// # Panics
    /// - if the previous function only resulted in a [`Operation::Bind`] since this should 
    ///     always followed by a [`Operation::Execute`]
    pub fn add_function(&'a mut self, function: &'a Function<V>)
    where
        V: Variable,
    {
        let mut b_write: Operation<'_, V> = Operation::BufferWrite {
            buffers: Vec::new(),
        };
        let mut bind: Operation<'_, V> = Operation::Bind {
            descriptor: Vec::new(),
        };
        let f_var = &function.variables;

        // For each variable in the function gets the position of the same variable
        // in the Algorithm Funcion list (if present)
        f_var.into_iter().for_each(|var| {
            let var_pos = self
                .variables
                .iter()
                .position(|&existing_var| existing_var == var);
            // If the variable is present, than if the bind group is different adds a
            // new bind variable
            if let Some(position) = var_pos {
                if var.get_bind() != self.variables[position].get_bind() {
                    bind.add_bind(var).unwrap();
                }
            } else {
                // if variable is not present it adds to the list and create a buffer write o
                // of it
                b_write.add_buffer(var.get_variable()).unwrap();
                self.variables.push(var);
            }
        });
        
        let exe: Operation<'a, V> = Operation::Execute {
            shader: function.shader,
            entry: function.entry_point,
        };

        if let Some(mut last_op) = self.operations.last() {
            match last_op {
                Operation::Bind { .. } => {
                    panic!("Somethig went wrong on the last function push, it was only a binding without any execution")
                }
                Operation::BufferWrite { buffers } => {
                    todo!()
                }
                Operation::Parallel(ops) => {
                    todo!()
                }
                Operation::Execute { shader, entry } => {
                    todo!()
                }
            }
        } else {
            self.operations.push(b_write);
            self.operations.push(bind);
            self.operations.push(exe);
        }
    }

    /// Consumes the [`Algorithm`] and gives back a [`Solver`]
    ///
    ///
    pub fn finish(self) -> Result<Solver<'static>, anyhow::Error> {
        todo!()
    }
}

impl<V> Function<'_, V>
where
    V: Variable,
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
            mutable: self.mutable.clone(),
        }
    }
}

impl<'a, V> VariableBind<'a, V, Mutable>
where
    V: Variable,
{
    /// Creates a new [`VariableBind`] from the variable and the binding group number
    ///
    /// This associated the variable, and thus will associate the correct buffer, to the
    /// bind group which has `bind_group` value inside the shader code.
    /// The variable is set as "mutable" by default, as it is considered [`unsafe`] for it to be immutable.
    /// To set as immuable use [`VariableBind::set_immutable`] method.
    /// Read [`VariableBind::is_mutable`] method for further explanation
    /// # Arguments
    /// * - `variable` - a reference to the variable to bind
    /// * - `bind_group` - the bind group number the variabe will be associated with
    pub fn new(variable: &'a V, bind_group: u32) -> VariableBind<'a, V, Mutable> {
        VariableBind {
            variable,
            bind_group,
            mutable: Default::default(),
        }
    }

    /// This method returns weather the variable is mutable or not.
    ///
    /// When the variable is set as immutable, it is supposed not to vary during GPU operation,
    /// i.e. it's a [`Variable`] which will be read only and never wrote to.
    pub fn is_mutable(&self) -> bool {
        true
    }

    /// Sets the [`VariableBind`] to be immutable, thus read only
    ///
    /// It is not unsafe per se, but set as such to warn about the possible implications of this.
    /// At the time of writing any variable can be set as read/write and set as immutable. This could potentially
    /// cause concurrency problems when queueing the pipelines on tha GPU.
    /// An immutable [`VariableBind`] is considered not to change during the calculation.
    pub unsafe fn set_immutable(self) -> VariableBind<'a, V, Immutable> {
        VariableBind {
            variable: self.variable,
            bind_group: self.bind_group,
            mutable: std::marker::PhantomData::<Immutable>,
        }
    }
}

impl<V, Type> VariableBind<'_, V, Type>
where
    V: Variable,
{
    /// gets the [`Variable`] from the [`VariableBind`]
    ///
    /// This is useful to perform actions on the variables, in particular for the [`Algorithm`] to
    /// create the associated [`wgpu::BufferDescriptor`] and/or optimize the solution of the
    pub fn get_variable(&self) -> &V {
        self.variable
    }

    /// Gets the bind group the [`Variable`] is set to
    pub fn get_bind(&self) -> u32 {
        self.bind_group
    }
}

impl<'a, V> VariableBind<'a, V, Immutable>
where
    V: Variable,
{
    /// Sets the variable as mutable
    ///
    /// This tells the [`Algorithm`] that the variable coulbe be muted by a function
    pub fn set_mutable(self) -> VariableBind<'a, V, Mutable> {
        VariableBind {
            variable: self.variable,
            bind_group: self.bind_group,
            mutable: std::marker::PhantomData::<Mutable>,
        }
    }
}

impl<V: Variable> PartialEq for VariableBind<'_, V, Mutable> {
    fn eq(&self, other: &Self) -> bool {
        self.variable == other.variable
            && self.bind_group == other.bind_group
            && self.mutable == other.mutable
    }
}

impl<'a, V: Variable> Operation<'a, V> {
    fn add_buffer(&mut self, var: &'a V) -> Result<(), anyhow::Error> {
        match self {
            Operation::BufferWrite { buffers } => {
                let descriptor = wgpu::BufferDescriptor {
                    label: var.get_name(),
                    size: var.byte_size(),
                    usage: wgpu::BufferUsages::STORAGE
                        | wgpu::BufferUsages::COPY_DST
                        | wgpu::BufferUsages::COPY_SRC,
                    mapped_at_creation: false,
                };

                buffers.push(descriptor);
                return Ok(());
            }
            _ => Err(anyhow::anyhow!(
                "Trying to add a buffer to any Operation other than BufferWrite"
            )),
        }
    }

    fn add_bind(&mut self, var_bind: &'a VariableBind<'a, V>) -> Result<(), anyhow::Error> {
        match self {
            Operation::Bind { descriptor } => {
                descriptor.push(var_bind);
                return Ok(());
            }
            _ => {
                return Err(anyhow::anyhow!(
                    "Trying to add a bind to any Operation other than Bind"
                ))
            }
        }
    }
    fn len(&self) -> Result<usize, anyhow::Error> {
        match self {
            Operation::Execute { .. } => Err(anyhow::anyhow!(
                "Cannot get length of Execute Operation, only one is present"
            )),
            Operation::Bind { descriptor } => Ok(descriptor.len()),
            Operation::BufferWrite { buffers } => Ok(buffers.len()),
            Operation::Parallel(ops) => Ok(ops.len()),
        }
    }
}

pub trait Variable
where
    Self: PartialEq + Debug,
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

    fn get_name(&self) -> Option<&str>;

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
    fn dimension_sizes(&self) -> [u32; 3];

    /// This method defines the workgroup count for the object
    ///
    /// It takes the dimension of the object and
    fn get_workgroup(&self) -> Result<[u32; 3], anyhow::Error>
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
                    return Err(VariableError::<u32>::WorkgroupDimensionError(id as u32).into());
                }
                (4194241..=16776960, 1 | 2) => {
                    // same as above
                    return Err(VariableError::<u32>::WorkgroupDimensionError(id as u32).into());
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
