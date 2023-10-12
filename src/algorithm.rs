//! This lib contains the interface to the user defined algorithms
//!
//! Its job is to translate an easier to write code into code which will get to the
//! GPU doing the job itself.
//!
#![allow(dead_code)]
use anyhow::Ok;
use wgpu::BufferDescriptor;

use crate::errors::VariableError;
use crate::interface::Executor;
use std::fmt::Debug;
use std::num::NonZeroU64;

use crate::coding::Shader;

#[derive(Clone)]
pub struct Algorithm<'a, V: Variable> {
    variables: Vec<&'a VariableBind<'a, V>>,
    modules: Vec<Module<'a>>,
    operations: Vec<Operation>,
    label: Option<&'a str>,
}

pub struct Function<'a, V: Variable> {
    shader: &'a Shader,
    entry_point: &'a str,
    variables: Vec<VariableBind<'a, V>>,
}
#[derive(Debug)]
pub struct Mutable;
#[derive(Debug)]
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
    buffer_descriptor: wgpu::BufferDescriptor<'a>,
    mutable: std::marker::PhantomData<Type>,
}
#[derive(Debug, PartialEq,Clone)]
pub(super) struct Module<'a> {
    shader: &'a Shader,
    entry_point: Vec<&'a str>,
}

#[derive(Debug,Clone)]
enum Operation {
    BufferWrite {
        variable_index: usize,
    },
    Bind {
        bind_index: Vec<usize>,
    },
    Execute {
        module_index: usize,
        entry_point_index: usize,
    },
}

impl<'a, V: Variable> Algorithm<'a, V> {
    /// Creates a new empty [`Algorithm`]
    ///
    /// The [`Algorithm`] will not be instantiated with any [`Operation`], will just be empty
    pub fn new(label: Option<&'a str>) -> Self {
        Algorithm {
            operations: Vec::new(),
            variables: Vec::new(),
            modules: Vec::new(),
            label,
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
    pub fn add_function(&mut self, function: &'a Function<'a,V>)
    where
        V: Variable,
    {
        let f_var = function.get_variables();
        let mut binds = Operation::Bind {
            bind_index: Vec::new(),
        };
        for var in f_var {
            if let Some(_) = self
                .variables
                .iter()
                .position(|&existing_var| existing_var == var)
            {
                self.variables.push(var);
                let index = self.variables.len() - 1;
                binds.add_bind(index).unwrap();
            } else {
                self.variables.push(var);
                let index = self.variables.len() - 1;
                self.operations.push(Operation::BufferWrite {
                    variable_index: index,
                });
                binds.add_bind(index).unwrap();
            }
        }

        self.operations.push(binds);

        if let Some(pos) = self
            .modules
            .iter()
            .position(|existing_module| existing_module.shader == function.shader)
        {
            if let Some(index) = self.modules[pos].find_entry_point(function.entry_point) {
                self.operations.push(Operation::Execute {
                    module_index: pos,
                    entry_point_index: index,
                })
            } else {
                self.modules[pos].add_entry_point(function.entry_point);
                self.operations.push(Operation::Execute {
                    module_index: pos,
                    entry_point_index: &self.modules[pos].entry_point.len() - 1,
                })
            }
        } else {
            self.modules.push(Module {
                shader: function.shader,
                entry_point: vec![function.entry_point],
            });
        }
    }

    pub fn optimize(&mut self) {
        todo!()
    }
    /// Consumes the [`Algorithm`] and gives back a [`Solver`]
    ///
    /// This is the moement where the machine is asked for a GPU device and the [`Operations`] are
    /// put in a pipeline after having been optimized
    pub async fn finish(&mut self) -> Result<Executor<'a>, anyhow::Error> {
        let mut executor = Executor::new(self.label).await?;

        // self.optimize();

        let mut buffers = Vec::new();
        let mut bind_layout_entries: Vec<wgpu::BindGroupLayoutEntry> = Vec::new();
        let mut workgroups = [0 as u32; 3];

        for variable in &self.variables {
            buffers.push(executor.get_buffer(&variable.buffer_descriptor));
            bind_layout_entries.push(variable.get_bind_group_layout_entry())
        }

        let mut operation_bind_layout_entries = Vec::new();
        let mut operation_bind_entries = Vec::new();

        for operation in &self.operations {
            match operation {
                Operation::Bind { bind_index } => {
                    operation_bind_layout_entries = Vec::new();
                    operation_bind_entries = Vec::new();
                    for index in bind_index {
                        // let layout_add = &mut operation_bind_layout_entries;
                        operation_bind_layout_entries.push(bind_layout_entries[*index]);
                        operation_bind_entries.push(wgpu::BindGroupEntry {
                            binding: self.variables[*index].bind_group,
                            resource: buffers[*index].as_entire_binding(),
                        });
                        workgroups = self.variables[*index].variable.get_workgroup()?;
                    }
                    continue;
                }
                Operation::BufferWrite { variable_index } => {
                    let variable = &self.variables[*variable_index].variable;
                    let buffer_descriptor = variable.to_buffer_descriptor(None);
                    let buffer = executor.get_buffer(&buffer_descriptor);
                    executor.write_buffer(&buffer, variable.byte_data());
                    continue;
                }
                Operation::Execute {
                    module_index,
                    entry_point_index,
                } => {
                    let shader = self.modules[*module_index].shader;
                    let entry_point = self.modules[*module_index].entry_point[*entry_point_index];

                    let bind_layout_descriptor = wgpu::BindGroupLayoutDescriptor {
                        label: None,
                        entries: &operation_bind_layout_entries,
                    };

                    let bind_layout = executor.get_bind_group_layout(&bind_layout_descriptor);
                    let bind_group_desriptor = wgpu::BindGroupDescriptor {
                        label: None,
                        layout: &bind_layout,
                        entries: &operation_bind_entries,
                    };
                    let bind_group = executor.get_bind_group(&bind_group_desriptor);

                    let pipeline_layout_descriptor = wgpu::PipelineLayoutDescriptor {
                        label: None,
                        bind_group_layouts: &[&bind_layout],
                        push_constant_ranges: &[],
                    };

                    let pipeline_layout = executor.get_pipeline_layout(&pipeline_layout_descriptor);

                    let shader_module = executor.get_shader_module(shader);

                    let pipeline_descriptor = wgpu::ComputePipelineDescriptor {
                        label: None,
                        layout: Some(&pipeline_layout),
                        module: &shader_module,
                        entry_point,
                    };
                    let pipeline: wgpu::ComputePipeline =
                        executor.get_pipeline(&pipeline_descriptor);

                    let command_buffer = [executor
                        .dispatch_bind_and_pipeline(&bind_group, &pipeline, &workgroups, None)
                        .finish()];

                    executor.execute(command_buffer.into_iter());
                    continue;
                }
            }
        }
        self.operations = Vec::new();
        return Ok(executor);
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

    fn get_variables<'a>(&'a self) -> &Vec<VariableBind<'a,V>>{
        &self.variables
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
            buffer_descriptor: self.buffer_descriptor.clone(),
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
    pub fn new(
        variable: &'a V,
        bind_group: u32,
        label: Option<&'a str>,
    ) -> VariableBind<'a, V, Mutable> {
        let buffer_descriptor = variable.to_buffer_descriptor(label);
        VariableBind {
            variable,
            bind_group,
            buffer_descriptor,
            mutable: Default::default(),
        }
    }

    /// Creates a [`wgpu::BindGroupLayoutEntry`] from [`self`]
    ///
    /// USeful to build the bind group layout for the executor to execute.
    pub fn get_bind_group_layout_entry(&self) -> wgpu::BindGroupLayoutEntry {
        wgpu::BindGroupLayoutEntry {
            binding: self.bind_group,
            visibility: wgpu::ShaderStages::COMPUTE,
            ty: wgpu::BindingType::Buffer {
                ty: wgpu::BufferBindingType::Storage { read_only: false },
                min_binding_size: NonZeroU64::new(self.buffer_descriptor.size),
                has_dynamic_offset: false,
            },
            count: None,
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
            buffer_descriptor: self.buffer_descriptor,
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
            buffer_descriptor: self.buffer_descriptor,
            mutable: std::marker::PhantomData::<Mutable>,
        }
    }

    pub fn get_bind_group_layout_entry(&self) -> wgpu::BindGroupLayoutEntry {
        wgpu::BindGroupLayoutEntry {
            binding: self.bind_group,
            visibility: wgpu::ShaderStages::COMPUTE,
            ty: wgpu::BindingType::Buffer {
                ty: wgpu::BufferBindingType::Storage { read_only: true },
                min_binding_size: NonZeroU64::new(self.buffer_descriptor.size),
                has_dynamic_offset: false,
            },
            count: None,
        }
    }
}

impl<V: Variable> PartialEq for VariableBind<'_, V> {
    fn eq(&self, other: &Self) -> bool {
        self.variable == other.variable
            && self.bind_group == other.bind_group
            && self.mutable == other.mutable
    }
}

impl<'a> Module<'a> {
    pub(super) fn new(shader: &'a Shader) -> Self {
        Self {
            shader,
            entry_point: Vec::new(),
        }
    }

    pub(super) fn add_entry_point(&mut self, e_p: &'a str) -> usize {
        self.entry_point.push(e_p);
        return self.entry_point.len() - 1;
    }

    pub(super) fn find_entry_point(&self, e_p: &'a str) -> Option<usize> {
        self.entry_point.iter().position(|&entry| entry == e_p)
    }
}

impl Operation {
    fn add_bind(&mut self, index: usize) -> Result<(), anyhow::Error> {
        match self {
            Operation::Bind { bind_index } => {
                bind_index.push(index);
                Ok(())
            }
            _ => Err(anyhow::anyhow!(
                "Can't add a bind to any operation which is not an [`Operation::Bind`]"
            )),
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

    fn get_data() {
        todo!()
    }
}
