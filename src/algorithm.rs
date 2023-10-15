//! This lib contains the interface to the user defined algorithms
//!
//! Its job is to translate an easier to write code into code which will get to the
//! GPU doing the job itself.
//!
//! All the structs in this libs are defined to operate with objects implementing
//! the [`Variable`] trait.
//!
//! # Principal components:
//! - [`Function`] is a struct which holds the definition of a function. It's a [`Shader`]
//!     with an ['entry point'] which performs operations on some [`Variable`]
//! - [`VariableBind`] is the link between a [`Variable`] and a 'bind group` defined in the shader`
//!     Each [`Variable`] can potentially have more than one bind in the [`Shader`] and the definition
//!     is held in this struct
//! - [`Algorithm`] is the operational part of this library, it collects instances of [`Function`]
//!     and tries to translate them as efficiently as possible to a series of [`Operation`]
//!     Once every function is inserted in the Algorithm, the [`Algorithm::finish`] method is used
//!     to actually send everything to the GPU and do the calculations
//!
//!
#![allow(dead_code)]
use anyhow::{anyhow, Ok};
use futures_intrusive::buffer;
use std::fmt::Debug;
use std::num::NonZeroU64;
use std::ops::Deref;
use std::sync::{Arc, Mutex};

use crate::coding::Shader;
use crate::interface::Executor;
use crate::variable::{self, Variable};

/// This struct is the container for the different operations to perform
///
/// It helds all the necessary components to bind each [`Variable`],
/// write the necessary buffers and execute the pipelines on the GPU.
///
/// The functions added to the [`Algorithm`] will be performed in series preserving
/// the desired output.
/// Some optimisation will be done in the future prior to executing the operations.
/// # Example
/// ```
/// use wgpu_calc::algorithm::{Algorithm,Function, VariableBind};
///
/// ```
#[derive(Debug)]
pub struct Algorithm<'a, V: Variable> {
    variables: Vec<StoredVariable<V>>,
    modules: Vec<Module<'a>>,
    operations: Vec<Operation<'a>>,
    label: Option<&'a str>,
    executor: Executor<'a>,
}

pub struct Function<'a, V: Variable> {
    shader: &'a Shader,
    entry_point: &'a str,
    variables: Vec<VariableBind<V>>,
}
#[derive(Debug)]
pub struct Mutable;
#[derive(Debug)]
pub struct Immutable;

#[derive(Debug)]
/// This struct holds a pair of (variable, bind_no)
///
/// It's useful for the Function to know which bind group to associate this
/// variable to.
pub struct VariableBind<V, Type = Mutable>
where
    V: Variable,
{
    variable: Arc<Mutex<V>>,
    bind_group: u32,
    mutable: std::marker::PhantomData<Type>,
}

#[derive(Debug)]
struct StoredVariable<V, Type = Mutable>
where
    V: Variable,
{
    variable: Arc<Mutex<V>>,
    // buffer_descriptor: wgpu::BufferDescriptor<'a>,
    buffer: Option<wgpu::Buffer>,
    mutable: std::marker::PhantomData<Type>,
}

#[derive(Debug, PartialEq, Clone)]
pub(super) struct Module<'a> {
    shader: &'a Shader,
    entry_point: Vec<&'a str>,
}

#[derive(Debug, Clone)]
pub enum Operation<'a> {
    BufferWrite {
        variable_index: usize,
    },
    Bind {
        bind_index: Vec<usize>,
        bind_groups: Vec<u32>,
    },
    Execute {
        module_index: usize,
        entry_point_index: usize,
        label: &'a str,
    },
}

impl<'a, V: Variable> Algorithm<'a, V> {
    /// Creates a new empty [`Algorithm`]
    ///
    /// The [`Algorithm`] will not be instantiated with any [`Operation`], will just be empty
    pub async fn new(label: Option<&'a str>) -> Result<Algorithm<'a, V>, anyhow::Error> {
        let executor = Executor::new(label).await?;
        Ok(Algorithm {
            operations: Vec::new(),
            variables: Vec::new(),
            modules: Vec::new(),
            label,
            executor,
        })
    }

    /// Adds a [`Function`] to the [`Algorithm`]
    ///
    /// This function takes the [`Function`] and translate it into [`Operation`], at the same time optimizing
    /// the calculation pipeline
    ///
    pub fn add_function(&mut self, function: Function<'a, V>)
    where
        V: Variable,
    {
        let f_label = stringify!(function);
        let f_var = function.variables;
        let mut binds = Operation::Bind {
            bind_index: Vec::new(),
            bind_groups: Vec::new(),
        };
        for var in f_var {
            if let Some(pos) = self.variables.iter().position(|existing_var| {
                Arc::ptr_eq(&existing_var.variable, &var.variable)
            }) {
                binds.add_bind(pos, var.bind_group).unwrap();
            } else {
                let var_ref= Arc::clone(&var.variable);
                let lock = var_ref.lock().unwrap();
                let buffer_descriptor = lock.to_buffer_descriptor();
                let buffer = self.executor.get_buffer(&buffer_descriptor);
                
                let sto_var = StoredVariable{
                    variable:Arc::clone(&var.variable),
                    buffer: Some(buffer), 
                    mutable: std::marker::PhantomData::<Mutable> };
                
                self.variables.push(sto_var);
                let index = self.variables.len() - 1;

                self.operations.push(Operation::BufferWrite {
                    variable_index: index,
                });
                binds.add_bind(index, var.bind_group).unwrap();
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
                    label: f_label,
                })
            } else {
                self.modules[pos].add_entry_point(function.entry_point);
                self.operations.push(Operation::Execute {
                    module_index: pos,
                    entry_point_index: &self.modules[pos].entry_point.len() - 1,
                    label: f_label,
                })
            }
        } else {
            self.modules.push(Module {
                shader: function.shader,
                entry_point: vec![function.entry_point],
            });
            self.operations.push(Operation::Execute {
                module_index: self.modules.len()- 1,
                entry_point_index: 0,
                label: f_label,
            })
        }
    }

    pub fn optimize(&mut self) {
        todo!()
    }
    
    // pub fn get_operations(&self)-> Vec<Operation> {
    //     self.operations.clone()
    // }

    /// 
    pub async fn finish(&mut self) -> Result<(), anyhow::Error> {
        // self.optimize();

        // let mut buffers = Vec::new();
        let mut workgroups = [0 as u32; 3];

        // for variable in &self.variables {
        //     let variable = variable.variable.lock().unwrap();
        //     buffers.push(self.executor.get_buffer(&variable.to_buffer_descriptor()));
        // }

        let mut operation_bind_layout_entries = Vec::new();
        let mut operation_bind_entries = Vec::new();

        for operation in &self.operations {
            match operation {
                Operation::Bind {
                    bind_index,
                    bind_groups,
                } => {
                    operation_bind_layout_entries = Vec::new();
                    operation_bind_entries = Vec::new();
                    for (index, group) in bind_index.iter().zip(bind_groups) {
                        let sto_var = &self.variables[*index];
                        // let layout_add = &mut operation_bind_layout_entries;
                        operation_bind_layout_entries
                            .push(sto_var.get_bind_group_layout_entry(*group));
                        // operation_bind_layout_entries.push(bind_layout_entries[*index]);
                        operation_bind_entries.push(wgpu::BindGroupEntry {
                            binding: *group,
                            resource: sto_var.buffer.as_ref().unwrap().as_entire_binding(),
                        });
                        workgroups = self.variables[*index]
                            .variable
                            .lock()
                            .unwrap()
                            .get_workgroup()?;
                    }
                }
                Operation::BufferWrite { variable_index } => {
                    let buffer = self.variables[*variable_index].buffer.as_ref().unwrap();
                    let sto_var = &self.variables[*variable_index];
                    let data_lock = sto_var.variable.lock().unwrap();
                    // let buffer = self.executor.get_buffer(&data_lock.to_buffer_descriptor());
                    
                    // // let buffer = &buffers[*variable_index];
                    self.executor.write_buffer(&buffer, data_lock.byte_data());
                    // self.variables[*variable_index].buffer = Some(buffer);
                                }
                Operation::Execute {
                    module_index,
                    entry_point_index,
                    label,
                } => {
                    let shader = self.modules[*module_index].shader;
                    let entry_point = self.modules[*module_index].entry_point[*entry_point_index];

                    let bind_layout_descriptor = wgpu::BindGroupLayoutDescriptor {
                        label: Some(label),
                        entries: &operation_bind_layout_entries,
                    };

                    let bind_layout = self.executor.get_bind_group_layout(&bind_layout_descriptor);
                    let bind_group_desriptor = wgpu::BindGroupDescriptor {
                        label: Some(label),
                        layout: &bind_layout,
                        entries: &operation_bind_entries,
                    };
                    let bind_group = self.executor.get_bind_group(&bind_group_desriptor);

                    let pipeline_layout_descriptor = wgpu::PipelineLayoutDescriptor {
                        label: Some(label),
                        bind_group_layouts: &[&bind_layout],
                        push_constant_ranges: &[],
                    };

                    let pipeline_layout = self
                        .executor
                        .get_pipeline_layout(&pipeline_layout_descriptor);

                    let shader_module = self.executor.get_shader_module(shader);

                    let pipeline_descriptor = wgpu::ComputePipelineDescriptor {
                        label: Some(label),
                        layout: Some(&pipeline_layout),
                        module: &shader_module,
                        entry_point,
                    };
                    let pipeline: wgpu::ComputePipeline =
                        self.executor.get_pipeline(&pipeline_descriptor);

                    let command_buffer = [self
                        .executor
                        .dispatch_bind_and_pipeline(&bind_group, &pipeline, &workgroups, None)
                        .finish()];

                    self.executor.execute(command_buffer.into_iter());
                }
            }
        }
        self.operations = Vec::new();
        return Ok(());
    }

    /// This method overwrite the [`Variable`] *`var` with the ouptut of the calculation
    ///
    /// The function returns an error if the variable is not found in the [`Algorithm`] or
    /// if the
    pub async fn get_output_unmap(&mut self, var: &Arc<Mutex<V>>) -> Result<(), anyhow::Error> {
        let mut var_pos = 0;
        if let Some(pos) = self
            .variables
            .iter()
            .position(|existing_var| Arc::ptr_eq(&existing_var.variable, var))
        {
            if let Some(_) = &self.variables[pos].buffer {
                var_pos = pos;
            } else {
                return Err(anyhow!(
                    "No buffer present for variable {:?}",
                    var.lock().unwrap().get_name()
                ));
            }
        } else {
            return Err(anyhow!(
                "Variable {:?} not found in {:?} Algorithm",
                var.lock().unwrap().get_name(),
                self.label
            ));
        }

        let buffer = self.variables[var_pos].buffer.as_ref().unwrap();
        let output = self.executor.read_buffer(&buffer).await;
        // let slice = 
        let mut var_write = self.variables[var_pos].variable.lock().unwrap();
        var_write.read_data(&output);

        // self.variables[var_pos].write_variable(output);
        return Ok(());
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
        variables: Vec<VariableBind<V>>,
    ) -> Function<'a, V> {
        Function {
            shader,
            entry_point,
            variables,
        }
    }

    // fn get_variables<'a>(&'a self) -> Vec<&'a mut V> {
    //     let mut vars = Vec::new();
    //     for variable in &self.variables {
    //         vars.push(variable.variable)
    //     }
    //     return vars;
    // }
}

// impl<'a, V> Clone for VariableBind<'a, V>
// where
//     V: Variable,
// {
//     fn clone(&self) -> Self {
//         Self {
//             variable: self.variable.clone(),
//             bind_group: self.bind_group.clone(),
//             mutable: self.mutable.clone(),
//         }
//     }
// }

impl<'a, V> VariableBind<V, Mutable>
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
    pub fn new(variable: Arc<Mutex<V>>, bind_group: u32) -> VariableBind<V, Mutable> {
        // let variable = Arc::clone(Mutex::new(*var));
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
    pub unsafe fn set_immutable(self) -> VariableBind<V, Immutable> {
        VariableBind {
            variable: self.variable,
            bind_group: self.bind_group,
            mutable: std::marker::PhantomData::<Immutable>,
        }
    }

    // pub fn get_var_mut(self)-> &'a mut V {
    //     self.variable.re
    // }
}

impl<V, Type> VariableBind<V, Type>
where
    V: Variable,
{
    // /// gets the [`Variable`] from the [`VariableBind`]
    // ///
    // /// This is useful to perform actions on the variables, in particular for the [`Algorithm`] to
    // /// create the associated [`wgpu::BufferDescriptor`] and/or optimize the solution of the
    // pub fn get_variable(&self) -> &V {
    //     self.variable.lock().unwrap().deref()
    // }

    // /// Gets the bind group the [`Variable`] is set to
    // pub fn get_bind(&self) -> u32 {
    //     self.bind_group
    // }
}

impl<'a, V> VariableBind<V, Immutable>
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
    // pub fn new_immutable(variable: &'a V, bind_group: u32) -> VariableBind<'a, V, Immutable> {
    //     VariableBind {
    //         variable,
    //         bind_group,
    //         mutable: std::marker::PhantomData::<Immutable>,
    //     }
    // }
    /// Sets the variable as mutable
    ///
    /// This tells the [`Algorithm`] that the variable coulbe be muted by a function
    pub fn set_mutable(self) -> VariableBind<V, Mutable> {
        VariableBind {
            variable: self.variable,
            bind_group: self.bind_group,
            mutable: std::marker::PhantomData::<Mutable>,
        }
    }
}

// impl<V: Variable> PartialEq for VariableBind<V> {
//     fn eq(&self, other: &Self) -> bool {
//         self.variable == other.variable
//             && self.bind_group == other.bind_group
//             && self.mutable == other.mutable
//     }
// }

impl<V: Variable> StoredVariable<V> {
    // fn get_var(&self) -> &V {
    //     let variable = Arc::clone(&self.variable);
    //     variable.lock().
    // }
    // fn get_buffer(&self) -> Option<&wgpu::Buffer> {
    //     self.buffer.as_ref()
    // }

    // fn write_variable(&mut self, vec: Vec<f32>) {
    //     self.variable.lock().unwrap().read_vec(vec)
    // }

    /// Creates a [`wgpu::BindGroupLayoutEntry`] from [`self`]
    ///
    /// USeful to build the bind group layout for the executor to execute.
    pub fn get_bind_group_layout_entry(&self, bind: u32) -> wgpu::BindGroupLayoutEntry {
        let size = self.variable.lock().unwrap().byte_size();
        wgpu::BindGroupLayoutEntry {
            binding: bind,
            visibility: wgpu::ShaderStages::COMPUTE,
            ty: wgpu::BindingType::Buffer {
                ty: wgpu::BufferBindingType::Storage { read_only: false },
                min_binding_size: NonZeroU64::new(size),
                has_dynamic_offset: false,
            },
            count: None,
        }
    }
}

impl<'a, V: Variable> StoredVariable<V, Mutable> {
    fn new(variable: Arc<Mutex<V>>) -> StoredVariable<V> {
        // let buffer_descriptor = var.to_buffer_descriptor();
        // let variable: Arc<Mutex<V>> = Arc::new(Mutex::new(*var));
        Self {
            variable,
            // buffer_descriptor,
            buffer: None,
            mutable: std::marker::PhantomData::<Mutable>,
        }
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

impl Operation<'_> {
    fn add_bind(&mut self, index: usize, bind_group: u32) -> Result<(), anyhow::Error> {
        match self {
            Operation::Bind {
                bind_index,
                bind_groups,
            } => {
                bind_index.push(index);
                bind_groups.push(bind_group);
                Ok(())
            }
            _ => Err(anyhow::anyhow!(
                "Can't add a bind to any operation which is not an [`Operation::Bind`]"
            )),
        }
    }
}
