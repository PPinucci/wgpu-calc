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
use futures_intrusive::buffer::RingBuf;
use std::fmt::Debug;
use std::num::NonZeroU64;
use std::ops::Deref;
use std::sync::{Arc, Mutex, MutexGuard};
use std::thread;

use crate::coding::Shader;
use crate::errors::VariableError;
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
    buffers:Vec<wgpu::Buffer>,
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
struct StoredVariable<V>
where
    V: Variable,
{
    variable: Arc<Mutex<V>>,
    binds: Vec<usize>,
    buffer_index: usize,
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
        buffer_index: usize,
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
            buffers: Vec::new(),
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
                    binds: vec![var.bind_group as usize],
                    buffer_index: self.buffers.len(),
                    // buffer: Some(buffer), 
                    // mutable: std::marker::PhantomData::<Mutable> 
                };
                
                self.variables.push(sto_var);
                let index = self.variables.len() - 1;

                self.buffers.push(buffer);

                self.operations.push(Operation::BufferWrite {
                    variable_index: index,
                    buffer_index: self.buffers.len()-1
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

    pub fn add_fun(&mut self, function: Function<'a,V>) {
        let f_label = stringify!(function);
        let f_var = function.variables;
        let mut command_encoder = self.executor.create_encoder(Some(f_label));
        let mut compute_pass = command_encoder.begin_compute_pass(&wgpu::ComputePassDescriptor { label:Some(f_label) });

        let mut new_vars = Vec::new();
        let mut new_binds = Vec::new();
        let mut new_vars_count = 0;



        for var in f_var {
            if let Some(pos) = self.variables.iter().position(|sto_var| {
                Arc::ptr_eq(&sto_var.variable, &var.variable)
            }) {
                new_binds.push([pos, var.bind_group as usize]);
            } else {
                new_vars. push(Arc::clone(&var.variable));
                new_binds.push([self.variables.len()+new_vars_count,var.bind_group as usize]);
                new_vars_count +=1;
            }
        }

            for (sto_var,[var_pos,var_bind]) in new_vars.iter().zip(&new_binds) {
                let var = Arc::clone(&sto_var);
                let var_lock = var.lock().unwrap();
                let buffer_descriptor = var_lock.to_buffer_descriptor();

                let buffer = self.executor.get_buffer(&buffer_descriptor);
                
                self.variables.push(StoredVariable { 
                        variable: Arc::clone(&var), 
                        binds: vec![*var_bind], 
                        buffer_index: self.buffers.len()-1
                     });
                self.executor.write_buffer(&buffer, var_lock.byte_data());
                self.buffers.push(buffer);
            }
            
            let mut operation_bind_layout_entries = Vec::new();
            let mut operation_bind_entries = Vec::new();

            for [var_pos,bind_group] in new_binds{
                let mut sto_var = &mut self.variables[var_pos];
                if bind_group == *sto_var.binds.last().unwrap() {
                    continue;
                }
                operation_bind_layout_entries.push(sto_var.get_bind_group_layout_entry(bind_group as u32));
                operation_bind_entries.push(wgpu::BindGroupEntry 
                    { 
                        binding: bind_group as u32, 
                        resource: self.buffers[sto_var.buffer_index].as_entire_binding() 
                    });
                
            }

            let bind_layout_descriptor = wgpu::BindGroupLayoutDescriptor {
                label: Some(f_label),
                entries: &operation_bind_layout_entries,
            };
            let bind_layout = self.executor.get_bind_group_layout(&bind_layout_descriptor);
            
            let bind_group_desriptor = wgpu::BindGroupDescriptor {
                label: Some(f_label),
                layout: &bind_layout,
                entries: &operation_bind_entries,
            };
            let bind_group = self.executor.get_bind_group(&bind_group_desriptor);
            
            let mut command_encoder = self.executor.create_encoder(Some(f_label));

            {
            let mut compute_pass = command_encoder.begin_compute_pass(&wgpu::ComputePassDescriptor { label:Some(f_label) });
        
            compute_pass.set_bind_group(0, &bind_group, &[]);
            }
             // are we sure about that? maybe we can just pass command encoder and compute 
             // pass in the Operation?
            command_encoder.finish(); 


        }
    


    pub fn get_operations(&self)-> Vec<Operation> {
        self.operations.clone()
    }


    /// 
    pub async fn finish(&mut self) -> Result<(), anyhow::Error> {
        // self.optimize();


        let mut workgroups = [0 as u32; 3];

        let mut operation_bind_layout_entries: Vec<wgpu::BindGroupLayoutEntry> = Vec::new();
        let mut operation_bind_entries_pointer = Vec::new();

        for operation in &self.operations {
            match operation {
                Operation::Bind {
                    bind_index,
                    bind_groups,
                } => {
                    operation_bind_layout_entries = Vec::new();
                    operation_bind_entries_pointer = Vec::new();
                    for (index, group) in bind_index.iter().zip(bind_groups) {
                        let sto_var = &self.variables[*index];
                        operation_bind_layout_entries
                            .push(sto_var.get_bind_group_layout_entry(*group));

                        operation_bind_entries_pointer.push([*group,sto_var.buffer_index as u32]);
                        //     binding: *group,
                        //     resource: buffer.as_entire_binding(),
                        // });
                        workgroups = self.variables[*index]
                            .variable
                            .lock()
                            .unwrap()
                            .get_workgroup()?;
                    }
                }
                Operation::BufferWrite { variable_index, buffer_index } => {
                    let buffer = &self.buffers[*buffer_index];
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
                    
                    let mut buffers = Vec::new();
                    for [_, buffer_index] in &operation_bind_entries_pointer {

                        buffers.push(&self.buffers[*buffer_index as usize])
                    }

                    let mut operation_bind_entries = Vec::new();

                    for (id,buffer) in buffers.iter().enumerate() {
                        // let lock = buffer.lock().unwrap();
                        operation_bind_entries.push(wgpu::BindGroupEntry 
                            { 
                                binding: operation_bind_entries_pointer[id][0], 
                                resource: buffer.as_entire_binding() 
                            });
                    }

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
        match self
            .variables
            .iter()
            .position(|existing_var| Arc::ptr_eq(&existing_var.variable, var)) {
                None => {
                return Err(anyhow!(
                    "Variable {:?} not found in {:?} Algorithm",
                    var.lock().unwrap().get_name(),
                    self.label
                ));
            },
            Some(index) => {
                let buffer_index = self.variables[index].buffer_index;
                let buffer = &self.buffers[buffer_index];
                let output = self.executor.read_buffer(&buffer).await;
                // let slice = 
                let mut var_write = self.variables[index].variable.lock().unwrap();
                var_write.read_data(&output);
                return Ok(());
            }
        }
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
}


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



impl<V: Variable> StoredVariable<V> {

    /// Creates a [`wgpu::BindGroupLayoutEntry`] from [`self`]
    ///
    /// Useful to build the bind group layout for the executor to execute.
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
