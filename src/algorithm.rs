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
//! - [`VariableBind`] is the link between a [`Variable`] and a `bind group` defined in the shader code.
//!     Each [`Variable`] can potentially have more than one bind in the [`Shader`] and the definition
//!     is held in this struct
//! - [`Algorithm`] is the operational part of this library, it collects instances of [`Function`]
//!     and tries to translate them as efficiently as possible to a series of [`Solver`].
//!     Once every function is inserted in the Algorithm, the [`Algorithm::run`] method is used
//!     to perform the calculation on the GPU
//!
//!
#![allow(dead_code)]
use anyhow::anyhow;
use std::fmt::Debug;
use std::num::NonZeroU64;
use std::sync::{Arc, Mutex};
use std::thread;

use crate::coding::Shader;
use crate::interface::Executor;
use crate::variable::Variable;

/// This struct is the container for the different operations to perform
///
/// It is used as a "container" for the [`Function`], which can be pushed inside it
/// to be performed on the GPU.
/// The functions added to the [`Algorithm`] will be performed in series preserving
/// the desired output.
/// Some optimisation will be done in the future prior to executing the operations.
///
/// The [`Algorithm`] takes also care of instantiating an [`Executor`] at creation, which is responsible for the
/// comunication with the GPU.
/// Ideally only one [`Algorithm`] should be created, and functions added to it sequentially to be executed a the correct time.
///
/// The struct is also reponsible of extracting the results of the calculation so that the data can be read back to the CPU at the end of the calculation.
#[derive(Debug)]
pub struct Algorithm<'a, V: Variable> {
    variables: Vec<StoredVariable<V>>,
    modules: Vec<Module<'a>>,
    buffers: Arc<Mutex<Vec<wgpu::Buffer>>>,
    // operations: Vec<Operation<'a>>,
    label: Option<&'a str>,
    executor: Arc<Mutex<Executor<'static>>>,
    solvers: Vec<Solver<V>>,
}

/// This struct is responsible of defining the operation to perform on the GPU
///
/// It combined the [`VariableBind`] with the defined [`Shader`] and his `entry point(s)` to
/// give the correct operation to execute on the GPU.
///
/// Notice that the [`Function`] is executed in the order on which it's added to the [`Algorithm`],
/// not in the order in which it is declared.
///
/// Multiple [`Function`]s can reference the same [`Shader`] and `entry point`, but one [`VariableBind`] must be
/// created for each of them
pub struct Function<'a, V: Variable> {
    shader: &'a Shader,
    entry_point: &'a str,
    variables: Vec<VariableBind<V>>,
}

/// Unit struct only for defining a [`VariableBind`] as mutable during the GPU calculations.
///
/// Currently all the [`VariableBind`] are created as mutable, until I become
/// smart enough to implement the immutable side of it, which could potentilly
/// make some more parallelisation possible
#[derive(Debug)]
pub struct Mutable;

/// Unit struct to define a [`VariableBind`] as immutable during the GPU calculations.
///
/// Currently it's impossible to create an immutable [`VariableBind`], but in the future it
/// might be possible
#[derive(Debug)]
pub struct Immutable;

/// This struct binds a [`Variable`] with a bind group in the shader
///
/// It holds an Arc<Mutex> to the [`Variable`] so that multiple binds can be created for the
/// same [`Variable`].
///
/// Currently all the [`VariableBind`] are [`Mutable`], i.e. they are trated like they will mutate during the
/// GPU operation.
#[derive(Debug)]
pub struct VariableBind<V, Type = Mutable>
where
    V: Variable,
{
    variable: Arc<Mutex<V>>,
    bind_group: u32,
    mutable: std::marker::PhantomData<Type>,
}

// holds the buffer references of the variable
#[derive(Debug)]
struct StoredVariable<V>
where
    V: Variable,
{
    variable: Arc<Mutex<V>>,
    binds: Vec<usize>,
    buffer_index: usize,
}

// holds the information of the inserted modules, shaders with different entry points
#[derive(Debug, PartialEq, Clone)]
struct Module<'a> {
    shader: &'a Shader,
    entry_point: Vec<&'a str>,
}

// Enum to deal in the future with the parallelisation of some [`Function`] execution
#[derive(Debug)]
enum Solver<V>
where
    V: Variable,
{
    Serial {
        command_encoder: wgpu::CommandEncoder,
        variables: Vec<Arc<Mutex<V>>>,
    },
    Parallel(Vec<Solver<V>>),

    ReadBuffer(usize),
}

impl<'a, V: Variable> Algorithm<'a, V> {
    /// Creates a new empty [`Algorithm`]
    ///
    /// Other than creating the struct, it also creates a new [`Executor`], which will be responsble of
    /// carrying out the operations.
    /// # Arguments
    ///* - `label` - an optional string reference to use for debugging purposes.
    ///
    /// Returns an [`anyhow::Error`] if the [`Executor`] fails to instantiate
    /// # Panics
    /// if the [`Executor`] initialisation
    pub async fn new(label: Option<&'static str>) -> Result<Algorithm<'a, V>, anyhow::Error> {
        let executor = Arc::new(Mutex::new(Executor::new(label).await?));
        Ok(Algorithm {
            variables: Vec::new(),
            modules: Vec::new(),
            buffers: Arc::new(Mutex::new(Vec::new())),
            solvers: Vec::new(),
            label,
            executor,
        })
    }

    /// This still needs implementations
    ///
    /// In the future will be responsible of optimizing the [`Algorithm`] in such a way that
    /// any operation which can be sent safely to the GPU in parallel (i.e. executing multiple parallel
    /// operations in parallel) will be done.
    pub fn optimize(&mut self) {
        todo!()
    }

    /// This method adds a [`Function`] to the [`Algorithm`], sheduling it for execution
    ///
    /// With this method the operation defined in the [`Function`] is added to the list of
    /// operations which will be carried on the GPU.
    /// The only action this takes is to write the [`Variable`] contained in the [`Function`] to
    /// the GPU buffer.
    ///
    /// Notice that buffer writing only takes place once for every builted [`Variable`], to avoid multiplication
    /// of this operation.
    ///
    /// Takes a mutable reference to `self`.
    ///
    /// # Arguments
    /// * - `function` - the [`Function`] to add to the [`Algorithm`]
    pub fn add_fun(&mut self, function: Function<'a, V>)
    where
        V: 'static,
    {
        let f_label = stringify!(function);
        let f_var = function.variables;
        let executor = self.executor.lock().unwrap();
        let mut command_encoder = executor.create_encoder(Some(f_label));
        // drop(executor);

        let variables: Vec<Arc<Mutex<V>>> =
            f_var.iter().map(|var| Arc::clone(&var.variable)).collect();

        let workgroups = variables[0].lock().unwrap().get_workgroup().unwrap();

        let mut new_vars = Vec::new();
        let mut new_binds = Vec::new();
        let mut new_vars_count = 0;

        for var in f_var {
            if let Some(pos) = self
                .variables
                .iter()
                .position(|sto_var| Arc::ptr_eq(&sto_var.variable, &var.variable))
            {
                new_binds.push([pos, var.bind_group as usize]);
            } else {
                new_vars.push(Arc::clone(&var.variable));
                new_binds.push([
                    self.variables.len() + new_vars_count,
                    var.bind_group as usize,
                ]);
                new_vars_count += 1;
            }
        }

        let mut var_buff_write = Vec::new();
        for (sto_var, [_, var_bind]) in new_vars.iter().zip(&new_binds) {
            // let exec = Arc::clone(&self.executor);
            let var = Arc::clone(&sto_var);

            let var_lock = var.lock().unwrap();
            let buffer_descriptor = var_lock.to_buffer_descriptor();
            //  drop(var_lock);
            let buffer = executor.get_buffer(&buffer_descriptor);

            self.variables.push(StoredVariable {
                variable: Arc::clone(&sto_var),
                binds: vec![*var_bind],
                buffer_index: self.buffers.lock().unwrap().len(),
            });
            var_buff_write.push(self.variables.len() - 1);
            self.buffers.lock().unwrap().push(buffer);
        }

        let mut operation_bind_layout_entries = Vec::new();
        let mut operation_bind_entries = Vec::new();
        let buffers = Arc::clone(&self.buffers);
        let buffer_lock = buffers.lock().unwrap();

        for [var_pos, bind_group] in new_binds {
            let sto_var = &mut self.variables[var_pos];
            operation_bind_layout_entries
                .push(sto_var.get_bind_group_layout_entry(bind_group as u32));
            // let buffer = &buffers[sto_var.buffer_index];

            operation_bind_entries.push(wgpu::BindGroupEntry {
                binding: bind_group as u32,
                resource: buffer_lock[sto_var.buffer_index].as_entire_binding(),
            });
        }

        let bind_layout_descriptor = wgpu::BindGroupLayoutDescriptor {
            label: Some(f_label),
            entries: &operation_bind_layout_entries,
        };
        let bind_layout = executor.get_bind_group_layout(&bind_layout_descriptor);

        let bind_group_desriptor = wgpu::BindGroupDescriptor {
            label: Some(f_label),
            layout: &bind_layout,
            entries: &operation_bind_entries,
        };
        let bind_group = executor.get_bind_group(&bind_group_desriptor);
        drop(buffer_lock);

        let module_pos;
        let entry_point_pos;

        if let Some(pos) = self
            .modules
            .iter()
            .position(|existing_module| existing_module.shader == function.shader)
        {
            module_pos = pos;
            if let Some(index) = self.modules[pos].find_entry_point(function.entry_point) {
                entry_point_pos = index;
            } else {
                self.modules[pos].add_entry_point(function.entry_point);
                entry_point_pos = self.modules[pos].entry_point.len() - 1;
            }
        } else {
            self.modules.push(Module {
                shader: function.shader,
                entry_point: vec![function.entry_point],
            });
            module_pos = self.modules.len() - 1;
            entry_point_pos = 0;
        }

        let shader = self.modules[module_pos].shader;
        let entry_point = self.modules[module_pos].entry_point[entry_point_pos];

        let pipeline_layout_descriptor = wgpu::PipelineLayoutDescriptor {
            label: Some(f_label),
            bind_group_layouts: &[&bind_layout],
            push_constant_ranges: &[],
        };

        let pipeline_layout = executor.get_pipeline_layout(&pipeline_layout_descriptor);

        let shader_module = executor.get_shader_module(shader);

        let pipeline_descriptor = wgpu::ComputePipelineDescriptor {
            label: Some(f_label),
            layout: Some(&pipeline_layout),
            module: &shader_module,
            entry_point,
        };
        let pipeline: wgpu::ComputePipeline = executor.get_pipeline(&pipeline_descriptor);
        {
            let mut compute_pass =
                command_encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                    label: Some(f_label),
                    timestamp_writes: None,
                });
            // if let Some(_) = bind_group {
            compute_pass.set_bind_group(0, &bind_group, &[]);
            // }

            compute_pass.set_pipeline(&pipeline);
            compute_pass.dispatch_workgroups(workgroups[0], workgroups[1], workgroups[2])
        }

        self.solvers.push(Solver::Serial {
            command_encoder,
            variables,
        });
        drop(executor);
        for index in var_buff_write {
            let buffers = Arc::clone(&self.buffers);
            let var2 = Arc::clone(&self.variables[index].variable);
            let exec = Arc::clone(&self.executor);
            let buff_index = self.variables[index].buffer_index;
            let var_lock = var2.lock().unwrap();
            let executor = exec.lock().unwrap();
            let bufflock = buffers.lock().unwrap();
            executor.write_buffer(&bufflock[buff_index], var_lock.byte_data());
            // thread::spawn(move || {
            // });
        }
    }

    /// This method executes the calculation defined in [`Algorithm`] on the GPU
    ///
    /// Notice this method consumes the list of operations sheduled during the [`Function`]s additions
    /// and performs all the calculations on the GPU as defined in the shaders on the [`Variable`]s bond to
    /// the bind groups as hey were defined in the [`Function`].
    ///
    /// This method doesn't perform any ouput operation, i.e. once the calculation have been run, you need to extract the
    /// [`Variable`] using the [`Algorithm::get_output_unmap`] method.
    /// This is done to assure that only the needed variables are brought back to the CPU memory, not spending any more time than needed on this
    /// operation.
    ///
    /// Takes a mutable reference to `self`
    pub async fn run(&mut self) -> Result<(), anyhow::Error> {
        for solver in &mut self.solvers.drain(0..) {
            match solver {
                Solver::Serial {
                    command_encoder, ..
                } => {
                    let mut executor = self.executor.lock().unwrap();
                    executor.execute([command_encoder.finish()].into_iter());
                }

                Solver::Parallel(solvers) => {
                    let mut buffers = Vec::new();
                    for serial in solvers {
                        match serial {
                            Solver::Serial {
                                command_encoder, ..
                            } => buffers.push(command_encoder.finish()),
                            _ => return Err(anyhow!("Cannot nest multiple parallel solvers!")),
                        }
                    }
                    let mut executor = self.executor.lock().unwrap();
                    executor.execute(buffers.into_iter());
                }

                Solver::ReadBuffer(index) => {
                    let buffer_index = self.variables[index].buffer_index;
                    let buffer = &self.buffers.lock().unwrap()[buffer_index];
                    let exec = Arc::clone(&self.executor);
                    
                    let mut var_write = self.variables[index].variable.lock().unwrap();
                    let exec_lock = exec.lock().unwrap();
                    
                    let result = exec_lock.read_buffer(buffer).await;
                    
                    var_write.read_data(&result);
                //     tokio::spawn(async move {
                // });
                }
            }
        }

        Ok(())
    }

    /// This method overwrite the [`Variable`] *`var` with the ouptut of the calculation
    ///
    /// reading from a GPU buffer is in general an expensive operation. This functions calls the
    /// correct method on the [`Executor`] to read the GPU buffer asycronously and with the least
    /// amount of effort possible.
    ///
    /// The function returns an error if the variable is not found in the [`Algorithm`] or
    pub fn read_variable(&mut self, var: &Arc<Mutex<V>>) -> Result<(), anyhow::Error> {
        match self
            .variables
            .iter()
            .position(|existing_var| Arc::ptr_eq(&existing_var.variable, var))
        {
            None => {
                return Err(anyhow!(
                    "Variable {:?} not found in {:?} Algorithm",
                    var.lock().unwrap().get_name(),
                    self.label
                ));
            }
            Some(index) => {
                self.solvers.push(Solver::ReadBuffer(index));
                // let buffer_index = self.variables[index].buffer_index;
                // let buffer = &self.buffers[buffer_index];
                // let output = self.executor.read_buffer(&buffer).await;
                // // let slice =
                // let mut var_write = self.variables[index].variable.lock().unwrap();
                // var_write.read_data(&output);
                return Ok(());
            }
        }
    }
}

impl<V> Function<'_, V>
where
    V: Variable,
{
    /// Creates a new function from a [`VariableBind`], a [`Shader`] and an `entry_point`
    ///
    /// Its primary purpose is organization of the code, bringing together the element which makes a GPU calculation possible.
    ///
    /// # Arguments
    /// * - `shader` - a reference to a [`Shader`] element, which contains the shader which will perform the operation
    /// * - `entry_point` - the name of the function inside the [`Shader`] which will execute the code
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
    /// This associated the defined [`Variable`] with the bind group number defined in the
    /// WGSL shader.
    /// This way the GPU will associate the correct bind group to the [`Variable`] and will perform
    /// the calculations on it.
    ///
    /// Notice that since the [`Variable`] is inside an Arc<Mutex>, it's not entirely cloned,
    /// only the pointer will be. This reduces the amount of oepration to perform, mantaining
    /// enough flexibility to reference the [`Variable`] from different points.
    ///
    /// # Arguments
    /// * - `variable` - an Arc<Mutex> of the variable which is used in a certain [`Function`]
    /// * - `bind_group` - the bind group number the variabe will be associated with in the WGSL shader
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

    // /// Sets the [`VariableBind`] to be immutable, thus read only
    // ///
    // /// It is not unsafe per se, but set as such to warn about the possible implications of this.
    // /// At the time of writing any variable can be set as read/write and set as immutable. This could potentially
    // /// cause concurrency problems when queueing the pipelines on tha GPU.
    // /// An immutable [`VariableBind`] is considered not to change during the calculation.
    // pub unsafe fn set_immutable(self) -> VariableBind<V, Immutable> {
    //     VariableBind {
    //         variable: self.variable,
    //         bind_group: self.bind_group,
    //         mutable: std::marker::PhantomData::<Immutable>,
    //     }
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
    ///
    /// # Arguments
    /// * - `variable` - a reference to the variable to bind
    /// * - `bind_group` - the bind group number the variabe will be associated with
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
    fn new(shader: &'a Shader) -> Self {
        Self {
            shader,
            entry_point: Vec::new(),
        }
    }

    fn add_entry_point(&mut self, e_p: &'a str) -> usize {
        self.entry_point.push(e_p);
        return self.entry_point.len() - 1;
    }

    fn find_entry_point(&self, e_p: &'a str) -> Option<usize> {
        self.entry_point.iter().position(|&entry| entry == e_p)
    }
}
