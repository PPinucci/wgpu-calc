//!This module contains the second layer from the Gpu interface.
//!
//! Its utility is to create an ['Algorithm`] (a series of [`Operation']) to be executed on the gpu in series.
//!
use crate::errors::OperationError;
use crate::interface::Executor;
use std::error::Error;

pub struct Solver<'a> {
    executor: Executor<'a>,
    operations: Option<Vec<Operation<'a>>>,
    label: Option<&'a str>,
    // add compute_pass and command_encoders?
}

impl<'a> Solver<'a> {
    /// This function creates a new computation struct
    ///
    /// When builted the struct contains the [`Executor`], but an empty set of operations.
    /// The operations can than be added with the [`Executor::add_operation()`] method.
    pub async fn new(label: Option<&str>) -> Result<Solver<'_>, Box<dyn Error>> {
        let executor = Executor::new(label).await?;
        Ok(Solver {
            label,
            executor,
            operations: None,
        })
    }

    pub fn add_operation(&mut self, operation: Operation<'a>) -> Result<(), Box<dyn Error>> {
        // let mut existing_operations = self.operations.take();
        if let Some(thing) = self.operations.as_mut() {
            thing.push(operation)
        } else {
            match operation {
                Operation::Pipeline { .. } => {
                    return Err(Box::new(OperationError::BindingNotPresent(
                        self.label.unwrap().to_string(),
                    )));
                }
                _ => {
                    self.operations = Some(Vec::new());
                    self.add_operation(operation)?;
                }
            }
        }
        Ok(())
    }
}

pub enum Operation<'a> {
    /// This [`Operation`] binds a bind group, initiates the pipeline and dispatches the calculation
    BindAndPipeline {
        bind_group: wgpu::BindGroup,
        pipeline: wgpu::ComputePipeline,
        workgroup: [u32; 3],
    },
    /// This [`Operation`]
    Pipeline {
        pipeline: wgpu::ComputePipeline,
        workgroup: [u32; 3],
    },
    Parallel {
        operations: Vec<Operation<'a>>,
    },
    BufferWrite {
        buffer: wgpu::Buffer,
        data: &'a [u8],
    },
}

impl<'a> Operation<'_> {
    pub(crate) fn from_bind_and_pipeline_descriptors(
        bind_desc: &wgpu::BindGroupDescriptor,
        pipeline_descriptor: &wgpu::ComputePipelineDescriptor,
        workgroup: [u32; 3],
        executor: &mut Executor,
    ) -> Self {
        let bind_group = executor.get_bind_group(bind_desc);
        let pipeline = executor.get_pipeline(pipeline_descriptor);
        Operation::BindAndPipeline {
            bind_group,
            pipeline,
            workgroup,
        }
    }

    pub(crate) fn add_to_compute_pass(
        &'a self,
        compute_pass: &'a mut wgpu::ComputePass<'a>,
    ) -> Result<(), Box<dyn Error>> {
        match self {
            Operation::BindAndPipeline {
                bind_group,
                pipeline,
                workgroup,
            } => {
                compute_pass.set_bind_group(0, bind_group, &[]);
                compute_pass.set_pipeline(pipeline);
                compute_pass.dispatch_workgroups(workgroup[0], workgroup[1], workgroup[2])
            }
            Operation::Pipeline {
                pipeline,
                workgroup,
            } => {
                compute_pass.set_pipeline(pipeline);
                compute_pass.dispatch_workgroups(workgroup[0], workgroup[1], workgroup[2])
            }
            Operation::Parallel { .. } => {
                return Err(Box::new(OperationError::ComputePassOnParallel))
            }
            Operation::BufferWrite { .. } => {
                return Err(Box::new(OperationError::ComputePassOnBuffer));
            }
        }
        todo!()
    }
}
