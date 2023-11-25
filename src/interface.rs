//! This module contains the code to interface with the GPU device.
//!
//! It's responsible of passing the calculation to the GPU in the correct format
//! using the [`wgpu`] crate and its functions.

#![allow(dead_code)]
use std::sync::{Arc, Mutex};

use crate::coding::Shader;
use anyhow::anyhow;
use wgpu::{util::DeviceExt, InstanceFlags};

/// Contains all the functions to interact with the GPU device in the machine.
///
/// It's responsible of creating the comunication with the GPU, binding the data to the buffers,
/// loading the shaders, filling the queue with the correct pipeline of commands and launching the calculations
/// in the device.
#[derive(Debug)]
pub struct Executor<'a> {
    adapter: wgpu::Adapter,
    device: wgpu::Device,
    queue: wgpu::Queue,
    label: Option<&'a str>,
}

impl Executor<'_> {
    /// This function creates sets up the connection with the GPU
    ///
    /// The struct is responsible than of the comunication with the GPU itself, both in termns of
    /// sending data back and forth, and in terms of creating all the pieces needed to perform the operation.
    ///
    /// It is contained inside a [`algorithm::Algorithm`] to perform operations on a set of [`Variable`], but could be
    /// used alone to perform some more basilar operations.
    ///
    /// # Example
    /// ```
    /// use wgpu_calc::interface::Executor;
    /// use pollster;
    ///
    /// let interface = pollster::block_on(Executor::new(Some("Label for debugging purposes"))).unwrap();
    /// ```
    /// # Arguments
    ///*- `label` - an optional label for debugging purposes
    ///
    /// # Panics
    /// - if no adapter is found (default settings, should be rare). Limits are furtherly restricted in case this is compiled for wasm32
    /// - if device don't match features and limits (default settings, should be very rare)
    pub async fn new(label: Option<&str>) -> Result<Executor<'_>, anyhow::Error> {
        if let Some(adapter) = Executor::find_adapter().await {
            let (device, queue) = adapter
                .request_device(
                    &wgpu::DeviceDescriptor {
                        features: wgpu::Features::empty(), // this can be set to various values https://docs.rs/wgpu/latest/wgpu/struct.Features.html
                        limits: if cfg!(target_arch = "wasm32") {
                            wgpu::Limits::downlevel_webgl2_defaults()
                        } else {
                            wgpu::Limits::default()
                        },
                        label,
                    },
                    None, // Trace path 'used for API call tracing', probably a sort of log
                )
                .await?;

            Ok(Executor {
                adapter,
                device,
                queue,
                label,
            })
        } else {
            return Err(anyhow!("No adapter found for this phisical device"));
        }
    }

    // This function finds the adapters and gives back an Option value. It's primary purpose is the use with [`GpuInterface::new`] function
    async fn find_adapter() -> Option<wgpu::Adapter> {
        let instance = wgpu::Instance::new(wgpu::InstanceDescriptor {
            backends: wgpu::Backends::all(), // this is to get all the possible backends
            dx12_shader_compiler: wgpu::Dx12Compiler::default(),
            flags: InstanceFlags::VALIDATION,
            gles_minor_version: wgpu::Gles3MinorVersion::Automatic,
        });

        let adapter = instance
            .request_adapter(
                // this asks between all the backends of the instance which is the one satisfying the requisites here under
                &wgpu::RequestAdapterOptions {
                    power_preference: wgpu::PowerPreference::HighPerformance, // this can be set to HighPerformance
                    compatible_surface: None, //this is to check the possibility of using the surface, not used as we want a compute shader
                    force_fallback_adapter: false, // this is incase we want to use a software back end instead of an hardware one
                },
            )
            .await?;
        return Some(adapter);
    }

    /// This function gets the bind gropu layout associated with the [`Executor`] device from a descriptor
    ///
    /// The bind layout will be associated with the device created with a new [`Executor`].
    ///
    /// # Example
    /// ```
    /// use wgpu_calc::interface::Executor;
    /// use pollster;
    ///   let executor = pollster::block_on(Executor::new(Some("Debug Label"))).unwrap();
    ///   let input_bind_group_layout_descriptor = &wgpu::BindGroupLayoutDescriptor {
    ///     label: Some("input bind group layout"),
    ///     entries: &[
    ///         wgpu::BindGroupLayoutEntry{
    ///             binding: 0,
    ///           visibility: wgpu::ShaderStages::COMPUTE,
    ///             ty: wgpu::BindingType::Buffer {
    ///                 ty: wgpu::BufferBindingType::Storage { read_only: false },
    ///                 has_dynamic_offset: false,
    ///                 min_binding_size: None
    ///             },
    ///             count: None,
    ///         },
    ///         wgpu::BindGroupLayoutEntry{
    ///             binding: 1,
    ///             visibility: wgpu::ShaderStages::COMPUTE,
    ///             ty: wgpu::BindingType::Buffer {
    ///                 ty: wgpu::BufferBindingType::Storage { read_only: true },
    ///                 has_dynamic_offset: false,
    ///                 min_binding_size: None
    ///             },
    ///             count: None,
    ///         }
    ///         ]
    ///     };

    /// let input_bind_layout = executor.get_bind_group_layout(input_bind_group_layout_descriptor);
    /// ```
    pub fn get_bind_group_layout(
        &self,
        layout_descriptor: &wgpu::BindGroupLayoutDescriptor,
    ) -> wgpu::BindGroupLayout {
        self.device.create_bind_group_layout(layout_descriptor)
    }

    /// This method gives back a bind group associated with the [`Executor`]
    ///
    /// It's useful to prepare the bind group descriptors and than call the bind group only when needed
    pub fn get_bind_group(
        &self,
        bind_group_descriptor: &wgpu::BindGroupDescriptor,
    ) -> wgpu::BindGroup {
        self.device.create_bind_group(bind_group_descriptor)
    }

    /// This methods gives a [`wgpu::Buffer`] from a [`wgpu::util::BufferInitDescriptor`] object
    ///
    /// This method instantiates the buffer on the GPU, thus is reatively costly in time
    pub fn get_buffer_init(
        &self,
        buffer_init_descriptor: &wgpu::util::BufferInitDescriptor,
    ) -> wgpu::Buffer {
        self.device.create_buffer_init(&buffer_init_descriptor)
    }

    /// This method gives a [`wgpu::Buffer`] from a [`wgpu::BufferDescriptor`]
    ///
    /// The buffer is not instantiated, nor written, which is useful if the buffer writing
    /// wants to be managd separately
    pub fn get_buffer(&self, buffer_descriptor: &wgpu::BufferDescriptor) -> wgpu::Buffer {
        self.device.create_buffer(&buffer_descriptor)
    }

    /// This method associates the [`Shader`] object to the executor, creating a module.
    ///
    /// At this stage the [`Shader`] must be valid WGSL code, otherwise it will cause the
    /// program to # panic
    pub fn get_shader_module(&self, shader: &Shader) -> wgpu::ShaderModule {
        self.device
            .create_shader_module(wgpu::ShaderModuleDescriptor {
                label: self.label,
                source: wgpu::ShaderSource::Wgsl(std::borrow::Cow::Borrowed(shader.get_content())),
            })
    }

    /// This method creates a pipeline layout associated with the [`Executor`] from a pipeline layout descriptor
    ///
    /// This can be useful to create a pipeline descriptor not associated with the [`Executor`] and create the pipeline
    /// layout only afterwards
    pub fn get_pipeline_layout(
        &self,
        pipeline_layout_descriptor: &wgpu::PipelineLayoutDescriptor,
    ) -> wgpu::PipelineLayout {
        self.device
            .create_pipeline_layout(pipeline_layout_descriptor)
    }

    // pub fn get_command_encoder(&self, label: Option<&str>) -> wgpu::CommandEncoder {
    //     self.device
    //         .create_command_encoder(&wgpu::CommandEncoderDescriptor { label })
    // }

    /// This method creates a [`wgpu::ComputePipeline`] from a pipeline descriptor
    ///
    /// This can be useful to manage the descriptor prior to the executor association
    pub fn get_pipeline(
        &self,
        pipeline_descriptor: &wgpu::ComputePipelineDescriptor,
    ) -> wgpu::ComputePipeline {
        self.device.create_compute_pipeline(pipeline_descriptor)
    }

    /// Gets a [`wgpu::CommandEncoder`] from the device associated with the [`Executor`]
    ///
    /// Takes an optional *`label` string for debugging purposes
    pub fn create_encoder(&self, label: Option<&str>) -> wgpu::CommandEncoder {
        self.device
            .create_command_encoder(&wgpu::CommandEncoderDescriptor { label })
    }

    /// This method adds a bind group and a pipeline to the [`Executor`] and calls the dispatch for the pipeline
    ///
    /// Note this is still not executing any opration, this only creates a command encoder, binds the [`wgpu::BindGroup`]
    /// and sets up the [`wgpu::ComputePipeline`].
    ///
    /// It returns the [`wgpu::CommandEncoder`] with all these operations already set.
    ///
    /// # Arguments
    /// *-`bind group` - the [`wgpu::BindGroup`] to bind
    /// *-`pipeline` - the [`wgpu::ComputePipeline`] to use for computation
    /// *-`worgroups` - the number of worgroups to use for dispatching the computation pipeline
    /// *-`label` - an optional string for debugging purposes
    pub fn dispatch_bind_and_pipeline(
        &mut self,
        bind_group: &wgpu::BindGroup,
        pipeline: &wgpu::ComputePipeline,
        workgroups: &[u32; 3],
        label: Option<&str>,
    ) -> wgpu::CommandEncoder {
        let mut encoder = self
            .device
            .create_command_encoder(&wgpu::CommandEncoderDescriptor { label });
        {
            let mut compute_pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: self.label,
                timestamp_writes: None,
            });
            compute_pass.set_bind_group(0, bind_group, &[]);
            compute_pass.set_pipeline(pipeline);
            compute_pass.dispatch_workgroups(workgroups[0], workgroups[1], workgroups[2]);
        }
        return encoder;
    }

    /// Dispatches a [`wgpu::ComputePipeline`] without setting the bindings.
    ///
    /// This should be used only when the bind group doesn't change between different dispatches.
    ///
    /// # Arguments
    /// *-`pipeline` - the [`wgpu::ComputePipeline`] to use for computation
    /// *-`worgroups` - the number of worgroups to use for dispatching the computation pipeline
    /// *-`label` - an optional string for debugging purposes
    pub fn dispatch_pipeline(
        &mut self,
        pipeline: &wgpu::ComputePipeline,
        workgroups: &[u32; 3],
        label: Option<&str>,
    ) -> wgpu::CommandEncoder {
        let mut encoder = self
            .device
            .create_command_encoder(&wgpu::CommandEncoderDescriptor { label });
        {
            let mut compute_pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label,
                timestamp_writes: None,
            });
            compute_pass.set_pipeline(pipeline);
            compute_pass.dispatch_workgroups(workgroups[0], workgroups[1], workgroups[2]);
        }
        return encoder;
    }

    /// Uses the queue associated to the [`Executor`] to write a [`wgpu::Buffer`] to the GPU
    pub fn write_buffer(&self, buffer: &wgpu::Buffer, data: &[u8]) {
        self.queue.write_buffer(buffer, 0, data);
    }

    /// Takes an Iterator of [`wgpu::CommandBuffer`] and submits the jobs to the
    /// queue of the [`Executor`]
    ///
    /// Note that all the [`wgpu::CommandBuffer`] in the [`Iterator`] will be executed in parallel
    /// in the GPU
    pub fn execute<I: IntoIterator<Item = wgpu::CommandBuffer>>(
        &mut self,
        command_buffers: I,
    ) -> wgpu::SubmissionIndex {
        self.queue.submit(command_buffers)
    }

    /// Reads a [`wgpu::Buffer`] back from the GPU to the CPU
    ///
    /// To do such it creates a staging buffer before writing back to the CPU.
    /// This allows the comunication to the CPU to happen in parallel with other GPU operations,
    /// but still need to copy the buffer from GPU to GPU before, blocking any other operation during the porcess.
    pub async fn read_buffer(&self, buffer: &wgpu::Buffer) -> Vec<u8> {
        let mut command_encoder =
            self.device
                .create_command_encoder(&wgpu::CommandEncoderDescriptor {
                    label: Some("copying command encoder"),
                });

        let staging_buffer = self.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("Staging Buffer"),
            usage: wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
            size: buffer.size(),
        });

        command_encoder.copy_buffer_to_buffer(buffer, 0, &staging_buffer, 0, staging_buffer.size());

        self.queue.submit(std::iter::once(command_encoder.finish()));

        let (sender, receiver) = futures_channel::oneshot::channel();
        staging_buffer
            .slice(..)
            .map_async(wgpu::MapMode::Read, |result| {
                let _ = sender.send(result);
            });
        self.device.poll(wgpu::Maintain::Wait); // TODO: poll in the background instead of blocking
        receiver
            .await
            .expect("communication failed")
            .expect("buffer reading failed");
        let slice: &[u8] = &staging_buffer.slice(..).get_mapped_range();
        return slice.to_owned();
    }

    pub async fn read_buffer_thread_safe(&self, buffer: Arc<Mutex<wgpu::Buffer>>) -> Vec<u8> {
        let mut command_encoder =
            self.device
                .create_command_encoder(&wgpu::CommandEncoderDescriptor {
                    label: Some("copying command encoder"),
                });

        let staging_buffer = self.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("Staging Buffer"),
            usage: wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
            size: buffer.lock().unwrap().size(),
        });

        command_encoder.copy_buffer_to_buffer(
            &buffer.lock().unwrap(),
            0,
            &staging_buffer,
            0,
            staging_buffer.size(),
        );

        self.queue.submit(std::iter::once(command_encoder.finish()));

        let (sender, receiver) = futures_channel::oneshot::channel();
        staging_buffer
            .slice(..)
            .map_async(wgpu::MapMode::Read, |result| {
                let _ = sender.send(result);
            });
        self.device.poll(wgpu::Maintain::Wait); // TODO: poll in the background instead of blocking
        receiver
            .await
            .expect("communication to GPU buffer failed")
            .expect("buffer reading failed");
        let slice: &[u8] = &staging_buffer.slice(..).get_mapped_range();
        return slice.to_owned();
    }
}

#[cfg(test)]
mod interface_test {
    use super::*;
    #[tokio::test]
    async fn base_calc() {
        let label = Some("Test executor");

        let mut executor = Executor::new(label).await.unwrap();

        let array: [f32; 10000] = [1.0; 10000];

        let workgroups: [u32; 3] = [10000, 1, 1];

        let shader = Shader::from_file_path("./tests/shaders/example_shader.wgsl").unwrap();
        let shader_module = executor.get_shader_module(&shader);
        let entry_point = "add";

        let input_bind_group_layout_descriptor = wgpu::BindGroupLayoutDescriptor {
            label,
            entries: &[
                wgpu::BindGroupLayoutEntry {
                    binding: 0, // this is where we will bind the input in the shader
                    visibility: wgpu::ShaderStages::COMPUTE, // the type of function this will be visible in
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Storage { read_only: false }, // Uniform buffer are faster than storage, but smaller in max size.
                        has_dynamic_offset: false,
                        min_binding_size: None, // this can be some like buffer size for performance?
                    },
                    count: None,
                },
                wgpu::BindGroupLayoutEntry {
                    binding: 1, // this is where we will bind the ioutput in the shader
                    visibility: wgpu::ShaderStages::COMPUTE, // the type of function this will be visible in
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Storage { read_only: false }, // Uniform buffer are faster than storage, but smaller in max size.
                        has_dynamic_offset: false,
                        min_binding_size: None, // this can be some like buffer size for performance?
                    },
                    count: None,
                },
            ],
        };

        let input_bind_layout = executor.get_bind_group_layout(&input_bind_group_layout_descriptor);

        let array1_buffer_descriptor = wgpu::BufferDescriptor {
            label,
            usage: wgpu::BufferUsages::STORAGE
                | wgpu::BufferUsages::COPY_DST
                | wgpu::BufferUsages::COPY_SRC,
            // | wgpu::BufferUsages::MAP_READ,
            size: (std::mem::size_of::<f32>() * array.len()) as u64,
            mapped_at_creation: false,
        };

        let array2_buffer_descriptor = wgpu::BufferDescriptor {
            label,
            usage: wgpu::BufferUsages::STORAGE
                | wgpu::BufferUsages::COPY_DST
                | wgpu::BufferUsages::COPY_SRC, // uniform is better in performance than Storaage, but has less storage space
            size: (std::mem::size_of::<f32>() * array.len()) as u64,
            mapped_at_creation: false, // uniform is better in performance than Storaage, but has less storage space
        };

        let array1_buffer = executor.get_buffer(&array1_buffer_descriptor);
        let array2_buffer = executor.get_buffer(&array2_buffer_descriptor);

        let bind_group_descriptor = wgpu::BindGroupDescriptor {
            label,
            layout: &input_bind_layout,
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: array1_buffer.as_entire_binding(), // this create the bindig resource from the entire buffer
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: array2_buffer.as_entire_binding(), // this create the bindig resource from the entire buffer
                },
            ],
        };

        let bind_group = executor.get_bind_group(&bind_group_descriptor);

        let pipeline_layout_descriptor = wgpu::PipelineLayoutDescriptor {
            label,
            bind_group_layouts: &[&input_bind_layout],
            push_constant_ranges: &[],
        };

        let pipeline_layout = executor.get_pipeline_layout(&pipeline_layout_descriptor);

        let pipeline_descriptor = wgpu::ComputePipelineDescriptor {
            label,
            layout: Some(&pipeline_layout),
            module: &shader_module,
            entry_point,
        };

        let pipeline: wgpu::ComputePipeline = executor.get_pipeline(&pipeline_descriptor);

        executor.write_buffer(&array1_buffer, bytemuck::cast_slice(&array));
        executor.write_buffer(&array2_buffer, bytemuck::cast_slice(&array));

        let command_encoder =
            executor.dispatch_bind_and_pipeline(&bind_group, &pipeline, &workgroups, label);
        let command_buffer = [command_encoder.finish()];

        executor.execute(command_buffer.into_iter());

        let output = executor.read_buffer(&array1_buffer).await;

        assert_eq!(bytemuck::cast_slice::<u8, f32>(&output), &[2.0; 10000])
    }
}
