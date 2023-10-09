//! This module contains all the specific error implementation for the crate
//!
//! Hopefully like this errors are easier to catch and manage from another crate

use std::fmt::Debug;
use thiserror::Error;

// type GpuResult<T> = Result<T, SizeError>;

/// These errors deals with the size of the operands of a function
///
/// They can be raised if the dimen
// #[derive(Debug, Error)]
// pub enum SizeError {
//     #[error("First operand has dimensions {:?}, second operand has dimension {:?}. The expected dimensions of the second operand are {:?}", [0], [1],[2])]
//     DimensionError((usize, usize), (usize, usize), (usize, usize)),
// }

#[derive(Debug, Error)]
pub enum OperationError {
    #[error("Operation field of {:?} is empty, please start with an operation which bind groups, create a pipeline and dispatch a worgroup before calling this method",[0])]
    BindingNotPresent(String),
    #[error("Can't add a parallel operation to a compute pipeline. A parallel operation needs to create a wgpu::CommandBuffer for every parallel operation to submits")]
    ComputePassOnParallel,
    #[error("Can't add a buffer write to a compute pipeline. the buffer writing needs to be called on the [`wgpu::Queue`] directly")]
    ComputePassOnBuffer,
}

#[derive(Debug, Error)]
pub enum VariableError<T: Debug> {
    #[error("Dimensions of the object {:?} is higher than 3, which is the max worksize group number",[0])]
    DimensionError(T),
    #[error("Variable has size in {:?} dimension which exceeds the max workgroup size. Please make sure you have more than one workgroup defined for this id",[0])]
    WorkgroupDimensionError(u32),
}
