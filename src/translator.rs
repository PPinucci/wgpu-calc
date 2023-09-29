//! This module contains all the structs which are responsible of having the GPU read common types
//!
//! Its responsibility is to translate certain common used type in a form which can be run in the GPU.
//! The final scope is for the user to build traits on top of this and extend the calculation library

#![allow(dead_code)]
use ndarray::Array2;
use std::borrow::Cow;
use wgpu;

/// This struct is responible to get to the GPU the data in the correct form and sequence.
///
/// It arranges the data so it's easier to comunicate it with the buffer.
///
/// Note: it helds references to the data, as as such the array  and the label must outlive this struct to be valid!
///
pub struct GpuArray2<'a> {
    buffer_descr: wgpu::BufferDescriptor<'a>,
    data: Cow<'a, [f32]>,
    nrows: usize,
    ncols: usize,
}

impl<'a> GpuArray2<'a> {
    /// This fucntion create an array from a borrowed [`ndarray::Array2`]
    ///
    /// It manupulates the content of the array so they're ready to be passed to a GpuBuffer.
    ///
    /// Note: it helds references to the data, and as such the array  and the label must live longer han this struct
    ///
    /// # Arguments:
    ///
    /// * -`array` - a reference to a [`ndarray::Array2`]. this data will be given to the GPU to execute the calculus
    /// * -`label` - an optional label used for debugging mostly. Any problem with the calculation pipeline will refer to this label in the panics messages
    ///
    /// # Example
    /// ```
    /// use ndarray::{array};
    /// use wgpu_matrix::translator::GpuArray2;
    ///
    /// let a = array![[0., 0., 0.], [1., 1., 1.], [2., 2., 2.]];
    /// let gpu_a = GpuArray2::from_ndarray(&a, Some("Matrix a"));
    /// ```
    ///
    pub fn from_ndarray(array: &'a Array2<f32>, label: Option<&'a str>) -> GpuArray2<'a> {
        let (ncols, nrows) = array.dim();
        let data = Cow::from(array.as_slice().unwrap());
        let buffer_descr = wgpu::BufferDescriptor {
            label: label,
            // contents: bytemuck::cast_slice&array.as_slice().unwrap()),
            usage: wgpu::BufferUsages::STORAGE
                | wgpu::BufferUsages::COPY_DST
                | wgpu::BufferUsages::COPY_SRC,
            size: (std::mem::size_of::<f32>() * data.len()) as u64,
            mapped_at_creation: false,
        };

        GpuArray2 {
            buffer_descr,
            data,
            nrows,
            ncols,
        }
    }

    /// This method returns the dimension of the array
    ///
    /// The dimension are in the same order as defined in [`ndarray`] crate.
    ///
    /// # Example
    /// ```
    /// use ndarray::{array};
    /// use wgpu_matrix::translator::GpuArray2;
    ///
    /// let a = array![[0., 0.], [1., 1.], [2., 2.]];
    /// let gpu_a = GpuArray2::from_ndarray(&a, Some("Matrix a"));
    /// let (ncols, nrows) = gpu_a.dim();
    /// assert_eq!((ncols,nrows), (3,2))
    /// ```
    pub fn dim(&self) -> (usize, usize) {
        (self.ncols, self.nrows)
    }

    pub fn extract_result(&self) -> Result<Array2<f32>, ()> {
        todo!()
    }
}

#[cfg(test)]
mod inputs_tests {
    // use super::*;
}
