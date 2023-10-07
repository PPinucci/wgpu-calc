/*!
This crate aim is to build an infrastructure to leverage the GPU for calculations

The project is based on the [`wgpu`] crate as base, and it builds on top of calc shader creations
a layer which makes hopefully easier to build and run calculations on the GPU using common structs which can be found in Crates.io
The first targets are the [`ndarray`] structs which can hold matrices of different dimensions.

There is also a small linear algebra set of operation for matrices built in the crate's shaders which can be useful for simple projects.
The aim is, anyway, fot the user to build its own calculation pipeline in a shader and use this crate to manage the I/O phase of the calculation.

Keeping in mind that this is an amateur project, it's trying to keep the calculations as efficient as possible, primarely focusing on avoiding
continuous data transfer between the host (the CPU machine) and the device (the GPU).
This is the primary reason why almost all the methods are "lazy" and never really perform the calculations before been
"executed". The aim is to make all the caclulations without waiting for the buffers to be read or write.
For this reason the intermediate poll of a result is possible, but not suggested if not for debugging purposes.

*/

#![allow(dead_code)]
#[cfg(target_arch = "wasm32")]
use wasm_bindgen::prelude::*;

pub mod algorithm;
pub mod coding;
pub mod errors;
pub mod interface;
pub mod solver;

#[cfg(test)]
mod tests;
