use crate::errors::VariableError;
use core::fmt::Debug;
use wgpu::BufferDescriptor;

/// This trait is the entry point to make a Rust type GPU compatible
///
/// It's still in early stage, but it contains all that is needed to a [`Function`] or
/// an [`Algorithm`] to perform the needed operations on the GPU.
///
/// It has some default implementations, but most of the critical pieces need still to be manually implemented,
/// since they're heavily dependent from the associated type.
///  
/// Please refer to the principal example to see an implementation example.
pub trait Variable
where
    Self: PartialEq + Debug + Send,
{
    /// This gets a buffer descriptor from the [`Variable`] itself
    ///
    /// It is useful to create the buffer, the bind group layouts and the ipelines which will be executed
    /// on the GPU
    fn to_buffer_descriptor(&self) -> BufferDescriptor {
        let label = self.get_name();
        return BufferDescriptor {
            label,
            mapped_at_creation: false,
            size: self.byte_size(),
            usage: wgpu::BufferUsages::STORAGE
                | wgpu::BufferUsages::COPY_DST
                | wgpu::BufferUsages::COPY_SRC,
        };
    }

    /// Gets an optional name associated with the [`Variable`]
    ///
    /// It is useful to always give variables a name for debugging purposes.
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

    /// This is the opposite of [`Variable::byte_data`] to get the data back
    ///
    /// The stream of data comes from the GPU as a Vec of f32, which needs to be translated into
    /// the Variable
    /// The data is returned in the same way as it's written, so the same logic which is
    /// implemented on {`Variable::byte_data`} should be implemented here
    fn read_data(&mut self, slice: &[u8]);

    /// This method is needed to better distribute the workload for the [`Variable`] calculation
    ///
    /// It returns the size in number of byte for each dimension of the [`Variable`], with its
    /// primary intent being the usage with matrices of maximum 3 dimensions ( see [`Variable::chek_dimensions`])
    /// Each dimension will be associated with a workgroup id in the GPU allowing the parallel execution of the calculus
    fn dimension_sizes(&self) -> [u32; 3];

    /// This method defines the workgroup count for the object
    ///
    /// It takes the dimension of the object and counts how many groups are needed to calculate the
    /// variable in parallel.
    ///
    /// # Errors
    /// - if the variable size in one or more direction is over the limit imposed by WGLS standard limits.
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
