use std::{error::Error, path::Path};

/// The [`Shader`] is a struct containing WGSL code
/// 
/// This struct is not able to read or check WGSL code, it's only purpose is to 
/// hold the content of it, evantually manipulate it, and feeding it to a [`wgpu_calc::function::Function`]
/// 
/// All the code checking is done at compile time by `Naga` (called by `wgpu`), and never through this stage of the 
/// program
/// On one side this doesn't avoid any wrong code to be submitted to a [`Function`], which will be catch only at runtime,
/// but at the same time it allows to write pseudo code and to manipulate it at runtime.
/// This allows to pass veriable length [`Variable`]s to the GPU without using some still unsupported (at the time of writing)
/// WGSL features
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Shader {
    content: String,
}

impl Shader {
    /// This method creates a shader from a string literal.
    ///
    /// No effort whatsoever is done at this stage to check the correctnes of the shader, which is only checked at compile time (by Naga)
    ///
    /// # Arguments
    /// * - `content` - a string holding the code of the shader
    ///
    /// #Example
    /// ```
    /// use wgpu_calc::coding::Shader;
    /// // This shader takes two 3x3 matrices and adds them
    ///  let shader = Shader::from_content("
    ///     struct Mat2 {
    ///         elements: array<array<f32,3>,3>,
    ///         }
    ///
    ///     @group(0) @binding(0)
    ///     var<storage,read_write>  a: Mat2;
    ///     @group(0) @binding(1)
    ///     var<storage,read_write>  b: Mat2;
    ///
    ///     @compute @workgroup_size(3,3)
    ///     fn add (@builtin(global_invocation_id) id: vec3<u32>) {
    ///         a.elements[id.x][id.y] = a.elements[id.x][id.y] + b.elements[id.x][id.y];
    ///     }
    /// ");
    /// ```
    pub fn from_content(content: &str) -> Self {
        Shader {
            content: content.to_string(),
        }
    }

    /// This functions reads a --wgls-- file to the shader content.
    ///
    /// It will open the file and simply put the content into the struct as a [`String`].
    /// Returns a [`std::error::Error`] if file is not existent of not readable.
    ///
    /// No effort whatsoever is done at this stage to check the correctnes of the shader, which is only checked at compile time (by Naga)
    ///
    /// # Arguments
    /// * - `path_to_module` - a string slice holding the path to the module
    ///
    /// # Example
    ///
    /// ```
    /// use wgpu_calc::coding::Shader;
    /// let shader = Shader::from_file_path("../shaders/example_shader.wgsl");
    /// ```

    pub fn from_file_path(path_to_module: &str) -> Result<Self, Box<dyn Error>> {
        let path = Path::new(path_to_module);
        let content = std::fs::read_to_string(path)?;

        Ok(Shader { content })
    }

    /// This function replace the `from` sring with the `to` string inside the [`Shader`]
    ///
    /// It replaces all the instances of the `from` string, so use this with caution, since no check on correctness of the final code
    /// is done in this function.
    /// It's useful to create tokens inside a file which are not valid wgls code, but will become once tokens are replaced.
    /// As an example, since Naga at the time of coding doesn't support the pipeline overridable variables in the shader, it can be used to adapt the
    /// size of an array<array,n_rows>,n_cols> to the size of the input matrix, only known at run time.
    ///
    /// # Examples
    /// ```
    /// use wgpu_calc::coding::Shader;
    /// // Notice the €rows €cols tokens which will be overwritten
    /// let mut shader = Shader::from_content("
    ///     struct Mat2 {
    ///         elements: array<array<f32,€rows>,€cols>,
    ///         }
    ///
    ///     @group(0) @binding(0)
    ///     var<storage,read_write>  a: Mat2;
    ///     @group(0) @binding(1)
    ///     var<storage,read_write>  b: Mat2;
    ///
    ///     @compute @workgroup_size(3,3)
    ///     fn add (@builtin(global_invocation_id) id: vec3<u32>) {
    ///         a.elements[id.x][id.y] = a.elements[id.x][id.y] + b.elements[id.x][id.y];
    ///     }
    /// ");
    /// let ncols = 4;
    /// let nrows = 5;
    /// shader.replace("€cols",ncols.to_string().as_str());
    /// shader.replace("€rows",nrows.to_string().as_str());
    ///
    /// let check_shader = Shader::from_content("
    ///     struct Mat2 {
    ///         elements: array<array<f32,5>,4>,
    ///         }
    ///
    ///     @group(0) @binding(0)
    ///     var<storage,read_write>  a: Mat2;
    ///     @group(0) @binding(1)
    ///     var<storage,read_write>  b: Mat2;
    ///
    ///     @compute @workgroup_size(3,3)
    ///     fn add (@builtin(global_invocation_id) id: vec3<u32>) {
    ///         a.elements[id.x][id.y] = a.elements[id.x][id.y] + b.elements[id.x][id.y];
    ///     }
    /// ");
    /// assert_eq!(shader, check_shader)
    /// ```
    pub fn replace(&mut self, from: &str, to: &str) {
        self.content = self.content.replace(from, to);
    }

    /// This methods gets the content of the [`Shader`] as a string reference
    /// 
    /// It can be used for debugging, checking or to manipulate the wgls shader before
    /// inserting it in another [`Shader`] instance.
    pub fn get_content(&self) -> &str {
        &self.content
    }
}
