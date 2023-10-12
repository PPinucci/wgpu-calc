use std::{error::Error, path::Path};

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

    /// This function replace the `from` sring with the `to` string
    ///
    /// It replaces all the instances of the `from` string, so use this with caution, since no check on correctness of the final code
    /// is done in this function.
    /// It's useful to create tokens inside a file which are not valid wgls code, but will become once tokens are replaced.
    /// As an example, since Naga at the time of coding doesn't support the pipeline overridable variables in the shader, it can be used to adapt the
    /// size of an array<array,n_rows>,n_cols> to the size of the input matriz, only known atrun time.
    ///
    /// # Examples
    /// ```
    /// use wgpu_calc::coding::Shader;
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

    pub fn get_content(&self) -> &str {
        &self.content
    }
}
