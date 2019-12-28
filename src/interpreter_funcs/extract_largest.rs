use std::error;
use std::fmt;

use crate::interpreter::{
    Func, FuncError, FuncFlags, FuncInfo, ParamInfo, ParamRefinement, Ty, Value,
};

#[derive(Debug, PartialEq)]
pub enum FuncExtractLargestError {
    Empty,
}

impl fmt::Display for FuncExtractLargestError {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        match self {
            Self::Empty => write!(f, "No mesh geometry contained in group"),
        }
    }
}

impl error::Error for FuncExtractLargestError {}

pub struct FuncExtractLargest;

impl Func for FuncExtractLargest {
    fn info(&self) -> &FuncInfo {
        &FuncInfo {
            name: "Extract Largest",
            return_value_name: "Extracted Mesh",
        }
    }

    fn flags(&self) -> FuncFlags {
        FuncFlags::PURE
    }

    fn param_info(&self) -> &[ParamInfo] {
        &[ParamInfo {
            name: "Group",
            refinement: ParamRefinement::MeshArray,
            optional: false,
        }]
    }

    fn return_ty(&self) -> Ty {
        Ty::Mesh
    }

    fn call(&mut self, values: &[Value]) -> Result<Value, FuncError> {
        let mesh_array = values[0].unwrap_mesh_array();

        if mesh_array.is_empty() {
            return Err(FuncError::new(FuncExtractLargestError::Empty));
        }

        let mut largest_face_count = 0;
        let mut mesh = mesh_array
            .get_refcounted(0)
            .expect("Array must not be empty");

        if mesh_array.len() > 1 {
            for current_index in 1..mesh_array.len() {
                let current_mesh = mesh_array
                    .get_refcounted(current_index)
                    .expect("Array must not be empty");
                let current_face_count = current_mesh.faces().len();
                if current_face_count > largest_face_count {
                    largest_face_count = current_face_count;
                    mesh = current_mesh;
                }
            }
        }

        Ok(Value::Mesh(mesh))
    }
}
