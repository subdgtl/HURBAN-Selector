use std::error;
use std::fmt;

use crate::analytics;
use crate::interpreter::{
    BooleanParamRefinement, Func, FuncError, FuncFlags, FuncInfo, LogMessage, ParamInfo,
    ParamRefinement, Ty, Value,
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
            description: "EXTRACT THE LARGEST MESH GEOMETRY FROM MESH GROUP\n\
                          \n\
                          Extracts the mesh geometry with the largest number of faces \
                          (considered to be the main geometry in the group) from a mesh group.\n\
                          \n\
                          Mesh group is displayed in the viewport as geometry but is \
                          a distinct data type. Only some operations, such as this one, \
                          can use mesh groups and most of them are intended to generate \
                          a proper mesh from the mesh group.\n\
                          \n\
                          The resulting mesh geometry will be named 'Extracted Mesh'.",
            return_value_name: "Extracted Mesh",
        }
    }

    fn flags(&self) -> FuncFlags {
        FuncFlags::PURE
    }

    fn param_info(&self) -> &[ParamInfo] {
        &[
            ParamInfo {
                name: "Group",
                description: "Input mesh group.",
                refinement: ParamRefinement::MeshArray,
                optional: false,
            },
            ParamInfo {
                name: "Analyze resulting mesh",
                description: "Reports detailed analytic information on the created mesh.\n\
                              The analysis may be slow, therefore it is by default off.",
                refinement: ParamRefinement::Boolean(BooleanParamRefinement {
                    default_value: false,
                }),
                optional: false,
            },
        ]
    }

    fn return_ty(&self) -> Ty {
        Ty::Mesh
    }

    fn call(
        &mut self,
        args: &[Value],
        log: &mut dyn FnMut(LogMessage),
    ) -> Result<Value, FuncError> {
        let mesh_array = args[0].unwrap_mesh_array();
        let analyze = args[1].unwrap_boolean();

        if mesh_array.is_empty() {
            return Err(FuncError::new(FuncExtractLargestError::Empty));
        }

        let mut mesh_iter = mesh_array.iter_refcounted();
        let mut mesh = mesh_iter.next().expect("Array must not be empty");
        let mut largest_face_count = mesh.faces().len();

        for current_mesh in mesh_iter {
            let current_face_count = current_mesh.faces().len();
            if current_face_count > largest_face_count {
                largest_face_count = current_face_count;
                mesh = current_mesh;
            }
        }

        if analyze {
            analytics::report_mesh_analysis(&mesh, log);
        }

        Ok(Value::Mesh(mesh))
    }
}
