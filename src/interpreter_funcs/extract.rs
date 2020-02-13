use std::error;
use std::fmt;

use crate::analytics;
use crate::interpreter::{
    BooleanParamRefinement, Func, FuncError, FuncFlags, FuncInfo, LogMessage, ParamInfo,
    ParamRefinement, Ty, UintParamRefinement, Value,
};

#[derive(Debug, PartialEq)]
pub enum FuncExtractError {
    Empty,
    IndexOutOfBounds(u32, u32),
}

impl fmt::Display for FuncExtractError {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        match self {
            Self::Empty => write!(f, "No mesh geometry contained in group"),
            Self::IndexOutOfBounds(index, length) => write!(
                f,
                "Can't extract mesh on index {} from group of {} meshes",
                index, length
            ),
        }
    }
}

impl error::Error for FuncExtractError {}

pub struct FuncExtract;

impl Func for FuncExtract {
    fn info(&self) -> &FuncInfo {
        &FuncInfo {
            name: "Extract from Group",
            description: "EXTRACT MESH GEOMETRY FROM MESH GROUP\n\
                          \n\
                          Each mesh geometry in mesh group is given an index. \
                          This operation extracts mesh geometry with a specified \
                          index from a mesh group.\n\
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
                name: "Index",
                description: "Index of mesh to be extracted from the mesh group.",
                refinement: ParamRefinement::Uint(UintParamRefinement {
                    default_value: Some(0),
                    min_value: Some(0),
                    max_value: None,
                }),
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
        let index = args[1].unwrap_uint();
        let analyze = args[2].unwrap_boolean();

        if mesh_array.is_empty() {
            let error = FuncError::new(FuncExtractError::Empty);
            log(LogMessage::error(format!("Error: {}", error)));
            return Err(error);
        }

        let group_length = mesh_array.len();
        if index >= group_length {
            let error = FuncError::new(FuncExtractError::IndexOutOfBounds(index, group_length));
            log(LogMessage::error(format!("Error: {}", error)));
            Err(error)
        } else {
            let value = mesh_array
                .get_refcounted(index)
                .expect("Array must not be empty");

            if analyze {
                analytics::report_mesh_analysis(&value, log);
            }

            Ok(Value::Mesh(value))
        }
    }
}
