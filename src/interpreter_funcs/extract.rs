use std::error;
use std::fmt;

use crate::interpreter::{
    analytics, BooleanParamRefinement, Func, FuncError, FuncFlags, FuncInfo, LogMessage, ParamInfo,
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
                refinement: ParamRefinement::MeshArray,
                optional: false,
            },
            ParamInfo {
                name: "Index",
                refinement: ParamRefinement::Uint(UintParamRefinement {
                    default_value: Some(0),
                    min_value: Some(0),
                    max_value: None,
                }),
                optional: false,
            },
            ParamInfo {
                name: "Analyze resulting mesh",
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
            return Err(FuncError::new(FuncExtractError::Empty));
        }

        let group_length = mesh_array.len();
        if index >= group_length {
            Err(FuncError::new(FuncExtractError::IndexOutOfBounds(
                index,
                group_length,
            )))
        } else {
            let value = mesh_array
                .get_refcounted(index)
                .expect("Array must not be empty");

            if analyze {
                analytics::report_mesh_analysis(&value)
                    .iter()
                    .for_each(|line| log(line.clone()));
            }

            Ok(Value::Mesh(value))
        }
    }
}
