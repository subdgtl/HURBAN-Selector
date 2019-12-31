use std::error;
use std::fmt;
use std::sync::Arc;

use crate::interpreter::{
    FloatParamRefinement, Func, FuncError, FuncFlags, FuncInfo, LogMessage, ParamInfo,
    ParamRefinement, Ty, Value,
};
use crate::mesh::tools;

#[derive(Debug, PartialEq)]
pub enum FuncWeldError {
    AllFacesDegenerate,
}

impl fmt::Display for FuncWeldError {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        match self {
            FuncWeldError::AllFacesDegenerate => {
                write!(f, "All faces remained degenerate after welding")
            }
        }
    }
}

impl error::Error for FuncWeldError {}

pub struct FuncWeld;

impl Func for FuncWeld {
    fn info(&self) -> &FuncInfo {
        &FuncInfo {
            name: "Weld",
            return_value_name: "Welded Mesh",
        }
    }

    fn flags(&self) -> FuncFlags {
        FuncFlags::empty()
    }

    fn param_info(&self) -> &[ParamInfo] {
        &[
            ParamInfo {
                name: "Mesh",
                refinement: ParamRefinement::Mesh,
                optional: false,
            },
            ParamInfo {
                name: "Tolerance",
                refinement: ParamRefinement::Float(FloatParamRefinement {
                    default_value: Some(0.001),
                    min_value: Some(0.0),
                    max_value: None,
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
        _log: &mut dyn FnMut(LogMessage),
    ) -> Result<Value, FuncError> {
        let mesh = args[0].unwrap_mesh();
        let tolerance = args[1].unwrap_float();

        if let Some(welded) = tools::weld(&mesh, tolerance) {
            Ok(Value::Mesh(Arc::new(welded)))
        } else {
            Err(FuncError::new(FuncWeldError::AllFacesDegenerate))
        }
    }
}
