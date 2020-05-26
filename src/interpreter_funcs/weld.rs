use std::error;
use std::fmt;
use std::sync::Arc;

use crate::analytics;
use crate::interpreter::{
    BooleanParamRefinement, FloatParamRefinement, Func, FuncError, FuncFlags, FuncInfo, LogMessage,
    ParamInfo, ParamRefinement, Ty, Value,
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
            description: "WELD MESH VERTICES\n\
                          \n\
                          Replaces mesh vertices that are close to each other with \
                          a single common vertex. Faces will then share the common vertices, \
                          which results in watertight mesh if the original geometry was visually \
                          closed. In specific cases (large, mostly convex volumes with no kinks \
                          or folds and with proportionally small faces), \
                          weld can be used for mesh vertex and face reduction. \
                          Weld may result in invalid (non-manifold or collapsed) mesh in cases, \
                          when the welding tolerance is too high.\n\
                          \n\
                          The input mesh will be marked used and thus invisible in the viewport. \
                          It can still be used in subsequent operations.\n\
                          \n\
                          The resulting mesh geometry will be named 'Welded Mesh'.",
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
                description: "Input mesh.",
                refinement: ParamRefinement::Mesh,
                optional: false,
            },
            ParamInfo {
                name: "Tolerance",
                description: "Limit distance of two vertices to be welded into one.\n\
                     \n\
                     Weld may result in invalid (non-manifold or collapsed) mesh in cases, \
                     when the welding tolerance is too high.",
                refinement: ParamRefinement::Float(FloatParamRefinement {
                    default_value: Some(0.001),
                    min_value: Some(0.0),
                    max_value: None,
                }),
                optional: false,
            },
            ParamInfo {
                name: "Mesh Analysis",
                description: "Reports detailed analytic information on the created mesh.\n\
                     The analysis may be slow but it is crucial to check the validity of welding.",
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
        let mesh = args[0].unwrap_mesh();
        let tolerance = args[1].unwrap_float();
        let analyze_mesh = args[2].unwrap_boolean();

        if let Some(value) = tools::weld(&mesh, tolerance) {
            if analyze_mesh {
                analytics::report_bounding_box_analysis(&value, log);
                analytics::report_mesh_analysis(&value, log);
            }
            Ok(Value::Mesh(Arc::new(value)))
        } else {
            let error = FuncError::new(FuncWeldError::AllFacesDegenerate);
            log(LogMessage::error(format!("Error: {}", error)));
            Err(error)
        }
    }
}
