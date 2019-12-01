use std::error;
use std::fmt;
use std::sync::Arc;

use crate::geometry;
use crate::interpreter::{
    Float3ParamRefinement, FloatParamRefinement, Func, FuncError, FuncFlags, FuncInfo, ParamInfo,
    ParamRefinement, Ty, UintParamRefinement, Value,
};

#[derive(Debug, PartialEq)]
pub enum FuncCreateUvSphereError {
    TooFewParallels { parallels_provided: u32 },
    TooFewMeridians { meridians_provided: u32 },
}

impl fmt::Display for FuncCreateUvSphereError {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        match self {
            FuncCreateUvSphereError::TooFewParallels { parallels_provided } => write!(
                f,
                "Create UV Sphere requires at least 2 parallels, but only {} provided",
                parallels_provided,
            ),
            FuncCreateUvSphereError::TooFewMeridians { meridians_provided } => write!(
                f,
                "Create UV Sphere requires at least 3 meridians, but only {} provided",
                meridians_provided,
            ),
        }
    }
}

impl error::Error for FuncCreateUvSphereError {}

pub struct FuncCreateUvSphere;

impl FuncCreateUvSphere {
    const MIN_PARALLELS: u32 = 2;
    const MIN_MERIDIANS: u32 = 3;
}

impl Func for FuncCreateUvSphere {
    fn info(&self) -> &FuncInfo {
        &FuncInfo {
            name: "Create UV Sphere",
            return_value_name: "Sphere",
        }
    }

    fn flags(&self) -> FuncFlags {
        FuncFlags::PURE
    }

    fn param_info(&self) -> &[ParamInfo] {
        &[
            ParamInfo {
                name: "Position",
                refinement: ParamRefinement::Float3(Float3ParamRefinement {
                    default_value_x: Some(0.0),
                    min_value_x: None,
                    max_value_x: None,
                    default_value_y: Some(0.0),
                    min_value_y: None,
                    max_value_y: None,
                    default_value_z: Some(0.0),
                    min_value_z: None,
                    max_value_z: None,
                }),
                optional: false,
            },
            ParamInfo {
                name: "Scale",
                refinement: ParamRefinement::Float(FloatParamRefinement {
                    default_value: Some(1.0),
                    min_value: Some(0.0),
                    max_value: None,
                }),
                optional: false,
            },
            ParamInfo {
                name: "Parallels",
                refinement: ParamRefinement::Uint(UintParamRefinement {
                    default_value: Some(Self::MIN_PARALLELS),
                    min_value: Some(Self::MIN_PARALLELS),
                    max_value: None,
                }),
                optional: false,
            },
            ParamInfo {
                name: "Meridians",
                refinement: ParamRefinement::Uint(UintParamRefinement {
                    default_value: Some(Self::MIN_MERIDIANS),
                    min_value: Some(Self::MIN_MERIDIANS),
                    max_value: None,
                }),
                optional: false,
            },
        ]
    }

    fn return_ty(&self) -> Ty {
        Ty::Geometry
    }

    fn call(&mut self, args: &[Value]) -> Result<Value, FuncError> {
        let position = args[0].unwrap_float3();
        let scale = args[1].unwrap_float();
        let n_parallels = args[2].unwrap_uint();
        let n_meridians = args[3].unwrap_uint();

        if n_parallels < Self::MIN_PARALLELS {
            return Err(FuncError::new(FuncCreateUvSphereError::TooFewParallels {
                parallels_provided: n_parallels,
            }));
        }

        if n_meridians < Self::MIN_MERIDIANS {
            return Err(FuncError::new(FuncCreateUvSphereError::TooFewMeridians {
                meridians_provided: n_meridians,
            }));
        }

        let value = geometry::uv_sphere(position, scale, n_parallels, n_meridians);
        Ok(Value::Geometry(Arc::new(value)))
    }
}
