use std::sync::Arc;

use crate::geometry;
use crate::interpreter::{
    Float3ParamRefinement, FloatParamRefinement, Func, FuncError, FuncFlags, FuncInfo, ParamInfo,
    ParamRefinement, Ty, Value,
};

pub struct FuncCreatePlane;

impl Func for FuncCreatePlane {
    fn info(&self) -> &FuncInfo {
        &FuncInfo {
            name: "Create Plane",
            return_value_name: "Plane",
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
                optional: true,
            },
            ParamInfo {
                name: "Scale",
                refinement: ParamRefinement::Float(FloatParamRefinement {
                    default_value: Some(1.0),
                    min_value: Some(0.0),
                    max_value: None,
                }),
                optional: true,
            },
        ]
    }

    fn return_ty(&self) -> Ty {
        Ty::Geometry
    }

    fn call(&mut self, values: &[Value]) -> Result<Value, FuncError> {
        let position = values[0].get_float3().unwrap_or([0.0; 3]);
        let scale = values[1].get_float().unwrap_or(1.0);

        let value = geometry::create_plane(position, scale);
        Ok(Value::Geometry(Arc::new(value)))
    }
}
