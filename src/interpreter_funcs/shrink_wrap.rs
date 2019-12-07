use std::iter;
use std::sync::Arc;

use crate::geometry;
use crate::interpreter::{
    Func, FuncError, FuncFlags, FuncInfo, ParamInfo, ParamRefinement, Ty, UintParamRefinement,
    Value,
};

pub struct FuncShrinkWrap;

impl Func for FuncShrinkWrap {
    fn info(&self) -> &FuncInfo {
        &FuncInfo {
            name: "Shrinkwrap",
            return_value_name: "Shrinkwrapped Mesh",
        }
    }

    fn flags(&self) -> FuncFlags {
        FuncFlags::PURE
    }

    fn param_info(&self) -> &[ParamInfo] {
        &[
            ParamInfo {
                name: "Mesh",
                refinement: ParamRefinement::Mesh,
                optional: false,
            },
            ParamInfo {
                name: "Density",
                refinement: ParamRefinement::Uint(UintParamRefinement {
                    default_value: Some(10),
                    min_value: Some(3),
                    max_value: None,
                }),
                optional: false,
            },
        ]
    }

    fn return_ty(&self) -> Ty {
        Ty::Mesh
    }

    fn call(&mut self, args: &[Value]) -> Result<Value, FuncError> {
        let mesh = args[0].unwrap_mesh();
        let sphere_density = args[1].unwrap_uint();

        let (center, radius) = geometry::compute_bounding_sphere(iter::once(mesh));
        let mut value = geometry::create_uv_sphere(
            center.coords.into(),
            radius,
            sphere_density,
            sphere_density,
        );

        for vertex in value.vertices_mut() {
            if let Some(closest) = geometry::find_closest_point(vertex, mesh) {
                vertex.coords = closest.coords;
            }
        }

        Ok(Value::Mesh(Arc::new(value)))
    }
}
