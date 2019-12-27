use std::iter;
use std::sync::Arc;

use nalgebra::{Rotation3, Vector3};

use crate::interpreter::{
    Func, FuncError, FuncFlags, FuncInfo, ParamInfo, ParamRefinement, Ty, UintParamRefinement,
    Value,
};
use crate::mesh::analysis::BoundingBox;
use crate::mesh::{analysis, primitive, NormalStrategy};

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

        let bounding_box = BoundingBox::from_meshes(iter::once(mesh));
        let mut value = primitive::create_uv_sphere(
            bounding_box.center().coords.into(),
            Rotation3::identity(),
            Vector3::new(
                bounding_box.diagonal_length() / 2.0,
                bounding_box.diagonal_length() / 2.0,
                bounding_box.diagonal_length() / 2.0,
            ),
            sphere_density,
            sphere_density,
            NormalStrategy::Sharp,
        );

        for vertex in value.vertices_mut() {
            if let Some(closest) = analysis::find_closest_point(vertex, mesh) {
                vertex.coords = closest.coords;
            }
        }

        Ok(Value::Mesh(Arc::new(value)))
    }
}
