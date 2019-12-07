use std::sync::Arc;

use nalgebra::base::{Matrix4, Vector3};
use nalgebra::geometry::{Rotation, Translation};

use crate::geometry::Mesh;
use crate::interpreter::{
    Float3ParamRefinement, Func, FuncError, FuncFlags, FuncInfo, ParamInfo, ParamRefinement, Ty,
    Value,
};

pub struct FuncTransform;

impl Func for FuncTransform {
    fn info(&self) -> &FuncInfo {
        &FuncInfo {
            name: "Transform",
            return_value_name: "Transformed Mesh",
        }
    }

    fn flags(&self) -> FuncFlags {
        FuncFlags::PURE
    }

    fn param_info(&self) -> &[ParamInfo] {
        &[
            ParamInfo {
                name: "Mesh",
                refinement: ParamRefinement::Geometry,
                optional: false,
            },
            ParamInfo {
                name: "Translate",
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
                name: "Rotate (deg)",
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
                refinement: ParamRefinement::Float3(Float3ParamRefinement {
                    default_value_x: Some(1.0),
                    min_value_x: None,
                    max_value_x: None,
                    default_value_y: Some(1.0),
                    min_value_y: None,
                    max_value_y: None,
                    default_value_z: Some(1.0),
                    min_value_z: None,
                    max_value_z: None,
                }),
                optional: true,
            },
        ]
    }

    fn return_ty(&self) -> Ty {
        Ty::Geometry
    }

    fn call(&mut self, args: &[Value]) -> Result<Value, FuncError> {
        let mesh = args[0].unwrap_mesh();

        let translate = args[1]
            .get_float3()
            .map(Vector3::from)
            .unwrap_or_else(Vector3::zeros);
        let rotate = args[2]
            .get_float3()
            .map(|rot| {
                [
                    rot[0].to_radians(),
                    rot[1].to_radians(),
                    rot[2].to_radians(),
                ]
            })
            .unwrap_or([0.0; 3]);
        let scale = args[3]
            .get_float3()
            .map(Vector3::from)
            .unwrap_or_else(|| Vector3::new(1.0, 1.0, 1.0));

        let translation = Translation::from(translate);
        let rotation = Rotation::from_euler_angles(rotate[0], rotate[1], rotate[2]);
        let scaling = Matrix4::new_nonuniform_scaling(&scale);

        let t = Matrix4::from(translation) * Matrix4::from(rotation) * scaling;
        let vertices_iter = mesh.vertices().iter().map(|v| t.transform_point(v));
        let normals_iter = mesh.normals().iter().map(|n| t.transform_vector(n));

        let value = Mesh::from_faces_with_vertices_and_normals(
            mesh.faces().iter().copied(),
            vertices_iter,
            normals_iter,
        );

        Ok(Value::Geometry(Arc::new(value)))
    }
}
