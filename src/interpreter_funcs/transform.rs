use std::sync::Arc;

use nalgebra::{Matrix4, Rotation, Vector3};

use crate::interpreter::{
    Float3ParamRefinement, Func, FuncError, FuncFlags, FuncInfo, ParamInfo, ParamRefinement, Ty,
    Value,
};
use crate::mesh::Mesh;

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
                refinement: ParamRefinement::Mesh,
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
                optional: false,
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
                optional: false,
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
                optional: false,
            },
        ]
    }

    fn return_ty(&self) -> Ty {
        Ty::Mesh
    }

    fn call(&mut self, args: &[Value]) -> Result<Value, FuncError> {
        let mesh = args[0].unwrap_mesh();

        let translate = Vector3::from(args[1].unwrap_float3());
        let rotate = args[2].unwrap_float3();
        let scale = Vector3::from(args[3].unwrap_float3());

        let translation = Matrix4::new_translation(&translate);
        let rotation = Rotation::from_euler_angles(
            rotate[0].to_radians(),
            rotate[1].to_radians(),
            rotate[2].to_radians(),
        );
        let scaling = Matrix4::new_nonuniform_scaling(&scale);

        let t = translation * Matrix4::from(rotation) * scaling;

        let vertices_iter = mesh.vertices().iter().map(|v| t.transform_point(v));
        let normals_iter = mesh.normals().iter().map(|n| t.transform_vector(n));

        let value = Mesh::from_faces_with_vertices_and_normals(
            mesh.faces().iter().copied(),
            vertices_iter,
            normals_iter,
        );

        Ok(Value::Mesh(Arc::new(value)))
    }
}
