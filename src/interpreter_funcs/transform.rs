use std::sync::Arc;

use nalgebra::{Matrix4, Rotation, Vector3};

use crate::analytics;
use crate::interpreter::{
    BooleanParamRefinement, Float3ParamRefinement, Func, FuncError, FuncFlags, FuncInfo,
    LogMessage, ParamInfo, ParamRefinement, Ty, Value,
};
use crate::mesh::Mesh;

pub struct FuncTransform;

impl Func for FuncTransform {
    fn info(&self) -> &FuncInfo {
        &FuncInfo {
            name: "Transform",
            description: "TRANSFORM: MOVE, ROTATE, SCALE\n\
                          \n\
                          Moves, rotates and scales the mesh geometry, either around local \
                          object center or global origin.\n\
                          \n\
                          The input mesh will be marked used and thus invisible in the viewport. \
                          It can still be used in subsequent operations.\n\
                          \n\
                          The resulting mesh geometry will be named 'Transformed Mesh'.",
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
                description: "Input mesh.",
                refinement: ParamRefinement::Mesh,
                optional: false,
            },
            ParamInfo {
                name: "Move",
                description: "Translation (movement) in X, Y and Z direction.",
                refinement: ParamRefinement::Float3(Float3ParamRefinement {
                    min_value: None,
                    max_value: None,
                    default_value_x: Some(0.0),
                    default_value_y: Some(0.0),
                    default_value_z: Some(0.0),
                }),
                optional: false,
            },
            ParamInfo {
                name: "Rotate (deg)",
                description: "Rotation around the X, Y and Z axis in degrees.",
                refinement: ParamRefinement::Float3(Float3ParamRefinement {
                    min_value: None,
                    max_value: None,
                    default_value_x: Some(0.0),
                    default_value_y: Some(0.0),
                    default_value_z: Some(0.0),
                }),
                optional: false,
            },
            ParamInfo {
                name: "Scale",
                description: "Relative scaling factors for the world X, Y and Z axis.",
                refinement: ParamRefinement::Float3(Float3ParamRefinement {
                    min_value: None,
                    max_value: None,
                    default_value_x: Some(1.0),
                    default_value_y: Some(1.0),
                    default_value_z: Some(1.0),
                }),
                optional: false,
            },
            ParamInfo {
                name: "Transform around object center",
                description: "Transforms the mesh geometry around the object's center \
                              instead of global world origin.",
                refinement: ParamRefinement::Boolean(BooleanParamRefinement {
                    default_value: true,
                }),
                optional: false,
            },
            ParamInfo {
                name: "Mesh Analysis",
                description: "Reports detailed analytic information on the created mesh.\n\
                              The analysis may be slow, turn it on only when needed.",
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

        let translate = Vector3::from(args[1].unwrap_float3());
        let rotate = args[2].unwrap_float3();
        let scale = Vector3::from(args[3].unwrap_float3());
        let transform_around_local_center = args[4].unwrap_boolean();

        let analyze_mesh = args[5].unwrap_boolean();

        let user_rotation = Rotation::from_euler_angles(
            rotate[0].to_radians(),
            rotate[1].to_radians(),
            rotate[2].to_radians(),
        );
        let user_scaling = Matrix4::new_nonuniform_scaling(&scale);
        let user_translation = Matrix4::new_translation(&translate);

        let value = if transform_around_local_center {
            // Move to the origin, scale and rotate, then move back and finally
            // move according to the user translation.
            let b_box = mesh.bounding_box();
            let center = b_box.center();
            let vector_to_origin = Vector3::zeros() - center.coords;

            let translation_to_origin = Matrix4::new_translation(&vector_to_origin);
            let user_transformation = Matrix4::from(user_rotation) * user_scaling;
            let translation_from_origin = Matrix4::new_translation(&(-1.0 * vector_to_origin));
            let final_translation = translation_from_origin * user_translation;

            let vertices_iter = mesh.vertices().iter().map(|v| {
                let v1 = translation_to_origin.transform_point(v);
                let v2 = user_transformation.transform_point(&v1);
                final_translation.transform_point(&v2)
            });
            let normals_iter = mesh.normals().iter().map(|n| {
                let n1 = translation_to_origin.transform_vector(n);
                let n2 = user_transformation.transform_vector(&n1);
                final_translation.transform_vector(&n2)
            });

            Mesh::from_faces_with_vertices_and_normals(
                mesh.faces().iter().copied(),
                vertices_iter,
                normals_iter,
            )
        } else {
            let t = user_translation * Matrix4::from(user_rotation) * user_scaling;

            let vertices_iter = mesh.vertices().iter().map(|v| t.transform_point(v));
            let normals_iter = mesh.normals().iter().map(|n| t.transform_vector(n));

            Mesh::from_faces_with_vertices_and_normals(
                mesh.faces().iter().copied(),
                vertices_iter,
                normals_iter,
            )
        };

        if analyze_mesh {
            analytics::report_bounding_box_analysis(&value, log);
            analytics::report_mesh_analysis(&value, log);
        }

        Ok(Value::Mesh(Arc::new(value)))
    }
}
