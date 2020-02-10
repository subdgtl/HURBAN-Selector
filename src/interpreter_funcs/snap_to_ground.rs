use std::sync::Arc;

use nalgebra::{Matrix4, Point3, Vector3};

use crate::analytics;
use crate::interpreter::{
    BooleanParamRefinement, Func, FuncError, FuncFlags, FuncInfo, LogMessage, ParamInfo,
    ParamRefinement, Ty, Value,
};
use crate::mesh::Mesh;

pub struct FuncSnapToGround;

impl Func for FuncSnapToGround {
    fn info(&self) -> &FuncInfo {
        &FuncInfo {
            name: "Snap To Ground",
            description: "",
            return_value_name: "Mesh on origin",
        }
    }

    fn flags(&self) -> FuncFlags {
        FuncFlags::PURE
    }

    fn param_info(&self) -> &[ParamInfo] {
        &[
            ParamInfo {
                name: "Mesh",
                description: "",
                refinement: ParamRefinement::Mesh,
                optional: false,
            },
            ParamInfo {
                name: "Move to origin",
                description: "",
                refinement: ParamRefinement::Boolean(BooleanParamRefinement {
                    default_value: true,
                }),
                optional: false,
            },
            ParamInfo {
                name: "Snap to ground",
                description: "",
                refinement: ParamRefinement::Boolean(BooleanParamRefinement {
                    default_value: true,
                }),
                optional: false,
            },
            ParamInfo {
                name: "Analyze resulting mesh",
                description: "Whether to output mesh analysis into console after running",
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
        let mesh = args[0].unwrap_refcounted_mesh();
        let move_to_origin = args[1].unwrap_boolean();
        let snap_to_ground = args[2].unwrap_boolean();
        let analyze = args[3].unwrap_boolean();

        let value = if move_to_origin || snap_to_ground {
            let bbox = mesh.bounding_box();

            let translation_vector = match (move_to_origin, snap_to_ground) {
                (true, true) => {
                    Point3::origin() - bbox.center()
                        + Vector3::new(0.0, 0.0, bbox.diagonal().z / 2.0)
                }
                (true, false) => Point3::origin() - bbox.center(),
                (false, true) => Vector3::new(0.0, 0.0, bbox.diagonal().z / 2.0 - bbox.center().z),
                _ => Vector3::zeros(),
            };

            let translation = Matrix4::new_translation(&translation_vector);

            let vertices_iter = mesh
                .vertices()
                .iter()
                .map(|v| translation.transform_point(v));

            let value = Mesh::from_faces_with_vertices_and_normals(
                mesh.faces().iter().copied(),
                vertices_iter,
                mesh.normals().iter().copied(),
            );

            Arc::new(value)
        } else {
            mesh
        };

        if analyze {
            analytics::report_mesh_analysis(&value, log);
        }
        Ok(Value::Mesh(value))
    }
}
