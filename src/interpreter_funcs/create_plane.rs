use std::sync::Arc;

use nalgebra::{Point3, Rotation3, Vector2, Vector3};

use crate::analytics;
use crate::interpreter::{
    BooleanParamRefinement, Float2ParamRefinement, Float3ParamRefinement, Func, FuncError,
    FuncFlags, FuncInfo, LogMessage, ParamInfo, ParamRefinement, Ty, Value,
};
use crate::mesh::primitive;
use crate::plane::Plane;

pub struct FuncCreatePlane;

impl Func for FuncCreatePlane {
    fn info(&self) -> &FuncInfo {
        &FuncInfo {
            name: "Create Plane",
            description: "CREATE MESH PLANE\n\
                          \n\
                          Creates a new mesh plane made of two welded triangles \
                          and four vertices. \
                          The default size of the plane is 1x1 model units.\n\
                          \n\
                          The resulting mesh geometry will be named 'Plane'.",
            return_value_name: "Plane",
        }
    }

    fn flags(&self) -> FuncFlags {
        FuncFlags::PURE
    }

    fn param_info(&self) -> &[ParamInfo] {
        &[
            ParamInfo {
                name: "Center",
                description: "Center of the plane in absolute model units.",
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
                description: "Rotation of the plane in degrees.",
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
                description: "Scale of the plane as a relative factor.\n\
                              The original size of the plane is 1x1 model units.",
                refinement: ParamRefinement::Float2(Float2ParamRefinement {
                    min_value: Some(0.0),
                    max_value: None,
                    default_value_x: Some(1.0),
                    default_value_y: Some(1.0),
                }),
                optional: false,
            },
            ParamInfo {
                name: "Analyze resulting mesh",
                description: "Reports detailed analytic information on the created mesh.\n\
                              The analysis may be slow, therefore it is by default off.",
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
        let center = args[0].unwrap_float3();
        let rotate = args[1].unwrap_float3();
        let scale = args[2].unwrap_float2();
        let analyze = args[3].unwrap_boolean();

        let rotation = Rotation3::from_euler_angles(
            rotate[0].to_radians(),
            rotate[1].to_radians(),
            rotate[2].to_radians(),
        );

        let plane = Plane::new(
            &Point3::from_slice(&center),
            &rotation.transform_vector(&Vector3::new(1.0, 0.0, 0.0)),
            &rotation.transform_vector(&Vector3::new(0.0, 1.0, 0.0)),
        );

        let value = primitive::create_mesh_plane(plane, Vector2::from(scale));

        if analyze {
            analytics::report_mesh_analysis(&value, log);
        }

        Ok(Value::Mesh(Arc::new(value)))
    }
}
