use std::sync::Arc;

use nalgebra::{Point3, Rotation3, Vector3};

use crate::analytics;
use crate::interpreter::{
    BooleanParamRefinement, Float3ParamRefinement, Func, FuncError, FuncFlags, FuncInfo,
    LogMessage, ParamInfo, ParamRefinement, Ty, Value,
};
use crate::mesh::primitive;

pub struct FuncCreateBox;

impl Func for FuncCreateBox {
    fn info(&self) -> &FuncInfo {
        &FuncInfo {
            name: "Create Box",
            description: "\
Lorem Ipsum is simply dummy text of the printing and typesetting industry. \n\
\n\

Lorem Ipsum has been the industry's standard dummy text ever since the \
1500s, when an unknown printer took a galley of type and scrambled it to make \
a type specimen book. It has survived not only five centuries, but also the leap \
into electronic typesetting, remaining essentially unchanged. It was popularised \
in the 1960s with the release of Letraset sheets containing Lorem Ipsum \
passages, and more recently with desktop publishing software like Aldus \
PageMaker including versions of Lorem Ipsum.",
            return_value_name: "Box",
        }
    }

    fn flags(&self) -> FuncFlags {
        FuncFlags::PURE
    }

    fn param_info(&self) -> &[ParamInfo] {
        &[
            ParamInfo {
                name: "Center",
                description: "",
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
                description: "",
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
                description: "",
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
            ParamInfo {
                name: "Analyze resulting mesh",
                description: "",
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
        let scale = args[2].unwrap_float3();
        let analyze = args[3].unwrap_boolean();

        let value = primitive::create_box(
            Point3::from(center),
            Rotation3::from_euler_angles(
                rotate[0].to_radians(),
                rotate[1].to_radians(),
                rotate[2].to_radians(),
            ),
            Vector3::from(scale),
        );

        if analyze {
            analytics::report_mesh_analysis(&value, log);
        }

        Ok(Value::Mesh(Arc::new(value)))
    }
}
