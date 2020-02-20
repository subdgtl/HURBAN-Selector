use std::error;
use std::fmt;
use std::sync::Arc;

use nalgebra::{Point3, Rotation3, Vector3};

use crate::analytics;
use crate::interpreter::{
    BooleanParamRefinement, Float3ParamRefinement, Func, FuncError, FuncFlags, FuncInfo,
    LogMessage, ParamInfo, ParamRefinement, Ty, UintParamRefinement, Value,
};
use crate::mesh::{primitive, NormalStrategy};

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
            description: "CREATE UV MESH SPHERE\n\
                          \n\
                          Creates a new mesh sphere made of segments ordered into \
                          parallels and meridians, which intersect on poles. \
                          The default size of the sphere is 1x1x1 model units. \
                          A high number of parallels and meridians will produce \
                          smoother sphere but heavier geometry.\n\
                          \n\
                          The resulting mesh geometry will be named 'Sphere'.",
            return_value_name: "Sphere",
        }
    }

    fn flags(&self) -> FuncFlags {
        FuncFlags::PURE
    }

    fn param_info(&self) -> &[ParamInfo] {
        &[
            ParamInfo {
                name: "Center",
                description: "Center of the sphere in absolute model units.",
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
                description: "Rotation of the sphere in degrees.",
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
                description: "Scale of the sphere as a relative factor.\n\
                The original size of the sphere is 1x1x1 model units.",
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
                name: "Parallels",
                description: "The number of parallels must be greater than 2.\n\
                A high number of parallels and meridians will produce smoother but heavier geometry.",
                refinement: ParamRefinement::Uint(UintParamRefinement {
                    default_value: Some(8),
                    min_value: Some(Self::MIN_PARALLELS),
                    max_value: None,
                }),
                optional: false,
            },
            ParamInfo {
                name: "Meridians",
                description: "The number of meridians must be greater than 3.\n\
                A high number of parallels and meridians will produce smoother but heavier geometry.",
                refinement: ParamRefinement::Uint(UintParamRefinement {
                    default_value: Some(8),
                    min_value: Some(Self::MIN_MERIDIANS),
                    max_value: None,
                }),
                optional: false,
            },
            ParamInfo {
                name: "Smooth normals",
                description: "Sets the per-vertex mesh normals to be interpolated from \
                connected face normals. As a result, the rendered geometry will have \
                a smooth surface material even though the mesh itself may be coarse.\n\
                \n\
                When disabled, the geometry will be rendered as angular: each face will \
                appear flat, exposing edges as sharp creases.\n\
                \n\
                The normal smoothing strategy does not affect the geometry itself.",
                refinement: ParamRefinement::Boolean(BooleanParamRefinement {
                    default_value: true,
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
        let scale = args[2].unwrap_float3();
        let n_parallels = args[3].unwrap_uint();
        let n_meridians = args[4].unwrap_uint();
        let smooth = args[5].unwrap_boolean();
        let analyze = args[6].unwrap_boolean();

        if n_parallels < Self::MIN_PARALLELS {
            let error = FuncError::new(FuncCreateUvSphereError::TooFewParallels {
                parallels_provided: n_parallels,
            });
            log(LogMessage::error(format!("Error: {}", error)));
            return Err(error);
        }

        if n_meridians < Self::MIN_MERIDIANS {
            let error = FuncError::new(FuncCreateUvSphereError::TooFewMeridians {
                meridians_provided: n_meridians,
            });
            log(LogMessage::error(format!("Error: {}", error)));
            return Err(error);
        }

        let normal_strategy = if smooth {
            NormalStrategy::Smooth
        } else {
            NormalStrategy::Sharp
        };

        let value = primitive::create_uv_sphere(
            Point3::from(center),
            Rotation3::from_euler_angles(
                rotate[0].to_radians(),
                rotate[1].to_radians(),
                rotate[2].to_radians(),
            ),
            Vector3::from(scale),
            n_parallels,
            n_meridians,
            normal_strategy,
        );

        if analyze {
            analytics::report_mesh_analysis(&value, log);
        }

        Ok(Value::Mesh(Arc::new(value)))
    }
}
