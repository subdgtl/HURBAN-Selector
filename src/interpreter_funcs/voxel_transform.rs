use std::error;
use std::f32;
use std::fmt;
use std::ops::Bound;
use std::sync::Arc;

use nalgebra::{Rotation, Vector3};

use crate::analytics;
use crate::convert::clamp_cast_u32_to_i16;
use crate::interpreter::{
    BooleanParamRefinement, Float3ParamRefinement, Func, FuncError, FuncFlags, FuncInfo,
    LogMessage, ParamInfo, ParamRefinement, Ty, UintParamRefinement, Value,
};
use crate::mesh::scalar_field::{self, ScalarField};

const VOXEL_COUNT_THRESHOLD: u32 = 50000;

#[derive(Debug, PartialEq)]
pub enum FuncVoxelTransformError {
    WeldFailed,
    TransformFailed,
    VoxelDimensionZero,
    TooManyVoxels(u32, f32, f32, f32),
}

impl fmt::Display for FuncVoxelTransformError {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        match self {
            FuncVoxelTransformError::WeldFailed => write!(
                f,
                "Welding of separate voxels failed due to high welding proximity tolerance"
            ),
            FuncVoxelTransformError::TransformFailed =>write!(f, "Scalar field transformation failed"),
            FuncVoxelTransformError::VoxelDimensionZero => write!(f, "Voxel dimension is not larger than zero"),
            FuncVoxelTransformError::TooManyVoxels(max_count, x, y, z) => write!(
                f,
                "Too many voxels. Limit set to {}. Try setting voxel size to [{:.3}, {:.3}, {:.3}] or more.",
                max_count, x, y, z
            ),
        }
    }
}

impl error::Error for FuncVoxelTransformError {}

pub struct FuncVoxelTransform;

impl Func for FuncVoxelTransform {
    fn info(&self) -> &FuncInfo {
        &FuncInfo {
            name: "Voxel Transform",
            description: "VOXELIZE, THEN TRANSFORM: MOVE, ROTATE, SCALE\n\
                          \n\
                          Converts the input mesh geometry into voxel cloud, then \
                          moves, rotates and scales the voxel cloud around its local \
                          center and eventually materializes the resulting voxel cloud \
                          into a welded mesh.\n\
                          \n\
                          Voxels are three-dimensional pixels. They exist in a regular \
                          three-dimensional grid of arbitrary dimensions (voxel size). \
                          The voxel can be turned on (be a volume) or off (be a void). \
                          The voxels can be materialized as rectangular blocks. \
                          Voxelized meshes can be effectively smoothened by Laplacian relaxation.\n\
                          \n\
                          The input mesh will be marked used and thus invisible in the viewport. \
                          It can still be used in subsequent operations.\n\
                          \n\
                          The resulting mesh geometry will be named 'Voxel Transformed Mesh'.",
            return_value_name: "Voxel Transformed Mesh",
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
                name: "Voxel Dimensions",
                description: "Size of a single cell in the regular three-dimensional voxel grid.\n\
                \n\
                High values produce coarser results, low values may increase precision but produce \
                heavier geometry that significantly affect performance. Too high values produce \
                single large voxel, too low values may generate holes in the resulting geometry.",
                refinement: ParamRefinement::Float3(Float3ParamRefinement {
                    min_value: Some(0.005),
                    max_value: None,
                    default_value_x: Some(0.25),
                    default_value_y: Some(0.25),
                    default_value_z: Some(0.25),
                }),
                optional: false,
            },
            ParamInfo {
                name: "Grow",
                description: "The voxelization algorithm puts voxels on the surface of \
                the input mesh geometries.\n\
                \n\
                The grow option adds several extra layers of voxels on both sides of such \
                voxel volumes. This option generates thicker voxelized meshes. \
                In some cases not growing the volume at all may result in \
                a non manifold voxelized mesh.",
                refinement: ParamRefinement::Uint(UintParamRefinement {
                    default_value: Some(0),
                    min_value: None,
                    max_value: None,
                }),
                optional: false,
            },
            ParamInfo {
                name: "Fill closed volumes",
                description: "Treats the insides of watertight mesh geometries as volumes.\n\
                \n\
                If this option is off, the resulting voxelized mesh geometries will have two \
                separate mesh shells: one for outer surface, the other for inner surface of \
                hollow watertight mesh.",
                refinement: ParamRefinement::Boolean(BooleanParamRefinement {
                    default_value: true,
                }),
                optional: false,
            },
            ParamInfo {
                name: "Move",
                description: "Translation (movement) in X, Y and Z direction.",
                refinement: ParamRefinement::Float3(Float3ParamRefinement {
                    min_value: None,
                    max_value: None,
                    default_value_y: Some(0.0),
                    default_value_x: Some(0.0),
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
                name: "Prevent Unsafe Settings",
                description: "Stop computation and throw error if the calculation may be too slow.",
                refinement: ParamRefinement::Boolean(BooleanParamRefinement {
                    default_value: true,
                }),
                optional: false,
            },
            ParamInfo {
                name: "Bounding Box Analysis",
                description: "Reports basic and quick analytic information on the created mesh.",
                refinement: ParamRefinement::Boolean(BooleanParamRefinement {
                    default_value: true,
                }),
                optional: false,
            },
            ParamInfo {
                name: "Detailed Mesh Analysis",
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
        let mesh = args[0].unwrap_mesh();
        let voxel_dimensions = Vector3::from(args[1].unwrap_float3());
        let growth_u32 = args[2].unwrap_uint();
        let growth_i16 = clamp_cast_u32_to_i16(growth_u32);
        let fill = args[3].unwrap_boolean();
        let translate = Vector3::from(args[4].unwrap_float3());
        let rotate = args[5].unwrap_float3();
        let scale = args[6].unwrap_float3();
        let error_if_large = args[7].unwrap_boolean();
        let analyze_bbox = args[8].unwrap_boolean();
        let analyze_mesh = args[9].unwrap_boolean();

        if voxel_dimensions.iter().any(|dim| dim <= &0.0) {
            return {
                let error = FuncError::new(FuncVoxelTransformError::VoxelDimensionZero);
                log(LogMessage::error(format!("Error: {}", error)));
                Err(error)
            };
        }

        let bbox = mesh.bounding_box();
        let voxel_count = scalar_field::evaluate_voxel_count(&bbox, &voxel_dimensions);

        log(LogMessage::info(format!("Voxel count = {}", voxel_count)));

        if error_if_large && voxel_count > VOXEL_COUNT_THRESHOLD {
            let suggested_voxel_size =
                scalar_field::suggest_voxel_size_to_fit_bbox_within_voxel_count2(
                    voxel_count,
                    &voxel_dimensions,
                    VOXEL_COUNT_THRESHOLD,
                );

            let error = FuncError::new(FuncVoxelTransformError::TooManyVoxels(
                VOXEL_COUNT_THRESHOLD,
                suggested_voxel_size.x,
                suggested_voxel_size.y,
                suggested_voxel_size.z,
            ));
            log(LogMessage::error(format!("Error: {}", error)));
            return Err(error);
        }

        let mut scalar_field = ScalarField::from_mesh(mesh, &voxel_dimensions, 0_i16, growth_u32);

        scalar_field.compute_distance_filed(&(0..=0));

        let rotation = Rotation::from_euler_angles(
            rotate[0].to_radians(),
            rotate[1].to_radians(),
            rotate[2].to_radians(),
        );

        let scaling = Vector3::from(scale);

        if let Some(transformed_sf) = ScalarField::from_scalar_field_transformed(
            &scalar_field,
            &(0..=0),
            &voxel_dimensions,
            &translate,
            &rotation,
            &scaling,
        ) {
            let meshing_range = if fill {
                (Bound::Unbounded, Bound::Included(growth_i16))
            } else {
                (Bound::Included(-growth_i16), Bound::Included(growth_i16))
            };

            match transformed_sf.to_mesh(&meshing_range) {
                Some(value) => {
                    if analyze_bbox {
                        analytics::report_bounding_box_analysis(&value, log);
                    }
                    if analyze_mesh {
                        analytics::report_mesh_analysis(&value, log);
                    }
                    Ok(Value::Mesh(Arc::new(value)))
                }
                None => {
                    let error = FuncError::new(FuncVoxelTransformError::WeldFailed);
                    log(LogMessage::error(format!("Error: {}", error)));
                    Err(error)
                }
            }
        } else {
            {
                let error = FuncError::new(FuncVoxelTransformError::TransformFailed);
                log(LogMessage::error(format!("Error: {}", error)));
                Err(error)
            }
        }
    }
}
