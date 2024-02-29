use std::error;
use std::f32;
use std::fmt;
use std::ops::Bound;
use std::sync::Arc;

use nalgebra::Vector3;

use crate::analytics;
use crate::interpreter::{
    BooleanParamRefinement, Float3ParamRefinement, Func, FuncError, FuncFlags, FuncInfo,
    LogMessage, ParamInfo, ParamRefinement, Ty, UintParamRefinement, Value,
};
use crate::mesh::voxel_cloud::{
    self, BinaryVoxelFunction, FalloffFunction, ScalarField, UnaryVoxelFunction,
};

const VOXEL_COUNT_THRESHOLD: u32 = 100_000;

#[derive(Debug, PartialEq)]
pub enum FuncVoxelizeError {
    WeldFailed,
    EmptyScalarField,
    VoxelDimensionsZeroOrLess,
    TooManyVoxels(u32, f32, f32, f32),
}

impl fmt::Display for FuncVoxelizeError {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        match self {
            FuncVoxelizeError::WeldFailed => write!(
                f,
                "Welding of separate voxels failed due to high welding proximity tolerance"
            ),
            FuncVoxelizeError::EmptyScalarField => write!(f, "The resulting scalar field is empty"),
            FuncVoxelizeError::VoxelDimensionsZeroOrLess => write!(f, "One or more voxel dimensions are zero or less"),
            FuncVoxelizeError::TooManyVoxels(max_count, x, y, z) => write!(
                f,
                "Too many voxels. Limit set to {}. Try setting voxel size to [{:.3}, {:.3}, {:.3}] or more.",
                max_count, x, y, z
            ),
        }
    }
}

impl error::Error for FuncVoxelizeError {}

pub struct FuncVoxelize;

impl Func for FuncVoxelize {
    fn info(&self) -> &FuncInfo {
        &FuncInfo {
            name: "Voxelize Mesh",
            description: "VOXELIZE MESH\n\
            \n\
            Converts the input mesh geometry into voxel cloud and \
            materializes the resulting voxel cloud into a welded mesh.\n\
            \n\
            Voxels are three-dimensional pixels. They exist in a regular three-dimensional \
            grid of arbitrary dimensions (voxel size). The voxel can be turned on \
            (be a volume) or off (be a void). The voxels can be materialized as \
            rectangular blocks. Voxelized meshes can be effectively smoothened by \
            Laplacian relaxation.
            \n\
            The input mesh will be marked used and thus invisible in the viewport. \
            It can still be used in subsequent operations.\n\
            \n\
            The resulting mesh geometry will be named 'Voxelized Mesh'.",
            return_value_name: "Voxelized Mesh",
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
                name: "Voxel Size",
                description: "Size of a single cell in the regular three-dimensional voxel grid.\n\
                \n\
                High values produce coarser results, low values may increase precision but produce \
                heavier geometry that significantly affects performance. Too high values produce \
                single large voxel, too low values may generate holes in the resulting geometry.",
                refinement: ParamRefinement::Float3(Float3ParamRefinement {
                    min_value: Some(0.005),
                    max_value: None,
                    default_value_x: Some(1.0),
                    default_value_y: Some(1.0),
                    default_value_z: Some(1.0),
                }),
                optional: false,
            },
            // FIXME: Consider changing to volume range for consistency.
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
                name: "Fill Closed Volumes",
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
                name: "Marching Cubes",
                description: "Smoother result.\n\
                \n\
                If checked, the result will be smoother, otherwise it will be blocky.",
                refinement: ParamRefinement::Boolean(BooleanParamRefinement {
                    default_value: true,
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
        let voxel_dimensions = Vector3::from(args[1].unwrap_float3());
        let growth_u32 = args[2].unwrap_uint();
        let growth_f32 = growth_u32 as f32;
        let fill = args[3].unwrap_boolean();
        let marching_cubes = args[4].unwrap_boolean();
        let error_if_large = args[5].unwrap_boolean();
        let analyze_mesh = args[6].unwrap_boolean();

        if voxel_dimensions.iter().any(|dimension| *dimension <= 0.0) {
            let error = FuncError::new(FuncVoxelizeError::VoxelDimensionsZeroOrLess);
            log(LogMessage::error(format!("Error: {}", error)));
            return Err(error);
        }

        let bbox = mesh.bounding_box();
        let voxel_count = voxel_cloud::evaluate_voxel_count(&bbox, &voxel_dimensions);

        log(LogMessage::info(format!("Voxel count = {}", voxel_count)));

        if error_if_large && voxel_count > VOXEL_COUNT_THRESHOLD {
            let suggested_voxel_size =
                voxel_cloud::suggest_voxel_size_to_fit_bbox_within_voxel_count(
                    voxel_count,
                    &voxel_dimensions,
                    VOXEL_COUNT_THRESHOLD,
                );

            let error = FuncError::new(FuncVoxelizeError::TooManyVoxels(
                VOXEL_COUNT_THRESHOLD,
                suggested_voxel_size.x,
                suggested_voxel_size.y,
                suggested_voxel_size.z,
            ));
            log(LogMessage::error(format!("Error: {}", error)));
            return Err(error);
        }

        let mut scalar_field = ScalarField::from_mesh(mesh, &voxel_dimensions, 0.0, growth_u32);

        let volume_value_range = 0.0..=0.0;

        scalar_field.compute_distance_field(&volume_value_range, 1.0, FalloffFunction::Linear);

        if fill {
            let mut scalar_field_inside =
                scalar_field.extract_voxels_inside_volumes(&volume_value_range);
            scalar_field_inside.apply_unary_voxel_function(UnaryVoxelFunction::SetConstant(0.0));
            scalar_field.apply_binary_voxel_function(
                &scalar_field_inside,
                BinaryVoxelFunction::ReplaceIfValue,
            );
        }

        let meshing_range = if fill {
            (Bound::Unbounded, Bound::Included(growth_f32))
        } else {
            (Bound::Included(-growth_f32), Bound::Included(growth_f32))
        };

        if !scalar_field.contains_voxels_within_range(&meshing_range) {
            let error = FuncError::new(FuncVoxelizeError::EmptyScalarField);
            log(LogMessage::error(format!("Error: {}", error)));
            return Err(error);
        }

        let meshing_output = if marching_cubes {
            scalar_field.to_marching_cubes(&meshing_range)
        } else {
            scalar_field.to_mesh(&meshing_range)
        };

        match meshing_output {
            Some(value) => {
                if analyze_mesh {
                    analytics::report_bounding_box_analysis(&value, log);
                    analytics::report_mesh_analysis(&value, log);
                }
                Ok(Value::Mesh(Arc::new(value)))
            }
            None => {
                let error = FuncError::new(FuncVoxelizeError::WeldFailed);
                log(LogMessage::error(format!("Error: {}", error)));
                Err(error)
            }
        }
    }
}
