use std::error;
use std::f32;
use std::fmt;
use std::sync::Arc;

use nalgebra::{Point3, Vector3};

use crate::analytics;
use crate::bounding_box::BoundingBox;
use crate::interpreter::{
    BooleanParamRefinement, Float2ParamRefinement, Float3ParamRefinement, FloatParamRefinement,
    Func, FuncError, FuncFlags, FuncInfo, LogMessage, ParamInfo, ParamRefinement, Ty, Value,
};
use crate::mesh::voxel_cloud::ScalarField;

const VOXEL_COUNT_THRESHOLD: u32 = 50000;

#[derive(Debug, PartialEq)]
pub enum FuncVoxelNoiseError {
    WeldFailed,
    EmptyScalarField,
    VoxelDimensionsZero,
    TooManyVoxels(u32, f32, f32, f32),
}

impl fmt::Display for FuncVoxelNoiseError {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        match self {
            FuncVoxelNoiseError::WeldFailed => write!(
                f,
                "Welding of separate voxels failed due to high welding proximity tolerance"
            ),
            FuncVoxelNoiseError::EmptyScalarField => write!(f, "The resulting scalar field is empty"),
            FuncVoxelNoiseError::VoxelDimensionsZero => write!(f, "One or more voxel dimensions are zero"),
            FuncVoxelNoiseError::TooManyVoxels(max_count, x, y, z) => write!(
                f,
                "Too many voxels. Limit set to {}. Try setting voxel size to [{:.3}, {:.3}, {:.3}] or more.",
                max_count, x, y, z
            ),
        }
    }
}

impl error::Error for FuncVoxelNoiseError {}

pub struct FuncVoxelNoise;

impl Func for FuncVoxelNoise {
    fn info(&self) -> &FuncInfo {
        &FuncInfo {
            name: "Voxel Noise",
            description: "",
            return_value_name: "Noise Mesh",
        }
    }

    fn flags(&self) -> FuncFlags {
        FuncFlags::PURE
    }

    fn param_info(&self) -> &[ParamInfo] {
        &[
            ParamInfo {
                name: "Block Start",
                description: "",
                refinement: ParamRefinement::Float3(Float3ParamRefinement {
                    min_value: None,
                    max_value: None,
                    default_value_x: Some(-1.0),
                    default_value_y: Some(-1.0),
                    default_value_z: Some(-1.0),
                }),
                optional: false,
            },
            ParamInfo {
                name: "Block End",
                description: "",
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
                name: "Voxel Size",
                description: "Size of a single cell in the regular three-dimensional voxel grid.\n\
                \n\
                High values produce coarser results, low values may increase precision but produce \
                heavier geometry that significantly affects performance. Too high values produce \
                single large voxel, too low values may generate holes in the resulting geometry.",
                refinement: ParamRefinement::Float3(Float3ParamRefinement {
                    min_value: Some(f32::MIN_POSITIVE),
                    max_value: None,
                    default_value_x: Some(0.1),
                    default_value_y: Some(0.1),
                    default_value_z: Some(0.1),
                }),
                optional: false,
            },
            ParamInfo {
                name: "Noise Scale",
                description: "",
                refinement: ParamRefinement::Float(FloatParamRefinement {
                    default_value: Some(0.1),
                    min_value: Some(f32::MIN_POSITIVE),
                    max_value: None,
                }),
                optional: false,
            },
            ParamInfo {
                name: "Time offset",
                description: "",
                refinement: ParamRefinement::Float(FloatParamRefinement {
                    default_value: Some(1.0),
                    min_value: None,
                    max_value: None,
                }),
                optional: false,
            },
            ParamInfo {
                name: "Volume range",
                description: "",
                refinement: ParamRefinement::Float2(Float2ParamRefinement {
                    min_value: Some(-1.0),
                    max_value: Some(1.0),
                    default_value_x: Some(-0.25),
                    default_value_y: Some(0.25),
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
        let block_start = Point3::from(args[0].unwrap_float3());
        let block_end = Point3::from(args[1].unwrap_float3());
        let voxel_dimensions = args[2].unwrap_float3();
        let noise_scale = args[3].unwrap_float();
        let time_offset = args[4].unwrap_float();
        let volume_range_raw = args[5].unwrap_float2();
        let error_if_large = args[6].unwrap_boolean();
        let analyze_bbox = args[7].unwrap_boolean();
        let analyze_mesh = args[8].unwrap_boolean();

        let meshing_range = volume_range_raw[0]..=volume_range_raw[1];

        if voxel_dimensions
            .iter()
            .any(|dimension| approx::relative_eq!(*dimension, 0.0))
        {
            let error = FuncError::new(FuncVoxelNoiseError::VoxelDimensionsZero);
            log(LogMessage::error(format!("Error: {}", error)));
            return Err(error);
        }

        let bbox = BoundingBox::new(&block_start, &block_end);

        let bbox_diagonal = bbox.diagonal();
        let voxel_count = (bbox_diagonal.x / voxel_dimensions[0]).ceil() as u32
            * (bbox_diagonal.y / voxel_dimensions[1]).ceil() as u32
            * (bbox_diagonal.z / voxel_dimensions[2]).ceil() as u32;

        log(LogMessage::info(format!("Voxel count = {}", voxel_count)));

        if error_if_large && voxel_count > VOXEL_COUNT_THRESHOLD {
            let vy_over_vx = voxel_dimensions[1] / voxel_dimensions[0];
            let vz_over_vx = voxel_dimensions[2] / voxel_dimensions[0];
            let vx = ((bbox_diagonal.x * bbox_diagonal.y * bbox_diagonal.z)
                / (VOXEL_COUNT_THRESHOLD as f32 * vy_over_vx * vz_over_vx))
                .cbrt();
            let vy = vx * vy_over_vx;
            let vz = vx * vz_over_vx;

            // The equation doesn't take rounding into consideration, hence the
            // arbitrary multiplication by 1.1.
            let error = FuncError::new(FuncVoxelNoiseError::TooManyVoxels(
                VOXEL_COUNT_THRESHOLD,
                vx * 1.1,
                vy * 1.1,
                vz * 1.1,
            ));
            log(LogMessage::error(format!("Error: {}", error)));
            return Err(error);
        }

        let mut scalar_field: ScalarField =
            ScalarField::from_cartesian_bounding_box(&bbox, &Vector3::from(voxel_dimensions));

        scalar_field.fill_with_noise(noise_scale, time_offset);

        if !scalar_field.contains_voxels_within_range(&meshing_range) {
            let error = FuncError::new(FuncVoxelNoiseError::EmptyScalarField);
            log(LogMessage::error(format!("Error: {}", error)));
            return Err(error);
        }

        match scalar_field.to_mesh(&meshing_range) {
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
                let error = FuncError::new(FuncVoxelNoiseError::WeldFailed);
                log(LogMessage::error(format!("Error: {}", error)));
                Err(error)
            }
        }
    }
}
