use std::error;
use std::f32;
use std::fmt;
use std::sync::Arc;

use nalgebra::{Point3, Vector3};
use noise::{NoiseFn, OpenSimplex};

use crate::analytics;
use crate::bounding_box::BoundingBox;
use crate::interpreter::{
    BooleanParamRefinement, Float2ParamRefinement, Float3ParamRefinement, FloatParamRefinement,
    Func, FuncError, FuncFlags, FuncInfo, LogMessage, ParamInfo, ParamRefinement, Ty, Value,
};
use crate::mesh::scalar_field::ScalarField;

#[derive(Debug, PartialEq)]
pub enum FuncVoxelNoiseError {
    WeldFailed,
    EmptyScalarField,
    VoxelDimensionsZero,
}

impl fmt::Display for FuncVoxelNoiseError {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        match self {
            FuncVoxelNoiseError::WeldFailed => write!(
                f,
                "Welding of separate voxels failed due to high welding proximity tolerance"
            ),
            FuncVoxelNoiseError::EmptyScalarField => {
                write!(f, "The resulting scalar field is empty")
            }
            FuncVoxelNoiseError::VoxelDimensionsZero => {
                write!(f, "One or more voxel dimensions are zero")
            }
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
                    default_value_x: Some(-1.0),
                    min_value_x: None,
                    max_value_x: None,
                    default_value_y: Some(-1.0),
                    min_value_y: None,
                    max_value_y: None,
                    default_value_z: Some(-1.0),
                    min_value_z: None,
                    max_value_z: None,
                }),
                optional: false,
            },
            ParamInfo {
                name: "Block End",
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
                name: "Voxel Size",
                description: "Size of a single cell in the regular three-dimensional voxel grid.\n\
                High values produce coarser results, low values may increase precision but produce \
                heavier geometry that significantly affect performance. Too high values produce \
                single large voxel, too low values may generate holes in the resulting geometry.",
                refinement: ParamRefinement::Float3(Float3ParamRefinement {
                    default_value_x: Some(0.1),
                    min_value_x: Some(f32::MIN_POSITIVE),
                    max_value_x: None,
                    default_value_y: Some(0.1),
                    min_value_y: Some(f32::MIN_POSITIVE),
                    max_value_y: None,
                    default_value_z: Some(0.1),
                    min_value_z: Some(f32::MIN_POSITIVE),
                    max_value_z: None,
                }),
                optional: false,
            },
            ParamInfo {
                name: "Noise Scale",
                description: "",
                refinement: ParamRefinement::Float(FloatParamRefinement {
                    default_value: Some(1.0),
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
                    min_value: Some(f32::MIN_POSITIVE),
                    max_value: None,
                }),
                optional: false,
            },
            ParamInfo {
                name: "Volume range",
                description: "",
                refinement: ParamRefinement::Float2(Float2ParamRefinement {
                    default_value_x: Some(-0.5),
                    min_value_x: Some(-1.0),
                    max_value_x: Some(1.0),
                    default_value_y: Some(0.5),
                    min_value_y: Some(-1.0),
                    max_value_y: Some(1.0),
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
        let block_start = Point3::from(args[0].unwrap_float3());
        let block_end = Point3::from(args[1].unwrap_float3());
        let voxel_dimensions = args[2].unwrap_float3();
        let noise_scale = args[3].unwrap_float() as f64;
        let time_offset = args[4].unwrap_float() as f64;
        let volume_range_raw = args[5].unwrap_float2();
        let analyze = args[6].unwrap_boolean();
        let meshing_range = (volume_range_raw[0] as f64)..=(volume_range_raw[1] as f64);

        if voxel_dimensions
            .iter()
            .any(|dimension| approx::relative_eq!(*dimension, 0.0))
        {
            let error = FuncError::new(FuncVoxelNoiseError::VoxelDimensionsZero);
            log(LogMessage::error(format!("Error: {}", error)));
            return Err(error);
        }

        let bbox = BoundingBox::new(&block_start, &block_end);

        let mut scalar_field: ScalarField<f64> =
            ScalarField::from_cartesian_bounding_box(&bbox, &Vector3::from(voxel_dimensions));

        let simplex = OpenSimplex::new();

        for z in scalar_field.absolute_range_z() {
            for y in scalar_field.absolute_range_y() {
                for x in scalar_field.absolute_range_x() {
                    let noise_value = simplex.get([
                        x as f64 * noise_scale,
                        y as f64 * noise_scale,
                        z as f64 * noise_scale,
                        time_offset,
                    ]);
                    scalar_field.set_value_at_absolute_voxel_coordinate(
                        &Point3::new(x, y, z),
                        Some(noise_value),
                    );
                }
            }
        }

        if !scalar_field.contains_voxels_within_range(&meshing_range) {
            let error = FuncError::new(FuncVoxelNoiseError::EmptyScalarField);
            log(LogMessage::error(format!("Error: {}", error)));
            return Err(error);
        }

        match scalar_field.to_mesh(&meshing_range) {
            Some(value) => {
                if analyze {
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
