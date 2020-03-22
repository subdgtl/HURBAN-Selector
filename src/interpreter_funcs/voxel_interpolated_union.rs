use std::error;
use std::f32;
use std::fmt;
use std::ops::Bound;
use std::sync::Arc;

use nalgebra::Vector3;

use crate::analytics;
use crate::bounding_box::BoundingBox;
use crate::interpreter::{
    BooleanParamRefinement, Float3ParamRefinement, FloatParamRefinement, Func, FuncError,
    FuncFlags, FuncInfo, LogMessage, ParamInfo, ParamRefinement, Ty, Value,
};
use crate::mesh::voxel_cloud::{self, ScalarField};

const VOXEL_COUNT_THRESHOLD: u32 = 50000;

#[derive(Debug, PartialEq)]
pub enum FuncInterpolatedUnionError {
    WeldFailed,
    EmptyScalarField,
    VoxelDimensionsZeroOrLess,
    TooManyVoxels(u32, f32, f32, f32),
}

impl fmt::Display for FuncInterpolatedUnionError {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        match self {
            FuncInterpolatedUnionError::WeldFailed => write!(
                f,
                "Welding of separate voxels failed due to high welding proximity tolerance"
            ),
            FuncInterpolatedUnionError::EmptyScalarField => write!(
                f,
                "Scalar field from input meshes or the resulting mesh is empty"
            ),
            FuncInterpolatedUnionError::VoxelDimensionsZeroOrLess => write!(f, "One or more voxel dimensions are zero or less"),
            FuncInterpolatedUnionError::TooManyVoxels(max_count, x, y, z) => write!(
                f,
                "Too many voxels. Limit set to {}. Try setting voxel size to [{:.3}, {:.3}, {:.3}] or more.",
                max_count, x, y, z
            ),
        }
    }
}

impl error::Error for FuncInterpolatedUnionError {}

pub struct FuncInterpolatedUnion;

impl Func for FuncInterpolatedUnion {
    fn info(&self) -> &FuncInfo {
        &FuncInfo {
            name: "Voxel Morph",
            description: "VOXEL INTERPOLATION MORPH FROM TWO MESH GEOMETRIES\n\
            \n\
            Converts the input mesh geometries into voxel clouds, resizes both voxel \
            clouds to be large enough to contain volumes from both of them \
            then computes distance fields for both voxel clouds, interpolates values \
            of the two distance fields by the given factor and eventually \
            materializes the resulting voxel cloud into a welded mesh.\n\
            \n\
            Voxels are three-dimensional pixels. They exist in a regular three-dimensional \
            grid of arbitrary dimensions (voxel size). The voxel can be turned on \
            (be a volume) or off (be a void). The voxels can be materialized as \
            rectangular blocks. Voxelized meshes can be effectively smoothened by \
            Laplacian relaxation.
            \n\
            The input meshes will be marked used and thus invisible in the viewport. \
            They can still be used in subsequent operations.\n\
            \n\
            The resulting mesh geometry will be named 'Interpolated Mesh'.",
            return_value_name: "Interpolated Mesh",
        }
    }

    fn flags(&self) -> FuncFlags {
        FuncFlags::PURE
    }

    fn param_info(&self) -> &[ParamInfo] {
        &[
            ParamInfo {
                name: "Mesh 1",
                description: "First input mesh.",
                refinement: ParamRefinement::Mesh,
                optional: false,
            },
            ParamInfo {
                name: "Mesh 2",
                description: "Second input mesh.",
                refinement: ParamRefinement::Mesh,
                optional: false,
            },
            ParamInfo {
                name: "Voxel Size",
                description: "Size of a single cell in the regular three-dimensional voxel grid.\n\
                \n\
                High values produce coarser results, low values may increase precision but produce \
                heavier geometry that significantly affect performance. Too high values produce \
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
                name: "Factor",
                description: "Proportional interpolation factor between the first and the second mesh.\n\
                \n\
                Factor = 0.0: the result is equal to the first mesh.\n\
                Factor = 0.5: the result is half way between the first and the second mesh.\n\
                Factor = 1.0: the result is equal to the second mesh.",
                refinement: ParamRefinement::Float(FloatParamRefinement {
                    default_value: Some(0.5),
                    min_value: Some(0.0),
                    max_value: Some(1.0),
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
        let mesh1 = args[0].unwrap_mesh();
        let mesh2 = args[1].unwrap_mesh();
        let voxel_dimensions = Vector3::from(args[2].unwrap_float3());
        let fill = args[3].unwrap_boolean();
        let interpolation_factor = args[4].unwrap_float();
        let error_if_large = args[5].unwrap_boolean();
        let analyze_bbox = args[6].unwrap_boolean();
        let analyze_mesh = args[7].unwrap_boolean();

        if voxel_dimensions.iter().any(|dimension| *dimension <= 0.0) {
            let error = FuncError::new(FuncInterpolatedUnionError::VoxelDimensionsZeroOrLess);
            log(LogMessage::error(format!("Error: {}", error)));
            return Err(error);
        }

        let bbox1 = mesh1.bounding_box();
        let bbox2 = mesh2.bounding_box();
        let bbox =
            BoundingBox::union([bbox1, bbox2].iter().copied()).expect("Failed to create union box");
        let voxel_count = voxel_cloud::evaluate_voxel_count(&bbox, &voxel_dimensions);

        log(LogMessage::info(format!("Voxel count = {}", voxel_count)));

        if error_if_large && voxel_count > VOXEL_COUNT_THRESHOLD {
            let suggested_voxel_size =
                voxel_cloud::suggest_voxel_size_to_fit_bbox_within_voxel_count(
                    voxel_count,
                    &voxel_dimensions,
                    VOXEL_COUNT_THRESHOLD,
                );

            let error = FuncError::new(FuncInterpolatedUnionError::TooManyVoxels(
                VOXEL_COUNT_THRESHOLD,
                suggested_voxel_size.x,
                suggested_voxel_size.y,
                suggested_voxel_size.z,
            ));
            log(LogMessage::error(format!("Error: {}", error)));
            return Err(error);
        }

        let mut voxel_cloud1 = ScalarField::from_mesh(mesh1, &voxel_dimensions, 0.0, 1);
        let mut voxel_cloud2 = ScalarField::from_mesh(mesh2, &voxel_dimensions, 0.0, 1);

        let volume_value_range = if fill {
            (Bound::Unbounded, Bound::Included(0.0))
        } else {
            (Bound::Included(0.0), Bound::Included(0.0))
        };

        let bounding_box_voxel_cloud1 = voxel_cloud1.bounding_box_cartesian_space();
        let bounding_box_voxel_cloud2 = voxel_cloud2.bounding_box_cartesian_space();

        if let Some(bounding_box) = BoundingBox::union(
            [bounding_box_voxel_cloud1, bounding_box_voxel_cloud2]
                .iter()
                .copied(),
        ) {
            voxel_cloud1.resize_to_bounding_box_cartesian_space(&bounding_box);
            voxel_cloud1.compute_distance_field(&volume_value_range);
            voxel_cloud2.compute_distance_field(&volume_value_range);

            voxel_cloud1.interpolate_to(&voxel_cloud2, interpolation_factor);
        } else {
            let error = FuncError::new(FuncInterpolatedUnionError::EmptyScalarField);
            log(LogMessage::error(format!("Error: {}", error)));
            return Err(error);
        }

        if !voxel_cloud1.contains_voxels_within_range(&volume_value_range) {
            let error = FuncError::new(FuncInterpolatedUnionError::EmptyScalarField);
            log(LogMessage::error(format!("Error: {}", error)));
            return Err(error);
        }

        match voxel_cloud1.to_mesh(&volume_value_range) {
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
                let error = FuncError::new(FuncInterpolatedUnionError::WeldFailed);
                log(LogMessage::error(format!("Error: {}", error)));
                Err(error)
            }
        }
    }
}
