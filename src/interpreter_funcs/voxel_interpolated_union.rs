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
use crate::mesh::scalar_field::ScalarField;

const VOXEL_COUNT_THRESHOLD: u32 = 50000;

#[derive(Debug, PartialEq)]
pub enum FuncInterpolatedUnionError {
    WeldFailed,
    EmptyScalarField,
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
            clouds to bel large enough to contain volumes from both of them \
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
                    min_value: Some(f32::MIN_POSITIVE),
                    max_value: None,
                    default_value_x: Some(0.25),
                    default_value_y: Some(0.25),
                    default_value_z: Some(0.25),
                }),
                optional: false,
            },
            ParamInfo {
                name: "Fill Closed Volumes",
                description: "Treats the insides of watertight mesh geometries as volumes.\n\
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
                    default_value: Some(0.0),
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
        let voxel_dimensions = args[2].unwrap_float3();
        let fill = args[3].unwrap_boolean();
        let factor = args[4].unwrap_float();
        let error_if_large = args[5].unwrap_boolean();
        let analyze_bbox = args[6].unwrap_boolean();
        let analyze_mesh = args[7].unwrap_boolean();

        let bbox1 = mesh1.bounding_box();
        let bbox2 = mesh2.bounding_box();
        let bbox =
            BoundingBox::union([bbox1, bbox2].iter().copied()).expect("Failed to create union box");
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
            let error = FuncError::new(FuncInterpolatedUnionError::TooManyVoxels(
                VOXEL_COUNT_THRESHOLD,
                vx * 1.1,
                vy * 1.1,
                vz * 1.1,
            ));
            log(LogMessage::error(format!("Error: {}", error)));
            return Err(error);
        }

        let mut scalar_field1 =
            ScalarField::from_mesh(mesh1, &Vector3::from(voxel_dimensions), 0_i16, 1);
        let scalar_field2 =
            ScalarField::from_mesh(mesh2, &Vector3::from(voxel_dimensions), 0_i16, 1);

        let boolean_union_range = 0..=0;

        let meshing_range = if fill {
            (Bound::Unbounded, Bound::Included(0))
        } else {
            (Bound::Included(0), Bound::Included(0))
        };

        scalar_field1.interpolated_union_of_distance_fields(
            &boolean_union_range,
            &scalar_field2,
            &boolean_union_range,
            factor,
        );

        if !scalar_field1.contains_voxels_within_range(&meshing_range) {
            let error = FuncError::new(FuncInterpolatedUnionError::EmptyScalarField);
            log(LogMessage::error(format!("Error: {}", error)));
            return Err(error);
        }

        match scalar_field1.to_mesh(&meshing_range) {
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
