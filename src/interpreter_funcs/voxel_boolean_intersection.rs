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
use crate::mesh::voxel_cloud::{self, FalloffFunction, ScalarField};

const VOXEL_COUNT_THRESHOLD: u32 = 100_000;

#[derive(Debug, PartialEq)]
pub enum FuncBooleanIntersectionError {
    WeldFailed,
    EmptyScalarField,
    VoxelDimensionsZeroOrLess,
    TooManyVoxels(u32, f32, f32, f32),
}

impl fmt::Display for FuncBooleanIntersectionError {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        match self {
            FuncBooleanIntersectionError::WeldFailed => write!(
                f,
                "Welding of separate voxels failed due to high welding proximity tolerance"
            ),
            FuncBooleanIntersectionError::EmptyScalarField => write!(f, "The resulting scalar field is empty"),
            FuncBooleanIntersectionError::VoxelDimensionsZeroOrLess => write!(f, "One or more voxel dimensions are zero or less"),
            FuncBooleanIntersectionError::TooManyVoxels(max_count, x, y, z) => write!(
                f,
                "Too many voxels. Limit set to {}. Try setting voxel size to [{:.3}, {:.3}, {:.3}] or more.",
                max_count, x, y, z
            ),
        }
    }
}

impl error::Error for FuncBooleanIntersectionError {}

pub struct FuncBooleanIntersection;

impl Func for FuncBooleanIntersection {
    fn info(&self) -> &FuncInfo {
        &FuncInfo {
            name: "Intersection",
            description: "BOOLEAN INTERSECTION OF VOXEL CLOUDS FROM TWO MESH GEOMETRIES\n\
            \n\
            Converts the input mesh geometries into voxel clouds, then performs \
            boolean intersection of the first and second voxel clouds and eventually \
            materializes the resulting voxel cloud into a welded mesh. \
            Boolean intersection keeps only those parts of the volume, which are common \
            to both input geometries. It is equivalent to logical AND operation.\n\
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
            The resulting mesh geometry will be named 'Intersection Mesh'.",
            return_value_name: "Intersection Mesh",
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
        let mesh1 = args[0].unwrap_mesh();
        let mesh2 = args[1].unwrap_mesh();
        let voxel_dimensions = Vector3::from(args[2].unwrap_float3());
        let growth_u32 = args[3].unwrap_uint();
        let growth_f32 = growth_u32 as f32;
        let fill = args[4].unwrap_boolean();
        let marching_cubes = args[5].unwrap_boolean();
        let error_if_large = args[6].unwrap_boolean();
        let analyze_mesh = args[7].unwrap_boolean();

        if voxel_dimensions.iter().any(|dimension| *dimension <= 0.0) {
            let error = FuncError::new(FuncBooleanIntersectionError::VoxelDimensionsZeroOrLess);
            log(LogMessage::error(format!("Error: {}", error)));
            return Err(error);
        }

        let bbox1 = mesh1.bounding_box();
        let voxel_count1 = voxel_cloud::evaluate_voxel_count(&bbox1, &voxel_dimensions);

        let bbox2 = mesh2.bounding_box();
        let voxel_count2 = voxel_cloud::evaluate_voxel_count(&bbox2, &voxel_dimensions);

        let voxel_count = voxel_count1.max(voxel_count2);

        log(LogMessage::info(format!("Voxel count = {}", voxel_count)));

        if error_if_large && voxel_count > VOXEL_COUNT_THRESHOLD {
            let suggested_voxel_size =
                voxel_cloud::suggest_voxel_size_to_fit_bbox_within_voxel_count(
                    voxel_count,
                    &voxel_dimensions,
                    VOXEL_COUNT_THRESHOLD,
                );

            let error = FuncError::new(FuncBooleanIntersectionError::TooManyVoxels(
                VOXEL_COUNT_THRESHOLD,
                suggested_voxel_size.x,
                suggested_voxel_size.y,
                suggested_voxel_size.z,
            ));
            log(LogMessage::error(format!("Error: {}", error)));
            return Err(error);
        }

        let mut voxel_cloud1 = ScalarField::from_mesh(mesh1, &voxel_dimensions, 0.0, growth_u32);
        let mut voxel_cloud2 = ScalarField::from_mesh(mesh2, &voxel_dimensions, 0.0, growth_u32);

        voxel_cloud1.compute_distance_field(&(0.0..=0.0), FalloffFunction::Linear(1.0));
        voxel_cloud2.compute_distance_field(&(0.0..=0.0), FalloffFunction::Linear(1.0));

        let meshing_range = if fill {
            (Bound::Unbounded, Bound::Included(growth_f32))
        } else {
            (Bound::Included(-growth_f32), Bound::Included(growth_f32))
        };

        voxel_cloud1.boolean_intersection(&meshing_range, &voxel_cloud2, &meshing_range);

        if !voxel_cloud1.contains_voxels_within_range(&meshing_range) {
            let error = FuncError::new(FuncBooleanIntersectionError::EmptyScalarField);
            log(LogMessage::error(format!("Error: {}", error)));
            return Err(error);
        }

        let meshing_output = if marching_cubes {
            voxel_cloud1.to_marching_cubes(&meshing_range)
        } else {
            voxel_cloud1.to_mesh(&meshing_range)
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
                let error = FuncError::new(FuncBooleanIntersectionError::WeldFailed);
                log(LogMessage::error(format!("Error: {}", error)));
                Err(error)
            }
        }
    }
}
