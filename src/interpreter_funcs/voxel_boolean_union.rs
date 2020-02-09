use std::error;
use std::f32;
use std::fmt;
use std::sync::Arc;

use nalgebra::Vector3;

use crate::analytics;
use crate::interpreter::{
    BooleanParamRefinement, Float3ParamRefinement, Func, FuncError, FuncFlags, FuncInfo,
    LogMessage, ParamInfo, ParamRefinement, Ty, UintParamRefinement, Value,
};
use crate::mesh::voxel_cloud::VoxelCloud;

#[derive(Debug, PartialEq)]
pub enum FuncBooleanUnionError {
    WeldFailed,
    EmptyVoxelCloud,
}

impl fmt::Display for FuncBooleanUnionError {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        match self {
            FuncBooleanUnionError::WeldFailed => write!(
                f,
                "Welding of separate voxels failed due to high welding proximity tolerance"
            ),
            FuncBooleanUnionError::EmptyVoxelCloud => write!(
                f,
                "A voxel cloud from input meshes or the resulting mesh is empty"
            ),
        }
    }
}

impl error::Error for FuncBooleanUnionError {}

pub struct FuncBooleanUnion;

impl Func for FuncBooleanUnion {
    fn info(&self) -> &FuncInfo {
        &FuncInfo {
            name: "Union",
            return_value_name: "Union Mesh",
        }
    }

    fn flags(&self) -> FuncFlags {
        FuncFlags::PURE
    }

    fn param_info(&self) -> &[ParamInfo] {
        &[
            ParamInfo {
                name: "Mesh 1",
                refinement: ParamRefinement::Mesh,
                optional: false,
            },
            ParamInfo {
                name: "Mesh 2",
                refinement: ParamRefinement::Mesh,
                optional: false,
            },
            ParamInfo {
                name: "Voxel Size",
                refinement: ParamRefinement::Float3(Float3ParamRefinement {
                    default_value_x: Some(1.0),
                    min_value_x: Some(f32::MIN_POSITIVE),
                    max_value_x: None,
                    default_value_y: Some(1.0),
                    min_value_y: Some(f32::MIN_POSITIVE),
                    max_value_y: None,
                    default_value_z: Some(1.0),
                    min_value_z: Some(f32::MIN_POSITIVE),
                    max_value_z: None,
                }),
                optional: false,
            },
            ParamInfo {
                name: "Grow",
                refinement: ParamRefinement::Uint(UintParamRefinement {
                    default_value: Some(1),
                    min_value: None,
                    max_value: None,
                }),
                optional: false,
            },
            ParamInfo {
                name: "Fill Closed Volumes",
                refinement: ParamRefinement::Boolean(BooleanParamRefinement {
                    default_value: true,
                }),
                optional: false,
            },
            ParamInfo {
                name: "Analyze resulting mesh",
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
        let growth_iterations = args[3].unwrap_uint();
        let fill = args[4].unwrap_boolean();
        let analyze = args[5].unwrap_boolean();

        let mut voxel_cloud1 = VoxelCloud::from_mesh(mesh1, &Vector3::from(voxel_dimensions));
        let mut voxel_cloud2 = VoxelCloud::from_mesh(mesh2, &Vector3::from(voxel_dimensions));

        for _ in 0..growth_iterations {
            voxel_cloud1.grow_volume();
            voxel_cloud2.grow_volume();
        }

        if fill {
            voxel_cloud1.fill_volumes();
            voxel_cloud2.fill_volumes();
        }

        voxel_cloud1.boolean_union(&voxel_cloud2);
        if !voxel_cloud1.contains_voxels() {
            return Err(FuncError::new(FuncBooleanUnionError::EmptyVoxelCloud));
        }

        match voxel_cloud1.to_mesh() {
            Some(value) => {
                if analyze {
                    analytics::report_mesh_analysis(&value, log);
                }
                Ok(Value::Mesh(Arc::new(value)))
            }
            None => Err(FuncError::new(FuncBooleanUnionError::WeldFailed)),
        }
    }
}
