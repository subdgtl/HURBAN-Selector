use std::error;
use std::f32;
use std::fmt;
use std::sync::Arc;

use nalgebra::Vector3;

use crate::interpreter::{
    Float3ParamRefinement, Func, FuncError, FuncFlags, FuncInfo, ParamInfo, ParamRefinement, Ty,
    UintParamRefinement, Value,
};
use crate::mesh::voxel_cloud::VoxelCloud;

#[derive(Debug, PartialEq)]
pub enum FuncVoxelizeError {
    WeldFailed,
}

impl fmt::Display for FuncVoxelizeError {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        match self {
            FuncVoxelizeError::WeldFailed => write!(
                f,
                "Welding of separate voxels failed due to high welding proximity tolerance"
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
            return_value_name: "Voxelized mesh",
        }
    }

    fn flags(&self) -> FuncFlags {
        FuncFlags::PURE
    }

    fn param_info(&self) -> &[ParamInfo] {
        &[
            ParamInfo {
                name: "Mesh",
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
                    default_value: Some(2),
                    min_value: None,
                    max_value: None,
                }),
                optional: false,
            },
        ]
    }

    fn return_ty(&self) -> Ty {
        Ty::Mesh
    }

    fn call(&mut self, args: &[Value]) -> Result<Value, FuncError> {
        let mesh = args[0].unwrap_mesh();
        let voxel_dimensions = args[1].unwrap_float3();
        let growth_iterations = args[2].unwrap_uint();

        let mut voxel_cloud = VoxelCloud::from_mesh(mesh, &Vector3::from(voxel_dimensions));
        for _ in 0..growth_iterations {
            voxel_cloud.grow_volume();
        }

        match voxel_cloud.to_mesh() {
            Some(value) => Ok(Value::Mesh(Arc::new(value))),
            None => Err(FuncError::new(FuncVoxelizeError::WeldFailed)),
        }
    }
}
