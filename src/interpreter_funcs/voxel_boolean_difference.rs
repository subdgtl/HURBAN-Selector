use std::error;
use std::f32;
use std::fmt;
use std::sync::Arc;

use nalgebra::Vector3;

use crate::interpreter::{
    Float3ParamRefinement, Func, FuncError, FuncFlags, FuncInfo, LogMessage, ParamInfo,
    ParamRefinement, Ty, UintParamRefinement, Value,
};
use crate::mesh::voxel_cloud::VoxelCloud;

#[derive(Debug, PartialEq)]
pub enum FuncBooleanDifferenceError {
    WeldFailed,
}

impl fmt::Display for FuncBooleanDifferenceError {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        match self {
            FuncBooleanDifferenceError::WeldFailed => write!(
                f,
                "Welding of separate voxels failed due to high welding proximity tolerance"
            ),
        }
    }
}

impl error::Error for FuncBooleanDifferenceError {}

pub struct FuncBooleanDifference;

impl Func for FuncBooleanDifference {
    fn info(&self) -> &FuncInfo {
        &FuncInfo {
            name: "Difference",
            return_value_name: "Difference Mesh",
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
        ]
    }

    fn return_ty(&self) -> Ty {
        Ty::Mesh
    }

    fn call(
        &mut self,
        args: &[Value],
        _log: &mut dyn FnMut(LogMessage),
    ) -> Result<Value, FuncError> {
        let mesh1 = args[0].unwrap_mesh();
        let mesh2 = args[1].unwrap_mesh();
        let voxel_dimensions = args[2].unwrap_float3();
        let growth_iterations = args[3].unwrap_uint();

        let mut voxel_cloud1 = VoxelCloud::from_mesh(mesh1, &Vector3::from(voxel_dimensions));
        let mut voxel_cloud2 = VoxelCloud::from_mesh(mesh2, &Vector3::from(voxel_dimensions));

        for _ in 0..growth_iterations {
            voxel_cloud1.grow_volume();
            voxel_cloud2.grow_volume();
        }

        voxel_cloud1.boolean_difference(&voxel_cloud2);
        match voxel_cloud1.to_mesh() {
            Some(value) => Ok(Value::Mesh(Arc::new(value))),
            None => Err(FuncError::new(FuncBooleanDifferenceError::WeldFailed)),
        }
    }
}
