use std::error;
use std::f32;
use std::fmt;
use std::sync::Arc;

use nalgebra::Vector3;

use crate::convert::clamp_cast_u32_to_i16;
use crate::interpreter::{
    BooleanParamRefinement, Float3ParamRefinement, Func, FuncError, FuncFlags, FuncInfo,
    LogMessage, ParamInfo, ParamRefinement, Ty, UintParamRefinement, Value,
};
use crate::mesh::scalar_field::ScalarField;

#[derive(Debug, PartialEq)]
pub enum FuncBooleanUnionError {
    WeldFailed,
    EmptyScalarField,
}

impl fmt::Display for FuncBooleanUnionError {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        match self {
            FuncBooleanUnionError::WeldFailed => write!(
                f,
                "Welding of separate voxels failed due to high welding proximity tolerance"
            ),
            FuncBooleanUnionError::EmptyScalarField => write!(
                f,
                "Scalar field from input meshes or the resulting mesh is empty"
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
                    default_value: Some(0),
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
        let growth_u32 = args[3].unwrap_uint();
        let growth_i16 = clamp_cast_u32_to_i16(growth_u32);
        let fill = args[4].unwrap_boolean();

        let mut scalar_field1 =
            ScalarField::from_mesh(mesh1, &Vector3::from(voxel_dimensions), 0_i16, growth_u32);
        let mut scalar_field2 =
            ScalarField::from_mesh(mesh2, &Vector3::from(voxel_dimensions), 0_i16, growth_u32);

        scalar_field1.compute_distance_filed(&(0..=0));
        scalar_field2.compute_distance_filed(&(0..=0));

        let boolean_union_range = &(-growth_i16..=growth_i16);

        scalar_field1.boolean_union(boolean_union_range, &scalar_field2, boolean_union_range);

        // FIXME: Return RangeBounds of the volume_value_range for both
        // if/else options and remove redundant code.
        if fill {
            if !scalar_field1.contains_voxels_within_range(&(..=growth_i16)) {
                return Err(FuncError::new(FuncBooleanUnionError::EmptyScalarField));
            }

            match scalar_field1.to_mesh(&(..=growth_i16)) {
                Some(value) => Ok(Value::Mesh(Arc::new(value))),
                None => Err(FuncError::new(FuncBooleanUnionError::WeldFailed)),
            }
        } else {
            if !scalar_field1.contains_voxels_within_range(&(-growth_i16..=growth_i16)) {
                return Err(FuncError::new(FuncBooleanUnionError::EmptyScalarField));
            }

            match scalar_field1.to_mesh(&(-growth_i16..=growth_i16)) {
                Some(value) => Ok(Value::Mesh(Arc::new(value))),
                None => Err(FuncError::new(FuncBooleanUnionError::WeldFailed)),
            }
        }
    }
}
