use std::error;
use std::f32;
use std::fmt;
use std::ops::Bound;
use std::sync::Arc;

use nalgebra::Vector3;

use crate::analytics;
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
            description: "",
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
                description: "",
                refinement: ParamRefinement::Mesh,
                optional: false,
            },
            ParamInfo {
                name: "Mesh 2",
                description: "",
                refinement: ParamRefinement::Mesh,
                optional: false,
            },
            ParamInfo {
                name: "Voxel Size",
                description: "",
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
                description: "",
                refinement: ParamRefinement::Uint(UintParamRefinement {
                    default_value: Some(0),
                    min_value: None,
                    max_value: None,
                }),
                optional: false,
            },
            ParamInfo {
                name: "Fill Closed Volumes",
                description: "",
                refinement: ParamRefinement::Boolean(BooleanParamRefinement {
                    default_value: true,
                }),
                optional: false,
            },
            ParamInfo {
                name: "Analyze resulting mesh",
                description: "",
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
        let growth_u32 = args[3].unwrap_uint();
        let growth_i16 = clamp_cast_u32_to_i16(growth_u32);
        let fill = args[4].unwrap_boolean();
        let analyze = args[5].unwrap_boolean();

        let mut scalar_field1 =
            ScalarField::from_mesh(mesh1, &Vector3::from(voxel_dimensions), 0_i16, growth_u32);
        let mut scalar_field2 =
            ScalarField::from_mesh(mesh2, &Vector3::from(voxel_dimensions), 0_i16, growth_u32);

        scalar_field1.compute_distance_filed(&(0..=0));
        scalar_field2.compute_distance_filed(&(0..=0));

        let boolean_union_range = &(-growth_i16..=growth_i16);

        scalar_field1.boolean_union(boolean_union_range, &scalar_field2, boolean_union_range);

        let meshing_range = if fill {
            (Bound::Unbounded, Bound::Included(growth_i16))
        } else {
            (Bound::Included(-growth_i16), Bound::Included(growth_i16))
        };

        if !scalar_field1.contains_voxels_within_range(&meshing_range) {
            let error = FuncError::new(FuncBooleanUnionError::EmptyScalarField);
            log(LogMessage::error(format!("Error: {}", error)));
            return Err(error);
        }

        match scalar_field1.to_mesh(&meshing_range) {
            Some(value) => {
                if analyze {
                    analytics::report_mesh_analysis(&value, log);
                }
                Ok(Value::Mesh(Arc::new(value)))
            }
            None => {
                let error = FuncError::new(FuncBooleanUnionError::WeldFailed);
                log(LogMessage::error(format!("Error: {}", error)));
                Err(error)
            }
        }
    }
}
