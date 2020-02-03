use std::error;
use std::f32;
use std::fmt;
use std::sync::Arc;

use nalgebra::Vector3;

use crate::interpreter::{
    BooleanParamRefinement, Float3ParamRefinement, FloatParamRefinement, Func, FuncError,
    FuncFlags, FuncInfo, LogMessage, ParamInfo, ParamRefinement, Ty, Value,
};
use crate::interval::Interval;
use crate::mesh::scalar_field::ScalarField;

#[derive(Debug, PartialEq)]
pub enum FuncInterpolatedUnionError {
    WeldFailed,
    EmptyScalarField,
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
        }
    }
}

impl error::Error for FuncInterpolatedUnionError {}

pub struct FuncInterpolatedUnion;

impl Func for FuncInterpolatedUnion {
    fn info(&self) -> &FuncInfo {
        &FuncInfo {
            name: "Interpolated",
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
                name: "Fill Closed Volumes",
                refinement: ParamRefinement::Boolean(BooleanParamRefinement {
                    default_value: true,
                }),
                optional: false,
            },
            ParamInfo {
                name: "Factor",
                refinement: ParamRefinement::Float(FloatParamRefinement {
                    default_value: Some(0.0),
                    min_value: Some(0.0),
                    max_value: Some(1.0),
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
        let fill = args[3].unwrap_boolean();
        let factor = args[4].unwrap_float();

        let mut scalar_field1 =
            ScalarField::from_mesh(mesh1, &Vector3::from(voxel_dimensions), 0_i16, 5);
        let scalar_field2 =
            ScalarField::from_mesh(mesh2, &Vector3::from(voxel_dimensions), 0_i16, 5);

        let boolean_union_interval = Interval::new(0, 0);

        let meshing_interval = if fill {
            Interval::new_left_infinite(0)
        } else {
            Interval::new(0, 0)
        };

        scalar_field1.interpolated_union_of_distance_fields(
            boolean_union_interval,
            &scalar_field2,
            boolean_union_interval,
            factor,
        );

        if !scalar_field1.contains_voxels_within_interval(meshing_interval) {
            return Err(FuncError::new(FuncInterpolatedUnionError::EmptyScalarField));
        }

        match scalar_field1.to_mesh(meshing_interval) {
            Some(value) => Ok(Value::Mesh(Arc::new(value))),
            None => Err(FuncError::new(FuncInterpolatedUnionError::WeldFailed)),
        }
    }
}
