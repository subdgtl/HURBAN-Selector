use std::cmp;
use std::error;
use std::fmt;
use std::sync::Arc;

use crate::interpreter::{
    Func, FuncError, FuncFlags, FuncInfo, ParamInfo, ParamRefinement, Ty, UintParamRefinement,
    Value,
};
use crate::mesh::{smoothing, topology};

#[derive(Debug, PartialEq)]
pub enum FuncLoopSubdivisionError {
    InvalidMesh,
}

impl fmt::Display for FuncLoopSubdivisionError {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        match self {
            Self::InvalidMesh => write!(
                f,
                "The mesh doesn't meet the loop subdivision prerequisites"
            ),
        }
    }
}

impl error::Error for FuncLoopSubdivisionError {}

pub struct FuncLoopSubdivision;

impl FuncLoopSubdivision {
    const MAX_ITERATIONS: u32 = 3;
}

impl Func for FuncLoopSubdivision {
    fn info(&self) -> &FuncInfo {
        &FuncInfo {
            name: "Loop Subdivision",
            return_value_name: "Subdivided Mesh",
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
                name: "Iterations",
                refinement: ParamRefinement::Uint(UintParamRefinement {
                    default_value: Some(1),
                    min_value: Some(0),
                    max_value: Some(Self::MAX_ITERATIONS),
                }),
                optional: false,
            },
        ]
    }

    fn return_ty(&self) -> Ty {
        Ty::Mesh
    }

    fn call(&mut self, args: &[Value]) -> Result<Value, FuncError> {
        let mesh = args[0].unwrap_refcounted_mesh();
        let iterations = cmp::min(args[1].unwrap_uint(), Self::MAX_ITERATIONS);

        if iterations == 0 {
            return Ok(Value::Mesh(mesh));
        }

        let mut v2v = topology::compute_vertex_to_vertex_topology(&mesh);
        let mut f2f = topology::compute_face_to_face_topology(&mesh);
        if let Some(mut current_mesh) = smoothing::loop_subdivision(&mesh, &v2v, &f2f) {
            for _ in 1..iterations {
                v2v = topology::compute_vertex_to_vertex_topology(&current_mesh);
                f2f = topology::compute_face_to_face_topology(&current_mesh);
                current_mesh = match smoothing::loop_subdivision(&current_mesh, &v2v, &f2f) {
                    Some(m) => m,
                    None => return Err(FuncError::new(FuncLoopSubdivisionError::InvalidMesh)),
                }
            }
            Ok(Value::Mesh(Arc::new(current_mesh)))
        } else {
            Err(FuncError::new(FuncLoopSubdivisionError::InvalidMesh))
        }
    }
}
