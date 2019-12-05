use std::cmp;
use std::sync::Arc;

use crate::interpreter::{
    Func, FuncError, FuncFlags, FuncInfo, ParamInfo, ParamRefinement, Ty, UintParamRefinement,
    Value,
};
use crate::mesh_smoothing;
use crate::mesh_topology_analysis;

pub struct FuncLoopSubdivision;

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
                refinement: ParamRefinement::Geometry,
                optional: false,
            },
            ParamInfo {
                name: "Iterations",
                refinement: ParamRefinement::Uint(UintParamRefinement {
                    default_value: Some(1),
                    min_value: Some(0),
                    max_value: Some(5),
                }),
                optional: false,
            },
        ]
    }

    fn return_ty(&self) -> Ty {
        Ty::Geometry
    }

    fn call(&mut self, args: &[Value]) -> Result<Value, FuncError> {
        // FIXME: add the max value to the param info so that that the
        // gui doesn't mislead
        const MAX_ITERATIONS: u32 = 3;

        let geometry = args[0].unwrap_refcounted_geometry();
        let iterations = cmp::min(args[1].unwrap_uint(), MAX_ITERATIONS);

        if iterations == 0 {
            return Ok(Value::Geometry(geometry));
        }

        let mut v2v = mesh_topology_analysis::vertex_to_vertex_topology(&geometry);
        let mut f2f = mesh_topology_analysis::face_to_face_topology(&geometry);
        let mut current_geometry = mesh_smoothing::loop_subdivision(&geometry, &v2v, &f2f);

        for _ in 1..iterations {
            v2v = mesh_topology_analysis::vertex_to_vertex_topology(&current_geometry);
            f2f = mesh_topology_analysis::face_to_face_topology(&current_geometry);
            current_geometry = mesh_smoothing::loop_subdivision(&current_geometry, &v2v, &f2f);
        }

        Ok(Value::Geometry(Arc::new(current_geometry)))
    }
}
