use std::cmp;
use std::sync::Arc;

use crate::interpreter::{
    Func, FuncError, FuncFlags, FuncInfo, ParamInfo, ParamRefinement, Ty, UintParamRefinement,
    Value,
};
use crate::mesh_smoothing;
use crate::mesh_topology_analysis;

pub struct FuncLaplacianSmoothing;

impl Func for FuncLaplacianSmoothing {
    fn info(&self) -> &FuncInfo {
        &FuncInfo {
            name: "Laplacian Smoothing",
            return_value_name: "Smoothed Mesh",
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
                    max_value: Some(255),
                }),
                optional: false,
            },
        ]
    }

    fn return_ty(&self) -> Ty {
        Ty::Geometry
    }

    fn call(&mut self, args: &[Value]) -> Result<Value, FuncError> {
        let geometry = args[0].unwrap_geometry();
        let iterations = args[1].unwrap_uint();

        let v2v = mesh_topology_analysis::vertex_to_vertex_topology(geometry);

        let (value, _, _) = mesh_smoothing::laplacian_smoothing(
            geometry,
            &v2v,
            cmp::min(255, iterations),
            &[],
            false,
        );
        Ok(Value::Geometry(Arc::new(value)))
    }
}
