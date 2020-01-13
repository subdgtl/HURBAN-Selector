use std::sync::Arc;

use crate::interpreter::{
    Func, FuncError, FuncFlags, FuncInfo, LogMessage, MeshArrayValue, ParamInfo, ParamRefinement,
    Ty, Value,
};
use crate::mesh::tools;

pub struct FuncDisjointMesh;

impl Func for FuncDisjointMesh {
    fn info(&self) -> &FuncInfo {
        &FuncInfo {
            name: "Disjoint Mesh",
            return_value_name: "Disjoint Group",
        }
    }

    fn flags(&self) -> FuncFlags {
        FuncFlags::empty()
    }

    fn param_info(&self) -> &[ParamInfo] {
        &[ParamInfo {
            name: "Mesh",
            refinement: ParamRefinement::Mesh,
            optional: false,
        }]
    }

    fn return_ty(&self) -> Ty {
        Ty::MeshArray
    }

    fn call(
        &mut self,
        args: &[Value],
        _log: &mut dyn FnMut(LogMessage),
    ) -> Result<Value, FuncError> {
        let mesh = args[0].unwrap_mesh();

        let meshes = tools::disjoint_mesh(&mesh);
        let value = MeshArrayValue::new(meshes.into_iter().map(Arc::new).collect());

        Ok(Value::MeshArray(Arc::new(value)))
    }
}
