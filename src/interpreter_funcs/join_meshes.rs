use std::sync::Arc;

use crate::interpreter::{
    Func, FuncError, FuncFlags, FuncInfo, LogMessage, ParamInfo, ParamRefinement, Ty, Value,
};
use crate::mesh::tools;

pub struct FuncJoinMeshes;

impl Func for FuncJoinMeshes {
    fn info(&self) -> &FuncInfo {
        &FuncInfo {
            name: "Join Meshes",
            return_value_name: "Joined Mesh",
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
        let meshes = args.iter().map(|a| a.unwrap_mesh());

        let value = tools::join_multiple_meshes(meshes);
        Ok(Value::Mesh(Arc::new(value)))
    }
}
