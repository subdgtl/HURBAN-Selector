use std::sync::Arc;

use crate::interpreter::{
    Func, FuncError, FuncFlags, FuncInfo, ParamInfo, ParamRefinement, Ty, Value,
};
use crate::mesh::tools;

pub struct FuncJoinGroup;

impl Func for FuncJoinGroup {
    fn info(&self) -> &FuncInfo {
        &FuncInfo {
            name: "Join Group",
            return_value_name: "Joined Mesh",
        }
    }

    fn flags(&self) -> FuncFlags {
        FuncFlags::PURE
    }

    fn param_info(&self) -> &[ParamInfo] {
        &[ParamInfo {
            name: "Group",
            refinement: ParamRefinement::MeshArray,
            optional: false,
        }]
    }

    fn return_ty(&self) -> Ty {
        Ty::Mesh
    }

    fn call(&mut self, args: &[Value]) -> Result<Value, FuncError> {
        let mesh_arc_array = args[0].unwrap_mesh_array();

        let meshes = mesh_arc_array.iter();
        let value = tools::join_multiple_meshes(meshes);
        Ok(Value::Mesh(Arc::new(value)))
    }
}
