use std::sync::Arc;

use crate::interpreter::{
    Func, FuncError, FuncFlags, FuncInfo, MeshArrayValue, ParamInfo, ParamRefinement, Ty, Value,
};
use crate::mesh_tools;

pub struct FuncSeparateIsolatedMeshes;

impl Func for FuncSeparateIsolatedMeshes {
    fn info(&self) -> &FuncInfo {
        &FuncInfo {
            name: "Separate Volumes",
            return_value_name: "Volumes Group",
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

    fn call(&mut self, args: &[Value]) -> Result<Value, FuncError> {
        let mesh = args[0].unwrap_mesh();

        let meshes = mesh_tools::separate_isolated_meshes(&mesh);
        let value = MeshArrayValue::new(meshes.into_iter().map(Arc::new).collect());

        Ok(Value::MeshArray(Arc::new(value)))
    }
}
