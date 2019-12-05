use std::sync::Arc;

use crate::interpreter::{
    Func, FuncError, FuncFlags, FuncInfo, ParamInfo, ParamRefinement, Ty, Value,
};
use crate::mesh_tools;

pub struct FuncRevertMeshFaces;

impl Func for FuncRevertMeshFaces {
    fn info(&self) -> &FuncInfo {
        &FuncInfo {
            name: "Revert Faces",
            return_value_name: "Reverted Mesh",
        }
    }

    fn flags(&self) -> FuncFlags {
        FuncFlags::PURE
    }

    fn param_info(&self) -> &[ParamInfo] {
        &[ParamInfo {
            name: "Mesh",
            refinement: ParamRefinement::Geometry,
            optional: false,
        }]
    }

    fn return_ty(&self) -> Ty {
        Ty::Geometry
    }

    fn call(&mut self, args: &[Value]) -> Result<Value, FuncError> {
        let geometry = args[0].unwrap_geometry();

        let value = mesh_tools::revert_mesh_faces(geometry);
        Ok(Value::Geometry(Arc::new(value)))
    }
}
