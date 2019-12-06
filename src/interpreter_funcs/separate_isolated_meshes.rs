use std::sync::Arc;

use crate::interpreter::{
    Func, FuncError, FuncFlags, FuncInfo, GeometryArrayValue, ParamInfo, ParamRefinement, Ty, Value,
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
            refinement: ParamRefinement::Geometry,
            optional: false,
        }]
    }

    fn return_ty(&self) -> Ty {
        Ty::GeometryArray
    }

    fn call(&mut self, args: &[Value]) -> Result<Value, FuncError> {
        let geometry = args[0].unwrap_geometry();

        let geometries = mesh_tools::separate_isolated_meshes(&geometry);
        let value = GeometryArrayValue::new(geometries.into_iter().map(Arc::new).collect());

        Ok(Value::GeometryArray(Arc::new(value)))
    }
}
