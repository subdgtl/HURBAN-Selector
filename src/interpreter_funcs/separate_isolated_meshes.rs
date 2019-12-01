use std::sync::Arc;

use crate::interpreter::{
    Func, FuncError, FuncFlags, FuncInfo, ParamInfo, ParamRefinement, Ty, Value,
};
use crate::mesh_tools;

pub struct FuncSeparateIsolatedMeshes;

impl Func for FuncSeparateIsolatedMeshes {
    fn info(&self) -> &FuncInfo {
        &FuncInfo {
            name: "Separate Volumes",
            return_value_name: "Separated Mesh",
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
        Ty::Geometry
    }

    fn call(&mut self, args: &[Value]) -> Result<Value, FuncError> {
        let geometry = args[0].unwrap_geometry();

        let values = mesh_tools::separate_isolated_meshes(&geometry);

        // FIXME: This returns a slice of Geometries. Return all of them
        let first_value = values
            .into_iter()
            .next()
            .expect("Need at least one geometry");
        Ok(Value::Geometry(Arc::new(first_value)))
    }
}
