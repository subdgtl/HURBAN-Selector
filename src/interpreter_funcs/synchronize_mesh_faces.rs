use std::sync::Arc;

use crate::edge_analysis;
use crate::interpreter::{
    Func, FuncError, FuncFlags, FuncInfo, ParamInfo, ParamRefinement, Ty, Value,
};
use crate::mesh_analysis;
use crate::mesh_tools;
use crate::mesh_topology_analysis;

pub struct FuncSynchronizeMeshFaces;

impl Func for FuncSynchronizeMeshFaces {
    fn info(&self) -> &FuncInfo {
        &FuncInfo {
            name: "Synchronize Faces",
            return_value_name: "Synchronized Mesh",
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
        let geometry = args[0].unwrap_refcounted_geometry();

        let oriented_edges: Vec<_> = geometry.oriented_edges_iter().collect();
        let edge_sharing_map = edge_analysis::edge_sharing(&oriented_edges);

        if !mesh_analysis::is_mesh_orientable(&edge_sharing_map)
            && mesh_analysis::is_mesh_manifold(&edge_sharing_map)
        {
            let face_to_face = mesh_topology_analysis::face_to_face_topology(&geometry);

            let value = Arc::new(mesh_tools::synchronize_mesh_winding(
                &geometry,
                &face_to_face,
            ));

            Ok(Value::Geometry(value))
        } else {
            Ok(Value::Geometry(geometry))
        }
    }
}
