use std::sync::Arc;

use crate::interpreter::{
    Func, FuncError, FuncFlags, FuncInfo, ParamInfo, ParamRefinement, Ty, Value,
};
use crate::mesh::tools;
use crate::mesh::{analysis, topology};

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
            refinement: ParamRefinement::Mesh,
            optional: false,
        }]
    }

    fn return_ty(&self) -> Ty {
        Ty::Mesh
    }

    fn call(&mut self, args: &[Value]) -> Result<Value, FuncError> {
        let mesh = args[0].unwrap_refcounted_mesh();

        let oriented_edges: Vec<_> = mesh.oriented_edges_iter().collect();
        let edge_sharing_map = analysis::edge_sharing(&oriented_edges);

        if !analysis::is_mesh_orientable(&edge_sharing_map)
            && analysis::is_mesh_manifold(&edge_sharing_map)
        {
            let face_to_face = topology::compute_face_to_face_topology(&mesh);

            let value = Arc::new(tools::synchronize_mesh_winding(&mesh, &face_to_face));

            Ok(Value::Mesh(value))
        } else {
            Ok(Value::Mesh(mesh))
        }
    }
}
