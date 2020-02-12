use std::error;
use std::fmt;
use std::sync::Arc;

use crate::analytics;
use crate::interpreter::{
    BooleanParamRefinement, Func, FuncError, FuncFlags, FuncInfo, LogMessage, ParamInfo,
    ParamRefinement, Ty, Value,
};
use crate::mesh::{analysis, tools, topology};

#[derive(Debug, PartialEq)]
pub enum FuncSynchronizeMeshFacesError {
    NonManifold,
}

impl fmt::Display for FuncSynchronizeMeshFacesError {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        match self {
            FuncSynchronizeMeshFacesError::NonManifold => write!(
                f,
                "The mesh is non-manifold. Some edges are shared by more than two faces."
            ),
        }
    }
}

impl error::Error for FuncSynchronizeMeshFacesError {}

pub struct FuncSynchronizeMeshFaces;

impl Func for FuncSynchronizeMeshFaces {
    fn info(&self) -> &FuncInfo {
        &FuncInfo {
            name: "Synchronize Faces",
            description: "",
            return_value_name: "Synchronized Mesh",
        }
    }

    fn flags(&self) -> FuncFlags {
        FuncFlags::PURE
    }

    fn param_info(&self) -> &[ParamInfo] {
        &[
            ParamInfo {
                name: "Mesh",
                description: "",
                refinement: ParamRefinement::Mesh,
                optional: false,
            },
            ParamInfo {
                name: "Analyze resulting mesh",
                description: "",
                refinement: ParamRefinement::Boolean(BooleanParamRefinement {
                    default_value: false,
                }),
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
        log: &mut dyn FnMut(LogMessage),
    ) -> Result<Value, FuncError> {
        let mesh = args[0].unwrap_refcounted_mesh();
        let analyze = args[1].unwrap_boolean();

        let oriented_edges: Vec<_> = mesh.oriented_edges_iter().collect();
        let edge_sharing_map = analysis::edge_sharing(&oriented_edges);

        let is_manifold = analysis::is_mesh_manifold(&edge_sharing_map);
        if !is_manifold {
            let error = FuncError::new(FuncSynchronizeMeshFacesError::NonManifold);
            log(LogMessage::error(format!("Error: {}", error)));
            return Err(error);
        }

        let value = if !analysis::is_mesh_orientable(&edge_sharing_map) {
            let vertex_to_face = topology::compute_vertex_to_face_topology(&mesh);
            let face_to_face = topology::compute_face_to_face_topology(&mesh, &vertex_to_face);

            Arc::new(tools::synchronize_mesh_winding(&mesh, &face_to_face))
        } else {
            mesh
        };

        if analyze {
            analytics::report_mesh_analysis(&value, log);
        }

        Ok(Value::Mesh(value))
    }
}
