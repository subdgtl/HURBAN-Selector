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
            description: "SYNCHRONIZE MESH FACES\n\
                          \n\
                          Change winding (vertex order) of all mesh faces so \
                          that they match winding of the first face of the mesh.\
                          This is a maintenance operation that helps to \
                          fix meshes with faces randomly facing inwards and outwards. \
                          The resulting mesh may face entirely inwards. In such case it \
                          is necessary to Revert mesh faces so that they face outwards. \
                          The operation is useful for synchronizing properties of multiple \
                          meshes before hybridization. Some hybridization algorithms \
                          require the faces of input meshes to face the same direction \
                          (inwards or outwards).\n\
                          \n\
                          This operation does not affect normals and mesh rendering.\n\
                          \n\
                          The input mesh will be marked used and thus invisible in the viewport. \
                          It can still be used in subsequent operations.\n\
                          \n\
                          The resulting mesh geometry will be named 'Synchronized Mesh'.",
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
                description: "Input mesh.\n\
                              The input mesh must be manifold.",
                refinement: ParamRefinement::Mesh,
                optional: false,
            },
            ParamInfo {
                name: "Mesh Analysis",
                description: "Reports detailed analytic information on the created mesh.\n\
                              The analysis may be slow, turn it on only when needed.",
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
        let analyze_mesh = args[1].unwrap_boolean();

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

        if analyze_mesh {
            analytics::report_bounding_box_analysis(&value, log);
            analytics::report_mesh_analysis(&value, log);
        }

        Ok(Value::Mesh(value))
    }
}
