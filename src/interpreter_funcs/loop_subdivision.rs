use std::cmp;
use std::error;
use std::fmt;
use std::sync::Arc;

use crate::analytics;
use crate::interpreter::{
    BooleanParamRefinement, Func, FuncError, FuncFlags, FuncInfo, LogMessage, ParamInfo,
    ParamRefinement, Ty, UintParamRefinement, Value,
};
use crate::mesh::{smoothing, topology, NormalStrategy};

#[derive(Debug, PartialEq)]
pub enum FuncLoopSubdivisionError {
    InvalidMesh,
}

impl fmt::Display for FuncLoopSubdivisionError {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        match self {
            Self::InvalidMesh => write!(
                f,
                "The mesh is either not triangulated, watertight or manifold."
            ),
        }
    }
}

impl error::Error for FuncLoopSubdivisionError {}

pub struct FuncLoopSubdivision;

impl FuncLoopSubdivision {
    const MAX_ITERATIONS: u32 = 3;
}

impl Func for FuncLoopSubdivision {
    fn info(&self) -> &FuncInfo {
        &FuncInfo {
            name: "Smoothen",
            description:
                "SMOOTHEN MESH WITH LOOP SUBDIVISION\n\
                 \n\
                 Creates a new smoothened mesh geometry using Loop subdivision algorithm. \
                 Loop subdivision surface is an approximating subdivision scheme developed \
                 by Charles Loop in 1987 for triangular meshes. Loop subdivision surfaces \
                 are defined recursively, dividing each triangle into four smaller ones. \
                 The vertex and face count will increase with each iteration (2 times \
                 for vertices, 4 times for faces). Too many iterations may take long time \
                 and produce an unnecessarily heavy mesh. Therefore the number of \
                 iterations is limited to 3. \
                 The output mesh will be recomputed with smooth normals.\n\
                 \n\
                 The Loop subdivision algorithm requires the input mesh to be triangulated, \
                 watertight and manifold. \
                 Smoothing operations are usually placed at the end of mesh manipulation \
                 pipeline because the subdivided dense meshes are not suitable for geometry \
                 manipulation.\n\
                 \n\
                 The input mesh will be marked used and thus invisible in the viewport. \
                 It can still be used in subsequent operations.\n\
                 \n\
                 The resulting mesh geometry will be named 'Smooth Mesh'.",
            return_value_name: "Smooth Mesh",
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
                              The mesh must be triangulated, watertight and manifold.",
                refinement: ParamRefinement::Mesh,
                optional: false,
            },
            ParamInfo {
                name: "Iterations",
                description:
                    "Number of iterations (repetitions) of the Loop subdivision algorithm.\n\
                     \n\
                     The number of iterations is limited to 3. \
                     Too many iterations may take long time and produce a heavy dense mesh.",
                refinement: ParamRefinement::Uint(UintParamRefinement {
                    default_value: Some(1),
                    min_value: Some(0),
                    max_value: Some(Self::MAX_ITERATIONS),
                }),
                optional: false,
            },
            ParamInfo {
                name: "Analyze resulting mesh",
                description: "Reports detailed analytic information on the created mesh.\n\
                              The analysis may be slow, therefore it is by default off.",
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
        let iterations = cmp::min(args[1].unwrap_uint(), Self::MAX_ITERATIONS);
        let analyze = args[2].unwrap_boolean();

        if iterations == 0 {
            log(LogMessage::info("Zero iterations, the mesh hasn't changed"));
            return Ok(Value::Mesh(mesh));
        }

        let mut v2v = topology::compute_vertex_to_vertex_topology(&mesh);
        let mut v2f = topology::compute_vertex_to_face_topology(&mesh);
        let mut f2f = topology::compute_face_to_face_topology(&mesh, &v2f);
        if let Some(mut current_mesh) =
            smoothing::loop_subdivision(&mesh, &v2v, &f2f, NormalStrategy::Smooth)
        {
            for _ in 1..iterations {
                v2v = topology::compute_vertex_to_vertex_topology(&current_mesh);
                v2f = topology::compute_vertex_to_face_topology(&current_mesh);
                f2f = topology::compute_face_to_face_topology(&current_mesh, &v2f);
                current_mesh = match smoothing::loop_subdivision(
                    &current_mesh,
                    &v2v,
                    &f2f,
                    NormalStrategy::Smooth,
                ) {
                    Some(m) => m,
                    None => {
                        let error = FuncError::new(FuncLoopSubdivisionError::InvalidMesh);
                        log(LogMessage::error(format!("Error: {}", error)));
                        return Err(error);
                    }
                }
            }

            if analyze {
                analytics::report_mesh_analysis(&current_mesh, log);
            }

            Ok(Value::Mesh(Arc::new(current_mesh)))
        } else {
            let error = FuncError::new(FuncLoopSubdivisionError::InvalidMesh);
            log(LogMessage::error(format!("Error: {}", error)));
            Err(error)
        }
    }
}
