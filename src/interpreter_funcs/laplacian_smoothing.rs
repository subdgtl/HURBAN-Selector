use std::cmp;
use std::sync::Arc;

use crate::analytics;
use crate::interpreter::{
    BooleanParamRefinement, Func, FuncError, FuncFlags, FuncInfo, LogMessage, ParamInfo,
    ParamRefinement, Ty, UintParamRefinement, Value,
};
use crate::mesh::{smoothing, topology, NormalStrategy};

pub struct FuncLaplacianSmoothing;

impl Func for FuncLaplacianSmoothing {
    fn info(&self) -> &FuncInfo {
        &FuncInfo {
            name: "Relax",
            description:
                "RELAX MESH WITH LAPLACIAN SMOOTHING\n\
                 \n\
                 Creates a new relaxed mesh geometry using laplacian smoothing algorithm. \
                 Laplacian smoothing is an algorithm to smoothen a polygonal mesh. \
                 For each vertex in a mesh, a new position is chosen based on local \
                 information (such as the position of neighbors) and the vertex is moved there. \
                 The vertex and face count will remain unchanged. \n\
                 \n\
                 Laplacian smoothing removes small details, grain and kinks of the original model. \
                 Too many iterations may reduce the mesh volume. \
                 The output mesh will be recomputed with smooth normals.\n\
                 \n\
                 The input mesh will be marked used and thus invisible in the viewport. \
                 It can still be used in subsequent operations.\n\
                 \n\
                 The resulting mesh geometry will be named 'Relaxed Mesh'.",
            return_value_name: "Relaxed Mesh",
        }
    }

    fn flags(&self) -> FuncFlags {
        FuncFlags::PURE
    }

    fn param_info(&self) -> &[ParamInfo] {
        &[
            ParamInfo {
                name: "Mesh",
                description: "Input mesh.",
                refinement: ParamRefinement::Mesh,
                optional: false,
            },
            ParamInfo {
                name: "Iterations",
                description:
                    "Number of iterations (repetitions) of the laplacian smoothing algorithm.\n\
                     Too many iterations may take long time and/or reduce the mesh volume.",
                refinement: ParamRefinement::Uint(UintParamRefinement {
                    default_value: Some(1),
                    min_value: Some(0),
                    max_value: Some(255),
                }),
                optional: false,
            },
            ParamInfo {
                name: "Bounding Box Analysis",
                description: "Reports basic and quick analytic information on the created mesh.",
                refinement: ParamRefinement::Boolean(BooleanParamRefinement {
                    default_value: true,
                }),
                optional: false,
            },
            ParamInfo {
                name: "Detailed Mesh Analysis",
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
        let mesh = args[0].unwrap_mesh();
        let iterations = args[1].unwrap_uint();
        let analyze_bbox = args[2].unwrap_boolean();
        let analyze_mesh = args[3].unwrap_boolean();

        let v2v = topology::compute_vertex_to_vertex_topology(mesh);

        let (value, _, _) = smoothing::laplacian_smoothing(
            mesh,
            &v2v,
            cmp::min(255, iterations),
            &[],
            false,
            NormalStrategy::Smooth,
        );

        if analyze_bbox {
            analytics::report_bounding_box_analysis(&value, log);
        }
        if analyze_mesh {
            analytics::report_mesh_analysis(&value, log);
        }

        Ok(Value::Mesh(Arc::new(value)))
    }
}
