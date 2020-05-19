use std::sync::Arc;

use crate::analytics;
use crate::interpreter::{
    BooleanParamRefinement, Func, FuncError, FuncFlags, FuncInfo, LogMessage, ParamInfo,
    ParamRefinement, Ty, Value,
};
use crate::mesh::tools;

pub struct FuncRevertMeshFaces;

impl Func for FuncRevertMeshFaces {
    fn info(&self) -> &FuncInfo {
        &FuncInfo {
            name: "Revert Faces",
            description: "REVERT MESH FACES\n\
                          \n\
                          Changes winding (vertex order) of all mesh faces.\n\
                          This is a maintenance operation that helps to \
                          synchronize properties of multiple meshes before \
                          hybridization. Some hybridization algorithms require \
                          the faces of input meshes to face the same direction \
                          (inwards or outwards). This operation is also useful \
                          after Synchronizing mesh faces, which can result in \
                          a mesh facing inwards.\n\
                          \n\
                          This operation does not affect normals and mesh rendering.\n\
                          \n\
                          The input mesh will be marked used and thus invisible in the viewport. \
                          It can still be used in subsequent operations.\n\
                          \n\
                          The resulting mesh geometry will be named 'Reverted Mesh'.",
            return_value_name: "Reverted Mesh",
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
        let mesh = args[0].unwrap_mesh();
        let analyze_mesh = args[1].unwrap_boolean();

        let value = tools::revert_mesh_faces(mesh);

        if analyze_mesh {
            analytics::report_bounding_box_analysis(&value, log);
            analytics::report_mesh_analysis(&value, log);
        }
        Ok(Value::Mesh(Arc::new(value)))
    }
}
