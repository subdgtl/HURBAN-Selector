use std::sync::Arc;

use crate::analytics;
use crate::interpreter::{
    BooleanParamRefinement, Func, FuncError, FuncFlags, FuncInfo, LogMessage, ParamInfo,
    ParamRefinement, Ty, Value,
};
use crate::mesh::tools;

pub struct FuncJoinGroup;

impl Func for FuncJoinGroup {
    fn info(&self) -> &FuncInfo {
        &FuncInfo {
            name: "Join Group",
            description: "JOIN MESH GROUP INTO A SINGLE MESH\n\
                 \n\
                 Joins all mesh geometries from a mesh group into a single mesh. \
                 Creates a new mesh containing vertices and triangles \
                 from all meshes in the mesh group. \
                 The meshes will not be welded.\n\
                 \n\
                 Mesh group is displayed in the viewport as geometry but is \
                 a distinct data type. Only some operations, such as this one, \
                 can use mesh groups and most of them are intended to generate a proper mesh \
                 from the mesh group.\n\
                 \n\
                 The resulting mesh geometry will be named 'Joined Mesh'.",
            return_value_name: "Joined Mesh",
        }
    }

    fn flags(&self) -> FuncFlags {
        FuncFlags::PURE
    }

    fn param_info(&self) -> &[ParamInfo] {
        &[
            ParamInfo {
                name: "Group",
                description: "Input mesh group.",
                refinement: ParamRefinement::MeshArray,
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
        let mesh_arc_array = args[0].unwrap_mesh_array();
        let analyze_mesh = args[1].unwrap_boolean();

        let meshes = mesh_arc_array.iter();
        let value = tools::join_multiple_meshes(meshes);

        if analyze_mesh {
            analytics::report_bounding_box_analysis(&value, log);
            analytics::report_mesh_analysis(&value, log);
        }

        Ok(Value::Mesh(Arc::new(value)))
    }
}
