use std::sync::Arc;

use crate::analytics;
use crate::interpreter::{
    BooleanParamRefinement, Func, FuncError, FuncFlags, FuncInfo, LogMessage, MeshArrayValue,
    ParamInfo, ParamRefinement, Ty, Value,
};
use crate::mesh::tools;

pub struct FuncDisjointMesh;

impl Func for FuncDisjointMesh {
    fn info(&self) -> &FuncInfo {
        &FuncInfo {
            name: "Disjoint Mesh",
            description:
                "DISJOINT MESH INTO A MESH GROUP\n\
                 \n\
                 Splits the visually separate / unwelded / island geometries \
                 from their common mesh geometry and stores the resulting \
                 separate mesh geometries in a mesh group.\n\
                 \n\
                 Mesh group is displayed in the viewport as geometry but is \
                 a distinct data type. Only some operations can use mesh groups \
                 and most of them are intended to generate a proper mesh from the mesh group.\n\
                 To use the content of the group it is necessary to Extract specific \
                 mesh from group, Extract largest mesh from group or Join mesh group \
                 into a single mesh.\n\
                 \n\
                 The resulting mesh group will be named 'Disjoint Group'.",
            return_value_name: "Disjoint Group",
        }
    }

    fn flags(&self) -> FuncFlags {
        FuncFlags::empty()
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
                name: "Analyze resulting group",
                description: "Reports detailed analytic information on the mesh group.",
                refinement: ParamRefinement::Boolean(BooleanParamRefinement {
                    default_value: true,
                }),
                optional: false,
            },
        ]
    }

    fn return_ty(&self) -> Ty {
        Ty::MeshArray
    }

    fn call(
        &mut self,
        args: &[Value],
        log: &mut dyn FnMut(LogMessage),
    ) -> Result<Value, FuncError> {
        let mesh = args[0].unwrap_mesh();
        let analyze = args[1].unwrap_boolean();

        let meshes = tools::disjoint_mesh(&mesh);
        let value = MeshArrayValue::new(meshes.into_iter().map(Arc::new).collect());

        if analyze {
            analytics::report_group_analysis(&value, log);
        }

        Ok(Value::MeshArray(Arc::new(value)))
    }
}
