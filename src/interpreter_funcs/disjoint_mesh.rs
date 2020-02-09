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
            description: "",
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
                description: "",
                refinement: ParamRefinement::Mesh,
                optional: false,
            },
            ParamInfo {
                name: "Analyze resulting group",
                description: "",
                refinement: ParamRefinement::Boolean(BooleanParamRefinement {
                    default_value: false,
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
