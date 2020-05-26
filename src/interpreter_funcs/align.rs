use std::sync::Arc;

use crate::analytics;
use crate::interpreter::{
    BooleanParamRefinement, Func, FuncError, FuncFlags, FuncInfo, LogMessage, ParamInfo,
    ParamRefinement, Ty, Value,
};
use crate::mesh::tools;

pub struct FuncAlign;

impl Func for FuncAlign {
    fn info(&self) -> &FuncInfo {
        &FuncInfo {
            name: "Align",
            description:
                "ALIGN MESH TO ANOTHER\n\
                 \n\
                 Moves and scales a mesh to match another mesh. 'Mesh to align' will be \
                 translated so that its center matches the center of 'Align to mesh'. \
                 'Mesh to align' will be scaled so that the diagonal of its bounding \
                 box will have the same length as the diagonal of the bounding box \
                 of 'Align to mesh'.\n\
                 \n\
                 Both input meshes will be marked used and thus invisible in the viewport. \
                 They can still be used in subsequent operations.\n\
                 \n\
                 The resulting mesh geometry will be named 'Aligned Mesh'.",
            return_value_name: "Aligned Mesh",
        }
    }

    fn flags(&self) -> FuncFlags {
        FuncFlags::PURE
    }

    fn param_info(&self) -> &[ParamInfo] {
        &[
            ParamInfo {
                name: "Mesh to align",
                description: "Mesh to be transformed to align to another mesh.",
                refinement: ParamRefinement::Mesh,
                optional: false,
            },
            ParamInfo {
                name: "Align to mesh",
                description: "The target mesh, towards which the source mesh should be aligned.",
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
        let mesh_to_align = args[0].unwrap_mesh();
        let align_to_mesh = args[1].unwrap_mesh();
        let analyze_mesh = args[2].unwrap_boolean();

        let value = tools::align_two_meshes(&mesh_to_align, &align_to_mesh);

        if analyze_mesh {
            analytics::report_bounding_box_analysis(&value, log);
            analytics::report_mesh_analysis(&value, log);
        }

        Ok(Value::Mesh(Arc::new(value)))
    }
}
