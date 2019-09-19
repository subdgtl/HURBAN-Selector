use std::sync::Arc;

use crate::convert::cast_u32;
use crate::geometry;
use crate::interpreter::{Func, FuncFlags, ParamInfo, Ty, Value};
use crate::operations::shrink_wrap::{self, ShrinkWrapParams};

pub struct FuncImplCreateUvSphere;
impl Func for FuncImplCreateUvSphere {
    fn flags(&self) -> FuncFlags {
        FuncFlags::PURE
    }

    fn param_info(&self) -> &[ParamInfo] {
        &[
            ParamInfo {
                ty: Ty::Float,
                optional: false,
            },
            ParamInfo {
                ty: Ty::Uint,
                optional: false,
            },
            ParamInfo {
                ty: Ty::Uint,
                optional: false,
            },
        ]
    }

    fn return_ty(&self) -> Ty {
        Ty::Geometry
    }

    fn call(&self, args: &[Value]) -> Value {
        let scale = args[0].unwrap_float();
        let n_parallels = args[1].unwrap_uint();
        let n_meridians = args[2].unwrap_uint();

        let value = geometry::uv_sphere([0.0, 0.0, 0.0], scale, n_parallels, n_meridians);

        Value::Geometry(Arc::new(value))
    }
}

pub struct FuncImplShrinkWrap;
impl Func for FuncImplShrinkWrap {
    fn flags(&self) -> FuncFlags {
        FuncFlags::PURE
    }

    fn param_info(&self) -> &[ParamInfo] {
        &[
            ParamInfo {
                ty: Ty::Geometry,
                optional: false,
            },
            ParamInfo {
                ty: Ty::Uint,
                optional: false,
            },
        ]
    }

    fn return_ty(&self) -> Ty {
        Ty::Geometry
    }

    fn call(&self, args: &[Value]) -> Value {
        let value = shrink_wrap::shrink_wrap(ShrinkWrapParams {
            geometry: args[0].unwrap_geometry(),
            sphere_density: cast_u32(args[1].unwrap_uint()),
        });

        Value::Geometry(Arc::new(value))
    }
}

pub fn global_definitions() -> Vec<Box<dyn Func>> {
    // IMPORTANT: Do not re-order, ever!
    // The index into this array serves as the unique function
    // identifier throughout the program.
    vec![
        Box::new(FuncImplCreateUvSphere),
        Box::new(FuncImplShrinkWrap),
    ]
}
