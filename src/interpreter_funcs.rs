use std::cmp;
use std::collections::HashMap;
use std::sync::Arc;

use nalgebra::base::Vector3;

use crate::convert::cast_u32;
use crate::geometry;
use crate::interpreter::{Func, FuncFlags, FuncIdent, ParamInfo, Ty, Value};
use crate::mesh_smoothing;
use crate::mesh_tools;
use crate::mesh_topology_analysis;
use crate::operations::shrink_wrap::{self, ShrinkWrapParams};
use crate::operations::transform;

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

pub struct FuncImplTransform;
impl Func for FuncImplTransform {
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
                ty: Ty::Float3,
                optional: true,
            },
            ParamInfo {
                ty: Ty::Float3,
                optional: true,
            },
            ParamInfo {
                ty: Ty::Float3,
                optional: true,
            },
        ]
    }

    fn return_ty(&self) -> Ty {
        Ty::Geometry
    }

    fn call(&self, args: &[Value]) -> Value {
        let geometry = args[0].unwrap_geometry();

        let translate = args[1].get_float3().map(Vector3::from);
        let rotate = args[2].get_float3().map(|rot| {
            [
                rot[0].to_radians(),
                rot[1].to_radians(),
                rot[2].to_radians(),
            ]
        });
        let scale = args[3].get_float3().map(Vector3::from);

        let value = transform::transform(
            &geometry,
            transform::TransformOptions {
                translate,
                rotate,
                scale,
            },
        );

        Value::Geometry(Arc::new(value))
    }
}

pub struct FuncImplLaplacianSmoothing;
impl Func for FuncImplLaplacianSmoothing {
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
        let geometry = args[0].unwrap_geometry();
        let iterations = args[1].unwrap_uint();
        let vertex_to_vertex_topology = mesh_topology_analysis::vertex_to_vertex_topology(geometry);

        let (g, _, _) = mesh_smoothing::laplacian_smoothing_with_anchors_full(
            geometry,
            vertex_to_vertex_topology,
            cmp::min(255, iterations),
            &[],
            false,
        );

        Value::Geometry(Arc::new(g))
    }
}

pub struct FuncImplSeparateIsolatedMeshes;
impl Func for FuncImplSeparateIsolatedMeshes {
    fn param_info(&self) -> &[ParamInfo] {
        &[ParamInfo {
            ty: Ty::Geometry,
            optional: false,
        }]
    }

    fn return_ty(&self) -> Ty {
        Ty::Geometry
    }

    fn call(&self, args: &[Value]) -> Value {
        let geometry = args[0].unwrap_geometry();

        let values = mesh_tools::separate_isolated_meshes(geometry);

        // FIXME: This returns a slice of Geometries. Return all of them
        Value::Geometry(Arc::new(values[0].clone()))
    }
}

// IMPORTANT: Do not change these IDs, ever! When adding a new
// function, always create a new, unique function identifier for it.

pub const FUNC_ID_CREATE_UV_SPHERE: FuncIdent = FuncIdent(0);
pub const FUNC_ID_SHRINK_WRAP: FuncIdent = FuncIdent(1);
pub const FUNC_ID_TRANSFORM: FuncIdent = FuncIdent(2);
pub const FUNC_ID_LAPLACIAN_SMOOTHING: FuncIdent = FuncIdent(3);
pub const FUNC_ID_SEPARATE_ISOLATED_MESHES: FuncIdent = FuncIdent(4);

/// The global set of function definitions available to the
/// interpreter and it's clients.
pub fn global_definitions() -> HashMap<FuncIdent, Box<dyn Func>> {
    let mut funcs: HashMap<FuncIdent, Box<dyn Func>> = HashMap::new();

    funcs.insert(FUNC_ID_CREATE_UV_SPHERE, Box::new(FuncImplCreateUvSphere));
    funcs.insert(FUNC_ID_SHRINK_WRAP, Box::new(FuncImplShrinkWrap));
    funcs.insert(FUNC_ID_TRANSFORM, Box::new(FuncImplTransform));
    funcs.insert(
        FUNC_ID_LAPLACIAN_SMOOTHING,
        Box::new(FuncImplLaplacianSmoothing),
    );
    funcs.insert(
        FUNC_ID_SEPARATE_ISOLATED_MESHES,
        Box::new(FuncImplSeparateIsolatedMeshes),
    );

    funcs
}
