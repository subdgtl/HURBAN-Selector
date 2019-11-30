use std::cmp;

use std::collections::BTreeMap;
use std::error;
use std::fmt;
use std::sync::Arc;

use nalgebra::base::Vector3;

use crate::edge_analysis;
use crate::geometry;
use crate::importer::{EndlessCache, Importer, ImporterError, ObjCache};
use crate::interpreter::{
    Float3ParamRefinement, FloatParamRefinement, Func, FuncError, FuncFlags, FuncIdent, FuncInfo,
    ParamInfo, ParamRefinement, Ty, UintParamRefinement, Value,
};
use crate::mesh_analysis;
use crate::mesh_smoothing;
use crate::mesh_tools;
use crate::mesh_topology_analysis;
use crate::operations::shrink_wrap::{self, ShrinkWrapParams};
use crate::operations::transform;

#[derive(Debug, PartialEq)]
pub enum FuncCreateUvSphereError {
    TooFewParallels { parallels_provided: u32 },
    TooFewMeridians { meridians_provided: u32 },
}

impl fmt::Display for FuncCreateUvSphereError {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        match self {
            FuncCreateUvSphereError::TooFewParallels { parallels_provided } => write!(
                f,
                "Create UV Sphere requires at least 2 parallels, but only {} provided",
                parallels_provided,
            ),
            FuncCreateUvSphereError::TooFewMeridians { meridians_provided } => write!(
                f,
                "Create UV Sphere requires at least 3 meridians, but only {} provided",
                meridians_provided,
            ),
        }
    }
}

impl error::Error for FuncCreateUvSphereError {}

pub struct FuncImplCreateUvSphere;

impl FuncImplCreateUvSphere {
    const MIN_PARALLELS: u32 = 2;
    const MIN_MERIDIANS: u32 = 3;
}

impl Func for FuncImplCreateUvSphere {
    fn info(&self) -> &FuncInfo {
        &FuncInfo {
            name: "Create UV Sphere",
            return_value_name: "Sphere",
        }
    }

    fn flags(&self) -> FuncFlags {
        FuncFlags::PURE
    }

    fn param_info(&self) -> &[ParamInfo] {
        &[
            ParamInfo {
                name: "Position",
                refinement: ParamRefinement::Float3(Float3ParamRefinement {
                    default_value_x: Some(0.0),
                    min_value_x: None,
                    max_value_x: None,
                    default_value_y: Some(0.0),
                    min_value_y: None,
                    max_value_y: None,
                    default_value_z: Some(0.0),
                    min_value_z: None,
                    max_value_z: None,
                }),
                optional: false,
            },
            ParamInfo {
                name: "Scale",
                refinement: ParamRefinement::Float(FloatParamRefinement {
                    default_value: Some(1.0),
                    min_value: Some(0.0),
                    max_value: None,
                }),
                optional: false,
            },
            ParamInfo {
                name: "Parallels",
                refinement: ParamRefinement::Uint(UintParamRefinement {
                    default_value: Some(Self::MIN_PARALLELS),
                    min_value: Some(Self::MIN_PARALLELS),
                    max_value: None,
                }),
                optional: false,
            },
            ParamInfo {
                name: "Meridians",
                refinement: ParamRefinement::Uint(UintParamRefinement {
                    default_value: Some(Self::MIN_MERIDIANS),
                    min_value: Some(Self::MIN_MERIDIANS),
                    max_value: None,
                }),
                optional: false,
            },
        ]
    }

    fn return_ty(&self) -> Ty {
        Ty::Geometry
    }

    fn call(&mut self, args: &[Value]) -> Result<Value, FuncError> {
        let position = args[0].unwrap_float3();
        let scale = args[1].unwrap_float();
        let n_parallels = args[2].unwrap_uint();
        let n_meridians = args[3].unwrap_uint();

        if n_parallels < Self::MIN_PARALLELS {
            return Err(FuncError::new(FuncCreateUvSphereError::TooFewParallels {
                parallels_provided: n_parallels,
            }));
        }

        if n_meridians < Self::MIN_MERIDIANS {
            return Err(FuncError::new(FuncCreateUvSphereError::TooFewMeridians {
                meridians_provided: n_meridians,
            }));
        }

        let value = geometry::uv_sphere(position, scale, n_parallels, n_meridians);
        Ok(Value::Geometry(Arc::new(value)))
    }
}

pub struct FuncImplShrinkWrap;
impl Func for FuncImplShrinkWrap {
    fn info(&self) -> &FuncInfo {
        &FuncInfo {
            name: "Shrinkwrap",
            return_value_name: "Shrinkwrapped Mesh",
        }
    }

    fn flags(&self) -> FuncFlags {
        FuncFlags::PURE
    }

    fn param_info(&self) -> &[ParamInfo] {
        &[
            ParamInfo {
                name: "Mesh",
                refinement: ParamRefinement::Geometry,
                optional: false,
            },
            ParamInfo {
                name: "Density",
                refinement: ParamRefinement::Uint(UintParamRefinement {
                    default_value: Some(10),
                    min_value: Some(FuncImplCreateUvSphere::MIN_MERIDIANS),
                    max_value: None,
                }),
                optional: false,
            },
        ]
    }

    fn return_ty(&self) -> Ty {
        Ty::Geometry
    }

    fn call(&mut self, args: &[Value]) -> Result<Value, FuncError> {
        let geometry = args[0].unwrap_geometry();
        let sphere_density = args[1].unwrap_uint();

        let value = shrink_wrap::shrink_wrap(ShrinkWrapParams {
            geometry,
            sphere_density,
        });
        Ok(Value::Geometry(Arc::new(value)))
    }
}

pub struct FuncImplTransform;
impl Func for FuncImplTransform {
    fn info(&self) -> &FuncInfo {
        &FuncInfo {
            name: "Transform",
            return_value_name: "Transformed Mesh",
        }
    }

    fn flags(&self) -> FuncFlags {
        FuncFlags::PURE
    }

    fn param_info(&self) -> &[ParamInfo] {
        &[
            ParamInfo {
                name: "Mesh",
                refinement: ParamRefinement::Geometry,
                optional: false,
            },
            ParamInfo {
                name: "Translate",
                refinement: ParamRefinement::Float3(Float3ParamRefinement {
                    default_value_x: Some(0.0),
                    min_value_x: None,
                    max_value_x: None,
                    default_value_y: Some(0.0),
                    min_value_y: None,
                    max_value_y: None,
                    default_value_z: Some(0.0),
                    min_value_z: None,
                    max_value_z: None,
                }),
                optional: true,
            },
            ParamInfo {
                name: "Rotate (deg)",
                refinement: ParamRefinement::Float3(Float3ParamRefinement {
                    default_value_x: Some(0.0),
                    min_value_x: None,
                    max_value_x: None,
                    default_value_y: Some(0.0),
                    min_value_y: None,
                    max_value_y: None,
                    default_value_z: Some(0.0),
                    min_value_z: None,
                    max_value_z: None,
                }),
                optional: true,
            },
            ParamInfo {
                name: "Scale",
                refinement: ParamRefinement::Float3(Float3ParamRefinement {
                    default_value_x: Some(1.0),
                    min_value_x: None,
                    max_value_x: None,
                    default_value_y: Some(1.0),
                    min_value_y: None,
                    max_value_y: None,
                    default_value_z: Some(1.0),
                    min_value_z: None,
                    max_value_z: None,
                }),
                optional: true,
            },
        ]
    }

    fn return_ty(&self) -> Ty {
        Ty::Geometry
    }

    fn call(&mut self, args: &[Value]) -> Result<Value, FuncError> {
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
            geometry,
            transform::TransformOptions {
                translate,
                rotate,
                scale,
            },
        );
        Ok(Value::Geometry(Arc::new(value)))
    }
}

pub struct FuncImplLaplacianSmoothing;
impl Func for FuncImplLaplacianSmoothing {
    fn info(&self) -> &FuncInfo {
        &FuncInfo {
            name: "Laplacian Smoothing",
            return_value_name: "Smoothed Mesh",
        }
    }

    fn flags(&self) -> FuncFlags {
        FuncFlags::PURE
    }

    fn param_info(&self) -> &[ParamInfo] {
        &[
            ParamInfo {
                name: "Mesh",
                refinement: ParamRefinement::Geometry,
                optional: false,
            },
            ParamInfo {
                name: "Iterations",
                refinement: ParamRefinement::Uint(UintParamRefinement {
                    default_value: Some(1),
                    min_value: Some(0),
                    max_value: Some(255),
                }),
                optional: false,
            },
        ]
    }

    fn return_ty(&self) -> Ty {
        Ty::Geometry
    }

    fn call(&mut self, args: &[Value]) -> Result<Value, FuncError> {
        let geometry = args[0].unwrap_geometry();
        let iterations = args[1].unwrap_uint();

        let v2v = mesh_topology_analysis::vertex_to_vertex_topology(geometry);

        let (value, _, _) = mesh_smoothing::laplacian_smoothing(
            geometry,
            &v2v,
            cmp::min(255, iterations),
            &[],
            false,
        );
        Ok(Value::Geometry(Arc::new(value)))
    }
}

pub struct FuncImplSeparateIsolatedMeshes;
impl Func for FuncImplSeparateIsolatedMeshes {
    fn info(&self) -> &FuncInfo {
        &FuncInfo {
            name: "Separate Volumes",
            return_value_name: "Separated Mesh",
        }
    }

    fn flags(&self) -> FuncFlags {
        FuncFlags::empty()
    }

    fn param_info(&self) -> &[ParamInfo] {
        &[ParamInfo {
            name: "Mesh",
            refinement: ParamRefinement::Geometry,
            optional: false,
        }]
    }

    fn return_ty(&self) -> Ty {
        Ty::Geometry
    }

    fn call(&mut self, args: &[Value]) -> Result<Value, FuncError> {
        let geometry = args[0].unwrap_geometry();

        let values = mesh_tools::separate_isolated_meshes(&geometry);

        // FIXME: This returns a slice of Geometries. Return all of them
        let first_value = values
            .into_iter()
            .next()
            .expect("Need at least one geometry");
        Ok(Value::Geometry(Arc::new(first_value)))
    }
}

pub struct FuncImplJoinMeshes;
impl Func for FuncImplJoinMeshes {
    fn info(&self) -> &FuncInfo {
        &FuncInfo {
            name: "Join Meshes",
            return_value_name: "Joined Mesh",
        }
    }

    fn flags(&self) -> FuncFlags {
        FuncFlags::PURE
    }

    fn param_info(&self) -> &[ParamInfo] {
        &[
            ParamInfo {
                name: "Mesh 1",
                refinement: ParamRefinement::Geometry,
                optional: false,
            },
            ParamInfo {
                name: "Mesh 2",
                refinement: ParamRefinement::Geometry,
                optional: false,
            },
        ]
    }

    fn return_ty(&self) -> Ty {
        Ty::Geometry
    }

    fn call(&mut self, args: &[Value]) -> Result<Value, FuncError> {
        let first_geometry = args[0].unwrap_geometry();
        let second_geometry = args[1].unwrap_geometry();

        let value = mesh_tools::join_meshes(first_geometry, second_geometry);
        Ok(Value::Geometry(Arc::new(value)))
    }
}

pub struct FuncImplWeld;
impl Func for FuncImplWeld {
    fn info(&self) -> &FuncInfo {
        &FuncInfo {
            name: "Weld",
            return_value_name: "Welded Mesh",
        }
    }

    fn flags(&self) -> FuncFlags {
        FuncFlags::empty()
    }

    fn param_info(&self) -> &[ParamInfo] {
        &[
            ParamInfo {
                name: "Mesh",
                refinement: ParamRefinement::Geometry,
                optional: false,
            },
            ParamInfo {
                name: "Tolerance",
                refinement: ParamRefinement::Float(FloatParamRefinement {
                    default_value: Some(1.0),
                    min_value: Some(0.0),
                    max_value: None,
                }),
                optional: false,
            },
        ]
    }

    fn return_ty(&self) -> Ty {
        Ty::Geometry
    }

    fn call(&mut self, args: &[Value]) -> Result<Value, FuncError> {
        let geometry = args[0].unwrap_geometry();
        let tolerance = args[1].unwrap_float();

        let value = mesh_tools::weld(geometry, tolerance);
        Ok(Value::Geometry(Arc::new(value)))
    }
}

pub struct FuncImplLoopSubdivision;
impl Func for FuncImplLoopSubdivision {
    fn info(&self) -> &FuncInfo {
        &FuncInfo {
            name: "Loop Subdivision",
            return_value_name: "Subdivided Mesh",
        }
    }

    fn flags(&self) -> FuncFlags {
        FuncFlags::PURE
    }

    fn param_info(&self) -> &[ParamInfo] {
        &[
            ParamInfo {
                name: "Mesh",
                refinement: ParamRefinement::Geometry,
                optional: false,
            },
            ParamInfo {
                name: "Iterations",
                refinement: ParamRefinement::Uint(UintParamRefinement {
                    default_value: Some(1),
                    min_value: Some(0),
                    max_value: Some(5),
                }),
                optional: false,
            },
        ]
    }

    fn return_ty(&self) -> Ty {
        Ty::Geometry
    }

    fn call(&mut self, args: &[Value]) -> Result<Value, FuncError> {
        // FIXME: add the max value to the param info so that that the
        // gui doesn't mislead
        const MAX_ITERATIONS: u32 = 3;

        let geometry = args[0].unwrap_refcounted_geometry();
        let iterations = cmp::min(args[1].unwrap_uint(), MAX_ITERATIONS);

        if iterations == 0 {
            return Ok(Value::Geometry(geometry));
        }

        let mut v2v = mesh_topology_analysis::vertex_to_vertex_topology(&geometry);
        let mut f2f = mesh_topology_analysis::face_to_face_topology(&geometry);
        let mut current_geometry = mesh_smoothing::loop_subdivision(&geometry, &v2v, &f2f);

        for _ in 1..iterations {
            v2v = mesh_topology_analysis::vertex_to_vertex_topology(&current_geometry);
            f2f = mesh_topology_analysis::face_to_face_topology(&current_geometry);
            current_geometry = mesh_smoothing::loop_subdivision(&current_geometry, &v2v, &f2f);
        }

        Ok(Value::Geometry(Arc::new(current_geometry)))
    }
}

pub struct FuncImplCreatePlane;
impl Func for FuncImplCreatePlane {
    fn info(&self) -> &FuncInfo {
        &FuncInfo {
            name: "Create Plane",
            return_value_name: "Plane",
        }
    }

    fn flags(&self) -> FuncFlags {
        FuncFlags::PURE
    }

    fn param_info(&self) -> &[ParamInfo] {
        &[
            ParamInfo {
                name: "Position",
                refinement: ParamRefinement::Float3(Float3ParamRefinement {
                    default_value_x: Some(0.0),
                    min_value_x: None,
                    max_value_x: None,
                    default_value_y: Some(0.0),
                    min_value_y: None,
                    max_value_y: None,
                    default_value_z: Some(0.0),
                    min_value_z: None,
                    max_value_z: None,
                }),
                optional: true,
            },
            ParamInfo {
                name: "Scale",
                refinement: ParamRefinement::Float(FloatParamRefinement {
                    default_value: Some(1.0),
                    min_value: Some(0.0),
                    max_value: None,
                }),
                optional: true,
            },
        ]
    }

    fn return_ty(&self) -> Ty {
        Ty::Geometry
    }

    fn call(&mut self, values: &[Value]) -> Result<Value, FuncError> {
        let position = values[0].get_float3().unwrap_or([0.0; 3]);
        let scale = values[1].get_float().unwrap_or(1.0);

        let value = geometry::plane(position, scale);
        Ok(Value::Geometry(Arc::new(value)))
    }
}

#[derive(Debug, PartialEq)]
pub enum FuncImportObjMeshError {
    Empty,
    Importer(ImporterError),
}

impl fmt::Display for FuncImportObjMeshError {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        match self {
            Self::Empty => write!(f, "No mesh geometry contained in OBJ"),
            Self::Importer(importer_error) => f.write_str(&importer_error.to_string()),
        }
    }
}

impl error::Error for FuncImportObjMeshError {}

pub struct FuncImplImportObjMesh<C: ObjCache> {
    importer: Importer<C>,
}

impl<C: ObjCache> Func for FuncImplImportObjMesh<C> {
    fn info(&self) -> &FuncInfo {
        &FuncInfo {
            name: "Import OBJ",
            return_value_name: "Imported Mesh",
        }
    }

    fn flags(&self) -> FuncFlags {
        FuncFlags::empty()
    }

    fn param_info(&self) -> &[ParamInfo] {
        &[ParamInfo {
            name: "Path",
            refinement: ParamRefinement::String,
            optional: false,
        }]
    }

    fn return_ty(&self) -> Ty {
        Ty::Geometry
    }

    fn call(&mut self, values: &[Value]) -> Result<Value, FuncError> {
        let path = values[0].unwrap_string();

        let result = self.importer.import_obj(path);
        match result {
            Ok(models) => {
                // FIXME: @Correctness Join all meshes into one once
                // we have join implemented for more than just 2
                // meshes

                let first_model = models.into_iter().next();
                if let Some(first_model) = first_model {
                    Ok(Value::Geometry(Arc::new(first_model.geometry)))
                } else {
                    Err(FuncError::new(FuncImportObjMeshError::Empty))
                }
            }
            Err(err) => Err(FuncError::new(FuncImportObjMeshError::Importer(err))),
        }
    }
}

pub struct FuncImplRevertMeshFaces;
impl Func for FuncImplRevertMeshFaces {
    fn info(&self) -> &FuncInfo {
        &FuncInfo {
            name: "Revert Faces",
            return_value_name: "Reverted Mesh",
        }
    }

    fn flags(&self) -> FuncFlags {
        FuncFlags::PURE
    }

    fn param_info(&self) -> &[ParamInfo] {
        &[ParamInfo {
            name: "Mesh",
            refinement: ParamRefinement::Geometry,
            optional: false,
        }]
    }

    fn return_ty(&self) -> Ty {
        Ty::Geometry
    }

    fn call(&mut self, args: &[Value]) -> Result<Value, FuncError> {
        let geometry = args[0].unwrap_geometry();

        let value = mesh_tools::revert_mesh_faces(geometry);
        Ok(Value::Geometry(Arc::new(value)))
    }
}

pub struct FuncImplSynchronizeMeshFaces;
impl Func for FuncImplSynchronizeMeshFaces {
    fn info(&self) -> &FuncInfo {
        &FuncInfo {
            name: "Synchronize Faces",
            return_value_name: "Synchronized Mesh",
        }
    }

    fn flags(&self) -> FuncFlags {
        FuncFlags::PURE
    }

    fn param_info(&self) -> &[ParamInfo] {
        &[ParamInfo {
            name: "Mesh",
            refinement: ParamRefinement::Geometry,
            optional: false,
        }]
    }

    fn return_ty(&self) -> Ty {
        Ty::Geometry
    }

    fn call(&mut self, args: &[Value]) -> Result<Value, FuncError> {
        let geometry = args[0].unwrap_refcounted_geometry();

        let oriented_edges: Vec<_> = geometry.oriented_edges_iter().collect();
        let edge_sharing_map = edge_analysis::edge_sharing(&oriented_edges);

        if !mesh_analysis::is_mesh_orientable(&edge_sharing_map)
            && mesh_analysis::is_mesh_manifold(&edge_sharing_map)
        {
            let face_to_face = mesh_topology_analysis::face_to_face_topology(&geometry);

            let value = Arc::new(mesh_tools::synchronize_mesh_winding(
                &geometry,
                &face_to_face,
            ));

            Ok(Value::Geometry(value))
        } else {
            Ok(Value::Geometry(geometry))
        }
    }
}

// IMPORTANT: Do not change these IDs, ever! When adding a new
// function, always create a new, unique function identifier for it.
// Also note: the number in the identifier currently also defines the
// order of the operation in the UI.

// Special funcs
pub const FUNC_ID_TRANSFORM: FuncIdent = FuncIdent(0000);

// Create funcs
pub const FUNC_ID_CREATE_UV_SPHERE: FuncIdent = FuncIdent(1000);
pub const FUNC_ID_CREATE_PLANE: FuncIdent = FuncIdent(1001);

// Import/Export funcs
pub const FUNC_ID_IMPORT_OBJ_MESH: FuncIdent = FuncIdent(2000);

// Smoothing funcs
pub const FUNC_ID_LAPLACIAN_SMOOTHING: FuncIdent = FuncIdent(3000);
pub const FUNC_ID_LOOP_SUBDIVISION: FuncIdent = FuncIdent(3001);

// Tool funcs
pub const FUNC_ID_SHRINK_WRAP: FuncIdent = FuncIdent(9000);
pub const FUNC_ID_SEPARATE_ISOLATED_MESHES: FuncIdent = FuncIdent(9001);
pub const FUNC_ID_JOIN_MESHES: FuncIdent = FuncIdent(9002);
pub const FUNC_ID_WELD: FuncIdent = FuncIdent(9003);
pub const FUNC_ID_REVERT_MESH_FACES: FuncIdent = FuncIdent(9004);
pub const FUNC_ID_SYNCHRONIZE_MESH_FACES: FuncIdent = FuncIdent(9005);

/// Returns the global set of function definitions available to the
/// editor.
///
/// Note that since funcs can have internal state such as a cache or
/// random state, two instances of the function table are not always
/// equivalent.
pub fn create_function_table() -> BTreeMap<FuncIdent, Box<dyn Func>> {
    let mut funcs: BTreeMap<FuncIdent, Box<dyn Func>> = BTreeMap::new();

    // Special funcs
    funcs.insert(FUNC_ID_TRANSFORM, Box::new(FuncImplTransform));

    // Create funcs
    funcs.insert(FUNC_ID_CREATE_UV_SPHERE, Box::new(FuncImplCreateUvSphere));
    funcs.insert(FUNC_ID_CREATE_PLANE, Box::new(FuncImplCreatePlane));

    // Import/Export funcs
    funcs.insert(
        FUNC_ID_IMPORT_OBJ_MESH,
        Box::new(FuncImplImportObjMesh {
            importer: Importer::new(EndlessCache::default()),
        }),
    );

    // Smoothing funcs
    funcs.insert(
        FUNC_ID_LAPLACIAN_SMOOTHING,
        Box::new(FuncImplLaplacianSmoothing),
    );
    funcs.insert(FUNC_ID_LOOP_SUBDIVISION, Box::new(FuncImplLoopSubdivision));

    // Tool funcs
    funcs.insert(FUNC_ID_SHRINK_WRAP, Box::new(FuncImplShrinkWrap));
    funcs.insert(
        FUNC_ID_SEPARATE_ISOLATED_MESHES,
        Box::new(FuncImplSeparateIsolatedMeshes),
    );
    funcs.insert(FUNC_ID_JOIN_MESHES, Box::new(FuncImplJoinMeshes));
    funcs.insert(FUNC_ID_WELD, Box::new(FuncImplWeld));
    funcs.insert(FUNC_ID_REVERT_MESH_FACES, Box::new(FuncImplRevertMeshFaces));
    funcs.insert(
        FUNC_ID_SYNCHRONIZE_MESH_FACES,
        Box::new(FuncImplSynchronizeMeshFaces),
    );

    funcs
}
