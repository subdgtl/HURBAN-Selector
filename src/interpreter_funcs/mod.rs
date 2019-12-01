use std::collections::BTreeMap;

use crate::importer::{EndlessCache, Importer};
use crate::interpreter::{Func, FuncIdent};

use self::create_plane::FuncCreatePlane;
use self::create_uv_sphere::FuncCreateUvSphere;
use self::import_obj_mesh::FuncImportObjMesh;
use self::join_meshes::FuncJoinMeshes;
use self::laplacian_smoothing::FuncLaplacianSmoothing;
use self::loop_subdivision::FuncLoopSubdivision;
use self::revert_mesh_faces::FuncRevertMeshFaces;
use self::separate_isolated_meshes::FuncSeparateIsolatedMeshes;
use self::shrink_wrap::FuncShrinkWrap;
use self::synchronize_mesh_faces::FuncSynchronizeMeshFaces;
use self::transform::FuncTransform;
use self::weld::FuncWeld;

mod create_plane;
mod create_uv_sphere;
mod import_obj_mesh;
mod join_meshes;
mod laplacian_smoothing;
mod loop_subdivision;
mod revert_mesh_faces;
mod separate_isolated_meshes;
mod shrink_wrap;
mod synchronize_mesh_faces;
mod transform;
mod weld;

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
    funcs.insert(FUNC_ID_TRANSFORM, Box::new(FuncTransform));

    // Create funcs
    funcs.insert(FUNC_ID_CREATE_UV_SPHERE, Box::new(FuncCreateUvSphere));
    funcs.insert(FUNC_ID_CREATE_PLANE, Box::new(FuncCreatePlane));

    // Import/Export funcs
    funcs.insert(
        FUNC_ID_IMPORT_OBJ_MESH,
        Box::new(FuncImportObjMesh::new(Importer::new(
            EndlessCache::default(),
        ))),
    );

    // Smoothing funcs
    funcs.insert(
        FUNC_ID_LAPLACIAN_SMOOTHING,
        Box::new(FuncLaplacianSmoothing),
    );
    funcs.insert(FUNC_ID_LOOP_SUBDIVISION, Box::new(FuncLoopSubdivision));

    // Tool funcs
    funcs.insert(FUNC_ID_SHRINK_WRAP, Box::new(FuncShrinkWrap));
    funcs.insert(
        FUNC_ID_SEPARATE_ISOLATED_MESHES,
        Box::new(FuncSeparateIsolatedMeshes),
    );
    funcs.insert(FUNC_ID_JOIN_MESHES, Box::new(FuncJoinMeshes));
    funcs.insert(FUNC_ID_WELD, Box::new(FuncWeld));
    funcs.insert(FUNC_ID_REVERT_MESH_FACES, Box::new(FuncRevertMeshFaces));
    funcs.insert(
        FUNC_ID_SYNCHRONIZE_MESH_FACES,
        Box::new(FuncSynchronizeMeshFaces),
    );

    funcs
}
