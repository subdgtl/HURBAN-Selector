use std::collections::BTreeMap;

use crate::importer::{EndlessCache, Importer};
use crate::interpreter::{Func, FuncIdent};

use self::align::FuncAlign;
use self::create_box::FuncCreateBox;
use self::create_plane::FuncCreatePlane;
use self::create_uv_sphere::FuncCreateUvSphere;
use self::disjoint_mesh::FuncDisjointMesh;
use self::extract::FuncExtract;
use self::extract_largest::FuncExtractLargest;
use self::import_obj_mesh::FuncImportObjMesh;
use self::join_group::FuncJoinGroup;
use self::join_meshes::FuncJoinMeshes;
use self::laplacian_smoothing::FuncLaplacianSmoothing;
use self::loop_subdivision::FuncLoopSubdivision;
use self::revert_mesh_faces::FuncRevertMeshFaces;
use self::snap_to_ground::FuncSnapToGround;
use self::synchronize_mesh_faces::FuncSynchronizeMeshFaces;
use self::transform::FuncTransform;
use self::voxel_boolean_difference::FuncBooleanDifference;
use self::voxel_boolean_intersection::FuncBooleanIntersection;
use self::voxel_boolean_union::FuncBooleanUnion;
use self::voxel_transform::FuncVoxelTransform;
use self::voxelize::FuncVoxelize;
use self::weld::FuncWeld;

mod align;
mod create_box;
mod create_plane;
mod create_uv_sphere;
mod disjoint_mesh;
mod extract;
mod extract_largest;
mod import_obj_mesh;
mod join_group;
mod join_meshes;
mod laplacian_smoothing;
mod loop_subdivision;
mod revert_mesh_faces;
mod snap_to_ground;
mod synchronize_mesh_faces;
mod transform;
mod voxel_boolean_difference;
mod voxel_boolean_intersection;
mod voxel_boolean_union;
mod voxel_transform;
mod voxelize;
mod weld;

// IMPORTANT: Do not change these IDs, ever! When adding a new
// function, always create a new, unique function identifier for it.
// Also note: the number in the identifier currently also defines the
// order of the operation in the UI.

// Manipulation funcs
pub const FUNC_ID_TRANSFORM: FuncIdent = FuncIdent(0);
pub const FUNC_ID_EXTRACT: FuncIdent = FuncIdent(1);
pub const FUNC_ID_EXTRACT_LARGEST: FuncIdent = FuncIdent(2);
pub const FUNC_ID_SNAP_TO_GROUND: FuncIdent = FuncIdent(3);
pub const FUNC_ID_ALIGN: FuncIdent = FuncIdent(4);

// Create funcs
pub const FUNC_ID_CREATE_UV_SPHERE: FuncIdent = FuncIdent(1000);
pub const FUNC_ID_CREATE_PLANE: FuncIdent = FuncIdent(1001);
pub const FUNC_ID_CREATE_BOX: FuncIdent = FuncIdent(1002);

// Import/Export funcs
pub const FUNC_ID_IMPORT_OBJ_MESH: FuncIdent = FuncIdent(2000);

// Smoothing funcs
pub const FUNC_ID_LAPLACIAN_SMOOTHING: FuncIdent = FuncIdent(3000);
pub const FUNC_ID_LOOP_SUBDIVISION: FuncIdent = FuncIdent(3001);

// Tool funcs
// FIXME: Fill id 9000
pub const FUNC_ID_DISJOINT_MESH: FuncIdent = FuncIdent(9001);
pub const FUNC_ID_JOIN_MESHES: FuncIdent = FuncIdent(9002);
pub const FUNC_ID_WELD: FuncIdent = FuncIdent(9003);
pub const FUNC_ID_REVERT_MESH_FACES: FuncIdent = FuncIdent(9004);
pub const FUNC_ID_SYNCHRONIZE_MESH_FACES: FuncIdent = FuncIdent(9005);
pub const FUNC_ID_JOIN_GROUP: FuncIdent = FuncIdent(9006);
pub const FUNC_ID_VOXELIZE: FuncIdent = FuncIdent(9007);
pub const FUNC_ID_BOOLEAN_INTERSECTION: FuncIdent = FuncIdent(9008);
pub const FUNC_ID_BOOLEAN_DIFFERENCE: FuncIdent = FuncIdent(9009);
pub const FUNC_ID_BOOLEAN_UNION: FuncIdent = FuncIdent(9010);
pub const FUNC_ID_VOXEL_TRANSFORM: FuncIdent = FuncIdent(9011);

/// Returns the global set of function definitions available to the
/// editor.
///
/// Note that since funcs can have internal state such as a cache or
/// random state, two instances of the function table are not always
/// equivalent.
pub fn create_function_table() -> BTreeMap<FuncIdent, Box<dyn Func>> {
    let mut funcs: BTreeMap<FuncIdent, Box<dyn Func>> = BTreeMap::new();

    // Manipulation funcs
    funcs.insert(FUNC_ID_TRANSFORM, Box::new(FuncTransform));
    funcs.insert(FUNC_ID_EXTRACT, Box::new(FuncExtract));
    funcs.insert(FUNC_ID_EXTRACT_LARGEST, Box::new(FuncExtractLargest));
    funcs.insert(FUNC_ID_SNAP_TO_GROUND, Box::new(FuncSnapToGround));
    funcs.insert(FUNC_ID_ALIGN, Box::new(FuncAlign));

    // Create funcs
    funcs.insert(FUNC_ID_CREATE_UV_SPHERE, Box::new(FuncCreateUvSphere));
    funcs.insert(FUNC_ID_CREATE_PLANE, Box::new(FuncCreatePlane));
    funcs.insert(FUNC_ID_CREATE_BOX, Box::new(FuncCreateBox));

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
    funcs.insert(FUNC_ID_DISJOINT_MESH, Box::new(FuncDisjointMesh));
    funcs.insert(FUNC_ID_JOIN_MESHES, Box::new(FuncJoinMeshes));
    funcs.insert(FUNC_ID_WELD, Box::new(FuncWeld));
    funcs.insert(FUNC_ID_REVERT_MESH_FACES, Box::new(FuncRevertMeshFaces));
    funcs.insert(
        FUNC_ID_SYNCHRONIZE_MESH_FACES,
        Box::new(FuncSynchronizeMeshFaces),
    );
    funcs.insert(FUNC_ID_JOIN_GROUP, Box::new(FuncJoinGroup));
    funcs.insert(FUNC_ID_VOXELIZE, Box::new(FuncVoxelize));
    funcs.insert(
        FUNC_ID_BOOLEAN_INTERSECTION,
        Box::new(FuncBooleanIntersection),
    );
    funcs.insert(FUNC_ID_BOOLEAN_DIFFERENCE, Box::new(FuncBooleanDifference));
    funcs.insert(FUNC_ID_BOOLEAN_UNION, Box::new(FuncBooleanUnion));
    funcs.insert(FUNC_ID_VOXEL_TRANSFORM, Box::new(FuncVoxelTransform));

    funcs
}
