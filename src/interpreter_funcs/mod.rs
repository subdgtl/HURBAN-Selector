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
use self::voxel_interpolated_union::FuncInterpolatedUnion;
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
mod voxel_interpolated_union;
mod voxel_transform;
mod voxelize;
mod weld;

// IMPORTANT: Do not change these IDs, ever! When adding a new
// function, always create a new, unique function identifier for it.
// Also note: the number in the identifier currently also defines the
// order of the operation in the UI.

// Creation funcs: 0xxx
pub const FUNC_ID_CREATE_PLANE: FuncIdent = FuncIdent(0);
pub const FUNC_ID_CREATE_BOX: FuncIdent = FuncIdent(1);
pub const FUNC_ID_CREATE_UV_SPHERE: FuncIdent = FuncIdent(2);

// Import/Export funcs: 2xxx
pub const FUNC_ID_IMPORT_OBJ_MESH: FuncIdent = FuncIdent(2000);
pub const FUNC_ID_EXTRACT: FuncIdent = FuncIdent(2001);
pub const FUNC_ID_EXTRACT_LARGEST: FuncIdent = FuncIdent(2002);

// Manipulation funcs: 4xxx
pub const FUNC_ID_TRANSFORM: FuncIdent = FuncIdent(4000);
pub const FUNC_ID_ALIGN: FuncIdent = FuncIdent(4001);
pub const FUNC_ID_SNAP_TO_GROUND: FuncIdent = FuncIdent(4002);

// Smoothing funcs: 6xxx
pub const FUNC_ID_LAPLACIAN_SMOOTHING: FuncIdent = FuncIdent(6000);
pub const FUNC_ID_LOOP_SUBDIVISION: FuncIdent = FuncIdent(6001);

// Voxel-based funcs: 8xxx
pub const FUNC_ID_VOXELIZE: FuncIdent = FuncIdent(8000);
pub const FUNC_ID_BOOLEAN_INTERSECTION: FuncIdent = FuncIdent(8001);
pub const FUNC_ID_BOOLEAN_UNION: FuncIdent = FuncIdent(8002);
pub const FUNC_ID_BOOLEAN_DIFFERENCE: FuncIdent = FuncIdent(8003);
pub const FUNC_ID_VOXEL_TRANSFORM: FuncIdent = FuncIdent(8004);

// Hybridization funcs: 10xxx
pub const FUNC_ID_INTERPOLATED_UNION: FuncIdent = FuncIdent(10000);

// Tool funcs: 12xxx
pub const FUNC_ID_DISJOINT_MESH: FuncIdent = FuncIdent(12000);
pub const FUNC_ID_JOIN_MESHES: FuncIdent = FuncIdent(12001);
pub const FUNC_ID_JOIN_GROUP: FuncIdent = FuncIdent(12002);
pub const FUNC_ID_WELD: FuncIdent = FuncIdent(12003);
pub const FUNC_ID_REVERT_MESH_FACES: FuncIdent = FuncIdent(12004);
pub const FUNC_ID_SYNCHRONIZE_MESH_FACES: FuncIdent = FuncIdent(12005);

/// Returns the global set of function definitions available to the
/// editor.
///
/// Note that since funcs can have internal state such as a cache or
/// random state, two instances of the function table are not always
/// equivalent.
pub fn create_function_table() -> BTreeMap<FuncIdent, Box<dyn Func>> {
    let mut funcs: BTreeMap<FuncIdent, Box<dyn Func>> = BTreeMap::new();

    // Create funcs
    funcs.insert(FUNC_ID_CREATE_PLANE, Box::new(FuncCreatePlane));
    funcs.insert(FUNC_ID_CREATE_BOX, Box::new(FuncCreateBox));
    funcs.insert(FUNC_ID_CREATE_UV_SPHERE, Box::new(FuncCreateUvSphere));

    // Import/Export funcs
    funcs.insert(
        FUNC_ID_IMPORT_OBJ_MESH,
        Box::new(FuncImportObjMesh::new(Importer::new(
            EndlessCache::default(),
        ))),
    );
    funcs.insert(FUNC_ID_EXTRACT, Box::new(FuncExtract));
    funcs.insert(FUNC_ID_EXTRACT_LARGEST, Box::new(FuncExtractLargest));

    // Manipulation funcs
    funcs.insert(FUNC_ID_TRANSFORM, Box::new(FuncTransform));
    funcs.insert(FUNC_ID_ALIGN, Box::new(FuncAlign));
    funcs.insert(FUNC_ID_SNAP_TO_GROUND, Box::new(FuncSnapToGround));

    // Smoothing funcs
    funcs.insert(
        FUNC_ID_LAPLACIAN_SMOOTHING,
        Box::new(FuncLaplacianSmoothing),
    );
    funcs.insert(FUNC_ID_LOOP_SUBDIVISION, Box::new(FuncLoopSubdivision));

    // Voxel-based funcs: 8xxx
    funcs.insert(FUNC_ID_VOXELIZE, Box::new(FuncVoxelize));
    funcs.insert(
        FUNC_ID_BOOLEAN_INTERSECTION,
        Box::new(FuncBooleanIntersection),
    );
    funcs.insert(FUNC_ID_BOOLEAN_UNION, Box::new(FuncBooleanUnion));
    funcs.insert(FUNC_ID_BOOLEAN_DIFFERENCE, Box::new(FuncBooleanDifference));

    // Hybridization funcs
    funcs.insert(FUNC_ID_INTERPOLATED_UNION, Box::new(FuncInterpolatedUnion));

    // Tool funcs
    funcs.insert(FUNC_ID_DISJOINT_MESH, Box::new(FuncDisjointMesh));
    funcs.insert(FUNC_ID_JOIN_MESHES, Box::new(FuncJoinMeshes));
    funcs.insert(FUNC_ID_JOIN_GROUP, Box::new(FuncJoinGroup));
    funcs.insert(FUNC_ID_WELD, Box::new(FuncWeld));
    funcs.insert(FUNC_ID_REVERT_MESH_FACES, Box::new(FuncRevertMeshFaces));
    funcs.insert(
        FUNC_ID_SYNCHRONIZE_MESH_FACES,
        Box::new(FuncSynchronizeMeshFaces),
    );

    funcs
}
