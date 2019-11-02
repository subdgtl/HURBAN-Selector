use std::collections::{HashMap, HashSet};

use smallvec::SmallVec;

use crate::convert::{cast_u32, cast_usize};
use crate::geometry::{Face, Geometry, OrientedEdge, TriangleFace, UnorientedEdge};
use crate::mesh_topology_analysis;

/// Make sure all the faces are oriented the same way - have the same winding
/// (vertex order).
///
/// This function crawls the mesh geometry and flips all the faces, which are
/// not facing the same way as the previous faces in the process, starting with
/// the first face in the list. As a result, the entre mesh can end up facing
/// inwards (be entirely reverted). At the moment we have no tools to detect
/// such a case automatically, so we need to rely on the user to check it and
/// potentially revert winding of the entire mesh.
///
/// The algorithm relies on the fact that in a proper non-manifold mesh, each
/// oriented edge has exactly one (for watertight mesh geometry) or none (for
/// mesh patches) counter-edge oriented the opposite direction. It crawls the
/// geometry and if a face neighboring the current one doesn't have the proper
/// winding, it's being reverted and only then triggers checking its own
/// neighbors.
///
/// This method, doesn't flip the normals associated with the face vertices, as
/// there is no unambiguous way to do so automatically.

// FIXME: Flip also vertex normals if the visual/practical tests prove it's
// needed
pub fn synchronize_mesh_winding(
    geometry: &Geometry,
    unoriented_edges: &[UnorientedEdge],
    edge_to_face_topology: &HashMap<u32, SmallVec<[u32; 8]>>,
) -> Geometry {
    // All faces in the original mesh geometry
    let original_triangle_faces: Vec<_> = geometry
        .faces()
        .iter()
        .map(|face| match face {
            Face::Triangle(f) => *f,
        })
        .collect();

    // item index = face index; TRUE = the face was already checked, FALSE = the
    // face hasn't yet been checked
    let mut face_treatment_pattern = vec![false; original_triangle_faces.len()];

    // Faces to be checked for winding, determined by the orientation of the
    // OrientedEdge of the neighbor, which triggered the check. Current face has
    // to contain a reverted edge to have the proper winding, otherwise it has
    // to be reverted.
    let mut edge_face_stack: Vec<(OrientedEdge, u32)> =
        Vec::with_capacity(original_triangle_faces.len() / 2);

    // Faces already checked and reverted if needed => all the faces in this
    // list have the same vertex winding.
    let mut synchronized_triangle_faces: Vec<TriangleFace> =
        Vec::with_capacity(original_triangle_faces.len());

    // Synchronize faces in all mesh geometry triangles
    while synchronized_triangle_faces.len() < original_triangle_faces.len() {
        // Start with the first untreated face in the list of the original mesh
        // geometry faces. The winding of this face also determines the winding
        // of the rest of the mesh geometry.
        let mut current_face_index: u32 = 0;
        while face_treatment_pattern[cast_usize(current_face_index)] {
            current_face_index += 1;
        }

        face_treatment_pattern[cast_usize(current_face_index)] = true;
        let mut current_test_edge: OrientedEdge =
            original_triangle_faces[cast_usize(current_face_index)].to_oriented_edges()[0];

        // Put the first edge-face couple to the stack
        edge_face_stack.push((current_test_edge, current_face_index));

        // Check and revert (if needed) faces in the stack as long as there is any
        while !edge_face_stack.is_empty() {
            // Get ready for the next iteration.
            let next_edge_face = edge_face_stack.pop().expect("Popping from an empty vector");
            current_test_edge = next_edge_face.0;
            current_face_index = next_edge_face.1;

            let current_face = original_triangle_faces[cast_usize(current_face_index)];
            // Use if contains reverted edge, if not, revert and use. It should be
            // safe to presume the face contains the edge one way or another because
            // it's given by the topology generator.
            let proper_current_face = if current_face.contains_oriented_edge(current_test_edge) {
                current_face
            } else {
                current_face.to_reverted()
            };
            synchronized_triangle_faces.push(proper_current_face);

            // Edge-to-index map for faster lookup
            let mut unoriented_edge_index_map: HashMap<UnorientedEdge, u32> = HashMap::new();
            for (unoriented_edge_index, unoriented_edge) in unoriented_edges.iter().enumerate() {
                unoriented_edge_index_map.insert(*unoriented_edge, cast_u32(unoriented_edge_index));
            }

            // Find the indices of edges of the current face in the list of edges,
            // from which the topology was created
            let face_unoriented_edges = proper_current_face.to_unoriented_edges();
            let face_unoriented_edge_indices =
                face_unoriented_edges.iter().map(|unoriented_edge| {
                    unoriented_edge_index_map
                        .get(unoriented_edge)
                        .expect("The current edge is not found in the edge collection")
                });

            // Convert into oriented edges so that the neighboring faces can check
            // for correct winding
            let face_oriented_edges: Vec<_> =
                face_unoriented_edges.iter().map(|u_e| u_e.0).collect();
            // For each face edge index
            for (i, face_unoriented_edge_index) in face_unoriented_edge_indices.enumerate() {
                // get the actual oriented edge
                let face_oriented_edge = face_oriented_edges[i];
                // and try to find it in the edge-to-face topology.
                if let Some(edge_in_faces) = edge_to_face_topology.get(&face_unoriented_edge_index)
                {
                    // If it exists, iterate the faces containing the edge
                    for face_index in edge_in_faces {
                        // and if it was not already added to the stack or even checked
                        if !face_treatment_pattern[cast_usize(*face_index)] {
                            // add it to the stack with the expected edge orientation
                            edge_face_stack.push((face_oriented_edge.to_reverted(), *face_index));
                            // and mark it treated.
                            face_treatment_pattern[cast_usize(*face_index)] = true;
                        }
                    }
                }
            }
        }
    }

    // Rebuild the mesh geometry with the new faces and original faces and normals.
    Geometry::from_triangle_faces_with_vertices_and_normals(
        synchronized_triangle_faces,
        geometry.vertices().to_vec(),
        geometry.normals().to_vec(),
    )
}

/// Reverts vertex and normal winding of all faces in the mesh geometry and
/// returns a reverted mesh geometry
pub fn revert_mesh_faces(geometry: &Geometry) -> Geometry {
    let reverted_faces = geometry.faces().iter().map(|face| match face {
        Face::Triangle(t_f) => t_f.to_reverted(),
    });
    Geometry::from_triangle_faces_with_vertices_and_normals(
        reverted_faces,
        geometry.vertices().to_vec(),
        geometry.normals().to_vec(),
    )
}

/// Crawls the geometry to find continuous patches of geometry.
/// Returns a vector of new separated geometries.
pub fn separate_isolated_meshes(geometry: &Geometry) -> Vec<Geometry> {
    let face_to_face = mesh_topology_analysis::face_to_face_topology(geometry);
    let mut available_face_indices: HashSet<u32> = face_to_face.keys().cloned().collect();
    let mut patches: Vec<Geometry> = Vec::new();

    while let Some(first_face_index) = available_face_indices.iter().next() {
        let connected_indices = crawl_faces(*first_face_index, &face_to_face);

        patches.push(
            Geometry::from_faces_with_vertices_and_normals_remove_orphans(
                connected_indices
                    .iter()
                    .map(|face_index| geometry.faces()[cast_usize(*face_index)]),
                geometry.vertices().to_vec(),
                geometry.normals().to_vec(),
            ),
        );

        for c in &connected_indices {
            available_face_indices.remove(c);
        }
    }

    patches
}

fn crawl_faces(
    start_face_index: u32,
    face_to_face: &HashMap<u32, SmallVec<[u32; 8]>>,
) -> HashSet<u32> {
    let mut index_stack = vec![start_face_index];
    index_stack.push(start_face_index);

    let mut connected_face_indices = HashSet::new();

    while let Some(current_face_index) = index_stack.pop() {
        if connected_face_indices.insert(current_face_index) {
            for neighbor in &face_to_face[&current_face_index] {
                index_stack.push(neighbor.clone());
            }
        }
    }

    connected_face_indices.shrink_to_fit();

    connected_face_indices
}

#[cfg(test)]
mod tests {
    use nalgebra::base::Vector3;
    use nalgebra::geometry::Point3;

    use crate::geometry::{self, Geometry, TriangleFace};
    use crate::mesh_analysis;

    use super::*;

    fn n(x: f32, y: f32, z: f32) -> Vector3<f32> {
        Vector3::new(x, y, z)
    }

    fn tessellated_triangle_geometry() -> Geometry {
        let vertices = vec![
            Point3::new(-2.0, -2.0, 0.0),
            Point3::new(0.0, -2.0, 0.0),
            Point3::new(2.0, -2.0, 0.0),
            Point3::new(-1.0, 0.0, 0.0),
            Point3::new(1.0, 0.0, 0.0),
            Point3::new(0.0, 2.0, 0.0),
        ];

        let vertex_normals = vec![n(0.0, 0.0, 1.0)];

        let faces = vec![
            TriangleFace::new_separate(0, 3, 1, 0, 0, 0),
            TriangleFace::new_separate(1, 3, 4, 0, 0, 0),
            TriangleFace::new_separate(1, 4, 2, 0, 0, 0),
            TriangleFace::new_separate(3, 5, 4, 0, 0, 0),
        ];

        Geometry::from_triangle_faces_with_vertices_and_normals(faces, vertices, vertex_normals)
    }

    fn tessellated_triangle_with_island_geometry() -> Geometry {
        let vertices = vec![
            Point3::new(-2.0, -2.0, 0.0),
            Point3::new(0.0, -2.0, 0.0),
            Point3::new(2.0, -2.0, 0.0),
            Point3::new(-1.0, 0.0, 0.0),
            Point3::new(1.0, 0.0, 0.0),
            Point3::new(0.0, 2.0, 0.0),
            Point3::new(-1.0, 0.0, 1.0),
            Point3::new(1.0, 0.0, 1.0),
            Point3::new(0.0, 2.0, 1.0),
        ];

        let vertex_normals = vec![n(0.0, 0.0, 1.0)];

        let faces = vec![
            TriangleFace::new_separate(0, 3, 1, 0, 0, 0),
            TriangleFace::new_separate(1, 3, 4, 0, 0, 0),
            TriangleFace::new_separate(1, 4, 2, 0, 0, 0),
            TriangleFace::new_separate(3, 5, 4, 0, 0, 0),
            TriangleFace::new_separate(6, 7, 8, 0, 0, 0),
        ];

        Geometry::from_triangle_faces_with_vertices_and_normals(faces, vertices, vertex_normals)
    }

    fn tessellated_triangle_with_island_geometry_with_flipped_face() -> Geometry {
        let vertices = vec![
            Point3::new(-2.0, -2.0, 0.0),
            Point3::new(0.0, -2.0, 0.0),
            Point3::new(2.0, -2.0, 0.0),
            Point3::new(-1.0, 0.0, 0.0),
            Point3::new(1.0, 0.0, 0.0),
            Point3::new(0.0, 2.0, 0.0),
            Point3::new(-1.0, 0.0, 1.0),
            Point3::new(1.0, 0.0, 1.0),
            Point3::new(0.0, 2.0, 1.0),
        ];

        let vertex_normals = vec![n(0.0, 0.0, 1.0)];

        let faces = vec![
            TriangleFace::new_separate(0, 3, 1, 0, 0, 0),
            TriangleFace::new_separate(1, 3, 4, 0, 0, 0),
            TriangleFace::new_separate(2, 4, 1, 0, 0, 0),
            TriangleFace::new_separate(3, 5, 4, 0, 0, 0),
            TriangleFace::new_separate(6, 7, 8, 0, 0, 0),
        ];

        Geometry::from_triangle_faces_with_vertices_and_normals(faces, vertices, vertex_normals)
    }

    fn triangular_island_geometry() -> Geometry {
        let vertices = vec![
            Point3::new(-1.0, 0.0, 1.0),
            Point3::new(1.0, 0.0, 1.0),
            Point3::new(0.0, 2.0, 1.0),
        ];

        let vertex_normals = vec![n(0.0, 0.0, 1.0)];

        let faces = vec![TriangleFace::new_separate(0, 1, 2, 0, 0, 0)];

        Geometry::from_triangle_faces_with_vertices_and_normals(faces, vertices, vertex_normals)
    }

    #[test]
    fn test_separate_isolated_meshes_returns_similar_for_tessellated_triangle() {
        let geometry = tessellated_triangle_geometry();

        let calculated_geometries = separate_isolated_meshes(&geometry);

        assert_eq!(calculated_geometries.len(), 1);

        assert!(mesh_analysis::are_similar(
            &calculated_geometries[0],
            &geometry
        ));
    }

    #[test]
    fn test_separate_isolated_meshes_returns_similar_for_cube() {
        let geometry = geometry::cube_sharp_var_len([0.0, 0.0, 0.0], 1.0);

        let calculated_geometries = separate_isolated_meshes(&geometry);

        assert_eq!(calculated_geometries.len(), 1);
        assert!(mesh_analysis::are_similar(
            &geometry,
            &calculated_geometries[0]
        ));
    }

    #[test]
    fn test_separate_isolated_meshes_returns_similar_for_tessellated_triangle_with_island() {
        let geometry = tessellated_triangle_with_island_geometry();
        let geometry_triangle_correct = tessellated_triangle_geometry();
        let geometry_island_correct = triangular_island_geometry();

        let calculated_geometries = separate_isolated_meshes(&geometry);

        assert_eq!(calculated_geometries.len(), 2);

        if mesh_analysis::are_similar(&calculated_geometries[0], &geometry_triangle_correct) {
            assert!(mesh_analysis::are_similar(
                &calculated_geometries[1],
                &geometry_island_correct
            ));
        } else {
            assert!(mesh_analysis::are_similar(
                &calculated_geometries[1],
                &geometry_triangle_correct
            ));
            assert!(mesh_analysis::are_similar(
                &calculated_geometries[0],
                &geometry_island_correct
            ));
        }
    }

    #[test]
    fn test_mesh_tools_revert_mesh_faces() {
        let test_geometry = geometry::cube_sharp_var_len([0.0, 0.0, 0.0], 1.0);

        let calculated_geometry = revert_mesh_faces(&test_geometry);

        assert_eq!(
            test_geometry.faces().len(),
            calculated_geometry.faces().len()
        );

        assert!(calculated_geometry.faces().iter().all(|face| match face {
            Face::Triangle(f) => test_geometry
                .faces()
                .iter()
                .any(|other_face| match other_face {
                    Face::Triangle(o_f) => o_f.is_reverted(*f),
                }),
        }));
    }

    #[test]
    fn test_mesh_tools_synchronize_mesh_winding() {
        let test_geometry = tessellated_triangle_with_island_geometry_with_flipped_face();
        let test_geometry_correct = tessellated_triangle_with_island_geometry();

        let unoriented_edges: Vec<_> = test_geometry.unoriented_edges_iter().collect();
        let edge_to_face =
            mesh_topology_analysis::edge_to_face_topology(&test_geometry, &unoriented_edges);

        let calculated_geometry =
            synchronize_mesh_winding(&test_geometry, &unoriented_edges, &edge_to_face);

        assert_eq!(test_geometry_correct, calculated_geometry);
    }
}
