use std::collections::{HashMap, HashSet};

use smallvec::SmallVec;

use crate::convert::cast_usize;
use crate::geometry::Geometry;
use crate::mesh_topology_analysis::face_to_face_topology;

/// Crawls the geometry to find continuous patches of geometry.
/// Returns a vector of new separated geometries.
#[allow(dead_code)]
pub fn separate_isolated_meshes(geometry: &Geometry) -> Vec<Geometry> {
    let face_to_face = face_to_face_topology(geometry);
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
    let mut index_stack = Vec::with_capacity(face_to_face.len() / 2);
    index_stack.push(start_face_index);

    let mut connected_face_indices = HashSet::with_capacity(face_to_face.len());

    while !index_stack.is_empty() {
        let current_face_index = index_stack.pop().unwrap();

        if !connected_face_indices.contains(&current_face_index) {
            connected_face_indices.insert(current_face_index);

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
    fn test_separate_isolated_meshes_returns_identical_for_tessellated_triangle() {
        let geometry = tessellated_triangle_geometry();

        let calculated_geometries = separate_isolated_meshes(&geometry);

        assert_eq!(calculated_geometries.len(), 1);

        assert!(mesh_analysis::are_visually_identical(
            &calculated_geometries[0],
            &geometry
        ));
    }

    #[test]
    fn test_separate_isolated_meshes_returns_identical_for_cube() {
        let geometry = geometry::cube_sharp_var_len([0.0, 0.0, 0.0], 1.0);

        let calculated_geometries = separate_isolated_meshes(&geometry);

        assert_eq!(calculated_geometries.len(), 1);
        assert!(mesh_analysis::are_visually_identical(
            &geometry,
            &calculated_geometries[0]
        ));
    }

    #[test]
    fn test_separate_isolated_meshes_returns_identical_for_tessellated_triangle_with_island() {
        let geometry = tessellated_triangle_with_island_geometry();
        let geometry_triangle_correct = tessellated_triangle_geometry();
        let geometry_island_correct = triangular_island_geometry();

        let calculated_geometries = separate_isolated_meshes(&geometry);

        assert_eq!(calculated_geometries.len(), 2);

        if mesh_analysis::are_visually_identical(
            &calculated_geometries[0],
            &geometry_triangle_correct,
        ) {
            assert!(mesh_analysis::are_visually_identical(
                &calculated_geometries[1],
                &geometry_island_correct
            ));
        } else {
            assert!(mesh_analysis::are_visually_identical(
                &calculated_geometries[1],
                &geometry_triangle_correct
            ));
            assert!(mesh_analysis::are_visually_identical(
                &calculated_geometries[0],
                &geometry_island_correct
            ));
        }
    }
}
