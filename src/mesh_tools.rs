use std::collections::HashMap;

use smallvec::SmallVec;

use crate::convert::{cast_u32, cast_usize};
use crate::geometry::Geometry;
use crate::mesh_topology_analysis::face_to_face_topology;

fn crawl_faces(
    current_face_index: u32,
    face_to_face: &HashMap<u32, SmallVec<[u32; 8]>>,
    available_faces: &mut Vec<u32>,
    separate_faces: &mut Vec<u32>,
) {
    separate_faces.push(current_face_index);

    if let Some(all_neighbors) = face_to_face.get(&current_face_index) {
        let neighbors: Vec<u32> = all_neighbors
            .iter()
            .copied()
            .filter(|n| available_faces.iter().any(|a_f| a_f == n))
            .collect();
        let af: Vec<_> = available_faces
            .iter()
            .copied()
            .filter(|f| neighbors.iter().all(|n| n != f))
            .collect();
        *available_faces = af;
        for neighbor_index in neighbors {
            crawl_faces(
                neighbor_index,
                face_to_face,
                available_faces,
                separate_faces,
            );
        }
    }
}

/// Crawls the geometry to find continuous patches of geometry.
/// Returns a vector of new separated geometries.
#[allow(dead_code)]
pub fn separate_isolated_meshes(geometry: &Geometry) -> Vec<Geometry> {
    let face_to_face = face_to_face_topology(geometry);
    let faces = geometry.faces();
    let vertices: Vec<_> = geometry.vertices().to_vec();
    let normals: Vec<_> = geometry.normals().to_vec();

    let mut available_faces: Vec<u32> = face_to_face.keys().map(|key| cast_u32(*key)).collect();

    let mut patches: Vec<Geometry> = Vec::new();

    while !available_faces.is_empty() {
        let mut separate_faces: Vec<u32> = Vec::new();

        if let Some(first_face_index) = available_faces.pop() {
            crawl_faces(
                first_face_index,
                &face_to_face,
                &mut available_faces,
                &mut separate_faces,
            );
        }

        let faces: Vec<_> = separate_faces
            .iter()
            .map(|face_index| faces[cast_usize(*face_index)])
            .collect();

        patches.push(
            Geometry::from_faces_with_vertices_and_normals_remove_orphans(
                faces,
                vertices.clone(),
                normals.clone(),
            ),
        );
    }

    patches
}

#[cfg(test)]
mod tests {
    use nalgebra::base::Vector3;
    use nalgebra::geometry::Point3;

    use crate::geometry::{Geometry, TriangleFace};
    use crate::mesh_analysis;

    use super::*;

    fn v(x: f32, y: f32, z: f32, translation: [f32; 3], scale: f32) -> Point3<f32> {
        Point3::new(
            scale * x + translation[0],
            scale * y + translation[1],
            scale * z + translation[2],
        )
    }

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

    pub fn cube_sharp_var_len(position: [f32; 3], scale: f32) -> Geometry {
        let vertex_positions = vec![
            // back
            v(-1.0, 1.0, -1.0, position, scale),
            v(-1.0, 1.0, 1.0, position, scale),
            v(1.0, 1.0, 1.0, position, scale),
            v(1.0, 1.0, -1.0, position, scale),
            // front
            v(-1.0, -1.0, -1.0, position, scale),
            v(1.0, -1.0, -1.0, position, scale),
            v(1.0, -1.0, 1.0, position, scale),
            v(-1.0, -1.0, 1.0, position, scale),
        ];

        let vertex_normals = vec![
            // back
            n(0.0, 1.0, 0.0),
            // front
            n(0.0, -1.0, 0.0),
            // top
            n(0.0, 0.0, 1.0),
            // bottom
            n(0.0, 0.0, -1.0),
            // right
            n(1.0, 0.0, 0.0),
            // left
            n(-1.0, 0.0, 0.0),
        ];

        let faces = vec![
            // back
            TriangleFace::new_separate(0, 1, 2, 0, 0, 0),
            TriangleFace::new_separate(2, 3, 0, 0, 0, 0),
            // front
            TriangleFace::new_separate(4, 5, 6, 1, 1, 1),
            TriangleFace::new_separate(6, 7, 4, 1, 1, 1),
            // top
            TriangleFace::new_separate(7, 6, 2, 2, 2, 2),
            TriangleFace::new_separate(2, 1, 7, 2, 2, 2),
            // bottom
            TriangleFace::new_separate(4, 0, 3, 3, 3, 3),
            TriangleFace::new_separate(3, 5, 4, 3, 3, 3),
            // right
            TriangleFace::new_separate(5, 3, 2, 4, 4, 4),
            TriangleFace::new_separate(2, 6, 5, 4, 4, 4),
            // left
            TriangleFace::new_separate(4, 7, 1, 5, 5, 5),
            TriangleFace::new_separate(1, 0, 4, 5, 5, 5),
        ];

        Geometry::from_triangle_faces_with_vertices_and_normals(
            faces,
            vertex_positions,
            vertex_normals,
        )
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
        let geometry = cube_sharp_var_len([0.0, 0.0, 0.0], 1.0);

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
