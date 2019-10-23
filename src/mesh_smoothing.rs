use nalgebra::geometry::Point3;

use crate::convert::{cast_u32, cast_usize};
use crate::geometry::{Face, Geometry, TriangleFace};
use crate::mesh_topology_analysis::vertex_to_vertex_topology;

/// Relaxes angles between mesh edges, resulting in a smoother geometry.
///
/// The number of vertices, faces and the overall topology remains unchanged. The
/// more iterations, the smoother result. Too many iterations may cause slow
/// calculation time.
///
/// The algorithm is based on replacing each vertex position with an average
/// position of its immediate neighbors
///
/// - `geometry` - mesh geometry to relax
/// - `iterations` - number of times the smoothing algorithm should relax the geometry
#[allow(dead_code)]
pub fn laplacian_smoothing(geometry: &Geometry, iterations: u8) -> Geometry {
    laplacian_smoothing_with_anchors(geometry, iterations, &[])
}

/// Relaxes angles between mesh edges, while keeping some vertices anchored,
/// resulting in a smoother geometry.
///
/// The number of vertices, faces and the overall topology remains unchanged. The
/// more iterations, the smoother result. Too many iterations may cause slow
/// calculation time.
///
/// The algorithm is based on replacing each vertex position with an average
/// position of its immediate neighbors
///
/// - `geometry` - mesh geometry to relax
/// - `iterations` - number of times the smoothing algorithm should relax the geometry
/// - `fixed_vertex_indices` - indices of vertices to keep fixed during the relaxation
#[allow(dead_code)]
pub fn laplacian_smoothing_with_anchors(
    geometry: &Geometry,
    iterations: u8,
    fixed_vertex_indices: &[u32],
) -> Geometry {
    if iterations == 0 {
        return geometry.clone();
    }

    let vertex_to_vertex_topology = vertex_to_vertex_topology(geometry);
    let mut vertices: Vec<Point3<f32>> = Vec::from(geometry.vertices());
    let mut geometry_vertices: Vec<Point3<f32>>;

    for _ in 0..iterations {
        geometry_vertices = vertices.clone();
        for (current_vertex_index, neighbors_indices) in vertex_to_vertex_topology.iter() {
            if fixed_vertex_indices
                .iter()
                .all(|i| *i != cast_u32(*current_vertex_index))
                || neighbors_indices.is_empty()
            {
                let mut average_position: Point3<f32> = Point3::origin();
                for neighbor_index in neighbors_indices {
                    average_position += geometry_vertices[cast_usize(*neighbor_index)].coords;
                }
                average_position /= neighbors_indices.len() as f32;
                vertices[cast_usize(*current_vertex_index)] = average_position;
            }
        }
    }

    let faces: Vec<_> = filter_triangular_faces(&geometry);

    // FIXME: Calculate smooth normals for the result once we support them
    Geometry::from_triangle_faces_with_vertices_and_normals(
        faces,
        vertices,
        geometry.normals().to_vec(),
    )
}

fn filter_triangular_faces(geometry: &Geometry) -> Vec<TriangleFace> {
    geometry
        .faces()
        .iter()
        .copied()
        .map(|face| match face {
            Face::Triangle(f) => f,
        })
        .collect()
}

#[cfg(test)]
mod tests {
    use std::iter::FromIterator;

    use nalgebra;

    use crate::edge_analysis;
    use crate::geometry::{Geometry, NormalStrategy, OrientedEdge, TriangleFace, Vertices};
    use crate::mesh_analysis;

    use super::*;

    // FIXME: Snapshot testing
    fn torus() -> (Vec<(u32, u32, u32)>, Vertices) {
        let vertices = vec![
            Point3::new(0.566987, -1.129e-11, 0.25),
            Point3::new(-0.716506, 1.241025, 0.25),
            Point3::new(-0.283494, 0.491025, 0.25),
            Point3::new(-0.716506, -1.241025, 0.25),
            Point3::new(-0.283494, -0.491025, 0.25),
            Point3::new(1.0, -1.129e-11, -0.5),
            Point3::new(1.433013, -1.129e-11, 0.25),
            Point3::new(-0.5, 0.866025, -0.5),
            Point3::new(-0.5, -0.866025, -0.5),
        ];

        let faces = vec![
            (4, 3, 6),
            (0, 6, 2),
            (2, 1, 3),
            (8, 4, 0),
            (3, 8, 6),
            (5, 0, 7),
            (6, 5, 7),
            (7, 2, 4),
            (1, 7, 8),
            (4, 6, 0),
            (6, 1, 2),
            (2, 3, 4),
            (8, 0, 5),
            (8, 5, 6),
            (0, 2, 7),
            (6, 7, 1),
            (7, 4, 8),
            (1, 8, 3),
        ];

        (faces, vertices)
    }

    fn triple_torus() -> (Vec<(u32, u32, u32)>, Vertices) {
        let vertices = vec![
            Point3::new(15.566987, -1.129e-11, 0.25),
            Point3::new(14.283494, 1.241025, 0.25),
            Point3::new(14.716506, 0.491025, 0.25),
            Point3::new(14.283494, -1.241025, 0.25),
            Point3::new(14.716506, -0.491025, 0.25),
            Point3::new(16.0, 0.75, 0.25),
            Point3::new(15.149519, 1.241025, 0.25),
            Point3::new(16.0, 1.732051, 0.25),
            Point3::new(16.108253, 0.1875, -0.5),
            Point3::new(16.433012, -1.129e-11, 0.25),
            Point3::new(14.716506, 1.991025, 0.25),
            Point3::new(15.566987, 2.482051, 0.25),
            Point3::new(14.283494, 3.723076, 0.25),
            Point3::new(14.716506, 2.973076, 0.25),
            Point3::new(14.554127, 1.334775, -0.5),
            Point3::new(14.5, -0.866025, -0.5),
            Point3::new(14.5, 3.348076, -0.5),
            Point3::new(16.108253, 2.294551, -0.5),
            Point3::new(16.433012, 2.482051, 0.25),
        ];

        let faces = vec![
            (4, 3, 0),
            (0, 9, 1),
            (2, 1, 3),
            (7, 5, 9),
            (5, 6, 9),
            (6, 7, 18),
            (15, 4, 0),
            (3, 15, 9),
            (10, 1, 11),
            (11, 18, 12),
            (13, 12, 1),
            (14, 2, 15),
            (1, 14, 15),
            (8, 0, 2),
            (8, 14, 6),
            (16, 13, 10),
            (12, 16, 1),
            (17, 8, 7),
            (18, 9, 8),
            (14, 17, 6),
            (17, 11, 16),
            (18, 17, 16),
            (14, 10, 17),
            (3, 9, 0),
            (0, 1, 2),
            (2, 3, 4),
            (7, 9, 18),
            (6, 1, 9),
            (6, 18, 1),
            (15, 0, 8),
            (15, 8, 9),
            (1, 18, 11),
            (11, 12, 13),
            (13, 1, 10),
            (2, 4, 15),
            (1, 15, 3),
            (8, 2, 14),
            (8, 6, 5),
            (16, 10, 14),
            (16, 14, 1),
            (8, 5, 7),
            (18, 8, 17),
            (17, 7, 6),
            (11, 13, 16),
            (18, 16, 12),
            (10, 11, 17),
        ];

        (faces, vertices)
    }

    fn triple_torus_laplacian_1_iteration() -> (Vec<(u32, u32, u32)>, Vertices) {
        let vertices = vec![
            Point3::new(15.005895, -0.096932, 0.035714),
            Point3::new(15.032319, 1.381472, 0.076923),
            Point3::new(14.85898, 0.023604, -0.071429),
            Point3::new(15.036084, 0.0625, 0.125),
            Point3::new(14.766747, -0.404006, 0.0625),
            Point3::new(15.922696, 0.790144, 0.0625),
            Point3::new(15.740019, 1.252744, -0.03125),
            Point3::new(16.038675, 1.159188, 0.0),
            Point3::new(15.546142, 0.945945, 0.025),
            Point3::new(15.369418, 0.614067, 0.083333),
            Point3::new(14.954895, 2.278926, -0.125),
            Point3::new(15.005895, 2.578983, 0.035714),
            Point3::new(15.1, 2.505256, 0.1),
            Point3::new(14.670096, 2.557051, 0.1),
            Point3::new(15.010316, 1.241025, -0.125),
            Point3::new(15.082797, 0.190284, 0.0625),
            Point3::new(15.082797, 2.315204, 0.0625),
            Point3::new(15.378551, 1.849819, -0.03125),
            Point3::new(15.381446, 1.805484, 0.0),
        ];

        let faces = vec![
            (4, 3, 0),
            (0, 9, 1),
            (2, 1, 3),
            (7, 5, 9),
            (5, 6, 9),
            (6, 7, 18),
            (15, 4, 0),
            (3, 15, 9),
            (10, 1, 11),
            (11, 18, 12),
            (13, 12, 1),
            (14, 2, 15),
            (1, 14, 15),
            (8, 0, 2),
            (8, 14, 6),
            (16, 13, 10),
            (12, 16, 1),
            (17, 8, 7),
            (18, 9, 8),
            (14, 17, 6),
            (17, 11, 16),
            (18, 17, 16),
            (14, 10, 17),
            (3, 9, 0),
            (0, 1, 2),
            (2, 3, 4),
            (7, 9, 18),
            (6, 1, 9),
            (6, 18, 1),
            (15, 0, 8),
            (15, 8, 9),
            (1, 18, 11),
            (11, 12, 13),
            (13, 1, 10),
            (2, 4, 15),
            (1, 15, 3),
            (8, 2, 14),
            (8, 6, 5),
            (16, 10, 14),
            (16, 14, 1),
            (8, 5, 7),
            (18, 8, 17),
            (17, 7, 6),
            (11, 13, 16),
            (18, 16, 12),
            (10, 11, 17),
        ];

        (faces, vertices)
    }

    fn triple_torus_laplacian_3_iterations() -> (Vec<(u32, u32, u32)>, Vertices) {
        let vertices = vec![
            Point3::new(15.151657, 0.617585, 0.028723),
            Point3::new(15.151801, 1.310818, 0.028865),
            Point3::new(15.125829, 0.671169, 0.02456),
            Point3::new(15.127048, 0.592897, 0.035489),
            Point3::new(15.066285, 0.408004, 0.039398),
            Point3::new(15.453969, 1.037088, 0.016901),
            Point3::new(15.381245, 1.231292, 0.013342),
            Point3::new(15.440678, 1.208557, 0.014121),
            Point3::new(15.327691, 1.020512, 0.021627),
            Point3::new(15.303463, 0.935796, 0.024711),
            Point3::new(15.140347, 1.774271, 0.009292),
            Point3::new(15.139611, 1.857752, 0.020585),
            Point3::new(15.130695, 1.858242, 0.023035),
            Point3::new(15.063364, 1.914324, 0.024865),
            Point3::new(15.19091, 1.26172, 0.012169),
            Point3::new(15.161481, 0.691733, 0.027817),
            Point3::new(15.150735, 1.794786, 0.020292),
            Point3::new(15.269145, 1.541168, 0.013697),
            Point3::new(15.27197, 1.492211, 0.016929),
        ];

        let faces = vec![
            (4, 3, 0),
            (0, 9, 1),
            (2, 1, 3),
            (7, 5, 9),
            (5, 6, 9),
            (6, 7, 18),
            (15, 4, 0),
            (3, 15, 9),
            (10, 1, 11),
            (11, 18, 12),
            (13, 12, 1),
            (14, 2, 15),
            (1, 14, 15),
            (8, 0, 2),
            (8, 14, 6),
            (16, 13, 10),
            (12, 16, 1),
            (17, 8, 7),
            (18, 9, 8),
            (14, 17, 6),
            (17, 11, 16),
            (18, 17, 16),
            (14, 10, 17),
            (3, 9, 0),
            (0, 1, 2),
            (2, 3, 4),
            (7, 9, 18),
            (6, 1, 9),
            (6, 18, 1),
            (15, 0, 8),
            (15, 8, 9),
            (1, 18, 11),
            (11, 12, 13),
            (13, 1, 10),
            (2, 4, 15),
            (1, 15, 3),
            (8, 2, 14),
            (8, 6, 5),
            (16, 10, 14),
            (16, 14, 1),
            (8, 5, 7),
            (18, 8, 17),
            (17, 7, 6),
            (11, 13, 16),
            (18, 16, 12),
            (10, 11, 17),
        ];

        (faces, vertices)
    }

    fn shape_for_smoothing_with_anchors() -> (Vec<(u32, u32, u32)>, Vertices) {
        let vertices = vec![
            Point3::new(30.21796, -6.119943, 0.0),
            Point3::new(32.031532, 1.328689, 0.0),
            Point3::new(33.875141, -3.522298, 3.718605),
            Point3::new(34.571838, -2.071111, 2.77835),
            Point3::new(34.778172, -5.285372, 3.718605),
            Point3::new(36.243252, -3.80194, 3.718605),
            Point3::new(36.741604, -10.146505, 0.0),
            Point3::new(39.676025, 1.905633, 0.0),
            Point3::new(42.587009, -5.186427, 0.0),
        ];

        let faces = vec![
            (4, 8, 5),
            (4, 6, 8),
            (5, 8, 7),
            (3, 5, 7),
            (0, 2, 1),
            (1, 2, 3),
            (0, 4, 2),
            (1, 3, 7),
            (0, 6, 4),
            (2, 4, 5),
            (2, 5, 3),
        ];

        (faces, vertices)
    }

    fn shape_for_smoothing_with_anchors_50_iterations() -> (Vec<(u32, u32, u32)>, Vertices) {
        let vertices = vec![
            Point3::new(30.21796, -6.119943, 0.0),
            Point3::new(32.031532, 1.328689, 0.0),
            Point3::new(34.491065, -2.551039, 0.0),
            Point3::new(36.00632, -0.404003, 0.0),
            Point3::new(36.372859, -5.260642, 0.0),
            Point3::new(37.826656, -2.299296, 0.0),
            Point3::new(36.741604, -10.146505, 0.0),
            Point3::new(39.676025, 1.905633, 0.0),
            Point3::new(42.587009, -5.186427, 0.0),
        ];

        let faces = vec![
            (4, 8, 5),
            (4, 6, 8),
            (5, 8, 7),
            (3, 5, 7),
            (0, 2, 1),
            (1, 2, 3),
            (0, 4, 2),
            (1, 3, 7),
            (0, 6, 4),
            (2, 4, 5),
            (2, 5, 3),
        ];

        (faces, vertices)
    }

    #[test]
    fn test_laplacian_smoothing_preserves_face_vertex_normal_count() {
        let (faces, vertices) = torus();
        let geometry = Geometry::from_triangle_faces_with_vertices_and_computed_normals(
            faces.clone(),
            vertices.clone(),
            NormalStrategy::Sharp,
        );
        let relaxed_geometry_0 = laplacian_smoothing(&geometry, 0);
        let relaxed_geometry_1 = laplacian_smoothing(&geometry, 1);
        let relaxed_geometry_10 = laplacian_smoothing(&geometry, 10);

        assert_eq!(relaxed_geometry_0.faces().len(), geometry.faces().len(),);
        assert_eq!(relaxed_geometry_1.faces().len(), geometry.faces().len(),);
        assert_eq!(relaxed_geometry_10.faces().len(), geometry.faces().len(),);
        assert_eq!(
            relaxed_geometry_0.vertices().len(),
            geometry.vertices().len(),
        );
        assert_eq!(
            relaxed_geometry_1.vertices().len(),
            geometry.vertices().len(),
        );
        assert_eq!(
            relaxed_geometry_10.vertices().len(),
            geometry.vertices().len(),
        );
        assert_eq!(relaxed_geometry_0.normals().len(), geometry.normals().len());
        assert_eq!(relaxed_geometry_1.normals().len(), geometry.normals().len());
        assert_eq!(
            relaxed_geometry_10.normals().len(),
            geometry.normals().len(),
        );
    }

    #[test]
    fn test_laplacian_smoothing() {
        let (faces, vertices) = triple_torus();
        let geometry = Geometry::from_triangle_faces_with_vertices_and_computed_normals(
            faces.clone(),
            vertices.clone(),
            NormalStrategy::Sharp,
        );

        let (faces_1_i, vertices_1_i) = triple_torus_laplacian_1_iteration();
        let test_geometry_1_i = Geometry::from_triangle_faces_with_vertices_and_computed_normals(
            faces_1_i.clone(),
            vertices_1_i.clone(),
            NormalStrategy::Sharp,
        );

        let (faces_3_i, vertices_3_i) = triple_torus_laplacian_3_iterations();
        let test_geometry_3_i = Geometry::from_triangle_faces_with_vertices_and_computed_normals(
            faces_3_i.clone(),
            vertices_3_i.clone(),
            NormalStrategy::Sharp,
        );

        let relaxed_geometry_0_i = laplacian_smoothing(&geometry, 0);
        let relaxed_geometry_1_i = laplacian_smoothing(&geometry, 1);
        let relaxed_geometry_3_i = laplacian_smoothing(&geometry, 3);

        let relaxed_geometry_0_i_faces: Vec<TriangleFace> =
            filter_triangular_faces(&relaxed_geometry_0_i);
        let relaxed_geometry_1_i_faces: Vec<TriangleFace> =
            filter_triangular_faces(&relaxed_geometry_1_i);
        let test_geometry_0_i_faces: Vec<TriangleFace> = filter_triangular_faces(&geometry);
        let test_geometry_1_i_faces: Vec<TriangleFace> =
            filter_triangular_faces(&test_geometry_1_i);
        let relaxed_geometry_3_i_faces: Vec<TriangleFace> =
            filter_triangular_faces(&relaxed_geometry_3_i);
        let test_geometry_3_i_faces: Vec<TriangleFace> =
            filter_triangular_faces(&test_geometry_3_i);

        assert_eq!(relaxed_geometry_0_i_faces, test_geometry_0_i_faces);
        assert_eq!(relaxed_geometry_1_i_faces, test_geometry_1_i_faces);
        assert_eq!(relaxed_geometry_3_i_faces, test_geometry_3_i_faces);

        const TOLERANCE_SQUARED: f32 = 0.01 * 0.01;

        let relaxed_geometry_0_i_vertices = relaxed_geometry_0_i.vertices();
        let test_geometry_0_i_vertices = geometry.vertices();

        for i in 0..test_geometry_0_i_vertices.len() {
            assert!(
                nalgebra::distance_squared(
                    &test_geometry_0_i_vertices[i],
                    &relaxed_geometry_0_i_vertices[i]
                ) < TOLERANCE_SQUARED
            );
        }

        let relaxed_geometry_1_i_vertices = relaxed_geometry_1_i.vertices();
        let test_geometry_1_i_vertices = test_geometry_1_i.vertices();

        for i in 0..test_geometry_1_i_vertices.len() {
            assert!(
                nalgebra::distance_squared(
                    &test_geometry_1_i_vertices[i],
                    &relaxed_geometry_1_i_vertices[i]
                ) < TOLERANCE_SQUARED
            );
        }

        let relaxed_geometry_3_i_vertices = relaxed_geometry_3_i.vertices();
        let test_geometry_3_i_vertices = test_geometry_3_i.vertices();

        for i in 0..test_geometry_3_i_vertices.len() {
            assert!(
                nalgebra::distance_squared(
                    &test_geometry_3_i_vertices[i],
                    &relaxed_geometry_3_i_vertices[i]
                ) < TOLERANCE_SQUARED
            );
        }
    }

    #[test]
    fn test_laplacian_smoothing_with_anchors() {
        let (faces, vertices) = shape_for_smoothing_with_anchors();
        let geometry = Geometry::from_triangle_faces_with_vertices_and_computed_normals(
            faces.clone(),
            vertices.clone(),
            NormalStrategy::Sharp,
        );

        let fixed_vertex_indices: Vec<u32> = vec![0, 1, 7, 8, 6];

        let (faces_correct, vertices_correct) = shape_for_smoothing_with_anchors_50_iterations();
        let test_geometry_correct =
            Geometry::from_triangle_faces_with_vertices_and_computed_normals(
                faces_correct.clone(),
                vertices_correct.clone(),
                NormalStrategy::Sharp,
            );

        let relaxed_geometry =
            laplacian_smoothing_with_anchors(&geometry, 50, &fixed_vertex_indices);

        let relaxed_geometry_faces: Vec<TriangleFace> = filter_triangular_faces(&relaxed_geometry);
        let test_geometry_faces: Vec<TriangleFace> =
            filter_triangular_faces(&test_geometry_correct);

        assert_eq!(relaxed_geometry_faces, test_geometry_faces);

        const TOLERANCE_SQUARED: f32 = 0.01 * 0.01;

        let relaxed_geometry_vertices = relaxed_geometry.vertices();
        let test_geometry_vertices = test_geometry_correct.vertices();

        for i in 0..test_geometry_vertices.len() {
            assert!(
                nalgebra::distance_squared(
                    &test_geometry_vertices[i],
                    &relaxed_geometry_vertices[i]
                ) < TOLERANCE_SQUARED
            );
        }
    }

    #[test]
    fn test_laplacian_smoothing_with_anchors_find_border_vertices() {
        let (faces, vertices) = shape_for_smoothing_with_anchors();
        let geometry = Geometry::from_triangle_faces_with_vertices_and_computed_normals(
            faces.clone(),
            vertices.clone(),
            NormalStrategy::Sharp,
        );

        let oriented_edges: Vec<OrientedEdge> = geometry.oriented_edges_iter().collect();
        let edge_sharing_map = edge_analysis::edge_sharing(&oriented_edges);
        let fixed_vertex_indices =
            Vec::from_iter(mesh_analysis::border_vertex_indices(&edge_sharing_map).into_iter());

        let (faces_correct, vertices_correct) = shape_for_smoothing_with_anchors_50_iterations();
        let test_geometry_correct =
            Geometry::from_triangle_faces_with_vertices_and_computed_normals(
                faces_correct.clone(),
                vertices_correct.clone(),
                NormalStrategy::Sharp,
            );

        let relaxed_geometry =
            laplacian_smoothing_with_anchors(&geometry, 50, &fixed_vertex_indices);

        let relaxed_geometry_faces: Vec<TriangleFace> = filter_triangular_faces(&relaxed_geometry);
        let test_geometry_faces: Vec<TriangleFace> =
            filter_triangular_faces(&test_geometry_correct);

        assert_eq!(relaxed_geometry_faces, test_geometry_faces);

        let relaxed_geometry_vertices = relaxed_geometry.vertices();
        let test_geometry_vertices = test_geometry_correct.vertices();

        for i in 0..test_geometry_vertices.len() {
            assert!(test_geometry_vertices[i].coords.relative_eq(
                &relaxed_geometry_vertices[i].coords,
                0.001,
                0.001,
            ));
        }
    }
}
