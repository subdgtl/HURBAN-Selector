use std::collections::HashSet;

use crate::convert::cast_i32;
use crate::edge_analysis::EdgeCountMap;
use crate::geometry::OrientedEdge;

/// Finds edges with a certain valency in a mesh edge collection
/// Valency indicates how many faces share the edge
fn find_edges_with_valency<'a>(
    edge_valencies: &'a EdgeCountMap,
    valency: usize,
) -> impl Iterator<Item = OrientedEdge> + 'a {
    edge_valencies
        .iter()
        .filter(move |(_, similar_edges)| {
            similar_edges.ascending_edges.len() + similar_edges.descending_edges.len() == valency
        })
        .flat_map(|(_, similar_edges)| {
            similar_edges
                .ascending_edges
                .iter()
                .copied()
                .chain(similar_edges.descending_edges.iter().copied())
        })
}

/// Finds border edges in a mesh edge collection
/// An edge is border when its valency is 1
#[allow(dead_code)]
pub fn border_edges<'a>(
    edge_valencies: &'a EdgeCountMap,
) -> impl Iterator<Item = OrientedEdge> + 'a {
    find_edges_with_valency(edge_valencies, 1)
}

/// Finds manifold (inner) edges in a mesh edge collection
/// An edge is manifold when its valency is 2
#[allow(dead_code)]
pub fn manifold_edges<'a>(
    edge_valencies: &'a EdgeCountMap,
) -> impl Iterator<Item = OrientedEdge> + 'a {
    find_edges_with_valency(edge_valencies, 2)
}

/// Finds non-manifold (errorneous) edges in a mesh edge collection
#[allow(dead_code)]
pub fn non_manifold_edges<'a>(
    edge_valencies: &'a EdgeCountMap,
) -> impl Iterator<Item = OrientedEdge> + 'a {
    edge_valencies
        .iter()
        .filter(|(_, edge_count)| {
            edge_count.ascending_edges.len() + edge_count.descending_edges.len() > 2
        })
        .flat_map(|(_, similar_edges)| {
            similar_edges
                .ascending_edges
                .iter()
                .copied()
                .chain(similar_edges.descending_edges.iter().copied())
        })
}

/// Finds border vertex indices in a mesh edge collection
/// A vertex is border when its edge's valency is 1
#[allow(dead_code)]
pub fn border_vertex_indices(edge_valencies: &EdgeCountMap) -> HashSet<u32> {
    let mut border_vertices = HashSet::new();

    border_edges(edge_valencies).for_each(|edge| {
        border_vertices.insert(edge.vertices.0);
        border_vertices.insert(edge.vertices.1);
    });
    border_vertices
}

/// Check if all the face normals point the same way.
/// In a proper watertight orientable mesh each oriented edge
/// should have a single counterpart in a reverted oriented edge.
/// In an open orientable mesh each internal edge has its counterpart
/// in a single reverted oriented edge and
/// the border edges don't have any counterpart.
#[allow(dead_code)]
pub fn is_mesh_orientable(edge_valencies: &EdgeCountMap) -> bool {
    edge_valencies.iter().all(|(_, edge_count)| {
        // Ascending_count and descending_count can never be both 0
        // at the same time because there is never a case that the
        // edge doesn't exist in any direction.
        // Even if this happens, it means that the tested edge
        // is non-existing and therefore doesn't affect edge winding.
        edge_count.ascending_edges.len() <= 1 && edge_count.descending_edges.len() <= 1
    })
}

/// The mesh is watertight if there is no border or non-manifold edge,
/// in other words, all edge valencies are 2
#[allow(dead_code)]
pub fn is_mesh_watertight(edge_valencies: &EdgeCountMap) -> bool {
    edge_valencies.iter().all(|(_, edge_count)| {
        edge_count.ascending_edges.len() == 1 && edge_count.descending_edges.len() == 1
    })
}

/// Genus of a mesh is the number of holes in topology / connectivity
/// The mesh must be triangular and watertight
/// V - E + F = 2 (1 - G)
#[allow(dead_code)]
pub fn mesh_genus(vertex_count: usize, edge_count: usize, face_count: usize) -> i32 {
    1 - (cast_i32(vertex_count) - cast_i32(edge_count) + cast_i32(face_count)) / 2
}

#[cfg(test)]
mod tests {
    use nalgebra::base::Vector3;
    use nalgebra::geometry::Point3;

    use crate::edge_analysis::edge_valencies;
    use crate::geometry::{
        self, cube_sharp_var_len, Geometry, NormalStrategy, TriangleFace, UnorientedEdge, Vertices,
    };

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

    pub fn quad() -> (Vec<(u32, u32, u32)>, Vertices) {
        let vertices = vec![
            v(-1.0, -1.0, 0.0, [0.0, 0.0, 0.0], 1.0),
            v(1.0, -1.0, 0.0, [0.0, 0.0, 0.0], 1.0),
            v(1.0, 1.0, 0.0, [0.0, 0.0, 0.0], 1.0),
            v(-1.0, 1.0, 0.0, [0.0, 0.0, 0.0], 1.0),
        ];

        let faces = vec![(0, 1, 2), (2, 3, 0)];

        (faces, vertices)
    }

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

    fn double_torus() -> (Vec<(u32, u32, u32)>, Vertices) {
        let vertices = vec![
            Point3::new(5.566988, -1.129e-11, 0.25),
            Point3::new(4.283494, 1.241025, 0.25),
            Point3::new(4.716506, 0.491025, 0.25),
            Point3::new(4.283494, -1.241025, 0.25),
            Point3::new(4.716506, -0.491025, 0.25),
            Point3::new(6.0, 0.75, 0.25),
            Point3::new(5.149519, 1.241025, 0.25),
            Point3::new(6.0, 1.732051, 0.25),
            Point3::new(4.608253, 1.053525, -0.5),
            Point3::new(4.5, -0.866025, -0.5),
            Point3::new(6.108253, 0.1875, -0.5),
            Point3::new(6.433012, -1.129e-11, 0.25),
            Point3::new(6.216506, 2.107051, -0.5),
            Point3::new(6.433012, 2.482051, 0.25),
        ];

        let faces = vec![
            (4, 3, 11),
            (0, 11, 2),
            (2, 1, 3),
            (7, 5, 11),
            (5, 6, 11),
            (6, 7, 13),
            (8, 2, 9),
            (1, 8, 9),
            (8, 12, 6),
            (1, 13, 12),
            (9, 4, 0),
            (3, 9, 11),
            (10, 0, 2),
            (12, 10, 5),
            (13, 11, 12),
            (10, 8, 6),
            (4, 11, 0),
            (11, 1, 2),
            (2, 3, 4),
            (7, 11, 13),
            (6, 1, 11),
            (6, 13, 1),
            (2, 4, 9),
            (1, 9, 3),
            (12, 7, 6),
            (1, 12, 8),
            (9, 0, 10),
            (9, 10, 11),
            (10, 2, 8),
            (12, 5, 7),
            (11, 10, 12),
            (10, 6, 5),
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

    pub fn non_manifold_shape() -> (Vec<(u32, u32, u32)>, Vertices) {
        let vertices = vec![
            v(-1.0, -1.0, 0.0, [0.0, 0.0, 0.0], 1.0),
            v(1.0, -1.0, 0.0, [0.0, 0.0, 0.0], 1.0),
            v(1.0, 1.0, 0.0, [0.0, 0.0, 0.0], 1.0),
            v(-1.0, 1.0, 0.0, [0.0, 0.0, 0.0], 1.0),
            v(0.0, 0.0, 1.0, [0.0, 0.0, 0.0], 1.0),
        ];

        let faces = vec![(0, 1, 2), (2, 3, 0), (2, 4, 0)];

        (faces, vertices)
    }

    pub fn cube_sharp_mismatching_winding(position: [f32; 3], scale: f32) -> Geometry {
        let vertex_positions = vec![
            // back
            v(-1.0, 1.0, -1.0, position, scale),
            v(-1.0, 1.0, 1.0, position, scale),
            v(1.0, 1.0, 1.0, position, scale),
            v(1.0, 1.0, -1.0, position, scale),
            // front
            v(-1.0, -1.0, -1.0, position, scale),
            v(-1.0, -1.0, 1.0, position, scale),
            v(1.0, -1.0, 1.0, position, scale),
            v(1.0, -1.0, -1.0, position, scale),
        ];

        let vertex_normals = vec![
            // back
            n(0.0, 1.0, 0.0),
            n(0.0, 1.0, 0.0),
            n(0.0, 1.0, 0.0),
            n(0.0, 1.0, 0.0),
            // front
            n(0.0, -1.0, 0.0),
            n(0.0, -1.0, 0.0),
            n(0.0, -1.0, 0.0),
            n(0.0, -1.0, 0.0),
            // top
            n(0.0, 0.0, 1.0),
            n(0.0, 0.0, 1.0),
            n(0.0, 0.0, 1.0),
            n(0.0, 0.0, 1.0),
            // bottom
            n(0.0, 0.0, -1.0),
            n(0.0, 0.0, -1.0),
            n(0.0, 0.0, -1.0),
            n(0.0, 0.0, -1.0),
            // right
            n(1.0, 0.0, 0.0),
            n(1.0, 0.0, 0.0),
            n(1.0, 0.0, 0.0),
            n(1.0, 0.0, 0.0),
            // left
            n(-1.0, 0.0, 0.0),
            n(-1.0, 0.0, 0.0),
            n(-1.0, 0.0, 0.0),
            n(-1.0, 0.0, 0.0),
        ];

        let faces = vec![
            // back
            TriangleFace::new(2, 1, 0),
            TriangleFace::new(2, 3, 0),
            // top
            TriangleFace::new(2, 1, 5),
            TriangleFace::new(2, 5, 6),
            // right
            TriangleFace::new(2, 6, 7),
            TriangleFace::new(7, 3, 2),
            // bottom
            TriangleFace::new(3, 7, 4),
            TriangleFace::new(4, 0, 3),
            // front
            TriangleFace::new(6, 4, 7),
            TriangleFace::new(4, 6, 5),
            // left
            TriangleFace::new(0, 4, 5),
            TriangleFace::new(5, 1, 0),
        ];

        Geometry::from_triangle_faces_with_vertices_and_normals(
            faces,
            vertex_positions,
            vertex_normals,
        )
    }

    #[test]
    fn test_mesh_analysis_find_edge_with_valency() {
        let (faces, vertices) = quad();
        let geometry = Geometry::from_triangle_faces_with_vertices_and_computed_normals(
            faces.clone(),
            vertices.clone(),
            NormalStrategy::Sharp,
        );
        let oriented_edges: Vec<OrientedEdge> = geometry.oriented_edges_iter().collect();
        let edge_valency_map = edge_valencies(&oriented_edges);

        let oriented_edges_with_valency_1_correct = vec![
            OrientedEdge::new(0, 1),
            OrientedEdge::new(1, 2),
            OrientedEdge::new(2, 3),
            OrientedEdge::new(3, 0),
        ];
        let oriented_edges_with_valency_2_correct =
            vec![OrientedEdge::new(2, 0), OrientedEdge::new(0, 2)];

        let oriented_edges_with_valency_1: Vec<_> =
            find_edges_with_valency(&edge_valency_map, 1).collect();
        let oriented_edges_with_valency_2: Vec<_> =
            find_edges_with_valency(&edge_valency_map, 2).collect();

        assert_eq!(oriented_edges_with_valency_1.len(), 4);
        assert_eq!(oriented_edges_with_valency_2.len(), 2);

        for o_e in oriented_edges_with_valency_1_correct {
            assert!(oriented_edges_with_valency_1.iter().any(|e| *e == o_e));
        }

        for o_e in oriented_edges_with_valency_2_correct {
            assert!(oriented_edges_with_valency_2.iter().any(|e| *e == o_e));
        }
    }

    #[test]
    fn test_mesh_analysis_border_edges() {
        let (faces, vertices) = quad();
        let geometry = Geometry::from_triangle_faces_with_vertices_and_computed_normals(
            faces.clone(),
            vertices.clone(),
            NormalStrategy::Sharp,
        );
        let oriented_edges: Vec<OrientedEdge> = geometry.oriented_edges_iter().collect();
        let edge_valency_map = edge_valencies(&oriented_edges);

        let oriented_edges_border_correct = vec![
            OrientedEdge::new(0, 1),
            OrientedEdge::new(1, 2),
            OrientedEdge::new(2, 3),
            OrientedEdge::new(3, 0),
        ];

        let oriented_edges_border_check: Vec<_> = border_edges(&edge_valency_map).collect();

        for o_e in &oriented_edges_border_correct {
            assert!(oriented_edges_border_check.iter().any(|e| e == o_e));
        }
    }

    #[test]
    fn test_mesh_analysis_manifold_edges() {
        let (faces, vertices) = quad();
        let geometry = Geometry::from_triangle_faces_with_vertices_and_computed_normals(
            faces.clone(),
            vertices.clone(),
            NormalStrategy::Sharp,
        );
        let oriented_edges: Vec<OrientedEdge> = geometry.oriented_edges_iter().collect();
        let edge_valency_map = edge_valencies(&oriented_edges);

        let oriented_edges_manifold_correct =
            vec![OrientedEdge::new(2, 0), OrientedEdge::new(0, 2)];

        let oriented_edges_manifold_check: Vec<_> = manifold_edges(&edge_valency_map).collect();

        for o_e in oriented_edges_manifold_correct {
            assert!(oriented_edges_manifold_check.iter().any(|e| *e == o_e));
        }
    }

    #[test]
    fn test_mesh_analysis_non_manifold_edges() {
        let (faces, vertices) = non_manifold_shape();
        let geometry = Geometry::from_triangle_faces_with_vertices_and_computed_normals(
            faces.clone(),
            vertices.clone(),
            NormalStrategy::Sharp,
        );
        let oriented_edges: Vec<OrientedEdge> = geometry.oriented_edges_iter().collect();
        let edge_valency_map = edge_valencies(&oriented_edges);

        let oriented_edges_non_manifold_correct =
            vec![OrientedEdge::new(2, 0), OrientedEdge::new(0, 2)];

        let oriented_edges_non_manifold_check: Vec<_> =
            non_manifold_edges(&edge_valency_map).collect();

        for o_e in &oriented_edges_non_manifold_correct {
            assert!(oriented_edges_non_manifold_check.iter().any(|e| e == o_e));
        }

        assert_eq!(oriented_edges_non_manifold_check.len(), 3);
    }

    #[test]
    fn test_mesh_analysis_border_vertex_indices() {
        let (faces, vertices) = quad();
        let geometry = Geometry::from_triangle_faces_with_vertices_and_computed_normals(
            faces.clone(),
            vertices.clone(),
            NormalStrategy::Sharp,
        );
        let oriented_edges: Vec<OrientedEdge> = geometry.oriented_edges_iter().collect();
        let edge_valency_map = edge_valencies(&oriented_edges);

        let border_vertex_indices_correct = vec![0, 1, 2, 3];

        let border_vertex_indices_check = border_vertex_indices(&edge_valency_map);

        for v_i in border_vertex_indices_correct {
            assert!(border_vertex_indices_check.iter().any(|v| *v == v_i));
        }
    }

    #[test]
    fn test_mesh_analysis_is_mesh_orientable_returns_true_watertight_mesh() {
        let geometry = geometry::cube_sharp_var_len([0.0, 0.0, 0.0], 1.0);
        let oriented_edges: Vec<OrientedEdge> = geometry.oriented_edges_iter().collect();
        let edge_valency_map = edge_valencies(&oriented_edges);

        assert!(is_mesh_orientable(&edge_valency_map));
    }

    #[test]
    fn test_mesh_analysis_is_mesh_orientable_returns_true_open_mesh() {
        let (faces, vertices) = quad();
        let geometry = Geometry::from_triangle_faces_with_vertices_and_computed_normals(
            faces.clone(),
            vertices.clone(),
            NormalStrategy::Sharp,
        );
        let oriented_edges: Vec<OrientedEdge> = geometry.oriented_edges_iter().collect();
        let edge_valency_map = edge_valencies(&oriented_edges);

        assert!(is_mesh_orientable(&edge_valency_map));
    }

    #[test]
    fn test_mesh_analysis_is_mesh_orientable_returns_false_for_nonorientable_mesh() {
        let geometry = cube_sharp_mismatching_winding([0.0, 0.0, 0.0], 1.0);

        let oriented_edges: Vec<OrientedEdge> = geometry.oriented_edges_iter().collect();
        let edge_valency_map = edge_valencies(&oriented_edges);

        assert!(!is_mesh_orientable(&edge_valency_map));
    }

    #[test]
    fn test_mesh_analysis_is_mesh_watertight_returns_true_for_watertight_mesh() {
        let geometry = geometry::cube_sharp_var_len([0.0, 0.0, 0.0], 1.0);
        let oriented_edges: Vec<OrientedEdge> = geometry.oriented_edges_iter().collect();
        let edge_valency_map = edge_valencies(&oriented_edges);

        assert!(is_mesh_watertight(&edge_valency_map));
    }

    #[test]
    fn test_mesh_analysis_is_mesh_watertight_returns_false_for_open_mesh() {
        let (faces, vertices) = quad();
        let geometry = Geometry::from_triangle_faces_with_vertices_and_computed_normals(
            faces.clone(),
            vertices.clone(),
            NormalStrategy::Sharp,
        );

        let oriented_edges: Vec<OrientedEdge> = geometry.oriented_edges_iter().collect();
        let edge_valency_map = edge_valencies(&oriented_edges);

        assert!(!is_mesh_watertight(&edge_valency_map));
    }

    #[test]
    fn test_geometry_mesh_genus_box_should_be_0() {
        let geometry = cube_sharp_var_len([0.0, 0.0, 0.0], 1.0);
        let edges: HashSet<UnorientedEdge> = geometry.unoriented_edges_iter().collect();

        let genus = mesh_genus(
            geometry.vertices().len(),
            edges.len(),
            geometry.triangle_faces_iter().count(),
        );
        assert_eq!(genus, 0);
    }

    #[test]
    fn test_geometry_mesh_genus_torus_should_be_1() {
        let (faces, vertices) = torus();
        let geometry = Geometry::from_triangle_faces_with_vertices_and_computed_normals(
            faces.clone(),
            vertices.clone(),
            NormalStrategy::Sharp,
        );
        let edges: HashSet<UnorientedEdge> = geometry.unoriented_edges_iter().collect();

        let genus = mesh_genus(
            geometry.vertices().len(),
            edges.len(),
            geometry.triangle_faces_iter().count(),
        );
        assert_eq!(genus, 1);
    }

    #[test]
    fn test_geometry_mesh_genus_double_torus_should_be_2() {
        let (faces, vertices) = double_torus();
        let geometry = Geometry::from_triangle_faces_with_vertices_and_computed_normals(
            faces.clone(),
            vertices.clone(),
            NormalStrategy::Sharp,
        );
        let edges: HashSet<UnorientedEdge> = geometry.unoriented_edges_iter().collect();

        let genus = mesh_genus(
            geometry.vertices().len(),
            edges.len(),
            geometry.triangle_faces_iter().count(),
        );
        assert_eq!(genus, 2);
    }

    #[test]
    fn test_geometry_mesh_genus_triple_torus_should_be_3() {
        let (faces, vertices) = triple_torus();
        let geometry = Geometry::from_triangle_faces_with_vertices_and_computed_normals(
            faces.clone(),
            vertices.clone(),
            NormalStrategy::Sharp,
        );
        let edges: HashSet<UnorientedEdge> = geometry.unoriented_edges_iter().collect();

        let genus = mesh_genus(
            geometry.vertices().len(),
            edges.len(),
            geometry.triangle_faces_iter().count(),
        );
        assert_eq!(genus, 3);
    }
}
