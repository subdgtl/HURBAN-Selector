use std::collections::HashSet;

use crate::edge_analysis::EdgeCountMap;
use crate::geometry::{Geometry, OrientedEdge};

/// Check if all the vertices of geometry are referenced in geometry's faces
#[allow(dead_code)]
pub fn has_no_orphans(geo: &Geometry) -> bool {
    let mut used_vertices = HashSet::new();
    for face in geo.triangle_faces_iter() {
        used_vertices.insert(face.vertices.0);
        used_vertices.insert(face.vertices.1);
        used_vertices.insert(face.vertices.2);
    }
    used_vertices.len() == geo.vertices().len()
}

/// Finds edges with a certain valency in a mesh edge collection
/// Valency indicated how many faces share the edge
fn find_edges_with_valency<'a>(
    edge_valencies: &'a EdgeCountMap,
    valency: u32,
) -> impl Iterator<Item = OrientedEdge> + 'a {
    edge_valencies
        .iter()
        .filter(move |(_edge_wrapper, edge_count)| {
            edge_count.ascending_count + edge_count.descending_count == valency
        })
        .map(|(edge_wrapper, _value)| edge_wrapper.0)
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
        .filter(move |(_edge_wrapper, edge_count)| {
            edge_count.ascending_count + edge_count.descending_count > 2
        })
        .map(|(edge_wrapper, _value)| edge_wrapper.0)
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
/// In a proper watertight mesh each oriented edge
/// should have a counterpart in a reverted oriented edge
// TODO: make this work for open meshes
#[allow(dead_code)]
pub fn is_mesh_orientable(edge_valencies: &EdgeCountMap) -> bool {
    edge_valencies.iter().all(|(_edge_wrapper, edge_count)| {
        edge_count.ascending_count == 1 && edge_count.descending_count == 1
    })
}

/// The mesh is watertight if there is no border or non-manifold edge,
/// in other words, all edge valencies are 2
#[allow(dead_code)]
pub fn is_mesh_watertight(edge_valencies: &EdgeCountMap) -> bool {
    edge_valencies.iter().all(|(_edge_wrapper, edge_count)| {
        edge_count.ascending_count + edge_count.descending_count == 2
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::edge_analysis::edge_valencies;
    use crate::geometry;
    use crate::geometry::{cube_sharp_same_len_open, NormalStrategy, TriangleFace};
    use nalgebra::{Point3, Vector3};

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

    fn quad() -> (Vec<(u32, u32, u32)>, Vec<Point3<f32>>) {
        #[rustfmt::skip]
            let vertices = vec![
            v(-1.0, -1.0,  0.0, [0.0, 0.0, 0.0], 1.0),
            v( 1.0, -1.0,  0.0, [0.0, 0.0, 0.0], 1.0),
            v( 1.0,  1.0,  0.0, [0.0, 0.0, 0.0], 1.0),
            v(-1.0,  1.0,  0.0, [0.0, 0.0, 0.0], 1.0),
        ];

        #[rustfmt::skip]
            let faces = vec![
            (0, 1, 2),
            (2, 3, 0),
        ];

        (faces, vertices)
    }

    fn non_manifold_shape() -> (Vec<(u32, u32, u32)>, Vec<Point3<f32>>) {
        #[rustfmt::skip]
            let vertices = vec![
            v(-1.0, -1.0,  0.0, [0.0, 0.0, 0.0], 1.0),
            v( 1.0, -1.0,  0.0, [0.0, 0.0, 0.0], 1.0),
            v( 1.0,  1.0,  0.0, [0.0, 0.0, 0.0], 1.0),
            v(-1.0,  1.0,  0.0, [0.0, 0.0, 0.0], 1.0),
            v(0.0,  0.0,  1.0, [0.0, 0.0, 0.0], 1.0),
        ];

        #[rustfmt::skip]
            let faces = vec![
            (0, 1, 2),
            (2, 3, 0),
            (2, 4, 0),
        ];

        (faces, vertices)
    }

    fn cube_sharp_same_len_broken_winding(position: [f32; 3], scale: f32) -> Geometry {
        #[rustfmt::skip]
            let vertex_positions = vec![
            // back
            v(-1.0,  1.0, -1.0, position, scale),
            v(-1.0,  1.0,  1.0, position, scale),
            v( 1.0,  1.0,  1.0, position, scale),
            v( 1.0,  1.0, -1.0, position, scale),
            // front
            v(-1.0, -1.0, -1.0, position, scale),
            v( -1.0, -1.0, 1.0, position, scale),
            v( 1.0, -1.0,  1.0, position, scale),
            v(1.0, -1.0,  -1.0, position, scale),
        ];

        #[rustfmt::skip]
            let vertex_normals = vec![
            // back
            n( 0.0,  1.0,  0.0),
            n( 0.0,  1.0,  0.0),
            n( 0.0,  1.0,  0.0),
            n( 0.0,  1.0,  0.0),
            // front
            n( 0.0, -1.0,  0.0),
            n( 0.0, -1.0,  0.0),
            n( 0.0, -1.0,  0.0),
            n( 0.0, -1.0,  0.0),
            // top
            n( 0.0,  0.0,  1.0),
            n( 0.0,  0.0,  1.0),
            n( 0.0,  0.0,  1.0),
            n( 0.0,  0.0,  1.0),
            // bottom
            n( 0.0,  0.0, -1.0),
            n( 0.0,  0.0, -1.0),
            n( 0.0,  0.0, -1.0),
            n( 0.0,  0.0, -1.0),
            // right
            n( 1.0,  0.0,  0.0),
            n( 1.0,  0.0,  0.0),
            n( 1.0,  0.0,  0.0),
            n( 1.0,  0.0,  0.0),
            // left
            n(-1.0,  0.0,  0.0),
            n(-1.0,  0.0,  0.0),
            n(-1.0,  0.0,  0.0),
            n(-1.0,  0.0,  0.0),
        ];

        #[rustfmt::skip]
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

        let oriented_edges_with_valency_1 = find_edges_with_valency(&edge_valency_map, 1);
        let oriented_edges_with_valency_2 = find_edges_with_valency(&edge_valency_map, 2);

        for o_e in oriented_edges_with_valency_1 {
            assert!(oriented_edges_with_valency_1_correct
                .iter()
                .any(|e| *e == o_e));
        }

        for o_e in oriented_edges_with_valency_2 {
            assert!(oriented_edges_with_valency_2_correct
                .iter()
                .any(|e| *e == o_e));
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

        let oriented_edges_border_check = border_edges(&edge_valency_map);

        for o_e in oriented_edges_border_check {
            assert!(oriented_edges_border_correct.iter().any(|e| *e == o_e));
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

        let oriented_edges_manifold_check = manifold_edges(&edge_valency_map);

        for o_e in oriented_edges_manifold_check {
            assert!(oriented_edges_manifold_correct.iter().any(|e| *e == o_e));
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

        for o_e in &oriented_edges_non_manifold_check {
            assert!(oriented_edges_non_manifold_correct.iter().any(|e| e == o_e));
        }

        assert_eq!(oriented_edges_non_manifold_check.len(), 1);
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

        let order_vertex_indices_check = border_vertex_indices(&edge_valency_map);

        for v_i in order_vertex_indices_check {
            assert!(border_vertex_indices_correct.iter().any(|v| *v == v_i));
        }
    }

    #[test]
    fn test_mesh_analysis_is_mesh_orientable_returns_true() {
        let geometry = geometry::cube_sharp_same_len_welded([0.0, 0.0, 0.0], 1.0);
        let oriented_edges: Vec<OrientedEdge> = geometry.oriented_edges_iter().collect();
        let edge_valency_map = edge_valencies(&oriented_edges);

        assert!(is_mesh_orientable(&edge_valency_map));
    }

    #[test]
    fn test_mesh_analysis_is_mesh_orientable_returns_false_because_is_wrong() {
        let geometry = cube_sharp_same_len_broken_winding([0.0, 0.0, 0.0], 1.0);

        let oriented_edges: Vec<OrientedEdge> = geometry.oriented_edges_iter().collect();
        let edge_valency_map = edge_valencies(&oriented_edges);

        assert!(!is_mesh_orientable(&edge_valency_map));
    }

    #[test]
    fn test_mesh_analysis_is_mesh_watertight_returns_true() {
        let geometry = geometry::cube_sharp_same_len_welded([0.0, 0.0, 0.0], 1.0);
        let oriented_edges: Vec<OrientedEdge> = geometry.oriented_edges_iter().collect();
        let edge_valency_map = edge_valencies(&oriented_edges);

        assert!(is_mesh_watertight(&edge_valency_map));
    }

    #[test]
    fn test_mesh_analysis_is_mesh_watertight_returns_false_because_is_open() {
        let geometry = cube_sharp_same_len_open([0.0, 0.0, 0.0], 1.0);

        let oriented_edges: Vec<OrientedEdge> = geometry.oriented_edges_iter().collect();
        let edge_valency_map = edge_valencies(&oriented_edges);

        assert!(!is_mesh_watertight(&edge_valency_map));
    }

}
