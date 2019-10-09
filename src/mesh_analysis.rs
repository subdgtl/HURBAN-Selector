use std::collections::HashSet;

use crate::edge_analysis::EdgeCountMap;
use crate::geometry::{OrientedEdge};

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
        .map(|(unoriented_edge, _)| unoriented_edge.0)
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

/// Genus of a mesh is the number of holes in topology / conectivity
/// The mesh must be triangular and watertight
/// V - E + F = 2 (1 - G)
#[allow(dead_code)]
pub fn mesh_genus(vertex_count: i32, edge_count: i32, face_count: i32) -> i32 {
    1 - (vertex_count - edge_count + face_count) / 2
}

#[cfg(test)]
mod tests {
    use crate::convert::cast_i32;
    use crate::edge_analysis::edge_valencies;
    use crate::geometry::{self, cube_sharp_var_len, NormalStrategy, UnorientedEdge, Geometry};
    use crate::test_geometry_fixtures::{
        cube_sharp_mismatching_winding, double_torus, non_manifold_shape, quad, torus, triple_torus,
    };

    use super::*;

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

        let mut oriented_edges_with_valency_1 = find_edges_with_valency(&edge_valency_map, 1);
        let mut oriented_edges_with_valency_2 = find_edges_with_valency(&edge_valency_map, 2);

        for o_e in oriented_edges_with_valency_1_correct {
            assert!(oriented_edges_with_valency_1.any(|e| e == o_e));
        }

        for o_e in oriented_edges_with_valency_2_correct {
            assert!(oriented_edges_with_valency_2.any(|e| e == o_e));
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
            cast_i32(geometry.vertices().len()),
            cast_i32(edges.len()),
            cast_i32(geometry.triangle_faces_iter().count()),
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
            cast_i32(geometry.vertices().len()),
            cast_i32(edges.len()),
            cast_i32(geometry.triangle_faces_iter().count()),
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
            cast_i32(geometry.vertices().len()),
            cast_i32(edges.len()),
            cast_i32(geometry.triangle_faces_iter().count()),
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
            cast_i32(geometry.vertices().len()),
            cast_i32(edges.len()),
            cast_i32(geometry.triangle_faces_iter().count()),
        );
        assert_eq!(genus, 3);
    }
}
