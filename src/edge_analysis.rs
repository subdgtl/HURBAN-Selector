use std::collections::HashMap;

use crate::geometry::{OrientedEdge, UnorientedEdge};

/// Used in EdgeCountMap
/// ascending_count contains number of edges oriented from lower index to higher
/// descending_count contains number of edges oriented from higher index to lower
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct EdgeCount {
    pub ascending_count: u32,
    pub descending_count: u32,
}

pub type EdgeCountMap = HashMap<UnorientedEdge, EdgeCount>;

/// Calculate edge valencies = number of faces sharing an edge
/// 1 -> border edge = acceptable but mesh is not watertight
/// 2 -> manifold edge = correct
/// 3 or more -> non-manifold edge = corrupted mesh
#[allow(dead_code)]
pub fn edge_valencies(oriented_edges: &[OrientedEdge]) -> EdgeCountMap {
    let mut edge_valency_map: EdgeCountMap = HashMap::new();
    for edge in oriented_edges {
        let unoriented_edge = UnorientedEdge(*edge);
        let mut ascending_count: u32 = 0;
        let mut descending_count: u32 = 0;
        if edge.vertices.0 < edge.vertices.1 {
            ascending_count += 1;
        } else {
            descending_count += 1;
        }
        if let Some(edge_count) = edge_valency_map.get(&unoriented_edge) {
            ascending_count += edge_count.ascending_count;
            descending_count += edge_count.descending_count;
        }

        let edge_count = EdgeCount {
            ascending_count,
            descending_count,
        };
        edge_valency_map.insert(unoriented_edge, edge_count);
    }

    edge_valency_map
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::geometry::{Geometry, NormalStrategy};
    use nalgebra::Point3;

    fn v(x: f32, y: f32, z: f32, translation: [f32; 3], scale: f32) -> Point3<f32> {
        Point3::new(
            scale * x + translation[0],
            scale * y + translation[1],
            scale * z + translation[2],
        )
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

    #[test]
    fn test_edge_analysis_edge_valencies() {
        let (faces, vertices) = quad();
        let geometry = Geometry::from_triangle_faces_with_vertices_and_computed_normals(
            faces.clone(),
            vertices.clone(),
            NormalStrategy::Sharp,
        );
        let oriented_edges: Vec<OrientedEdge> = geometry.oriented_edges_iter().collect();
        let edge_valency_map = edge_valencies(&oriented_edges);
        let unoriented_edges_one_way_correct = vec![
            UnorientedEdge(OrientedEdge::new(0, 1)),
            UnorientedEdge(OrientedEdge::new(1, 2)),
            UnorientedEdge(OrientedEdge::new(2, 3)),
            UnorientedEdge(OrientedEdge::new(3, 0)),
        ];
        let unoriented_edges_two_ways_correct = vec![
            UnorientedEdge(OrientedEdge::new(2, 0)),
            UnorientedEdge(OrientedEdge::new(0, 2)),
        ];
        let test_edge_count_ascending = EdgeCount {
            ascending_count: 1,
            descending_count: 0,
        };
        let test_edge_count_descending = EdgeCount {
            ascending_count: 0,
            descending_count: 1,
        };
        let test_edge_count_two_ways = EdgeCount {
            ascending_count: 1,
            descending_count: 1,
        };

        for u_e in unoriented_edges_one_way_correct {
            if u_e.0.vertices.0 < u_e.0.vertices.1 {
                assert_eq!(edge_valency_map.get(&u_e), Some(&test_edge_count_ascending));
            } else {
                assert_eq!(
                    edge_valency_map.get(&u_e),
                    Some(&test_edge_count_descending)
                );
            }
        }

        for u_e in unoriented_edges_two_ways_correct {
            assert_eq!(edge_valency_map.get(&u_e), Some(&test_edge_count_two_ways));
        }
    }
}
