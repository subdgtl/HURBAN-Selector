use std::collections::HashMap;

use crate::geometry::{OrientedEdge, UnorientedEdge};

/// Used in EdgeSharingMap
/// ascending_edges contains edges oriented from lower index to higher
/// descending_edges contains edges oriented from higher index to lower
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct SharedEdges {
    pub ascending_edges: Vec<OrientedEdge>,
    pub descending_edges: Vec<OrientedEdge>,
}

pub type EdgeSharingMap = HashMap<UnorientedEdge, SharedEdges>;

/// Calculate edge sharing and pack shared edges into SharedEdges
/// Edge valency = number of faces sharing an edge
/// 1 -> border edge = acceptable but mesh is not watertight
/// 2 -> manifold edge = correct
/// 3 or more -> non-manifold edge = corrupted mesh
// FIXME: Implement also for UnorientedEdges
#[allow(dead_code)]
pub fn edge_sharing<'a, I: IntoIterator<Item = &'a OrientedEdge>>(
    oriented_edges: I,
) -> EdgeSharingMap {
    let mut edge_sharing_map: EdgeSharingMap = HashMap::new();
    for edge in oriented_edges {
        let unoriented_edge = UnorientedEdge(*edge);
        let ascending_edges: Vec<OrientedEdge> = Vec::new();
        let descending_edges: Vec<OrientedEdge> = Vec::new();

        let shared_edges = edge_sharing_map
            .entry(unoriented_edge)
            .or_insert(SharedEdges {
                ascending_edges,
                descending_edges,
            });

        if edge.vertices.0 < edge.vertices.1 {
            shared_edges.ascending_edges.push(*edge);
        } else {
            shared_edges.descending_edges.push(*edge);
        }
    }

    edge_sharing_map
}

#[cfg(test)]
mod tests {
    use nalgebra::Point3;

    use crate::geometry::{Geometry, NormalStrategy, Vertices};

    use super::*;

    fn v(x: f32, y: f32, z: f32, translation: [f32; 3], scale: f32) -> Point3<f32> {
        Point3::new(
            scale * x + translation[0],
            scale * y + translation[1],
            scale * z + translation[2],
        )
    }

    fn quad() -> (Vec<(u32, u32, u32)>, Vertices) {
        #[rustfmt::skip]
        let vertices = vec![
            v(-1.0, -1.0, 0.0, [0.0, 0.0, 0.0], 1.0),
            v(1.0, -1.0, 0.0, [0.0, 0.0, 0.0], 1.0),
            v(1.0, 1.0, 0.0, [0.0, 0.0, 0.0], 1.0),
            v(-1.0, 1.0, 0.0, [0.0, 0.0, 0.0], 1.0),
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
        let edge_sharing_map = edge_sharing(&oriented_edges);
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

        for u_e in unoriented_edges_one_way_correct {
            assert!(edge_sharing_map.contains_key(&u_e));
            if u_e.0.vertices.0 < u_e.0.vertices.1 {
                assert_eq!(edge_sharing_map.get(&u_e).unwrap().ascending_edges.len(), 1);
                assert_eq!(
                    edge_sharing_map.get(&u_e).unwrap().descending_edges.len(),
                    0
                );
            } else {
                assert_eq!(edge_sharing_map.get(&u_e).unwrap().ascending_edges.len(), 0);
                assert_eq!(
                    edge_sharing_map.get(&u_e).unwrap().descending_edges.len(),
                    1
                );
            }
        }

        for u_e in unoriented_edges_two_ways_correct {
            assert!(edge_sharing_map.contains_key(&u_e));
            assert_eq!(edge_sharing_map.get(&u_e).unwrap().ascending_edges.len(), 1);
            assert_eq!(
                edge_sharing_map.get(&u_e).unwrap().descending_edges.len(),
                1
            );
        }
    }
}
