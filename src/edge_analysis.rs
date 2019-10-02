use std::collections::HashMap;

use crate::geometry::{EdgeCount, EdgeCountMap, UnorientedEdge, OrientedEdge};

//let map: HashMap<EdgeWrapper, EdgeInfo>;

/// Calculate edge valencies = number of faces sharing an edge
/// 1 -> border edge = acceptable but mesh is not watertight
/// 2 -> manifold edge = correct
/// 3 or more -> non-manifold edge = corrupted mesh
#[allow(dead_code)]
pub fn edge_valencies(edges: &[OrientedEdge]) -> EdgeCountMap {
    let mut edge_valency_map: EdgeCountMap = HashMap::new();
    for edge in edges {
        let edge_wrapped = UnorientedEdge(*edge);
        let mut ascending_count: u32 = 0;
        let mut descending_count: u32 = 0;
        if edge.vertices.0 < edge.vertices.1 {
            ascending_count += 1;
        } else {
            descending_count += 1;
        }
        if let Some(edge_count) = edge_valency_map.get(&edge_wrapped) {
            ascending_count += edge_count.ascending_count;
            descending_count += edge_count.descending_count;
        }

        let edge_count = EdgeCount {
            ascending_count,
            descending_count,
        };
        edge_valency_map.insert(edge_wrapped, edge_count);
    }

    edge_valency_map
}
