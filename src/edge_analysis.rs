use std::cmp;
use std::collections::HashMap;

use crate::geometry::Edge;

//struct EdgeWrapper (OrientedEdge);
//struct EdgeInfo {
//    one_way_count: u32,
//    other_way_count: u32,
//};
//
//let map: HashMap<EdgeWrapper, EdgeInfo>;

/// Calculate edge valencies = number of faces sharing an edge
/// 1 -> border edge = acceptable but mesh is not watertight
/// 2 -> manifold edge = correct
/// 3 or more -> non-manifold edge = corrupted mesh
#[allow(dead_code)]
pub fn edge_valencies(edges: &[Edge]) -> HashMap<Edge, u32> {
    let mut edge_valency_map: HashMap<Edge, u32> = HashMap::new();
    for edge in edges {
        match edge {
            Edge::Unoriented(_) => {
                let valencies = match edge_valency_map.get(edge) {
                    Some(v) => *v + 1,
                    None => 1,
                };
                edge_valency_map.insert(*edge, valencies);
            }
            Edge::Oriented(oriented_edge) => {
                let one_way_valencies = match edge_valency_map.get(edge) {
                    Some(v) => *v + 1,
                    None => 1,
                };
                let other_edge = Edge::Oriented(oriented_edge.reverted());
                let other_way_valencies = match edge_valency_map.get(&other_edge) {
                    Some(v) => *v + 1,
                    None => 0,
                };
                let valencies = cmp::max(one_way_valencies, other_way_valencies);
                edge_valency_map.insert(*edge, valencies);
                edge_valency_map.insert(other_edge, valencies);
            }
        }
    }
    edge_valency_map
}
