use std::cmp;
use std::collections::HashMap;

use crate::geometry::{Edge, OrientedEdge};

/// Calculate edge valencies = number of faces sharing an edge
/// 1 -> border edge = acceptable but mesh is not watertight
/// 2 -> manifold edge = correct
/// 3 or more -> non-manifold edge = corrupted mesh
#[allow(dead_code)]
pub fn edge_valencies(edges: &[Edge]) -> HashMap<Edge, usize> {
    let mut edge_valency_map: HashMap<Edge, usize> = HashMap::new();
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
                let other_edge = Edge::Oriented(OrientedEdge::from((
                    oriented_edge.vertices.1,
                    oriented_edge.vertices.0,
                )));
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
