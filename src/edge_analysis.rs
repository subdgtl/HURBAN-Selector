use std::collections::HashMap;

use crate::geometry::Edge;

/// Calculate edge valencies = number of faces sharing an edge
/// 1 -> border edge = acceptable but mesh is not watertight
/// 2 -> manifold edge = correct
/// 3 or more -> non-manifold edge = corrupted mesh
#[allow(dead_code)]
pub fn edge_valencies(edges: &[Edge]) -> HashMap<Edge, usize> {
    let mut edge_valency_map: HashMap<Edge, usize> = HashMap::new();
    for edge in edges {
        let valencies = match edge_valency_map.get(edge) {
            Some(v) => *v + 1,
            None => 1,
        };
        edge_valency_map.insert(*edge, valencies);
    }
    edge_valency_map
}
