use std::cmp;
use std::collections::HashMap;

use crate::geometry::{Edge, HalfEdge};

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

/// Calculate half-edge valencies = number of faces sharing a half-edge and its reversed half-edge
/// 1 -> border half-edge = acceptable but mesh is not watertight
/// 2 -> manifold half-edge = correct
/// 3 or more -> non-manifold half-edge = corrupted mesh
#[allow(dead_code)]
pub fn half_edge_valencies(half_edges: &[HalfEdge]) -> HashMap<HalfEdge, usize> {
    let mut half_edge_valency_map: HashMap<HalfEdge, usize> = HashMap::new();
    for half_edge in half_edges {
        let one_way_valencies = match half_edge_valency_map.get(half_edge) {
            Some(v) => *v + 1,
            None => 1,
        };
        let other_half_edge = HalfEdge::from((half_edge.vertices.1, half_edge.vertices.0));
        let other_way_valencies = match half_edge_valency_map.get(&other_half_edge) {
            Some(v) => *v + 1,
            None => 0,
        };
        let valencies = cmp::max(one_way_valencies, other_way_valencies);
        half_edge_valency_map.insert(*half_edge, valencies);
        half_edge_valency_map.insert(other_half_edge, valencies);
    }
    half_edge_valency_map
}
