use std::cmp;
use std::collections::HashMap;

use crate::geometry::{Edge, OrientedEdge, UnorientedEdge};

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

/// Calculate unoriented edge valencies = number of faces sharing an unoriented edge
/// 1 -> border unoriented edge = acceptable but mesh is not watertight
/// 2 -> manifold unoriented edge = correct
/// 3 or more -> non-manifold unoriented edge = corrupted mesh
#[allow(dead_code)]
pub fn unoriented_edge_valencies(
    unoriented_edges: &[UnorientedEdge],
) -> HashMap<UnorientedEdge, usize> {
    let mut unoriented_edge_valency_map: HashMap<UnorientedEdge, usize> = HashMap::new();
    for unoriented_edge in unoriented_edges {
        let valencies = match unoriented_edge_valency_map.get(unoriented_edge) {
            Some(v) => *v + 1,
            None => 1,
        };
        unoriented_edge_valency_map.insert(*unoriented_edge, valencies);
    }
    unoriented_edge_valency_map
}

/// Calculate oriented edge valencies = number of faces sharing a oriented edge and its reversed oriented edge
/// 1 -> border oriented edge = acceptable but mesh is not watertight
/// 2 -> manifold oriented edge = correct
/// 3 or more -> non-manifold oriented edge = corrupted mesh
#[allow(dead_code)]
pub fn oriented_edge_valencies(oriented_edges: &[OrientedEdge]) -> HashMap<OrientedEdge, usize> {
    let mut oriented_edge_valency_map: HashMap<OrientedEdge, usize> = HashMap::new();
    for oriented_edge in oriented_edges {
        let one_way_valencies = match oriented_edge_valency_map.get(oriented_edge) {
            Some(v) => *v + 1,
            None => 1,
        };
        let other_oriented_edge =
            OrientedEdge::from((oriented_edge.vertices.1, oriented_edge.vertices.0));
        let other_way_valencies = match oriented_edge_valency_map.get(&other_oriented_edge) {
            Some(v) => *v + 1,
            None => 0,
        };
        let valencies = cmp::max(one_way_valencies, other_way_valencies);
        oriented_edge_valency_map.insert(*oriented_edge, valencies);
        oriented_edge_valency_map.insert(other_oriented_edge, valencies);
    }
    oriented_edge_valency_map
}
