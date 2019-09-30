use std::collections::{HashMap, HashSet};

use crate::geometry::{Edge, Geometry, HalfEdge};

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

/// Finds border edges in a mesh edge collection
/// An edge is border when its valency is 1
#[allow(dead_code)]
pub fn border_edges(edge_valencies: HashMap<Edge, usize>) -> Vec<Edge> {
    let keys = edge_valencies.keys();
    keys.filter(|key| edge_valencies[key] == 1)
        .copied()
        .collect()
}

/// Finds border edges in a mesh half-edge collection
/// A half-edge is border when its valency is 1
#[allow(dead_code)]
pub fn border_half_edges(half_edge_valencies: HashMap<HalfEdge, usize>) -> Vec<HalfEdge> {
    let keys = half_edge_valencies.keys();
    keys.filter(|key| half_edge_valencies[key] == 1)
        .copied()
        .collect()
}

/// Finds border vertex indices in a mesh half-edge collection
/// A vertex is border when its half-edge's valency is 1
#[allow(dead_code)]
pub fn border_vertex_indices_from_half_edges(
    half_edge_valencies: HashMap<HalfEdge, usize>,
) -> Vec<u32> {
    let keys = half_edge_valencies.keys();
    let mut border_vertices = HashSet::new();
    keys.filter(|key| half_edge_valencies[key] == 1)
        .for_each(|half_edge| {
            border_vertices.insert(half_edge.vertices.0);
            border_vertices.insert(half_edge.vertices.1);
        });
    border_vertices.iter().copied().collect()
}

/// Finds border vertex indices in a mesh edge collection
/// A vertex is border when its edge's valency is 1
#[allow(dead_code)]
pub fn border_vertex_indices_from_edges(edge_valencies: HashMap<Edge, usize>) -> Vec<u32> {
    let keys = edge_valencies.keys();
    let mut border_vertices = HashSet::new();
    keys.filter(|key| edge_valencies[key] == 1)
        .for_each(|edge| {
            border_vertices.insert(edge.vertices.0);
            border_vertices.insert(edge.vertices.1);
        });
    border_vertices.iter().copied().collect()
}
