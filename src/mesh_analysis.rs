use std::collections::{HashMap, HashSet};

use crate::geometry::{Geometry, OrientedEdge, UnorientedEdge};

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

/// Finds border edges in a mesh unoriented edge collection
/// An unoriented edge is border when its valency is 1
#[allow(dead_code)]
pub fn border_edges(
    unoriented_edge_valencies: HashMap<UnorientedEdge, usize>,
) -> Vec<UnorientedEdge> {
    let keys = unoriented_edge_valencies.keys();
    keys.filter(|key| unoriented_edge_valencies[key] == 1)
        .copied()
        .collect()
}

/// Finds border edges in a mesh oriented edge collection
/// A oriented edge is border when its valency is 1
#[allow(dead_code)]
pub fn border_oriented_edges(
    oriented_edge_valencies: HashMap<OrientedEdge, usize>,
) -> Vec<OrientedEdge> {
    let keys = oriented_edge_valencies.keys();
    keys.filter(|key| oriented_edge_valencies[key] == 1)
        .copied()
        .collect()
}

/// Finds border vertex indices in a mesh oriented edge collection
/// A vertex is border when its oriented edge's valency is 1
#[allow(dead_code)]
pub fn border_vertex_indices_from_oriented_edges(
    oriented_edge_valencies: HashMap<OrientedEdge, usize>,
) -> Vec<u32> {
    let keys = oriented_edge_valencies.keys();
    let mut border_vertices = HashSet::new();
    keys.filter(|key| oriented_edge_valencies[key] == 1)
        .for_each(|oriented_edge| {
            border_vertices.insert(oriented_edge.vertices.0);
            border_vertices.insert(oriented_edge.vertices.1);
        });
    border_vertices.iter().copied().collect()
}

/// Finds border vertex indices in a mesh unoriented edge collection
/// A vertex is border when its unoriented edge's valency is 1
#[allow(dead_code)]
pub fn border_vertex_indices_from_edges(
    unoriented_edge_valencies: HashMap<UnorientedEdge, usize>,
) -> Vec<u32> {
    let keys = unoriented_edge_valencies.keys();
    let mut border_vertices = HashSet::new();
    keys.filter(|key| unoriented_edge_valencies[key] == 1)
        .for_each(|unoriented_edge| {
            border_vertices.insert(unoriented_edge.vertices.0);
            border_vertices.insert(unoriented_edge.vertices.1);
        });
    border_vertices.iter().copied().collect()
}
