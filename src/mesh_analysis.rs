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
pub fn border_edges(edges: HashMap<Edge, usize>) -> Vec<Edge> {
    let keys = edges.keys();
    keys.filter(|key| edges[key] == 1).copied().collect()
}

/// Finds border edges in a mesh half-edge collection
/// A half-edge is border when its valency is 1
#[allow(dead_code)]
pub fn border_half_edges(half_edges: HashMap<HalfEdge, usize>) -> Vec<HalfEdge> {
    let keys = half_edges.keys();
    keys.filter(|key| half_edges[key] == 1).copied().collect()
}
