use std::collections::HashSet;

use crate::geometry::{EdgeCountMap, Geometry, OrientedEdge};

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

/// Finds edges with a certain valency in a mesh edge collection
/// Valency indicated how many faces share the edge
fn find_edge_with_valency<'a>(
    edge_valencies: &'a EdgeCountMap,
    valency: u32,
) -> impl Iterator<Item = OrientedEdge> + 'a {
    edge_valencies
        .iter()
        .filter(move |(_edge_wrapper, edge_count)| {
            edge_count.ascending_count + edge_count.descending_count == valency
        })
        .map(|(edge_wrapper, _value)| edge_wrapper.0)
}

/// Finds border edges in a mesh edge collection
/// An edge is border when its valency is 1
#[allow(dead_code)]
pub fn border_edges<'a>(
    edge_valencies: &'a EdgeCountMap,
) -> impl Iterator<Item = OrientedEdge> + 'a {
    find_edge_with_valency(edge_valencies, 1)
}

/// Finds manifold (inner) edges in a mesh edge collection
/// An edge is manifold when its valency is 2
#[allow(dead_code)]
pub fn manifold_edges<'a>(
    edge_valencies: &'a EdgeCountMap,
) -> impl Iterator<Item = OrientedEdge> + 'a {
    find_edge_with_valency(edge_valencies, 2)
}

/// Finds non-manifold (errorneous) edges in a mesh edge collection
#[allow(dead_code)]
pub fn non_manifold_edges<'a>(
    edge_valencies: &'a EdgeCountMap,
) -> impl Iterator<Item = OrientedEdge> + 'a {
    edge_valencies
        .iter()
        .filter(move |(_edge_wrapper, edge_count)| {
            edge_count.ascending_count + edge_count.descending_count > 2
        })
        .map(|(edge_wrapper, _value)| edge_wrapper.0)
}

/// Finds border vertex indices in a mesh edge collection
/// A vertex is border when its edge's valency is 1
#[allow(dead_code)]
pub fn border_vertex_indices_from_edges(edge_valencies: &EdgeCountMap) -> HashSet<u32> {
    let mut border_vertices = HashSet::new();

    border_edges(edge_valencies).for_each(|edge| {
        border_vertices.insert(edge.vertices.0);
        border_vertices.insert(edge.vertices.1);
    });
    border_vertices
}

/// Check if all the face normals point the same way.
/// In a proper watertight mesh each oriented edge
/// should have a counterpart in a reverted oriented edge
// TODO: make this work for open meshes
#[allow(dead_code)]
pub fn is_mesh_orientable(edge_valencies: &EdgeCountMap) -> bool {
    edge_valencies.iter().all(|(_edge_wrapper, edge_count)| {
        edge_count.ascending_count == 1 && edge_count.descending_count == 1
    })
}

/// The mesh is watertight if there is no border or non-manifold edge,
/// in other words, all edge valencies are 2
#[allow(dead_code)]
pub fn is_mesh_watertight(edge_valencies: &EdgeCountMap) -> bool {
    edge_valencies.iter().all(|(_edge_wrapper, edge_count)| {
        edge_count.ascending_count + edge_count.descending_count == 2
    })
}
