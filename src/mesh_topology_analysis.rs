use std::collections::HashMap;

use smallvec::SmallVec;

use crate::convert::cast_u32;
use crate::geometry::{Face, Geometry, OrientedEdge, UnorientedEdge};

/// Calculates topological relations (neighborhood) of mesh face -> faces.
/// Returns a Map (key: face index, value: list of its neighboring faces indices)
#[allow(dead_code)]
pub fn face_to_face_topology(geometry: &Geometry) -> HashMap<u32, SmallVec<[u32; 1]>> {
    let mut f2f: HashMap<u32, SmallVec<[u32; 1]>> = HashMap::new();

    for (from_face_index, from_face) in geometry.faces().iter().enumerate() {
        let from_face_index_u32 = cast_u32(from_face_index);
        let mut neighbors = SmallVec::new();

        match from_face {
            Face::Triangle(triangle_face) => {
                let [e1, e2, e3] = triangle_face.to_unoriented_edges();
                for (to_face_index, to_face) in geometry.faces().iter().enumerate() {
                    let to_face_index_u32 = cast_u32(to_face_index);

                    match to_face {
                        Face::Triangle(triangle_face) => {
                            let face_contains_edge = triangle_face.contains_unoriented_edge(e1)
                                || triangle_face.contains_unoriented_edge(e2)
                                || triangle_face.contains_unoriented_edge(e3);
                            if face_contains_edge
                                && from_face_index != to_face_index
                                && !neighbors.contains(&to_face_index_u32)
                            {
                                neighbors.push(to_face_index_u32);
                            }
                        }
                    }
                }
            }
        }

        f2f.insert(from_face_index_u32, neighbors);
    }

    f2f
}

/// Calculates topological relations (neighborhood) of mesh edge -> faces.
/// Returns a Map (key: edge index, value: list of its neighboring faces indices)
#[allow(dead_code)]
pub fn edge_to_face_topology(
    geometry: &Geometry,
    edges: &[UnorientedEdge],
) -> HashMap<u32, SmallVec<[u32; 8]>> {
    let mut e2f: HashMap<u32, SmallVec<[u32; 8]>> = HashMap::new();

    for (from_edge_index, from_edge) in edges.iter().enumerate() {
        let from_edge_index_u32 = cast_u32(from_edge_index);
        let mut neighbors = SmallVec::new();

        for (to_face_index, to_face) in geometry.faces().iter().enumerate() {
            let to_face_index_u32 = cast_u32(to_face_index);

            match to_face {
                Face::Triangle(triangle_face) => {
                    if triangle_face.contains_unoriented_edge(*from_edge)
                        && !neighbors.contains(&to_face_index_u32)
                    {
                        neighbors.push(to_face_index_u32);
                    }
                }
            }
        }

        e2f.insert(from_edge_index_u32, neighbors);
    }

    e2f
}

/// Calculates topological relations (containment) of mesh face -> its oriented
/// edges.
///
/// Returns an iterator (index: face index, value: list of indices of oriented
/// edges it contains
#[allow(dead_code)]
pub fn face_to_oriented_edge_topology<'a>(
    geometry: &'a Geometry,
    oriented_edges: &'a [OrientedEdge],
) -> impl Iterator<Item = SmallVec<[u32; 3]>> + 'a {
    geometry.faces().iter().map(move |f| match f {
        Face::Triangle(t_f) => SmallVec::from_vec(
            t_f.to_oriented_edges()
                .iter()
                .map(|o_e| {
                    cast_u32(
                        oriented_edges
                            .iter()
                            .position(|e| e == o_e)
                            .expect("The edge not found in the list"),
                    )
                })
                .collect(),
        ),
    })
}

/// Calculates topological relations (containment) of mesh oriented edge ->
/// faces containing it
///
/// Returns an iterator (index: oriented edge index, value: list of faces
/// containing it
#[allow(dead_code)]
pub fn unoriented_edge_to_face_topology<'a>(
    geometry: &'a Geometry,
    edges: &'a [UnorientedEdge],
) -> impl Iterator<Item = SmallVec<[u32; 2]>> + 'a {
    edges.iter().map(move |e| {
        SmallVec::from_vec(
            geometry
                .faces()
                .iter()
                .enumerate()
                .filter(|(_, f)| match f {
                    Face::Triangle(t_f) => t_f.contains_unoriented_edge(*e),
                })
                .map(|(i, _)| cast_u32(i))
                .collect(),
        )
    })
}

/// Calculates topological relations (neighborhood) of mesh vertex -> faces.
/// Returns a Map (key: vertex index, value: list of its neighboring faces indices)
#[allow(dead_code)]
pub fn vertex_to_face_topology(geometry: &Geometry) -> HashMap<u32, SmallVec<[u32; 8]>> {
    let mut v2f: HashMap<u32, SmallVec<[u32; 8]>> = HashMap::new();

    for (to_face_index, to_face) in geometry.faces().iter().enumerate() {
        let to_face_index_u32 = cast_u32(to_face_index);

        match to_face {
            Face::Triangle(triangle_face) => {
                let vertices = &triangle_face.vertices;

                for from_vertex in &[vertices.0, vertices.1, vertices.2] {
                    let neighbors = v2f.entry(*from_vertex).or_insert_with(SmallVec::new);

                    if !neighbors.contains(&to_face_index_u32) {
                        neighbors.push(to_face_index_u32);
                    }
                }
            }
        }
    }

    v2f
}

/// Calculates topological relations (neighborhood) of mesh vertex -> edge.
/// Returns a Map (key: vertex index, value: list of its neighboring edge indices)
#[allow(dead_code)]
pub fn vertex_to_edge_topology(edges: &[UnorientedEdge]) -> HashMap<u32, SmallVec<[u32; 8]>> {
    let mut v2e: HashMap<u32, SmallVec<[u32; 8]>> = HashMap::new();

    for (to_edge_index, to_edge) in edges.iter().enumerate() {
        let to_edge_index_u32 = cast_u32(to_edge_index);

        for from_vertex in &[to_edge.0.vertices.0, to_edge.0.vertices.1] {
            let neighbors = v2e.entry(*from_vertex).or_insert_with(SmallVec::new);

            if !neighbors.contains(&to_edge_index_u32) {
                neighbors.push(to_edge_index_u32);
            }
        }
    }

    v2e
}

/// Calculates topological relations (neighborhood) of mesh vertex -> vertex.
/// Returns a Map (key: vertex index, value: list of its neighboring vertices indices)
#[allow(dead_code)]
pub fn vertex_to_vertex_topology(geometry: &Geometry) -> HashMap<u32, SmallVec<[u32; 8]>> {
    let mut v2v: HashMap<u32, SmallVec<[u32; 8]>> = HashMap::new();

    for face in geometry.faces() {
        match face {
            Face::Triangle(f) => {
                let vertex_indices = &[f.vertices.0, f.vertices.1, f.vertices.2];
                for i in 0..vertex_indices.len() {
                    let neighbors = v2v.entry(vertex_indices[i]).or_insert_with(SmallVec::new);

                    let neighbor_candidate1 = vertex_indices[(i + 1) % 3];
                    let neighbor_candidate2 = vertex_indices[(i + 2) % 3];

                    if !neighbors.contains(&neighbor_candidate1) {
                        neighbors.push(neighbor_candidate1)
                    }
                    if !neighbors.contains(&neighbor_candidate2) {
                        neighbors.push(neighbor_candidate2)
                    }
                }
            }
        }
    }

    v2v
}

#[cfg(test)]
mod tests {
    use std::collections::HashSet;

    use nalgebra::geometry::Point3;
    use smallvec::smallvec;

    use crate::geometry::NormalStrategy;

    use super::*;

    fn tessellated_triangle() -> (Vec<(u32, u32, u32)>, Vec<Point3<f32>>) {
        let vertices = vec![
            Point3::new(-2.0, -2.0, 0.0),
            Point3::new(0.0, -2.0, 0.0),
            Point3::new(2.0, -2.0, 0.0),
            Point3::new(-1.0, 0.0, 0.0),
            Point3::new(1.0, 0.0, 0.0),
            Point3::new(0.0, 2.0, 0.0),
        ];

        let faces = vec![(0, 3, 1), (1, 3, 4), (1, 4, 2), (3, 5, 4)];

        (faces, vertices)
    }

    // FIXME: make proptests on all topologies verifying that self is
    // not included in neighbors, similar to what
    // test_geometry_face_to_face_topology_does_not_include_self_in_neighbors
    // does

    #[test]
    fn test_geometry_face_to_face_topology_does_not_include_self_in_neighbors() {
        let (faces, vertices) = tessellated_triangle();
        let geometry = Geometry::from_triangle_faces_with_vertices_and_computed_normals(
            faces,
            vertices,
            NormalStrategy::Sharp,
        );
        let face_to_face_topology = face_to_face_topology(&geometry);

        for (key, value) in face_to_face_topology {
            assert!(!value.contains(&key));
        }
    }

    #[test]
    fn test_geometry_face_to_face_topology_from_tessellated_triangle() {
        let (faces, vertices) = tessellated_triangle();
        let geometry = Geometry::from_triangle_faces_with_vertices_and_computed_normals(
            faces.clone(),
            vertices.clone(),
            NormalStrategy::Sharp,
        );
        let mut face_to_face_topology_correct: HashMap<u32, SmallVec<[u32; 1]>> = HashMap::new();
        face_to_face_topology_correct.insert(0, smallvec![1]);
        face_to_face_topology_correct.insert(1, smallvec![0, 2, 3]);
        face_to_face_topology_correct.insert(2, smallvec![1]);
        face_to_face_topology_correct.insert(3, smallvec![1]);

        let face_to_face_topology_calculated = face_to_face_topology(&geometry);

        assert_eq!(
            face_to_face_topology_calculated,
            face_to_face_topology_correct,
        );
    }

    #[test]
    fn test_geometry_edge_to_face_topology_from_tessellated_triangle() {
        let (faces, vertices) = tessellated_triangle();
        let geometry = Geometry::from_triangle_faces_with_vertices_and_computed_normals(
            faces.clone(),
            vertices.clone(),
            NormalStrategy::Sharp,
        );
        let edge_set: HashSet<_> = geometry.unoriented_edges_iter().collect();
        let edges: Vec<_> = edge_set.iter().cloned().collect();

        let edge_to_face_topology_calculated = edge_to_face_topology(&geometry, &edges);

        let in_one_face_count = edge_to_face_topology_calculated
            .iter()
            .filter(|(_, to)| to.len() == 1)
            .count();
        let in_two_faces_count = edge_to_face_topology_calculated
            .iter()
            .filter(|(_, to)| to.len() == 2)
            .count();

        assert_eq!(edges.len(), 9);
        assert_eq!(in_one_face_count, 6);
        assert_eq!(in_two_faces_count, 3);
    }

    #[test]
    fn test_geometry_vertex_to_face_topology_from_tessellated_triangle() {
        let (faces, vertices) = tessellated_triangle();
        let geometry = Geometry::from_triangle_faces_with_vertices_and_computed_normals(
            faces.clone(),
            vertices.clone(),
            NormalStrategy::Sharp,
        );

        let vertex_to_face_topology_calculated = vertex_to_face_topology(&geometry);

        let in_one_face_count = vertex_to_face_topology_calculated
            .iter()
            .filter(|(_, to)| to.len() == 1)
            .count();
        let in_three_faces_count = vertex_to_face_topology_calculated
            .iter()
            .filter(|(_, to)| to.len() == 3)
            .count();

        assert_eq!(in_one_face_count, 3);
        assert_eq!(in_three_faces_count, 3);
    }

    #[test]
    fn test_geometry_vertex_to_edge_topology_from_tessellated_triangle() {
        let (faces, vertices) = tessellated_triangle();
        let geometry = Geometry::from_triangle_faces_with_vertices_and_computed_normals(
            faces.clone(),
            vertices.clone(),
            NormalStrategy::Sharp,
        );

        let edge_set: HashSet<_> = geometry.unoriented_edges_iter().collect();
        let edges: Vec<_> = edge_set.iter().cloned().collect();

        let vertex_to_edge_topology_calculated = vertex_to_edge_topology(&edges);

        let in_two_edges_count = vertex_to_edge_topology_calculated
            .iter()
            .filter(|(_, to)| to.len() == 2)
            .count();
        let in_four_edges_count = vertex_to_edge_topology_calculated
            .iter()
            .filter(|(_, to)| to.len() == 4)
            .count();

        assert_eq!(in_two_edges_count, 3);
        assert_eq!(in_four_edges_count, 3);
    }

    #[test]
    fn test_geometry_vertex_to_vertex_topology_from_tessellated_triangle() {
        let (faces, vertices) = tessellated_triangle();
        let geometry = Geometry::from_triangle_faces_with_vertices_and_computed_normals(
            faces.clone(),
            vertices.clone(),
            NormalStrategy::Sharp,
        );

        let vertex_to_vertex_topology_calculated = vertex_to_vertex_topology(&geometry);

        let two_neighbors_count = vertex_to_vertex_topology_calculated
            .iter()
            .filter(|(_, to)| to.len() == 2)
            .count();
        let four_neighbors_count = vertex_to_vertex_topology_calculated
            .iter()
            .filter(|(_, to)| to.len() == 4)
            .count();

        assert_eq!(two_neighbors_count, 3);
        assert_eq!(four_neighbors_count, 3);
    }
}
