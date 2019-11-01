use std::collections::HashMap;

use smallvec::SmallVec;

use crate::convert::cast_u32;
use crate::geometry::{Face, Geometry, UnorientedEdge};

/// Calculates topological relations (neighborhood) of mesh face -> faces.
/// Returns a Map (key: face index, value: list of its neighboring faces indices)
#[allow(dead_code)]
pub fn face_to_face_topology(geometry: &Geometry) -> HashMap<u32, SmallVec<[u32; 8]>> {
    let mut f2f: HashMap<u32, SmallVec<[u32; 8]>> = HashMap::new();

    for (from_counter, face) in geometry.faces().iter().enumerate() {
        match face {
            Face::Triangle(f) => {
                let from_counter_u32 = cast_u32(from_counter);
                let [f_e_0, f_e_1, f_e_2] = f.to_unoriented_edges();
                for (to_counter, to_face) in geometry.faces().iter().enumerate() {
                    let to_counter_u32 = cast_u32(to_counter);

                    match to_face {
                        Face::Triangle(t_f) => {
                            if from_counter != to_counter
                                && (t_f.contains_unoriented_edge(f_e_0)
                                    || t_f.contains_unoriented_edge(f_e_1)
                                    || t_f.contains_unoriented_edge(f_e_2))
                            {
                                let neighbors =
                                    f2f.entry(from_counter_u32).or_insert_with(SmallVec::new);

                                if !neighbors.contains(&to_counter_u32) {
                                    neighbors.push(to_counter_u32);
                                }
                            }
                        }
                    }
                }
            }
        }
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

    for (from_counter, e) in edges.iter().enumerate() {
        for (to_counter, face) in geometry.faces().iter().enumerate() {
            let to_counter_u32 = cast_u32(to_counter);

            match face {
                Face::Triangle(t_f) => {
                    if t_f.contains_unoriented_edge(*e) {
                        let neighbors = e2f
                            .entry(cast_u32(from_counter))
                            .or_insert_with(SmallVec::new);

                        if !neighbors.contains(&to_counter_u32) {
                            neighbors.push(to_counter_u32);
                        }
                    }
                }
            }
        }
    }

    e2f
}

/// Calculates topological relations (neighborhood) of mesh vertex -> faces.
/// Returns a Map (key: vertex index, value: list of its neighboring faces indices)
#[allow(dead_code)]
pub fn vertex_to_face_topology(geometry: &Geometry) -> HashMap<u32, SmallVec<[u32; 8]>> {
    let mut v2f: HashMap<u32, SmallVec<[u32; 8]>> = HashMap::new();

    for (to_face, face) in geometry.faces().iter().enumerate() {
        match face {
            Face::Triangle(t_f) => {
                let to_face_u32 = cast_u32(to_face);

                for from_vertex in &[t_f.vertices.0, t_f.vertices.1, t_f.vertices.2] {
                    let neighbors = v2f.entry(*from_vertex).or_insert_with(SmallVec::new);

                    if !neighbors.contains(&to_face_u32) {
                        neighbors.push(to_face_u32);
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

    for (to_edge, e) in edges.iter().enumerate() {
        let to_edge_u32 = cast_u32(to_edge);

        for from_vertex in &[e.0.vertices.0, e.0.vertices.1] {
            let neighbors = v2e.entry(*from_vertex).or_insert_with(SmallVec::new);

            if !neighbors.contains(&to_edge_u32) {
                neighbors.push(to_edge_u32);
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

    #[test]
    fn test_geometry_face_to_face_topology_from_tessellated_triangle() {
        let (faces, vertices) = tessellated_triangle();
        let geometry = Geometry::from_triangle_faces_with_vertices_and_computed_normals(
            faces.clone(),
            vertices.clone(),
            NormalStrategy::Sharp,
        );
        let mut face_to_face_topology_correct: HashMap<u32, Vec<u32>> = HashMap::new();
        face_to_face_topology_correct.insert(0, vec![1]);
        face_to_face_topology_correct.insert(1, vec![0, 2, 3]);
        face_to_face_topology_correct.insert(2, vec![1]);
        face_to_face_topology_correct.insert(3, vec![1]);

        let face_to_face_topology_calculated = face_to_face_topology(&geometry);

        assert!(face_to_face_topology_correct
            .iter()
            .all(|(face_index, neighbors)| {
                if let Some(neighbors_calculated) = face_to_face_topology_calculated.get(face_index)
                {
                    neighbors
                        .iter()
                        .all(|n| neighbors_calculated.iter().any(|n_c| n_c == n))
                } else {
                    false
                }
            }));
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
