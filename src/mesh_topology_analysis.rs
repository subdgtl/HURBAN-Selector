use std::collections::HashMap;

use smallvec::SmallVec;

use crate::convert::{cast_u32, cast_usize};
use crate::geometry::{Face, Geometry, UnorientedEdge};

/// Calculates topological relations (neighborhood) of mesh face -> faces.
/// Returns a Map (key: face index, value: list of its neighboring faces indices)
#[allow(dead_code)]
pub fn face_to_face_topology(geometry: &Geometry) -> HashMap<usize, SmallVec<[usize; 8]>> {
    let mut f2f: HashMap<usize, SmallVec<[usize; 8]>> = HashMap::new();
    for (from_counter, face) in geometry.faces().iter().enumerate() {
        match face {
            Face::Triangle(f) => {
                let [f_e_0, f_e_1, f_e_2] = f.to_unoriented_edges();
                for (to_counter, to_face) in geometry.faces().iter().enumerate() {
                    match to_face {
                        Face::Triangle(t_f) => {
                            if from_counter != to_counter && t_f.contains_unoriented_edge(f_e_0)
                                || t_f.contains_unoriented_edge(f_e_1)
                                || t_f.contains_unoriented_edge(f_e_2)
                            {
                                let neighbors =
                                    f2f.entry(from_counter).or_insert_with(SmallVec::new);
                                if neighbors.iter().all(|value| *value != to_counter) {
                                    neighbors.push(to_counter);
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
) -> HashMap<usize, SmallVec<[usize; 8]>> {
    let mut e2f: HashMap<usize, SmallVec<[usize; 8]>> = HashMap::new();
    for (from_counter, e) in edges.iter().enumerate() {
        for (to_counter, face) in geometry.faces().iter().enumerate() {
            match face {
                Face::Triangle(t_f) => {
                    if t_f.contains_unoriented_edge(*e) {
                        let neighbors = e2f.entry(from_counter).or_insert_with(SmallVec::new);
                        if neighbors.iter().all(|value| *value != to_counter) {
                            neighbors.push(to_counter);
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
pub fn vertex_to_face_topology(geometry: &Geometry) -> HashMap<usize, SmallVec<[usize; 8]>> {
    let mut v2f: HashMap<usize, SmallVec<[usize; 8]>> = HashMap::new();
    for from_counter in 0..geometry.vertices().len() {
        for (to_counter, face) in geometry.faces().iter().enumerate() {
            match face {
                Face::Triangle(t_f) => {
                    if t_f.contains_vertex(cast_u32(from_counter)) {
                        let neighbors = v2f.entry(from_counter).or_insert_with(SmallVec::new);
                        if neighbors.iter().all(|value| *value != to_counter) {
                            neighbors.push(to_counter);
                        }
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
pub fn vertex_to_edge_topology(
    geometry: &Geometry,
    edges: &[UnorientedEdge],
) -> HashMap<usize, SmallVec<[usize; 8]>> {
    let mut v2e: HashMap<usize, SmallVec<[usize; 8]>> = HashMap::new();
    for from_counter in 0..geometry.vertices().len() {
        for (to_counter, e) in edges.iter().enumerate() {
            if e.0.contains_vertex(cast_u32(from_counter)) {
                let neighbors = v2e.entry(from_counter).or_insert_with(SmallVec::new);
                if neighbors.iter().all(|value| *value != to_counter) {
                    neighbors.push(to_counter);
                }
            }
        }
    }
    v2e
}

/// Calculates topological relations (neighborhood) of mesh vertex -> vertex.
/// Returns a Map (key: vertex index, value: list of its neighboring vertices indices)
#[allow(dead_code)]
pub fn vertex_to_vertex_topology(geometry: &Geometry) -> HashMap<usize, SmallVec<[usize; 8]>> {
    let mut v2v: HashMap<usize, SmallVec<[usize; 8]>> = HashMap::new();
    for face in geometry.faces() {
        match face {
            Face::Triangle(f) => {
                let neighbors_0 = v2v
                    .entry(cast_usize(f.vertices.0))
                    .or_insert_with(SmallVec::new);
                if neighbors_0
                    .iter()
                    .all(|value| *value != cast_usize(f.vertices.1))
                {
                    neighbors_0.push(cast_usize(f.vertices.1));
                }
                if neighbors_0
                    .iter()
                    .all(|value| *value != cast_usize(f.vertices.2))
                {
                    neighbors_0.push(cast_usize(f.vertices.2));
                }
                let neighbors_1 = v2v
                    .entry(cast_usize(f.vertices.1))
                    .or_insert_with(SmallVec::new);
                if neighbors_1
                    .iter()
                    .all(|value| *value != cast_usize(f.vertices.0))
                {
                    neighbors_1.push(cast_usize(f.vertices.0));
                }
                if neighbors_1
                    .iter()
                    .all(|value| *value != cast_usize(f.vertices.2))
                {
                    neighbors_1.push(cast_usize(f.vertices.2));
                }
                let neighbors_2 = v2v
                    .entry(cast_usize(f.vertices.2))
                    .or_insert_with(SmallVec::new);
                if neighbors_2
                    .iter()
                    .all(|value| *value != cast_usize(f.vertices.0))
                {
                    neighbors_2.push(cast_usize(f.vertices.0));
                }
                if neighbors_2
                    .iter()
                    .all(|value| *value != cast_usize(f.vertices.1))
                {
                    neighbors_2.push(cast_usize(f.vertices.1));
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
        let mut face_to_face_topology_correct: HashMap<usize, Vec<usize>> = HashMap::new();
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

        let vertex_to_edge_topology_calculated = vertex_to_edge_topology(&geometry, &edges);

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
