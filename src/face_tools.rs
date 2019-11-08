use nalgebra::base::Vector3;

use crate::convert::cast_usize;
use crate::geometry::{Geometry, TriangleFace};

#[allow(dead_code)]
pub fn calculate_triangle_face_normal(
    triangle_face: &TriangleFace,
    geometry: &Geometry,
) -> Vector3<f32> {
    let all_vertices = geometry.vertices();
    let triangle_vertices = (
        all_vertices[cast_usize(triangle_face.vertices.0)],
        all_vertices[cast_usize(triangle_face.vertices.1)],
        all_vertices[cast_usize(triangle_face.vertices.2)],
    );
    let edge_1: Vector3<f32> = triangle_vertices.1 - triangle_vertices.0;
    let edge_2: Vector3<f32> = triangle_vertices.1 - triangle_vertices.2;
    edge_2.cross(&edge_1).normalize()
}

#[cfg(test)]
mod tests {
    use nalgebra::geometry::Point3;

    use crate::geometry::{Face, NormalStrategy};

    use super::*;

    #[test]
    fn test_face_tools_calculate_triangle_face_normal_returns_z_vector_for_horizontal_triangle() {
        let faces = vec![(0, 1, 2)];
        let vertices = vec![
            Point3::new(0.0, 1.0, 0.0),
            Point3::new(-0.866025, -0.5, 0.0),
            Point3::new(0.866025, -0.5, 0.0),
        ];
        let geometry = Geometry::from_triangle_faces_with_vertices_and_computed_normals(
            faces,
            vertices,
            NormalStrategy::Sharp,
        );

        let normal_correct = Vector3::new(0.0, 0.0, 1.0);

        let Face::Triangle(t_f) = geometry.faces()[0];
        let normal_calculated = calculate_triangle_face_normal(&t_f, &geometry);

        assert_eq!(normal_correct, normal_calculated);
    }

    #[test]
    fn test_face_tools_calculate_triangle_face_normal_returns_x_vector_for_vertical_triangle() {
        let faces = vec![(0, 1, 2)];
        let vertices = vec![
            Point3::new(0.0, 1.0, 0.0),
            Point3::new(0.0, -0.5, 0.866025),
            Point3::new(0.0, -0.5, -0.866025),
        ];
        let geometry = Geometry::from_triangle_faces_with_vertices_and_computed_normals(
            faces,
            vertices,
            NormalStrategy::Sharp,
        );

        let normal_correct = Vector3::new(1.0, 0.0, 0.0);

        let Face::Triangle(t_f) = geometry.faces()[0];
        let normal_calculated = calculate_triangle_face_normal(&t_f, &geometry);

        assert_eq!(normal_correct, normal_calculated);
    }

    #[test]
    fn test_face_tools_calculate_triangle_face_normal_returns_vector_for_arbitrary_triangle() {
        let faces = vec![(0, 1, 2)];
        let vertices = vec![
            Point3::new(0.268023, 0.8302, 0.392469),
            Point3::new(-0.870844, -0.462665, 0.215034),
            Point3::new(0.334798, -0.197734, -0.999972),
        ];
        let geometry = Geometry::from_triangle_faces_with_vertices_and_computed_normals(
            faces,
            vertices,
            NormalStrategy::Sharp,
        );

        let normal_correct = Vector3::new(0.62270945, -0.614937, 0.48382375);

        let Face::Triangle(t_f) = geometry.faces()[0];
        let normal_calculated = calculate_triangle_face_normal(&t_f, &geometry);

        assert_eq!(normal_correct, normal_calculated);
    }
}
