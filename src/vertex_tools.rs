use std::f32;

use nalgebra;
use nalgebra::base::Vector3;
use nalgebra::geometry::Point3;

use crate::convert::cast_usize;
use crate::face_tools;
use crate::geometry::{Face, Geometry, UnorientedEdge};

/// The Möller–Trumbore ray-triangle intersection algorithm is a fast method for
/// calculating the intersection of a ray and a triangle in three dimensions
/// without needing precomputation of the plane equation of the plane containing
/// the triangle.
///
/// https://en.wikipedia.org/wiki/Möller–Trumbore_intersection_algorithm
/// https://www.scratchapixel.com/lessons/3d-basic-rendering/ray-tracing-rendering-a-triangle/moller-trumbore-ray-triangle-intersection
#[allow(dead_code)]
pub fn ray_intersects_triangle(
    ray_origin: &Point3<f32>,
    ray_vector: &Vector3<f32>,
    triangle_vertices: (&Point3<f32>, &Point3<f32>, &Point3<f32>),
) -> Option<Point3<f32>> {
    let edge_1_vector = triangle_vertices.1 - triangle_vertices.0;
    let edge_2_vector = triangle_vertices.2 - triangle_vertices.0;
    let perpendicular_vector = ray_vector.cross(&edge_2_vector);
    let determinant = edge_1_vector.dot(&perpendicular_vector);
    // This ray is parallel to this triangle.
    if approx::relative_eq!(determinant, 0.0) {
        return None;
    }
    let inverse_determinant = 1.0 / determinant;
    let tangent_vector = ray_origin - triangle_vertices.0;
    let u_parameter = inverse_determinant * tangent_vector.dot(&perpendicular_vector);
    if u_parameter < 0.0 || u_parameter > 1.0 {
        return None;
    }
    let q_vector = tangent_vector.cross(&edge_1_vector);
    let v_parameter = inverse_determinant * ray_vector.dot(&q_vector);
    if v_parameter < 0.0 || u_parameter + v_parameter > 1.0 {
        return None;
    }
    // At this stage we can compute t_parameter to find out where the
    // intersection point is on the line.
    let t_parameter = inverse_determinant * edge_2_vector.dot(&q_vector);
    // Ray-triangle intersection
    if t_parameter > f32::EPSILON && t_parameter < 1.0 / f32::EPSILON {
        Some(ray_origin + ray_vector * t_parameter)
    } else {
        None
    }
}

#[allow(dead_code)]
pub struct PointOnLine {
    clamped: Point3<f32>,
    unclamped: Point3<f32>,
}

/// Pulls arbitrary point to an arbitrary line.
///
/// Returns both, a clamped point on the line adn an unclamped points on an
/// endless ray.
///
/// https://stackoverflow.com/questions/3120357/get-closest-point-to-a-line
pub fn pull_point_to_line(
    point: &Point3<f32>,
    line_points: (&Point3<f32>, &Point3<f32>),
) -> PointOnLine {
    let line_start = line_points.0;
    let line_end = line_points.1;
    let start_to_point = point - line_start;
    let start_to_end = line_end - line_start;

    let square_magnitude_start_to_end = nalgebra::distance_squared(line_start, line_end);
    let start_to_point_dot_start_to_end = start_to_point.dot(&start_to_end);
    let line_parameter = start_to_point_dot_start_to_end / square_magnitude_start_to_end;

    let closest_point_on_line_unclamped =
        Point3::from(line_start.coords.lerp(&line_end.coords, line_parameter));
    let closest_point_on_line_clamped = if line_parameter < 0.0 {
        *line_start
    } else if line_parameter > 1.0 {
        *line_end
    } else {
        closest_point_on_line_unclamped
    };

    PointOnLine {
        clamped: closest_point_on_line_clamped,
        unclamped: closest_point_on_line_unclamped,
    }
}

#[allow(dead_code)]
pub struct PulledPoint {
    closest_point: Point3<f32>,
    distance: f32,
}

/// Pulls arbitrary point to the closest point of a mesh geometry.
///
/// Cast a ray from the point perpendicular to each mesh face and if there is an
/// intersection, measure the distance. Also measure a distance to the closest
/// point on each mesh geometry edge. Pick the closest point of them all as the
/// pulled point.
#[allow(dead_code)]
pub fn pull_point_to_mesh(
    point: &Point3<f32>,
    geometry: &Geometry,
    unoriented_edges: &[UnorientedEdge],
) -> PulledPoint {
    let vertices = geometry.vertices();
    let all_mesh_faces_with_normals = geometry.faces().iter().map(|Face::Triangle(t_f)| {
        (
            (
                &vertices[cast_usize(t_f.vertices.0)],
                &vertices[cast_usize(t_f.vertices.1)],
                &vertices[cast_usize(t_f.vertices.2)],
            ),
            face_tools::calculate_triangle_face_normal(t_f, geometry),
        )
    });
    let points_pulled_to_faces =
        all_mesh_faces_with_normals.filter_map(|(unwrapped_face, face_normal)| {
            ray_intersects_triangle(point, &(-1.0 * face_normal), unwrapped_face)
        });
    let unwrapped_edges = unoriented_edges.iter().map(|u_e| {
        (
            &vertices[cast_usize(u_e.0.vertices.0)],
            &vertices[cast_usize(u_e.0.vertices.1)],
        )
    });
    let points_pulled_to_edges = unwrapped_edges.map(|e| {
        let closest_point = pull_point_to_line(point, e);
        closest_point.clamped
    });
    let all_pulled_points = points_pulled_to_faces.chain(points_pulled_to_edges);

    let (closest_point_distance_squared, closest_point_option) =
        all_pulled_points.fold((f32::MAX, None), |(cls_dist_sq, cls_pt_opt), current_pt| {
            let current_dist_sq = nalgebra::distance_squared(point, &current_pt);
            if current_dist_sq < cls_dist_sq {
                (current_dist_sq, Some(current_pt))
            } else {
                (cls_dist_sq, cls_pt_opt)
            }
        });

    PulledPoint {
        closest_point: closest_point_option.expect("Invalid point on mesh"),
        distance: closest_point_distance_squared.sqrt(),
    }
}

#[cfg(test)]
mod tests {
    use crate::geometry;

    use super::*;

    #[test]
    fn test_vertex_tools_ray_intersects_triangle_for_horizontal_triangle_returns_point_inside() {
        let face_points = (
            &Point3::new(0.0, 1.0, 0.0),
            &Point3::new(-0.866025, -0.5, 0.0),
            &Point3::new(0.866025, -0.5, 0.0),
        );

        let ray_origin = Point3::new(0.25, 0.25, 0.25);
        let ray_vector = Vector3::new(0.0, 0.0, -1.0);

        let point_on_triangle_correct = Point3::new(0.25, 0.25, 0.0);
        let point_on_triangle_calculated =
            ray_intersects_triangle(&ray_origin, &ray_vector, face_points)
                .expect("Point is not on the triangle.");

        assert_eq!(point_on_triangle_correct, point_on_triangle_calculated);
    }

    #[test]
    fn test_vertex_tools_ray_intersects_triangle_for_horizontal_triangle_returns_none_because_ray_misses(
    ) {
        let face_points = (
            &Point3::new(0.0, 1.0, 0.0),
            &Point3::new(-0.866025, -0.5, 0.0),
            &Point3::new(0.866025, -0.5, 0.0),
        );

        let ray_origin = Point3::new(1.25, 0.25, 0.25);
        let ray_vector = Vector3::new(0.0, 0.0, -1.0);

        let point_on_triangle_calculated =
            ray_intersects_triangle(&ray_origin, &ray_vector, face_points);

        // There is no intersection of the triangle and ray
        assert_eq!(None, point_on_triangle_calculated);
    }

    #[test]
    fn test_vertex_tools_ray_intersects_triangle_for_horizontal_triangle_returns_none_because_ray_parallel(
    ) {
        let face_points = (
            &Point3::new(0.0, 1.0, 0.0),
            &Point3::new(-0.866025, -0.5, 0.0),
            &Point3::new(0.866025, -0.5, 0.0),
        );

        let ray_origin = Point3::new(0.25, 0.25, 0.25);
        let ray_vector = Vector3::new(1.0, 0.0, 0.0);

        let point_on_triangle_calculated =
            ray_intersects_triangle(&ray_origin, &ray_vector, face_points);

        // There is no intersection of the triangle and ray
        assert_eq!(None, point_on_triangle_calculated);
    }

    #[test]
    fn test_vertex_tools_ray_intersects_triangle_for_arbitrary_triangle_returns_point_inside() {
        let face_points = (
            &Point3::new(0.268023, 0.8302, 0.392469),
            &Point3::new(-0.870844, -0.462665, 0.215034),
            &Point3::new(0.334798, -0.197734, -0.999972),
        );

        let ray_origin = Point3::new(0.25, 0.25, 0.25);
        let ray_vector = Vector3::new(-0.622709, 0.614937, -0.483823);

        let point_on_triangle_correct = Point3::new(0.07773775, 0.42011225, 0.11615828);
        let point_on_triangle_calculated =
            ray_intersects_triangle(&ray_origin, &ray_vector, face_points)
                .expect("Point is not on the triangle.");

        assert_eq!(point_on_triangle_correct, point_on_triangle_calculated);
    }

    #[test]
    fn test_vertex_tools_pull_point_to_line_center() {
        let line_points = (&Point3::new(-1.0, 0.0, 0.0), &Point3::new(1.0, 0.0, 0.0));
        let test_point = Point3::new(0.0, 1.0, 1.0);

        let point_on_line_clamped_correct = Point3::new(0.0, 0.0, 0.0);
        let point_on_line_unclamped_correct = Point3::new(0.0, 0.0, 0.0);

        let point_on_line_calculated = pull_point_to_line(&test_point, line_points);

        assert_eq!(
            point_on_line_clamped_correct,
            point_on_line_calculated.clamped
        );
        assert_eq!(
            point_on_line_unclamped_correct,
            point_on_line_calculated.unclamped
        );
    }

    #[test]
    fn test_vertex_tools_pull_point_to_line_quarter() {
        let line_points = (&Point3::new(-1.0, 0.0, 0.0), &Point3::new(1.0, 0.0, 0.0));
        let test_point = Point3::new(-0.5, 1.0, 1.0);

        let point_on_line_clamped_correct = Point3::new(-0.5, 0.0, 0.0);
        let point_on_line_unclamped_correct = Point3::new(-0.5, 0.0, 0.0);

        let point_on_line_calculated = pull_point_to_line(&test_point, line_points);

        assert_eq!(
            point_on_line_clamped_correct,
            point_on_line_calculated.clamped
        );
        assert_eq!(
            point_on_line_unclamped_correct,
            point_on_line_calculated.unclamped
        );
    }

    #[test]
    fn test_vertex_tools_pull_point_to_line_below_start() {
        let line_points = (&Point3::new(-1.0, 0.0, 0.0), &Point3::new(1.0, 0.0, 0.0));
        let test_point = Point3::new(-2.0, 1.0, 1.0);

        let point_on_line_clamped_correct = Point3::new(-1.0, 0.0, 0.0);
        let point_on_line_unclamped_correct = Point3::new(-2.0, 0.0, 0.0);

        let point_on_line_calculated = pull_point_to_line(&test_point, line_points);

        assert_eq!(
            point_on_line_clamped_correct,
            point_on_line_calculated.clamped
        );
        assert_eq!(
            point_on_line_unclamped_correct,
            point_on_line_calculated.unclamped
        );
    }

    #[test]
    fn test_vertex_tools_pull_point_to_line_beyond_end() {
        let line_points = (&Point3::new(-1.0, 0.0, 0.0), &Point3::new(1.0, 0.0, 0.0));
        let test_point = Point3::new(2.0, 1.0, 1.0);

        let point_on_line_clamped_correct = Point3::new(1.0, 0.0, 0.0);
        let point_on_line_unclamped_correct = Point3::new(2.0, 0.0, 0.0);

        let point_on_line_calculated = pull_point_to_line(&test_point, line_points);

        assert_eq!(
            point_on_line_clamped_correct,
            point_on_line_calculated.clamped
        );
        assert_eq!(
            point_on_line_unclamped_correct,
            point_on_line_calculated.unclamped
        );
    }

    #[test]
    fn test_vertex_tools_pull_point_to_mesh_cube_point_inside_left() {
        let cube_geometry = geometry::cube_sharp_var_len([0.0, 0.0, 0.0], 1.0);
        let test_point = Point3::new(-0.25, 0.0, 0.0);
        let unoriented_edges: Vec<_> = cube_geometry.unoriented_edges_iter().collect();

        let point_on_mesh_correct = Point3::new(-1.0, 0.0, 0.0);
        let pulled_point_on_mesh_calculated =
            pull_point_to_mesh(&test_point, &cube_geometry, &unoriented_edges);

        assert_eq!(
            point_on_mesh_correct,
            pulled_point_on_mesh_calculated.closest_point
        );
        assert_eq!(0.75, pulled_point_on_mesh_calculated.distance);
    }

    #[test]
    fn test_vertex_tools_pull_point_to_mesh_cube_point_inside_top_front_right() {
        let cube_geometry = geometry::cube_sharp_var_len([0.0, 0.0, 0.0], 1.0);
        let test_point = Point3::new(0.25, 0.25, 0.25);
        let unoriented_edges: Vec<_> = cube_geometry.unoriented_edges_iter().collect();

        // any fo the following points on mesh would be correct
        let points_on_mesh_correct = vec![
            Point3::new(1.0, 0.25, 0.25),
            Point3::new(0.25, 1.0, 0.25),
            Point3::new(0.25, 0.25, 1.0),
        ];
        let pulled_point_on_mesh_calculated =
            pull_point_to_mesh(&test_point, &cube_geometry, &unoriented_edges);

        assert!(points_on_mesh_correct
            .iter()
            .any(|p| *p == pulled_point_on_mesh_calculated.closest_point));

        assert_eq!(0.75, pulled_point_on_mesh_calculated.distance);
    }

    #[test]
    fn test_vertex_tools_pull_point_to_mesh_cube_point_outside_top_front_right() {
        let cube_geometry = geometry::cube_sharp_var_len([0.0, 0.0, 0.0], 1.0);
        let test_point = Point3::new(1.25, 1.25, 1.25);
        let unoriented_edges: Vec<_> = cube_geometry.unoriented_edges_iter().collect();

        // corner
        let point_on_mesh_correct = Point3::new(1.0, 1.0, 1.0);
        let pulled_point_on_mesh_calculated =
            pull_point_to_mesh(&test_point, &cube_geometry, &unoriented_edges);

        assert_eq!(
            point_on_mesh_correct,
            pulled_point_on_mesh_calculated.closest_point
        );
        assert_eq!(
            ((3.0 * 0.25 * 0.25) as f32).sqrt(),
            pulled_point_on_mesh_calculated.distance
        );
    }

    #[test]
    fn test_vertex_tools_pull_point_to_mesh_cube_point_outside_front_right() {
        let cube_geometry = geometry::cube_sharp_var_len([0.0, 0.0, 0.0], 1.0);
        let test_point = Point3::new(1.25, 1.25, 0.25);
        let unoriented_edges: Vec<_> = cube_geometry.unoriented_edges_iter().collect();

        // on the edge
        let point_on_mesh_correct = Point3::new(1.0, 1.0, 0.25);
        let pulled_point_on_mesh_calculated =
            pull_point_to_mesh(&test_point, &cube_geometry, &unoriented_edges);

        assert_eq!(
            point_on_mesh_correct,
            pulled_point_on_mesh_calculated.closest_point
        );
        assert_eq!(
            ((2.0 * 0.25 * 0.25) as f32).sqrt(),
            pulled_point_on_mesh_calculated.distance
        );
    }
}
