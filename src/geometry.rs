use nalgebra::base::Vector3;
use nalgebra::geometry::Point3;

pub fn compute_triangle_normal(
    p1: &Point3<f32>,
    p2: &Point3<f32>,
    p3: &Point3<f32>,
) -> Vector3<f32> {
    let u = p2 - p1;
    let v = p3 - p1;

    Vector3::cross(&u, &v).normalize()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_compute_triangle_normal_returns_z_vector_for_horizontal_triangle() {
        let normal_correct = Vector3::new(0.0, 0.0, 1.0);

        let normal_calculated = compute_triangle_normal(
            &Point3::new(0.0, 1.0, 0.0),
            &Point3::new(-0.866025, -0.5, 0.0),
            &Point3::new(0.866025, -0.5, 0.0),
        );

        assert_eq!(normal_correct, normal_calculated);
    }

    #[test]
    fn test_compute_triangle_normal_returns_x_vector_for_vertical_triangle() {
        let normal_correct = Vector3::new(1.0, 0.0, 0.0);

        let normal_calculated = compute_triangle_normal(
            &Point3::new(0.0, 1.0, 0.0),
            &Point3::new(0.0, -0.5, 0.866025),
            &Point3::new(0.0, -0.5, -0.866025),
        );

        assert_eq!(normal_correct, normal_calculated);
    }

    #[test]
    fn test_compute_triangle_normal_returns_vector_for_arbitrary_triangle() {
        let normal_correct = Vector3::new(0.62270945, -0.614937, 0.48382375);

        let normal_calculated = compute_triangle_normal(
            &Point3::new(0.268023, 0.8302, 0.392469),
            &Point3::new(-0.870844, -0.462665, 0.215034),
            &Point3::new(0.334798, -0.197734, -0.999972),
        );

        assert_eq!(normal_correct, normal_calculated);
    }
}
