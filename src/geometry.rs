use nalgebra::{Point2, Point3, Vector3};

/// Computes a (normalized) normal vector for a triangle.
pub fn compute_triangle_normal(
    p1: &Point3<f32>,
    p2: &Point3<f32>,
    p3: &Point3<f32>,
) -> Vector3<f32> {
    let u = p2 - p1;
    let v = p3 - p2;

    Vector3::cross(&u, &v).normalize()
}

/// Computes barycentric coordinates of point P in triangle A, B,
/// C. Returns `None` for degenerate triangles.
pub fn compute_barycentric_coords(
    a: Point2<f32>,
    b: Point2<f32>,
    c: Point2<f32>,
    p: Point2<f32>,
) -> Option<Point3<f32>> {
    let ab = b - a;
    let ac = c - a;
    let pa = a - p;
    let xs = Vector3::new(ac.x, ab.x, pa.x);
    let ys = Vector3::new(ac.y, ab.y, pa.y);
    let ortho = xs.cross(&ys);

    if approx::relative_eq!(f32::abs(ortho.z), 0.0) {
        None
    } else {
        Some(Point3::new(
            1.0 - (ortho.x + ortho.y) / ortho.z,
            ortho.y / ortho.z,
            ortho.x / ortho.z,
        ))
    }
}

/// Converts barycentric coordinates into cartesian.
pub fn barycentric_to_cartesian(
    barycentric_coords: &Point3<f32>,
    a: &Point3<f32>,
    b: &Point3<f32>,
    c: &Point3<f32>,
) -> Point3<f32> {
    Point3::from(
        barycentric_coords.x * a.coords
            + barycentric_coords.y * b.coords
            + barycentric_coords.z * c.coords,
    )
}

/// Checks if all three points lay on the same line.
///
/// http://www.ambrsoft.com/TrigoCalc/Line3D/LineColinear.htm
pub fn are_points_collinear(v0: &Point3<f32>, v1: &Point3<f32>, v2: &Point3<f32>) -> bool {
    if approx::relative_eq!(v0, v1) || approx::relative_eq!(v0, v2) {
        return true;
    }
    let n1 = v1 - v0;
    let n2 = v2 - v0;
    let cross = n1.cross(&n2);
    approx::relative_eq!(cross, Vector3::zeros())
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

    #[test]
    fn test_compute_barycentric_coords_for_point_inside() {
        let triangle_points = (
            Point2::new(0.0, 1.0),
            Point2::new(-0.866025, -0.5),
            Point2::new(0.866025, -0.5),
        );

        let test_point = Point2::new(0.0, 0.0);
        let barycentric_calculated = compute_barycentric_coords(
            triangle_points.0,
            triangle_points.1,
            triangle_points.2,
            test_point,
        )
        .expect("Failed to calculate barycentric coords");

        let barycentric_correct = Point3::new(0.333333, 0.333333, 0.333333);

        assert!(barycentric_calculated.coords.relative_eq(
            &barycentric_correct.coords,
            0.001,
            0.001
        ));
    }

    #[test]
    fn test_compute_barycentric_coords_for_point_outside() {
        let triangle_points = (
            Point2::new(0.0, 1.0),
            Point2::new(-0.866025, -0.5),
            Point2::new(0.866025, -0.5),
        );

        let test_point = Point2::new(0.0, 2.0);
        let barycentric_calculated = compute_barycentric_coords(
            triangle_points.0,
            triangle_points.1,
            triangle_points.2,
            test_point,
        )
        .expect("Failed to calculate barycentric coords");

        let barycentric_correct = Point3::new(1.6666667, -0.33333334, -0.33333334);

        assert!(barycentric_calculated.coords.relative_eq(
            &barycentric_correct.coords,
            0.001,
            0.001
        ));
    }
}
