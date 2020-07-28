use std::cmp::Ordering;
use std::f32;

use nalgebra::{Matrix4, Point3, Vector3};

use crate::math::{clamp, TAU};

const ZOOM_SPEED_BASE: f32 = 0.95;

#[derive(Debug, Clone, Copy, PartialEq)]
pub struct CameraOptions {
    pub radius_max: f32,
    pub radius_min: f32,
    pub polar_angle_distance_min: f32,
    pub speed_pan: f32,
    pub speed_rotate: f32,
    pub speed_zoom: f32,
    pub speed_zoom_step: f32,
    pub fovy: f32,
    pub znear: f32,
    pub zfar: f32,
}

#[derive(Debug, Clone, PartialEq)]
pub struct Camera {
    screen_width: u32,
    screen_height: u32,
    radius: f32,
    azimuthal_angle: f32,
    polar_angle: f32,
    origin: Point3<f32>,
    options: CameraOptions,
}

impl Camera {
    /// Creates a new camera with screen dimensions, radius, azimuthal and polar angle.
    ///
    /// Azimuthal angle is computed counter-clockwise from the unit vector
    /// `[1,0]` lying on the XY plane. Polar angle is the angle between the Z
    /// axis and the current camera position.
    pub fn new(
        screen_width: u32,
        screen_height: u32,
        radius: f32,
        azimuthal_angle: f32,
        polar_angle: f32,
        options: CameraOptions,
    ) -> Camera {
        Camera {
            screen_width,
            screen_height,
            radius: clamp(radius, options.radius_min, options.radius_max),
            azimuthal_angle: azimuthal_angle % TAU,
            polar_angle: clamp(
                polar_angle,
                options.polar_angle_distance_min,
                f32::consts::PI - options.polar_angle_distance_min,
            ),
            origin: Point3::origin(),
            options,
        }
    }

    pub fn position(&self) -> Point3<f32> {
        let x = self.radius * self.azimuthal_angle.cos() * self.polar_angle.sin();
        let y = self.radius * self.azimuthal_angle.sin() * self.polar_angle.sin();
        let z = self.radius * self.polar_angle.cos();

        self.origin + Vector3::new(x, y, z)
    }

    pub fn set_screen_dimensions(&mut self, screen_width: u32, screen_height: u32) {
        self.screen_width = screen_width;
        self.screen_height = screen_height;
    }

    pub fn screen_aspect_ratio(&self) -> f32 {
        self.screen_width as f32 / self.screen_height as f32
    }

    pub fn set_radius_min(&mut self, radius_min: f32) {
        self.options.radius_min = radius_min;
    }

    pub fn set_radius_max(&mut self, radius_max: f32) {
        self.options.radius_max = radius_max;
    }

    pub fn set_znear(&mut self, znear: f32) {
        self.options.znear = znear;
    }

    pub fn set_zfar(&mut self, zfar: f32) {
        self.options.zfar = zfar;
    }

    /// Pans the camera by changing the camera position against the ground
    /// plane.
    ///
    /// Interescts rays from screenspace points `(old_x,old_y)` and
    /// `(new_x,new_y)` with the ground plane and moves camera by the opposite
    /// of the difference of their intersections.
    pub fn pan_ground(&mut self, old_x: f32, old_y: f32, new_x: f32, new_y: f32) {
        let projection_matrix_inverse = self.projection_matrix().try_inverse();
        let view_matrix_inverse = self.view_matrix().try_inverse();

        if let (Some(proj_inv), Some(view_inv)) = (projection_matrix_inverse, view_matrix_inverse) {
            let screen_width = self.screen_width as f32;
            let screen_height = self.screen_height as f32;

            let old_x_ndc = old_x / screen_width * 2.0 - 1.0;
            let old_y_ndc = (screen_height - old_y) / screen_height * 2.0 - 1.0;
            let new_x_ndc = new_x / screen_width * 2.0 - 1.0;
            let new_y_ndc = (screen_height - new_y) / screen_height * 2.0 - 1.0;

            let (old_near_ndc, old_far_ndc, new_near_ndc, new_far_ndc) = (
                Point3::new(old_x_ndc, old_y_ndc, 1.0),
                Point3::new(old_x_ndc, old_y_ndc, -1.0),
                Point3::new(new_x_ndc, new_y_ndc, 1.0),
                Point3::new(new_x_ndc, new_y_ndc, -1.0),
            );

            let (old_near_eye, old_far_eye, new_near_eye, new_far_eye) = (
                proj_inv.transform_point(&old_near_ndc),
                proj_inv.transform_point(&old_far_ndc),
                proj_inv.transform_point(&new_near_ndc),
                proj_inv.transform_point(&new_far_ndc),
            );

            let (old_near_world, old_far_world, new_near_world, new_far_world) = (
                view_inv.transform_point(&old_near_eye),
                view_inv.transform_point(&old_far_eye),
                view_inv.transform_point(&new_near_eye),
                view_inv.transform_point(&new_far_eye),
            );

            let cos_75_deg: f32 = 75_f32.to_radians().cos();
            let ray_origin = self.position();
            let plane_origin = Point3::new(0.0, 0.0, 0.0);
            let plane_normal = Vector3::new(0.0, 0.0, 1.0);

            let old_ray_world = old_far_world - old_near_world;
            let new_ray_world = new_far_world - new_near_world;

            let old_plane_point = ray_plane_intersection(
                &ray_origin,
                &old_ray_world,
                &plane_origin,
                &plane_normal,
                cos_75_deg,
            );
            let new_plane_point = ray_plane_intersection(
                &ray_origin,
                &new_ray_world,
                &plane_origin,
                &plane_normal,
                cos_75_deg,
            );

            if let (Some(old), Some(new)) = (old_plane_point, new_plane_point) {
                let old_to_new = new - old;
                self.origin -= old_to_new;
            }
        }
    }

    /// Pans the camera by changing the camera position against the plane
    /// originating in the point the camera is looking at and is parallel to the
    /// screen plane.
    ///
    /// Interescts rays from screenspace points `(old_x,old_y)` and
    /// `(new_x,new_y)` with the constructed plane and moves camera by the
    /// opposite of the difference of their intersections.
    pub fn pan_screen(&mut self, old_x: f32, old_y: f32, new_x: f32, new_y: f32) {
        let projection_matrix_inverse = self.projection_matrix().try_inverse();
        let view_matrix_inverse = self.view_matrix().try_inverse();

        if let (Some(proj_inv), Some(view_inv)) = (projection_matrix_inverse, view_matrix_inverse) {
            let screen_width = self.screen_width as f32;
            let screen_height = self.screen_height as f32;

            let old_x_ndc = old_x / screen_width * 2.0 - 1.0;
            let old_y_ndc = (screen_height - old_y) / screen_height * 2.0 - 1.0;
            let new_x_ndc = new_x / screen_width * 2.0 - 1.0;
            let new_y_ndc = (screen_height - new_y) / screen_height * 2.0 - 1.0;

            let (old_near_ndc, old_far_ndc, new_near_ndc, new_far_ndc) = (
                Point3::new(old_x_ndc, old_y_ndc, 1.0),
                Point3::new(old_x_ndc, old_y_ndc, -1.0),
                Point3::new(new_x_ndc, new_y_ndc, 1.0),
                Point3::new(new_x_ndc, new_y_ndc, -1.0),
            );

            let (old_near_eye, old_far_eye, new_near_eye, new_far_eye) = (
                proj_inv.transform_point(&old_near_ndc),
                proj_inv.transform_point(&old_far_ndc),
                proj_inv.transform_point(&new_near_ndc),
                proj_inv.transform_point(&new_far_ndc),
            );

            let (old_near_world, old_far_world, new_near_world, new_far_world) = (
                view_inv.transform_point(&old_near_eye),
                view_inv.transform_point(&old_far_eye),
                view_inv.transform_point(&new_near_eye),
                view_inv.transform_point(&new_far_eye),
            );

            let cos_75_deg: f32 = 75_f32.to_radians().cos();
            let ray_origin = self.position();
            let plane_origin = self.origin;
            let plane_normal = ray_origin - plane_origin;

            let old_ray_world = old_far_world - old_near_world;
            let new_ray_world = new_far_world - new_near_world;

            let old_plane_point = ray_plane_intersection(
                &ray_origin,
                &old_ray_world,
                &plane_origin,
                &plane_normal,
                cos_75_deg,
            );
            let new_plane_point = ray_plane_intersection(
                &ray_origin,
                &new_ray_world,
                &plane_origin,
                &plane_normal,
                cos_75_deg,
            );

            if let (Some(old), Some(new)) = (old_plane_point, new_plane_point) {
                let old_to_new = new - old;
                self.origin -= old_to_new;
            }
        }
    }

    /// Rotates the camera by changing azimuthal (theta) and polar (phi)
    /// angles. `dx` and `dy` are in screen space.
    pub fn rotate(&mut self, dx: f32, dy: f32) {
        let dtheta = -dx * self.options.speed_rotate;
        let dphi = -dy * self.options.speed_rotate;

        self.azimuthal_angle = (self.azimuthal_angle + dtheta) % TAU;
        self.polar_angle = clamp(
            self.polar_angle + dphi,
            self.options.polar_angle_distance_min,
            f32::consts::PI - self.options.polar_angle_distance_min,
        );
    }

    pub fn zoom(&mut self, zoom_scale: f32) {
        let zoom_speed = ZOOM_SPEED_BASE.powf(self.options.speed_zoom * zoom_scale.abs());
        let new_radius = match zoom_scale.partial_cmp(&0.0) {
            Some(Ordering::Greater) => self.radius * zoom_speed,
            Some(Ordering::Less) => self.radius / zoom_speed,
            _ => self.radius,
        };

        self.radius = clamp(new_radius, self.options.radius_min, self.options.radius_max);
    }

    pub fn zoom_step(&mut self, zoom_steps: i32) {
        let zoom_speed = ZOOM_SPEED_BASE.powf(self.options.speed_zoom_step);

        let mut new_radius = self.radius;
        match zoom_steps.cmp(&0) {
            Ordering::Greater => {
                for _ in 0..zoom_steps {
                    new_radius *= zoom_speed;
                }
            }
            Ordering::Less => {
                for _ in zoom_steps..0 {
                    new_radius /= zoom_speed;
                }
            }
            _ => (),
        }

        self.radius = clamp(new_radius, self.options.radius_min, self.options.radius_max);
    }

    /// A sphere completely visible by this camera, no matter the rotation.
    pub fn visible_sphere(&self) -> (Point3<f32>, f32) {
        const MARGIN_MULTIPLIER: f32 = 1.005;
        let angle = self.compute_visible_sphere_angle();

        let sphere_radius = self.radius / MARGIN_MULTIPLIER * angle.tan();

        (self.origin, sphere_radius)
    }

    /// Attempt to fit a sphere into camera view, no matter the rotation.
    ///
    /// Camera options may affect the outcome. A too small
    /// `radius_max` or a too large `radius_min` may cause the result
    /// to be not zoomed out enough, or not zoomed in enough.
    pub fn zoom_to_fit_visible_sphere(&mut self, sphere_origin: Point3<f32>, sphere_radius: f32) {
        const MARGIN_MULTIPLIER: f32 = 1.005;
        let angle = self.compute_visible_sphere_angle();

        // Compute the distance needed from the sphere for it to fit
        // inside the camera frustum
        let new_radius = MARGIN_MULTIPLIER * sphere_radius / angle.tan();

        self.origin = sphere_origin;
        self.radius = clamp(new_radius, self.options.radius_min, self.options.radius_max);
    }

    pub fn view_matrix(&self) -> Matrix4<f32> {
        Matrix4::look_at_rh(&self.position(), &self.origin, &Vector3::z())
    }

    pub fn projection_matrix(&self) -> Matrix4<f32> {
        Matrix4::new_perspective(
            self.screen_aspect_ratio(),
            self.options.fovy,
            self.options.znear,
            self.options.zfar,
        )
    }

    fn compute_visible_sphere_angle(&self) -> f32 {
        let fovy = self.options.fovy;
        let fovx = fovy * self.screen_aspect_ratio();
        let fov = fovy.min(fovx);

        fov / 2.0
    }
}

fn ray_plane_intersection(
    ray_origin: &Point3<f32>,
    ray_direction: &Vector3<f32>,
    plane_origin: &Point3<f32>,
    plane_normal: &Vector3<f32>,
    min_angle_cos_abs: f32,
) -> Option<Point3<f32>> {
    assert!(ray_direction.norm_squared() > 0.0, "Ray vector zero");

    let direction = ray_direction.normalize();
    let normal = plane_normal.normalize();

    let denominator = direction.dot(&normal);

    if denominator.abs() > min_angle_cos_abs {
        let plane_distance_from_origin = plane_origin.coords.norm();
        let numerator = ray_origin.coords.dot(&normal) + plane_distance_from_origin;
        let t = -numerator / denominator;

        Some(ray_origin + t * direction)
    } else {
        // The ray direction and the plane normal are too close to being
        // parallel - the cosine of their angle is too close to 0.
        None
    }
}
