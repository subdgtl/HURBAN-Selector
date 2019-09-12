pub use self::renderer::{GpuBackend, Msaa, PresentMode};

use std::time::{Duration, Instant};

use nalgebra::geometry::Point3;

use crate::geometry::{self, Geometry};
use crate::math::{self, CubicBezierEasing};

use self::camera::{Camera, CameraOptions};
use self::importer::ImporterWorker;
use self::renderer::{Renderer, RendererOptions, SceneRendererGeometry, SceneRendererGeometryId};
use self::ui::Ui;

pub mod importer;

mod camera;
mod renderer;
mod ui;

const CAMERA_INTERPOLATION_DURATION: Duration = Duration::from_millis(1000);

#[derive(Debug, Clone, Copy, PartialEq, Default)]
pub struct UiInput {}

#[derive(Debug, Clone, Copy, PartialEq, Default)]
pub struct Input {
    pub mouse_move: [f64; 2],
    pub mouse_wheel: i32,
    pub lmb_down: bool,
    pub rmb_down: bool,
    pub meta_down: bool,
    pub shift_down: bool,
    pub ctrl_down: bool,
    pub alt_down: bool,
    pub key_a_pressed: bool,
    pub key_o_pressed: bool,
    pub key_q_pressed: bool,
    pub close_requested: bool,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct AppOptions {
    pub msaa: Msaa,
    pub present_mode: PresentMode,

    /// If present, try to force a gpu backend for the renderer to use
    pub gpu_backend: Option<GpuBackend>,
}

pub struct App {
    window: winit::window::Window,

    time_start: Instant,
    time: Instant,

    wants_close: bool,
    active_camera_interpolation: Option<CameraInterpolation>,
    scene_geometries: Vec<Geometry>,
    scene_renderer_geometry_ids: Vec<SceneRendererGeometryId>,

    cubic_bezier: CubicBezierEasing,
    camera: Camera,
    importer_worker: ImporterWorker,
    ui: Ui,
    renderer: Renderer,
}

impl App {
    pub fn new(time_start: Instant, window: winit::window::Window, options: AppOptions) -> Self {
        let window_size = window.inner_size().to_physical(window.hidpi_factor());

        let camera = Camera::new(
            window_size,
            5.0,
            45f32.to_radians(),
            60f32.to_radians(),
            CameraOptions {
                radius_min: 1.0,
                radius_max: 10000.0,
                polar_angle_distance_min: 1f32.to_radians(),
                speed_pan: 10.0,
                speed_rotate: 0.005,
                speed_zoom: 0.01,
                speed_zoom_step: 1.0,
                fovy: 45f32.to_radians(),
                znear: 0.01,
                zfar: 1000.0,
            },
        );

        let importer_worker = ImporterWorker::new();
        let mut ui = Ui::new(&window);
        let mut renderer = Renderer::new(
            &window,
            &camera.projection_matrix(),
            &camera.view_matrix(),
            ui.fonts(),
            RendererOptions {
                // FIXME: @Correctness Msaa X4 is the only value currently
                // working on all devices we tried. Once device
                // capabilities are queryable with wgpu `Limits`, we
                // should have a chain of options the renderer tries
                // before giving up.
                msaa: options.msaa,
                present_mode: PresentMode::Vsync,
                gpu_backend: options.gpu_backend,
            },
        );

        let scene_geometries = vec![
            // FIXME: This is just temporary code so that we can see
            // something in the scene and know the renderer works.
            geometry::uv_cube_var_len([0.0, 0.0, 0.0], 0.5),
            geometry::uv_cube_same_len([0.0, 50.0, 0.0], 5.0),
            geometry::uv_cube_same_len([50.0, 0.0, 0.0], 5.0),
            geometry::cube_same_len([0.0, 5.0, 0.0], 0.9),
            geometry::cube_same_len([0.0, 0.0, 5.0], 1.5),
            geometry::plane_same_len([0.0, 0.0, 20.0], 10.0),
            geometry::plane_var_len([0.0, 0.0, -20.0], 10.0),
        ];
        let mut scene_renderer_geometry_ids = Vec::with_capacity(scene_geometries.len());
        for geometry in &scene_geometries {
            let renderer_geometry = SceneRendererGeometry::from_geometry(geometry);
            let renderer_geometry_id = renderer
                .add_scene_geometry(&renderer_geometry)
                .expect("Failed to add geometry to renderer");
            scene_renderer_geometry_ids.push(renderer_geometry_id);
        }

        Self {
            window,

            time_start,
            time: time_start,

            wants_close: false,
            active_camera_interpolation: None,
            scene_geometries,
            scene_renderer_geometry_ids,

            cubic_bezier: CubicBezierEasing::new([0.7, 0.0], [0.3, 1.0]),
            camera,
            importer_worker,
            ui,
            renderer,
        }
    }

    // FIXME: this just plain sucks
    pub fn update_ui<T>(&mut self, event: &winit::event::Event<T>) {
        self.ui.handle_event(&self.window, event);
    }

    pub fn ui_captured_keyboard(&self) -> bool {
        self.ui.want_capture_keyboard()
    }

    pub fn ui_captured_mouse(&self) -> bool {
        self.ui.want_capture_mouse()
    }

    pub fn update_and_render(&mut self, time: Instant, input: &Input) {
        let (duration_last_frame, _duration_running) = {
            let duration_last_frame = time.duration_since(self.time);
            let duration_running = time.duration_since(self.time_start);

            self.time = time;

            (duration_last_frame, duration_running)
        };

        // FIXME: Use `Duration::as_secs_f32` instead once it's stabilized.
        let duration_last_frame_s = duration_last_frame.as_secs() as f32
            + duration_last_frame.subsec_nanos() as f32 / 1_000_000_000.0;

        self.ui.set_delta_time(duration_last_frame_s);

        let app_input = self.process_input(&input);

        let [pan_ground_x, pan_ground_y] = app_input.camera_pan_ground;
        let [pan_screen_x, pan_screen_y] = app_input.camera_pan_screen;
        let [rotate_x, rotate_y] = app_input.camera_rotate;

        self.camera.pan_ground(pan_ground_x, pan_ground_y);
        self.camera.pan_screen(pan_screen_x, pan_screen_y);
        self.camera.rotate(rotate_x, rotate_y);
        self.camera.zoom(app_input.camera_zoom);
        self.camera.zoom_step(app_input.camera_zoom_steps);

        if app_input.camera_reset_viewport {
            self.active_camera_interpolation = Some(CameraInterpolation::new(
                &self.camera,
                &self.scene_geometries,
                time,
            ));
        }

        {
            if app_input.import_requested {
                if let Some(path) = tinyfiledialogs::open_file_dialog(
                    "Open",
                    "",
                    Some((&["*.obj"], "Wavefront (.obj)")),
                ) {
                    self.importer_worker.import_obj(&path);
                }
            }
        }

        if let Some(parsed_models) = self.importer_worker.parsed_obj() {
            match parsed_models {
                Ok(models) => {
                    // Clear existing scene first...
                    self.scene_geometries.clear();
                    for geometry in self.scene_renderer_geometry_ids.drain(..) {
                        self.renderer.remove_scene_geometry(geometry);
                    }

                    // ... and add everything we found to it
                    for model in models {
                        let geometry = model.geometry;
                        let renderer_geometry = SceneRendererGeometry::from_geometry(&geometry);
                        let renderer_geometry_id = self
                            .renderer
                            .add_scene_geometry(&renderer_geometry)
                            .expect("Failed to add geometry to renderer");

                        self.scene_geometries.push(geometry);
                        self.scene_renderer_geometry_ids.push(renderer_geometry_id);
                    }
                }
                Err(err) => {
                    tinyfiledialogs::message_box_ok(
                        "Error",
                        &format!("{}", err),
                        tinyfiledialogs::MessageBoxIcon::Error,
                    );
                }
            }

            self.active_camera_interpolation = Some(CameraInterpolation::new(
                &self.camera,
                &self.scene_geometries,
                time,
            ));
        }

        if app_input.close_requested {
            self.wants_close = true;
        }

        if let Some(interp) = self.active_camera_interpolation {
            if interp.target_time > time {
                let (sphere_origin, sphere_radius) = interp.update(time, &self.cubic_bezier);
                self.camera
                    .zoom_to_fit_visible_sphere(sphere_origin, sphere_radius);
            } else {
                self.camera
                    .zoom_to_fit_visible_sphere(interp.target_origin, interp.target_radius);
                self.active_camera_interpolation = None;
            }
        }

        // Camera matrices have to be uploaded when either window
        // resizes or the camera moves. We do it every frame for
        // simplicity.
        self.renderer
            .set_camera_matrices(&self.camera.projection_matrix(), &self.camera.view_matrix());

        let ui_frame = self.ui.prepare_frame(&self.window);

        #[cfg(debug_assertions)]
        ui_frame.draw_fps_window();

        let imgui_draw_data = ui_frame.render(&self.window);

        let mut render_pass = self.renderer.begin_render_pass();

        render_pass.draw_geometry(&self.scene_renderer_geometry_ids[..]);
        render_pass.draw_ui(&imgui_draw_data);

        render_pass.submit();
    }

    pub fn set_window_size(&mut self, logical_size: winit::dpi::LogicalSize) {
        let physical_size = logical_size.to_physical(self.window.hidpi_factor());
        log::debug!(
            "Window resized to new size: logical [{},{}], physical [{},{}]",
            logical_size.width,
            logical_size.height,
            physical_size.width,
            physical_size.height,
        );

        self.camera.set_window_size(physical_size);
        self.renderer.set_window_size(physical_size);
    }

    pub fn wants_close(&self) -> bool {
        self.wants_close
    }

    fn process_input(&mut self, input: &Input) -> AppInput {
        let no_modifiers =
            !input.meta_down && !input.shift_down && !input.ctrl_down && !input.alt_down;

        #[cfg(target_os = "macos")]
        let close_requested_macos =
            input.key_q_pressed && input.meta_down && !input.shift_down && !input.ctrl_down;
        #[cfg(not(target_os = "macos"))]
        let close_requested_macos = false;

        let mouse_move_x = input.mouse_move[0] as f32;
        let mouse_move_y = input.mouse_move[1] as f32;

        let mut camera_pan_ground = [0.0; 2];
        let mut camera_pan_screen = [0.0; 2];
        let mut camera_rotate = [0.0; 2];
        let mut camera_zoom = 0.0;

        if input.lmb_down && input.rmb_down {
            camera_zoom = -mouse_move_y;
        } else if input.lmb_down {
            camera_rotate[0] = -mouse_move_x;
            camera_rotate[1] = -mouse_move_y;
        } else if input.rmb_down {
            if input.shift_down {
                camera_pan_ground[0] = mouse_move_x;
                camera_pan_ground[1] = -mouse_move_y;
            } else {
                camera_pan_screen[0] = mouse_move_x;
                camera_pan_screen[1] = -mouse_move_y;
            }
        }

        AppInput {
            // TODO: mouse/kbd captured by gui?
            camera_pan_ground,
            camera_pan_screen,
            camera_rotate,
            camera_zoom,
            camera_zoom_steps: input.mouse_wheel,
            camera_reset_viewport: input.key_a_pressed && no_modifiers,
            import_requested: input.key_o_pressed && no_modifiers,
            close_requested: input.close_requested || close_requested_macos,
        }
    }
}

struct AppInput {
    camera_pan_ground: [f32; 2],
    camera_pan_screen: [f32; 2],
    camera_rotate: [f32; 2],
    camera_zoom: f32,
    camera_zoom_steps: i32,
    camera_reset_viewport: bool,
    import_requested: bool,
    close_requested: bool,
}

#[derive(Debug, Clone, Copy)]
struct CameraInterpolation {
    source_origin: Point3<f32>,
    source_radius: f32,
    target_origin: Point3<f32>,
    target_radius: f32,
    target_time: Instant,
}

impl CameraInterpolation {
    fn new(camera: &Camera, scene_geometries: &[geometry::Geometry], time: Instant) -> Self {
        let (source_origin, source_radius) = camera.visible_sphere();
        let (target_origin, target_radius) = geometry::compute_bounding_sphere(&scene_geometries);

        CameraInterpolation {
            source_origin,
            source_radius,
            target_origin,
            target_radius,
            target_time: time + CAMERA_INTERPOLATION_DURATION,
        }
    }

    fn update(&self, time: Instant, easing: &CubicBezierEasing) -> (Point3<f32>, f32) {
        let duration_left = duration_as_secs_f32(self.target_time.duration_since(time));
        let whole_duration = duration_as_secs_f32(CAMERA_INTERPOLATION_DURATION);
        let t = easing.apply(1.0 - duration_left / whole_duration);

        let sphere_origin = Point3::from(
            self.source_origin
                .coords
                .lerp(&self.target_origin.coords, t),
        );
        let sphere_radius = math::lerp(self.source_radius, self.target_radius, t);

        (sphere_origin, sphere_radius)
    }
}

fn duration_as_secs_f32(duration: Duration) -> f32 {
    // FIXME: Use `Duration::as_secs_f32` instead once it's stabilized.
    duration.as_secs() as f32 + duration.subsec_nanos() as f32 / 1_000_000_000.0
}
