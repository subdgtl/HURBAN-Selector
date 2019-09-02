#![windows_subsystem = "windows"]

use std::collections::HashMap;
use std::env;
use std::fs;
use std::time::{Duration, Instant};

use nalgebra::geometry::Point3;
use wgpu::winit;

use hurban_selector::camera::{Camera, CameraOptions};
use hurban_selector::geometry;
use hurban_selector::importer::ImporterWorker;
use hurban_selector::input::InputManager;
use hurban_selector::math;
use hurban_selector::renderer::{Msaa, Renderer, RendererOptions, SceneRendererGeometry};
use hurban_selector::ui;

const CAMERA_INTERPOLATION_DURATION: Duration = Duration::from_millis(1000);

fn main() {
    env_logger::init();

    let mut event_loop = winit::EventsLoop::new();
    // let monitor_id = event_loop.get_primary_monitor();
    let window = winit::WindowBuilder::new()
        .with_title("H.U.R.B.A.N. Selector")
        // .with_fullscreen(Some(monitor_id))
        // .with_maximized(true)
        // .with_multitouch()
        .build(&event_loop)
        .expect("Failed to create window");

    let window_size = window
        .get_inner_size()
        .expect("Failed to get window inner size")
        .to_physical(window.get_hidpi_factor());

    let wgpu_instance = wgpu::Instance::new();

    let mut input_manager = InputManager::new();
    let cubic_bezier = math::CubicBezierEasing::new([0.7, 0.0], [0.3, 1.0]);

    let mut camera = Camera::new(
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

    let (mut imgui_context, mut imgui_winit_platform) = ui::init(&window);

    let mut renderer = Renderer::new(
        &wgpu_instance,
        &window,
        &camera.projection_matrix(),
        &camera.view_matrix(),
        imgui_context.fonts(),
        RendererOptions {
            // FIXME: @Correctness Msaa X4 is the only value currently
            // working on all devices we tried. Once device
            // capabilities are queryable with wgpu `Limits`, we
            // should have a chain of options the renderer tries
            // before giving up.
            msaa: Msaa::X4,
        },
    );

    let mut scene_geometries = vec![
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

    let time_start = Instant::now();
    let mut time = time_start;

    // Temporary model list

    let models_dir = env::var_os("MODELS_DIR").unwrap_or_else(|| {
        env::current_dir()
            .expect("Failed to load current dir")
            .into()
    });
    let obj_dir_entry_results =
        fs::read_dir(models_dir).expect("Failed to read directory with obj files");
    let mut obj_file_paths: HashMap<String, String> = HashMap::new();

    for obj_dir_entry_result in obj_dir_entry_results {
        let obj_dir_entry = obj_dir_entry_result.expect("Failed to read directory entry");
        let obj_path = obj_dir_entry.path();

        if let Some(ext) = obj_path.extension() {
            if ext == "obj" {
                let filename = obj_dir_entry
                    .file_name()
                    .into_string()
                    .expect("Filename UTF-8 conversion failed");
                let filepath = obj_path
                    .into_os_string()
                    .into_string()
                    .expect("File path UTF-8 conversion failed");

                obj_file_paths.insert(filename, filepath);
            }
        }
    }

    let mut obj_filenames: Vec<String> = obj_file_paths.keys().cloned().collect();
    obj_filenames.sort();

    let importer_worker = ImporterWorker::new();

    let mut is_importing = false;
    let mut import_progress = 1.0;

    let mut camera_interpolation: Option<CameraInterpolation> = None;

    let mut running = true;

    while running {
        let (duration_last_frame, _duration_running) = {
            let now = Instant::now();
            let duration_last_frame = now.duration_since(time);
            let duration_running = now.duration_since(time_start);
            time = now;

            (duration_last_frame, duration_running)
        };

        let duration_last_frame_s = duration_as_secs_f32(duration_last_frame);
        imgui_context.io_mut().delta_time = duration_last_frame_s;

        import_progress = if !is_importing {
            1.0
        } else {
            math::decay(import_progress, 1.0, 0.5, duration_last_frame_s)
        };

        // Since input manager needs to process events separately after imgui
        // handles them, this buffer with copies of events is needed.
        let mut input_events = vec![];

        event_loop.poll_events(|event| {
            input_events.push(event.clone());
            imgui_winit_platform.handle_event(imgui_context.io_mut(), &window, &event);
        });

        // Start UI and input manger frames
        imgui_winit_platform
            .prepare_frame(imgui_context.io_mut(), &window)
            .expect("Failed to start imgui frame");
        let imgui_ui = imgui_context.frame();
        let imgui_io = imgui_ui.io();

        input_manager.start_frame();

        // Imgui's IO is updated after current frame starts, else it'd contain
        // outdated values.
        let ui_captured_keyboard = imgui_io.want_capture_keyboard;
        let ui_captured_mouse = imgui_io.want_capture_mouse;

        for event in input_events {
            input_manager.process_event(event, ui_captured_keyboard, ui_captured_mouse);
        }

        let input_state = input_manager.input_state();

        let [pan_ground_x, pan_ground_y] = input_state.camera_pan_ground;
        let [pan_screen_x, pan_screen_y] = input_state.camera_pan_screen;
        let [rotate_x, rotate_y] = input_state.camera_rotate;

        camera.pan_ground(pan_ground_x, pan_ground_y);
        camera.pan_screen(pan_screen_x, pan_screen_y);
        camera.rotate(rotate_x, rotate_y);
        camera.zoom(input_state.camera_zoom);
        camera.zoom_step(input_state.camera_zoom_steps);

        if input_state.camera_reset_viewport {
            camera_interpolation = Some(CameraInterpolation::new(&camera, &scene_geometries, time));
        }

        #[cfg(debug_assertions)]
        {
            if input_state.import_requested {
                if let Some(path) = tinyfiledialogs::open_file_dialog(
                    "Open",
                    "",
                    Some((&["*.obj"], "Wavefront (.obj)")),
                ) {
                    importer_worker.import_obj(&path);

                    is_importing = true;
                    import_progress = 0.0;
                }
            }
        }

        if let Some(parsed_models) = importer_worker.parsed_obj() {
            is_importing = false;
            import_progress = 1.0;

            match parsed_models {
                Ok(models) => {
                    // Clear existing scene first...
                    scene_geometries.clear();
                    for geometry in scene_renderer_geometry_ids.drain(..) {
                        renderer.remove_scene_geometry(geometry);
                    }

                    // ... and add everything we found to it
                    for model in models {
                        let geometry = model.geometry;
                        let renderer_geometry = SceneRendererGeometry::from_geometry(&geometry);
                        let renderer_geometry_id = renderer
                            .add_scene_geometry(&renderer_geometry)
                            .expect("Failed to add geometry to renderer");

                        scene_geometries.push(geometry);
                        scene_renderer_geometry_ids.push(renderer_geometry_id);
                    }

                    camera_interpolation =
                        Some(CameraInterpolation::new(&camera, &scene_geometries, time));
                }
                Err(err) => {
                    tinyfiledialogs::message_box_ok(
                        "Error",
                        &format!("{}", err),
                        tinyfiledialogs::MessageBoxIcon::Error,
                    );
                }
            }
        }

        if input_state.close_requested {
            running = false;
        }

        if let Some(logical_size) = input_state.window_resized {
            let physical_size = logical_size.to_physical(window.get_hidpi_factor());
            log::debug!(
                "Window resized to new size: logical [{},{}], physical [{},{}]",
                logical_size.width,
                logical_size.height,
                physical_size.width,
                physical_size.height,
            );

            camera.set_window_size(physical_size);
            renderer.set_window_size(physical_size);
        }

        if let Some(interp) = camera_interpolation {
            if interp.target_time > time {
                let duration_left = interp.duration_left(time);
                let whole_duration = duration_as_secs_f32(CAMERA_INTERPOLATION_DURATION);
                let t = cubic_bezier.apply(1.0 - duration_left / whole_duration);

                let sphere_origin = interp.sphere_origin(t);
                let sphere_radius = interp.sphere_radius(t);

                camera.zoom_to_fit_visible_sphere(sphere_origin, sphere_radius);
            } else {
                camera.zoom_to_fit_visible_sphere(interp.target_origin, interp.target_radius);
                camera_interpolation = None;
            }
        }

        // Camera matrices have to be uploaded when either window
        // resizes or the camera moves. We do it every frame for
        // simplicity.
        renderer.set_camera_matrices(&camera.projection_matrix(), &camera.view_matrix());

        #[cfg(debug_assertions)]
        ui::draw_fps_window(&imgui_ui);

        // Clicking any of the models in the list means clearing everything out
        // of the scene, importing given model and pushing it into scene.
        if let Some(clicked_model) =
            ui::draw_model_window(&imgui_ui, &obj_filenames, import_progress)
        {
            let clicked_model_path: &str = &obj_file_paths[&clicked_model];
            importer_worker.import_obj(&clicked_model_path);

            is_importing = true;
            import_progress = 0.0;
        }

        imgui_winit_platform.prepare_render(&imgui_ui, &window);
        let imgui_draw_data = imgui_ui.render();

        let mut render_pass = renderer.begin_render_pass();

        render_pass.draw_geometry(&scene_renderer_geometry_ids[..]);
        render_pass.draw_ui(&imgui_draw_data);

        render_pass.submit();
    }

    for renderer_geometry_id in scene_renderer_geometry_ids {
        renderer.remove_scene_geometry(renderer_geometry_id);
    }
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

    fn duration_left(&self, time: Instant) -> f32 {
        duration_as_secs_f32(self.target_time.duration_since(time))
    }

    fn sphere_origin(&self, t: f32) -> Point3<f32> {
        Point3::from(
            self.source_origin
                .coords
                .lerp(&self.target_origin.coords, t),
        )
    }

    fn sphere_radius(&self, t: f32) -> f32 {
        math::lerp(self.source_radius, self.target_radius, t)
    }
}

fn duration_as_secs_f32(duration: Duration) -> f32 {
    // FIXME: Use `Duration::as_secs_f32` instead once it's stabilized.
    duration.as_secs() as f32 + duration.subsec_nanos() as f32 / 1_000_000_000.0
}
