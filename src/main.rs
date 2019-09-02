#![windows_subsystem = "windows"]

use std::collections::HashMap;
use std::env;
use std::fs;
use std::time::Instant;

use wgpu::winit;

use hurban_selector::camera::{Camera, CameraOptions};
use hurban_selector::geometry;
use hurban_selector::importer::ImporterWorker;
use hurban_selector::input::InputManager;
use hurban_selector::math::decay;
use hurban_selector::renderer::{Msaa, Renderer, RendererOptions, SceneRendererGeometry};
use hurban_selector::ui::Ui;

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

    let mut ui = Ui::new(&window);

    let mut renderer = Renderer::new(
        &wgpu_instance,
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
                let filename = obj_path
                    .file_stem()
                    .expect("Failed to extract obj file stem")
                    .to_str()
                    .expect("Filename UTF-8 conversion failed");
                let filepath = obj_path
                    .to_str()
                    .expect("Failed to convert Path to str")
                    .to_string();

                obj_file_paths.insert(filename.to_uppercase(), filepath);
            }
        }
    }

    let mut obj_filenames: Vec<String> = obj_file_paths.keys().cloned().collect();
    obj_filenames.sort();

    let importer_worker = ImporterWorker::new();
    let mut is_importing = false;
    let mut import_progress = 1.0;
    let mut selected_model = String::from("");
    let mut running = true;

    while running {
        let now = Instant::now();
        let duration_last_frame = now.duration_since(time);
        let _duration_running = now.duration_since(time_start);
        time = now;

        // FIXME: Use `Duration::as_secs_f32` instead once it's stabilized.
        let duration_last_frame_s = duration_last_frame.as_secs() as f32
            + duration_last_frame.subsec_nanos() as f32 / 1_000_000_000.0;

        ui.imgui_context().io_mut().delta_time = duration_last_frame_s;

        import_progress = if !is_importing {
            1.0
        } else {
            decay(import_progress, 1.0, 0.5, duration_last_frame_s)
        };

        // Since input manager needs to process events separately after imgui
        // handles them, this buffer with copies of events is needed.
        let mut input_events = vec![];

        event_loop.poll_events(|event| {
            input_events.push(event.clone());
            ui.handle_event(&event);
        });

        // Start UI and input manger frames
        let ui_frame = ui.prepare_frame();

        input_manager.start_frame();

        // Imgui's IO is updated after current frame starts, else it'd contain
        // outdated values.
        let ui_captured_keyboard = ui_frame.io().want_capture_keyboard;
        let ui_captured_mouse = ui_frame.io().want_capture_mouse;

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
            let (origin, radius) = geometry::compute_bounding_sphere(&scene_geometries[..]);
            camera.zoom_to_fit_sphere(&origin, radius);
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

        // Camera matrices have to be uploaded when either window
        // resizes or the camera moves. We do it every frame for
        // simplicity.
        renderer.set_camera_matrices(&camera.projection_matrix(), &camera.view_matrix());

        #[cfg(debug_assertions)]
        ui_frame.draw_fps_window();

        // Clicking any of the models in the list means clearing everything out
        // of the scene, importing given model and pushing it into scene.
        if let Some(clicked_model) =
            ui_frame.draw_model_window(&obj_filenames, &selected_model, import_progress)
        {
            let clicked_model_path: &str = &obj_file_paths[&clicked_model];
            importer_worker.import_obj(&clicked_model_path);

            selected_model = clicked_model;
            is_importing = true;
            import_progress = 0.0;
        }

        let imgui_draw_data = ui_frame.render();

        let mut render_pass = renderer.begin_render_pass();

        render_pass.draw_geometry(&scene_renderer_geometry_ids[..]);
        render_pass.draw_ui(&imgui_draw_data);

        render_pass.submit();
    }

    for renderer_geometry_id in scene_renderer_geometry_ids {
        renderer.remove_scene_geometry(renderer_geometry_id);
    }
}
