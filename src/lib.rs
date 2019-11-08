pub use crate::renderer::{GpuBackend, Msaa, PresentMode};

use std::time::{Duration, Instant};

use nalgebra::geometry::Point3;

use crate::camera::{Camera, CameraOptions};
use crate::importer::ImporterWorker;
use crate::input::InputManager;
use crate::interpreter::ast;
use crate::interpreter_funcs as funcs;
use crate::interpreter_server::{InterpreterRequest, InterpreterResponse, InterpreterServer};
use crate::renderer::{DrawGeometryMode, GpuGeometry, Options as RendererOptions, Renderer};
use crate::ui::Ui;

pub mod geometry;
pub mod importer;
pub mod logger;
pub mod operations;
pub mod renderer;

mod camera;
mod convert;
mod edge_analysis;
mod face_tools;
mod input;
mod interpreter;
mod interpreter_funcs;
mod interpreter_server;
mod math;
mod mesh_analysis;
mod mesh_smoothing;
mod mesh_tools;
mod mesh_topology_analysis;
mod platform;
mod ui;
mod vertex_tools;

const CAMERA_INTERPOLATION_DURATION: Duration = Duration::from_millis(1000);

#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Options {
    /// Which multi-sampling setting to use.
    pub msaa: Msaa,
    /// Whether to run with VSync or not.
    pub present_mode: PresentMode,
    /// Whether to select an explicit gpu backend for the renderer to use.
    pub gpu_backend: Option<GpuBackend>,
}

/// Initialize the window and run in infinite loop.
///
/// Will continue running until a close request is received from the
/// created window.
pub fn init_and_run(options: Options) -> ! {
    let event_loop = winit::event_loop::EventLoop::new();
    // let monitor_id = event_loop.primary_monitor();
    let window = winit::window::WindowBuilder::new()
        .with_title("H.U.R.B.A.N. Selector")
        // .with_fullscreen(Some(monitor_id))
        .build(&event_loop)
        .expect("Failed to create window");

    let window_size = window.inner_size().to_physical(window.hidpi_factor());

    let importer_worker = ImporterWorker::new();
    let mut interpreter_server = InterpreterServer::new();
    let mut input_manager = InputManager::new();
    let mut ui = Ui::new(&window);

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

    let mut renderer_draw_geometry_mode = DrawGeometryMode::Shaded;
    let mut renderer = Renderer::new(
        &window,
        &camera.projection_matrix(),
        &camera.view_matrix(),
        ui.fonts(),
        RendererOptions {
            // FIXME: @Correctness Msaa X4 is the only value currently
            // working on all devices we tried. Once msaa capabilities
            // are queryable with wgpu `Limits`, we should have a
            // chain of options the renderer tries before giving up,
            // and this field should be renamed to `desired_msaa`.
            msaa: options.msaa,
            present_mode: options.present_mode,
            gpu_backend: options.gpu_backend,
        },
    );

    let mut scene_geometries = vec![
        // FIXME: This is just temporary code so that we can see
        // something in the scene and know the renderer works.
        geometry::uv_sphere([0.0, 0.0, 0.0], 5.0, 2, 3),
        geometry::uv_sphere([20.0, 0.0, 0.0], 5.0, 3, 3),
        geometry::uv_sphere([40.0, 0.0, 0.0], 5.0, 4, 4),
        geometry::uv_sphere([60.0, 0.0, 0.0], 5.0, 5, 5),
        geometry::uv_sphere([80.0, 0.0, 0.0], 5.0, 6, 6),
        geometry::uv_sphere([100.0, 0.0, 0.0], 5.0, 7, 7),
        geometry::uv_sphere([120.0, 0.0, 0.0], 5.0, 8, 8),
        geometry::uv_sphere([170.0, 0.0, 0.0], 20.0, 20, 20),
        geometry::cube_sharp_var_len([220.0, 0.0, 0.0], 5.0),
        geometry::cube_smooth_same_len([260.0, 0.0, 0.0], 5.0),
        geometry::cube_smooth_same_len([280.0, 0.0, 0.0], 5.0),
        geometry::plane_same_len([300.0, 0.0, 0.0], 5.0),
        geometry::plane_var_len([320.0, 0.0, 0.0], 5.0),
    ];

    let mut scene_renderer_geometry_ids = Vec::with_capacity(scene_geometries.len());
    for geometry in &scene_geometries {
        let renderer_geometry = GpuGeometry::from_geometry(geometry);
        let renderer_geometry_id = renderer
            .add_scene_geometry(&renderer_geometry)
            .expect("Failed to add geometry to renderer");
        scene_renderer_geometry_ids.push(renderer_geometry_id);
    }

    let cubic_bezier = math::CubicBezierEasing::new([0.7, 0.0], [0.3, 1.0]);

    let time_start = Instant::now();
    let mut time = time_start;

    let mut camera_interpolation: Option<CameraInterpolation> = None;

    // Since input manager needs to process events separately after imgui
    // handles them, this buffer with copies of events is needed.
    let mut input_events: Vec<winit::event::Event<_>> = Vec::with_capacity(16);

    event_loop.run(move |event, _, control_flow| {
        *control_flow = winit::event_loop::ControlFlow::Poll;

        match event {
            winit::event::Event::EventsCleared => {
                let (duration_last_frame, _duration_running) = {
                    let now = Instant::now();
                    let duration_last_frame = now.duration_since(time);
                    let duration_running = now.duration_since(time_start);
                    time = now;

                    (duration_last_frame, duration_running)
                };

                ui.set_delta_time(duration_last_frame.as_secs_f32());

                let ui_frame = ui.prepare_frame(&window);
                input_manager.start_frame();

                for event in input_events.drain(..) {
                    input_manager.process_event(
                        &event,
                        ui_frame.want_capture_keyboard(),
                        ui_frame.want_capture_mouse(),
                    );
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
                    camera_interpolation =
                        Some(CameraInterpolation::new(&camera, &scene_geometries, time));
                }

                if input_state.import_requested {
                    if let Some(path) = tinyfiledialogs::open_file_dialog(
                        "Open",
                        "",
                        Some((&["*.obj"], "Wavefront (.obj)")),
                    ) {
                        importer_worker.import_obj(&path);
                    }
                }

                if let Some(parsed_models) = importer_worker.parsed_obj() {
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
                                let renderer_geometry = GpuGeometry::from_geometry(&geometry);
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

                if input_state.tmp_submit_prog_and_run {
                    let prog = ast::Prog::new(vec![
                        ast::Stmt::VarDecl(ast::VarDeclStmt::new(
                            ast::VarIdent(0),
                            ast::CallExpr::new(
                                funcs::FUNC_ID_CREATE_UV_SPHERE,
                                vec![
                                    ast::Expr::Lit(ast::LitExpr::Float(500.0)),
                                    ast::Expr::Lit(ast::LitExpr::Uint(20)),
                                    ast::Expr::Lit(ast::LitExpr::Uint(20)),
                                ],
                            ),
                        )),
                        ast::Stmt::VarDecl(ast::VarDeclStmt::new(
                            ast::VarIdent(1),
                            ast::CallExpr::new(
                                funcs::FUNC_ID_SHRINK_WRAP,
                                vec![
                                    ast::Expr::Var(ast::VarExpr::new(ast::VarIdent(0))),
                                    ast::Expr::Lit(ast::LitExpr::Uint(30)),
                                ],
                            ),
                        )),
                        ast::Stmt::VarDecl(ast::VarDeclStmt::new(
                            ast::VarIdent(2),
                            ast::CallExpr::new(
                                funcs::FUNC_ID_TRANSFORM,
                                vec![
                                    ast::Expr::Var(ast::VarExpr::new(ast::VarIdent(1))),
                                    ast::Expr::Lit(ast::LitExpr::Float3([2000.0, 0.0, 0.0])),
                                    ast::Expr::Lit(ast::LitExpr::Nil),
                                    ast::Expr::Lit(ast::LitExpr::Nil),
                                ],
                            ),
                        )),
                        ast::Stmt::VarDecl(ast::VarDeclStmt::new(
                            ast::VarIdent(3),
                            ast::CallExpr::new(
                                funcs::FUNC_ID_TRANSFORM,
                                vec![
                                    ast::Expr::Var(ast::VarExpr::new(ast::VarIdent(1))),
                                    ast::Expr::Lit(ast::LitExpr::Float3([4000.0, 0.0, 0.0])),
                                    ast::Expr::Lit(ast::LitExpr::Float3([30.0, 0.0, 0.0])),
                                    ast::Expr::Lit(ast::LitExpr::Nil),
                                ],
                            ),
                        )),
                        ast::Stmt::VarDecl(ast::VarDeclStmt::new(
                            ast::VarIdent(4),
                            ast::CallExpr::new(
                                funcs::FUNC_ID_TRANSFORM,
                                vec![
                                    ast::Expr::Var(ast::VarExpr::new(ast::VarIdent(1))),
                                    ast::Expr::Lit(ast::LitExpr::Float3([6000.0, 0.0, 0.0])),
                                    ast::Expr::Lit(ast::LitExpr::Nil),
                                    ast::Expr::Lit(ast::LitExpr::Float3([2.0, 2.0, 5.0])),
                                ],
                            ),
                        )),
                    ]);

                    interpreter_server.submit_request(InterpreterRequest::SetProg(prog));
                    interpreter_server.submit_request(InterpreterRequest::Interpret);
                }

                while let Ok((request_id, response)) = interpreter_server.poll_response() {
                    match response {
                        InterpreterResponse::Completed => {
                            log::info!("Interpreter completed request {:?}", request_id,)
                        }
                        InterpreterResponse::CompletedWithResult(result) => {
                            log::info!(
                                "Interpreter completed request {:?} with result",
                                request_id,
                            );

                            let value_set = result.unwrap();

                            for (_, value) in &value_set.used_values {
                                let geometry = value.unwrap_geometry().clone();
                                let renderer_geometry = GpuGeometry::from_geometry(&geometry);
                                let renderer_geometry_id =
                                    renderer.add_scene_geometry(&renderer_geometry).unwrap();

                                scene_geometries.push(geometry);
                                scene_renderer_geometry_ids.push(renderer_geometry_id);
                            }

                            for (_, value) in &value_set.unused_values {
                                let geometry = value.unwrap_geometry().clone();
                                let renderer_geometry = GpuGeometry::from_geometry(&geometry);
                                let renderer_geometry_id =
                                    renderer.add_scene_geometry(&renderer_geometry).unwrap();

                                scene_geometries.push(geometry);
                                scene_renderer_geometry_ids.push(renderer_geometry_id);
                            }
                        }
                    }
                }

                if input_state.close_requested {
                    *control_flow = winit::event_loop::ControlFlow::Exit;
                }

                if let Some(logical_size) = input_state.window_resized {
                    let physical_size = logical_size.to_physical(window.hidpi_factor());
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
                        let (sphere_origin, sphere_radius) = interp.update(time, &cubic_bezier);
                        camera.zoom_to_fit_visible_sphere(sphere_origin, sphere_radius);
                    } else {
                        camera
                            .zoom_to_fit_visible_sphere(interp.target_origin, interp.target_radius);
                        camera_interpolation = None;
                    }
                }

                // Camera matrices have to be uploaded when either window
                // resizes or the camera moves. We do it every frame for
                // simplicity.
                renderer.set_camera_matrices(&camera.projection_matrix(), &camera.view_matrix());

                ui_frame.draw_renderer_settings_window(&mut renderer_draw_geometry_mode);

                let imgui_draw_data = ui_frame.render(&window);

                let mut render_pass = renderer.begin_render_pass();

                render_pass.draw_geometry(
                    &scene_renderer_geometry_ids[..],
                    renderer_draw_geometry_mode,
                );
                render_pass.draw_ui(imgui_draw_data);

                render_pass.submit();
            }

            winit::event::Event::WindowEvent {
                event: winit::event::WindowEvent::RedrawRequested,
                ..
            } => {
                // We don't answer redraw requests and instead draw in
                // the "events cleared" for 2 reasons:
                //
                // 1) Doing it with VSync is challenging - redrawing
                //    at the OS's whim while knowing that each redraw
                //    will block until the monitor flips sounds
                //    dangerous. We could still redraw here only when
                //    we were running without VSync, but...
                //
                // 2) ImGui produces a draw list with `render()`. The
                //    drawlist shares the lifetime of the `Ui` frame
                //    context, which is dropped at the end of "events
                //    cleared". We could copy the draw list and stash
                //    it for our subsequent handling of redraw
                //    requests, but it contains raw pointers to the
                //    `Ui` frame context which I am not sure are alive
                //    (or even contain correct data) by the time we
                //    get here. We could also try to prolong the
                //    lifetime of the `Ui` by not dropping it, but
                //    this is Rust...
            }

            winit::event::Event::WindowEvent { .. } => {
                ui.handle_event(&window, &event);
                input_events.push(event.clone());
            }

            _ => (),
        }
    });
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

    fn update(&self, time: Instant, easing: &math::CubicBezierEasing) -> (Point3<f32>, f32) {
        let duration_left = self.target_time.duration_since(time).as_secs_f32();
        let whole_duration = CAMERA_INTERPOLATION_DURATION.as_secs_f32();
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
