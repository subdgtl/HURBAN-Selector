use std::time::Instant;

use wgpu::winit;

use hurban_selector::camera::{Camera, CameraOptions};
use hurban_selector::imgui_renderer::ImguiRenderer;
use hurban_selector::importer::Importer;
use hurban_selector::input::InputManager;
use hurban_selector::primitives;
use hurban_selector::ui;
use hurban_selector::viewport_renderer::{
    Geometry, Msaa, ViewportRenderer, ViewportRendererOptions,
};

const SWAP_CHAIN_FORMAT: wgpu::TextureFormat = wgpu::TextureFormat::Bgra8Unorm;

fn main() {
    env_logger::init();

    let mut event_loop = winit::EventsLoop::new();
    let window = winit::Window::new(&event_loop).expect("Failed to create window.");
    let window_size = window
        .get_inner_size()
        .expect("Failed to get window inner size")
        .to_physical(window.get_hidpi_factor());

    let wgpu_instance = wgpu::Instance::new();
    let surface = wgpu_instance.create_surface(&window);
    let adapter = wgpu_instance.get_adapter(&wgpu::AdapterDescriptor {
        power_preference: wgpu::PowerPreference::HighPerformance,
    });

    let mut device = adapter.request_device(&wgpu::DeviceDescriptor {
        extensions: wgpu::Extensions {
            anisotropic_filtering: false,
        },
        limits: wgpu::Limits::default(),
    });
    let mut swap_chain = create_swap_chain(&device, &surface, window_size);

    let mut input_manager = InputManager::new();
    let mut camera = Camera::new(
        [window_size.width as f32, window_size.height as f32],
        5.0,
        45f32.to_radians(),
        60f32.to_radians(),
        CameraOptions {
            radius_min: 1.0,
            radius_max: 500.0,
            polar_angle_distance_min: 1f32.to_radians(),
            speed_pan: 0.1,
            speed_rotate: 0.005,
            speed_zoom: 0.01,
            speed_zoom_step: 1.0,
            fovy: 45f32.to_radians(),
            znear: 0.01,
            zfar: 1000.0,
        },
    );
    let mut viewport_renderer = ViewportRenderer::new(
        &mut device,
        window_size,
        &camera.projection_matrix(),
        &camera.view_matrix(),
        ViewportRendererOptions {
            // FIXME: Msaa X4 is the only value currently working on
            // all devices we tried. We should query the device
            // capabilities (but how?!) to select the proper MSAA
            // value.
            msaa: Msaa::X4,
            output_format: SWAP_CHAIN_FORMAT,
        },
    );

    // FIXME: This is just temporary code so that we can see something
    // in the scene and know the renderer works.
    let cube1_g = primitives::uv_cube([0.0, 0.0, 0.0], 0.5);
    let cube2_g = primitives::uv_cube([5.0, 5.0, 0.0], 0.7);
    let cube3_g = primitives::uv_cube([5.0, 0.0, 0.0], 1.0);
    let cube4_g = primitives::cube([0.0, 5.0, 0.0], 0.9);
    let cube5_g = primitives::cube([0.0, 0.0, 5.0], 1.5);
    let plane1_g = primitives::plane([0.0, 0.0, 20.0], 10.0);
    let plane2_g = primitives::plane([0.0, 0.0, -20.0], 10.0);

    let cube1 = viewport_renderer.add_geometry(&device, &cube1_g).unwrap();
    let cube2 = viewport_renderer.add_geometry(&device, &cube2_g).unwrap();
    let cube3 = viewport_renderer.add_geometry(&device, &cube3_g).unwrap();
    let cube4 = viewport_renderer.add_geometry(&device, &cube4_g).unwrap();
    let cube5 = viewport_renderer.add_geometry(&device, &cube5_g).unwrap();
    let plane1 = viewport_renderer.add_geometry(&device, &plane1_g).unwrap();
    let plane2 = viewport_renderer.add_geometry(&device, &plane2_g).unwrap();

    let mut dynamic_models = Vec::new();

    let (mut imgui_context, mut winit_platform) = ui::init(&window);
    let mut imgui_renderer = ImguiRenderer::new(
        &mut imgui_context,
        &mut device,
        wgpu::TextureFormat::Bgra8Unorm,
        None,
    )
    .expect("Failed to create imgui renderer");

    let time_start = Instant::now();
    let mut time = time_start;

    let mut importer = Importer::new();
    let mut running = true;

    while running {
        let (_duration_last_frame, _duration_running) = {
            let now = Instant::now();
            let duration_last_frame = now.duration_since(time);
            let duration_running = now.duration_since(time_start);
            time = now;

            // FIXME: Use `Duration::as_secs_f32` instead once it's stabilized.
            let duration_last_frame_s = duration_last_frame.as_secs() as f32
                + duration_last_frame.subsec_nanos() as f32 / 1_000_000_000.0;

            imgui_context.io_mut().delta_time = duration_last_frame_s;

            (duration_last_frame, duration_running)
        };

        // Since input manager needs to process events separately after imgui
        // handles them, this buffer with copies of events is needed.
        let mut input_events = vec![];

        event_loop.poll_events(|event| {
            input_events.push(event.clone());
            winit_platform.handle_event(imgui_context.io_mut(), &window, &event);
        });

        // Start UI and input manger frames
        winit_platform
            .prepare_frame(imgui_context.io_mut(), &window)
            .expect("Failed to start imgui frame");
        let imgui_ui = imgui_context.frame();
        let imgui_ui_io = imgui_ui.io();

        input_manager.start_frame();

        // Imgui's IO is updated after current frame starts, else it'd contain
        // outdated values.
        let ui_captured_keyboard = imgui_ui_io.want_capture_keyboard;
        let ui_captured_mouse = imgui_ui_io.want_capture_mouse;

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
            // FIXME: For now this is a cheap man's version of:
            // https://trello.com/c/JFNQ6GyJ/85-center-viewport
            camera.reset_origin();
        }

        if input_state.import_requested {
            if let Some(path) = tinyfiledialogs::open_file_dialog(
                "Open",
                "",
                Some((&["*.obj"], "Wavefront (.obj)")),
            ) {
                match importer.import_obj(&path) {
                    Ok(models) => {
                        for model in models {
                            let geometry = Geometry::from_positions_and_normals_indexed(
                                model.indices,
                                model.vertex_positions,
                                model.vertex_normals,
                            );
                            let model_handle =
                                viewport_renderer.add_geometry(&device, &geometry).unwrap();
                            dynamic_models.push(model_handle);
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

            camera.set_screen_size([physical_size.width as f32, physical_size.height as f32]);

            swap_chain = create_swap_chain(&device, &surface, physical_size);
            viewport_renderer.set_screen_size(&mut device, physical_size);
        }

        // Camera matrices have to be uploaded when either window
        // resizes, or anything the camera moves. Seems easier to just
        // upload them always.
        viewport_renderer.set_camera_matrices(
            &mut device,
            &camera.projection_matrix(),
            &camera.view_matrix(),
        );

        ui::draw_fps_window(&imgui_ui);

        winit_platform.prepare_render(&imgui_ui, &window);
        let imgui_draw_data = imgui_ui.render();

        let frame = swap_chain.get_next_texture();
        let mut encoder =
            device.create_command_encoder(&wgpu::CommandEncoderDescriptor { todo: 0 });

        viewport_renderer.draw_geometry(
            &mut encoder,
            &frame.view,
            &[cube1, cube2, cube3, cube4, cube5, plane1, plane2],
        );

        if !dynamic_models.is_empty() {
            viewport_renderer.draw_geometry(&mut encoder, &frame.view, &dynamic_models[..]);
        }

        imgui_renderer
            .render(&mut device, &mut encoder, &frame.view, imgui_draw_data)
            .expect("Should render an imgui frame");

        device.get_queue().submit(&[encoder.finish()]);
    }

    viewport_renderer.remove_geometry(cube1);
    viewport_renderer.remove_geometry(cube2);
    viewport_renderer.remove_geometry(cube3);
    viewport_renderer.remove_geometry(cube4);
    viewport_renderer.remove_geometry(cube5);
    viewport_renderer.remove_geometry(plane1);
    viewport_renderer.remove_geometry(plane2);
}

fn create_swap_chain(
    device: &wgpu::Device,
    surface: &wgpu::Surface,
    window_size: winit::dpi::PhysicalSize,
) -> wgpu::SwapChain {
    device.create_swap_chain(
        &surface,
        &wgpu::SwapChainDescriptor {
            usage: wgpu::TextureUsage::OUTPUT_ATTACHMENT,
            format: SWAP_CHAIN_FORMAT,
            width: window_size.width.round() as u32,
            height: window_size.height.round() as u32,
            present_mode: wgpu::PresentMode::Vsync,
        },
    )
}
