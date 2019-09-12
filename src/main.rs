#![windows_subsystem = "windows"]

use std::env;
use std::time::Instant;

use hurban_selector::app::{App, AppOptions, GpuBackend, Msaa, PresentMode};
use hurban_selector::platform::InputManager;

fn main() {
    env_logger::init();

    let event_loop = winit::event_loop::EventLoop::new();
    // let monitor_id = event_loop.primary_monitor();
    let window = winit::window::WindowBuilder::new()
        .with_title("H.U.R.B.A.N. Selector")
        // .with_fullscreen(Some(monitor_id))
        .build(&event_loop)
        .expect("Failed to create window");

    let msaa = env::var("RTY_MSAA")
        .ok()
        .map(|msaa| match msaa.as_str() {
            "0" => Msaa::Disabled,
            "4" => Msaa::X4,
            "8" => Msaa::X8,
            "16" => Msaa::X16,
            unsupported_msaa => panic!("Unsupported MSAA value requested: {}", unsupported_msaa),
        })
        .unwrap_or(Msaa::Disabled);

    let present_mode = env::var("RTY_VSYNC")
        .ok()
        .map(|vsync| match vsync.as_str() {
            "1" => PresentMode::Vsync,
            "0" => PresentMode::NoVsync,
            unsupported_vsync => panic!(
                "Unsupported vsync behavior requested: {}",
                unsupported_vsync,
            ),
        })
        .unwrap_or(PresentMode::Vsync);

    let gpu_backend = env::var("RTY_GPU_BACKEND")
        .ok()
        .map(|backend| match backend.as_str() {
            "vulkan" => GpuBackend::Vulkan,
            "d3d12" => GpuBackend::D3d12,
            "metal" => GpuBackend::Metal,
            _ => panic!("Unknown gpu backend requested: {}", backend),
        });

    let time_start = Instant::now();

    let mut input_manager = InputManager::new();
    let mut app = App::new(
        time_start,
        window,
        AppOptions {
            msaa,
            present_mode,
            gpu_backend,
        },
    );

    // Since input manager needs to process events separately after imgui
    // handles them, this buffer with copies of events is needed.
    let mut input_events: Vec<winit::event::Event<_>> = Vec::new();
    let mut window_resized: Option<winit::dpi::LogicalSize> = None;

    event_loop.run(move |event, _, control_flow| {
        *control_flow = winit::event_loop::ControlFlow::Poll;

        let time = Instant::now();

        match event {
            winit::event::Event::EventsCleared => {
                if let Some(logical_size) = window_resized {
                    app.set_window_size(logical_size);
                }

                for event in &input_events {
                    app.update_ui(event);
                }

                input_manager.start_frame();
                for event in input_events.drain(..) {
                    input_manager.process_event(&event);
                }

                app.update_and_render(time, input_manager.input());

                if app.wants_close() {
                    *control_flow = winit::event_loop::ControlFlow::Exit;
                }
            }

            winit::event::Event::WindowEvent {
                event: winit::event::WindowEvent::RedrawRequested,
                ..
            } => {
                // V-SYNC makes it challenging to answer redraw
                // requests. Instead we do rendering after processing
                // piled up events.
            }

            winit::event::Event::WindowEvent {
                event: winit::event::WindowEvent::Resized(logical_size),
                ..
            } => {
                // Even if the window resized multiple times, only
                // take the last one into account.
                window_resized = Some(logical_size);
            }

            winit::event::Event::WindowEvent { .. } => {
                input_events.push(event.clone());
            }

            _ => (),
        }
    });
}
