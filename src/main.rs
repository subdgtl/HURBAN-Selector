#![windows_subsystem = "windows"]

use std::cmp::Ordering;
use std::env;
use std::time::Instant;

use hurban_selector::app::{App, AppOptions, GpuBackend, Input, Msaa, PresentMode};

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

                let ui_captured_keyboard = app.ui_captured_keyboard();
                let ui_captured_mouse = app.ui_captured_mouse();
                input_manager.start_frame();
                for event in input_events.drain(..) {
                    input_manager.process_event(&event, ui_captured_keyboard, ui_captured_mouse);
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

#[derive(Debug, Default)]
struct InputManager {
    lmb_down: bool,
    rmb_down: bool,
    shift_down: bool,
    input: Input,
    window_mouse_x: f64,
    window_mouse_y: f64,
}

impl InputManager {
    pub fn new() -> Self {
        Self {
            lmb_down: false,
            rmb_down: false,
            shift_down: false,
            input: Input::default(),
            window_mouse_x: 0.0,
            window_mouse_y: 0.0,
        }
    }

    pub fn input(&self) -> &Input {
        &self.input
    }

    pub fn start_frame(&mut self) {
        self.input = Input::default();
    }

    pub fn process_event<T>(
        &mut self,
        event: &winit::event::Event<T>,
        ui_captured_keyboard: bool,
        ui_captured_mouse: bool,
    ) {
        const MODIFIERS_NONE: winit::event::ModifiersState = winit::event::ModifiersState {
            logo: false,
            shift: false,
            ctrl: false,
            alt: false,
        };

        if let winit::event::Event::WindowEvent { event, .. } = event {
            match event {
                winit::event::WindowEvent::CloseRequested => {
                    self.input.close_requested = true;
                }

                winit::event::WindowEvent::KeyboardInput { input, .. } => {
                    let winit::event::KeyboardInput {
                        virtual_keycode,
                        state,
                        modifiers,
                        ..
                    } = input;

                    // We respond to some events unconditionally, even if GUI has focus.
                    match (virtual_keycode, state, modifiers) {
                        // Cmd+Q for macOS
                        #[cfg(target_os = "macos")]
                        (
                            Some(winit::event::VirtualKeyCode::Q),
                            winit::event::ElementState::Pressed,
                            winit::event::ModifiersState {
                                logo: true,
                                shift: false,
                                ctrl: false,
                                ..
                            },
                        ) => {
                            self.input.close_requested = true;
                        }
                        (
                            Some(winit::event::VirtualKeyCode::LShift),
                            winit::event::ElementState::Pressed,
                            _,
                        ) => {
                            self.shift_down = true;
                        }
                        (
                            Some(winit::event::VirtualKeyCode::LShift),
                            winit::event::ElementState::Released,
                            _,
                        ) => {
                            self.shift_down = false;
                        }
                        (
                            Some(winit::event::VirtualKeyCode::RShift),
                            winit::event::ElementState::Pressed,
                            _,
                        ) => {
                            self.shift_down = true;
                        }
                        (
                            Some(winit::event::VirtualKeyCode::RShift),
                            winit::event::ElementState::Released,
                            _,
                        ) => {
                            self.shift_down = false;
                        }
                        _ => (),
                    };

                    // These events are responded to only when gui doesn't have focus
                    if !ui_captured_keyboard {
                        match (virtual_keycode, state, modifiers) {
                            (
                                Some(winit::event::VirtualKeyCode::A),
                                winit::event::ElementState::Pressed,
                                &MODIFIERS_NONE,
                            ) => {
                                self.input.camera_reset_viewport = true;
                            }
                            (
                                Some(winit::event::VirtualKeyCode::O),
                                winit::event::ElementState::Pressed,
                                &MODIFIERS_NONE,
                            ) => {
                                self.input.import_requested = true;
                            }
                            _ => (),
                        }
                    }
                }

                winit::event::WindowEvent::MouseInput { state, button, .. } => {
                    match (state, button) {
                        (winit::event::ElementState::Pressed, winit::event::MouseButton::Left) => {
                            self.lmb_down = true;
                        }
                        (winit::event::ElementState::Released, winit::event::MouseButton::Left) => {
                            self.lmb_down = false;
                        }
                        (winit::event::ElementState::Pressed, winit::event::MouseButton::Right) => {
                            self.rmb_down = true;
                        }
                        (
                            winit::event::ElementState::Released,
                            winit::event::MouseButton::Right,
                        ) => {
                            self.rmb_down = false;
                        }
                        (_, _) => (),
                    }
                }

                winit::event::WindowEvent::CursorMoved { position, .. } => {
                    if !ui_captured_mouse {
                        let x = position.x;
                        let y = position.y;
                        let x_prev = self.window_mouse_x;
                        let y_prev = self.window_mouse_y;
                        self.window_mouse_x = x;
                        self.window_mouse_y = y;

                        let dx = (x - x_prev) as f32;
                        let dy = (y - y_prev) as f32;

                        if self.lmb_down && self.rmb_down {
                            self.input.camera_zoom -= dy;
                        } else if self.lmb_down {
                            self.input.camera_rotate[0] -= dx;
                            self.input.camera_rotate[1] -= dy;
                        } else if self.rmb_down {
                            if self.shift_down {
                                self.input.camera_pan_ground[0] += dx;
                                self.input.camera_pan_ground[1] -= dy;
                            } else {
                                self.input.camera_pan_screen[0] += dx;
                                self.input.camera_pan_screen[1] -= dy;
                            }
                        }
                    }
                }

                winit::event::WindowEvent::MouseWheel { delta, .. } => match delta {
                    winit::event::MouseScrollDelta::PixelDelta(winit::dpi::LogicalPosition {
                        y,
                        ..
                    }) => {
                        if !ui_captured_mouse {
                            match y.partial_cmp(&0.0) {
                                Some(Ordering::Greater) => self.input.camera_zoom_steps += 1,
                                Some(Ordering::Less) => self.input.camera_zoom_steps -= 1,
                                _ => (),
                            }
                        }
                    }

                    winit::event::MouseScrollDelta::LineDelta(_, y) => {
                        if !ui_captured_mouse {
                            match y.partial_cmp(&0.0) {
                                Some(Ordering::Greater) => self.input.camera_zoom_steps += 1,
                                Some(Ordering::Less) => self.input.camera_zoom_steps -= 1,
                                _ => (),
                            }
                        }
                    }
                },
                _ => (),
            }
        }
    }
}
