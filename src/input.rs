use std::cmp::Ordering;

use wgpu::winit;

#[derive(Debug, Clone, Copy, PartialEq, Default)]
pub struct InputState {
    pub camera_pan_ground: [f32; 2],
    pub camera_pan_screen: [f32; 2],
    pub camera_rotate: [f32; 2],
    pub camera_zoom: f32,
    pub camera_zoom_steps: i32,
    pub camera_reset_viewport: bool,
    pub close_requested: bool,
    pub import_requested: bool,
    pub window_resized: Option<winit::dpi::LogicalSize>,
}

#[derive(Debug, Default)]
pub struct InputManager {
    lmb_down: bool,
    rmb_down: bool,
    shift_down: bool,
    input_state: InputState,
    window_mouse_x: f64,
    window_mouse_y: f64,
}

impl InputManager {
    pub fn new() -> Self {
        Self {
            lmb_down: false,
            rmb_down: false,
            shift_down: false,
            input_state: InputState::default(),
            window_mouse_x: 0.0,
            window_mouse_y: 0.0,
        }
    }

    pub fn input_state(&self) -> &InputState {
        &self.input_state
    }

    pub fn start_frame(&mut self) {
        self.input_state = InputState::default();
    }

    pub fn process_event(
        &mut self,
        ev: winit::Event,
        ui_captured_keyboard: bool,
        ui_captured_mouse: bool,
    ) {
        const MODIFIERS_NONE: winit::ModifiersState = winit::ModifiersState {
            logo: false,
            shift: false,
            ctrl: false,
            alt: false,
        };

        if let winit::Event::WindowEvent { event, .. } = ev {
            match event {
                winit::WindowEvent::CloseRequested => {
                    self.input_state.close_requested = true;
                }

                winit::WindowEvent::KeyboardInput { input, .. } => {
                    let winit::KeyboardInput {
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
                            Some(winit::VirtualKeyCode::Q),
                            winit::ElementState::Pressed,
                            winit::ModifiersState {
                                logo: true,
                                shift: false,
                                ctrl: false,
                                ..
                            },
                        ) => {
                            self.input_state.close_requested = true;
                        }
                        (Some(winit::VirtualKeyCode::LShift), winit::ElementState::Pressed, _) => {
                            self.shift_down = true;
                        }
                        (Some(winit::VirtualKeyCode::LShift), winit::ElementState::Released, _) => {
                            self.shift_down = false;
                        }
                        (Some(winit::VirtualKeyCode::RShift), winit::ElementState::Pressed, _) => {
                            self.shift_down = true;
                        }
                        (Some(winit::VirtualKeyCode::RShift), winit::ElementState::Released, _) => {
                            self.shift_down = false;
                        }
                        _ => (),
                    };

                    // These events are responded to only when gui doesn't have focus
                    if !ui_captured_keyboard {
                        match (virtual_keycode, state, modifiers) {
                            (
                                Some(winit::VirtualKeyCode::A),
                                winit::ElementState::Pressed,
                                MODIFIERS_NONE,
                            ) => {
                                self.input_state.camera_reset_viewport = true;
                            }
                            (
                                Some(winit::VirtualKeyCode::O),
                                winit::ElementState::Pressed,
                                MODIFIERS_NONE,
                            ) => {
                                self.input_state.import_requested = true;
                            }
                            _ => (),
                        }
                    }
                }

                winit::WindowEvent::MouseInput { state, button, .. } => match (state, button) {
                    (winit::ElementState::Pressed, winit::MouseButton::Left) => {
                        self.lmb_down = true;
                    }
                    (winit::ElementState::Released, winit::MouseButton::Left) => {
                        self.lmb_down = false;
                    }
                    (winit::ElementState::Pressed, winit::MouseButton::Right) => {
                        self.rmb_down = true;
                    }
                    (winit::ElementState::Released, winit::MouseButton::Right) => {
                        self.rmb_down = false;
                    }
                    (_, _) => (),
                },

                winit::WindowEvent::CursorMoved { position, .. } => {
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
                            self.input_state.camera_zoom -= dy;
                        } else if self.lmb_down {
                            self.input_state.camera_rotate[0] -= dx;
                            self.input_state.camera_rotate[1] -= dy;
                        } else if self.rmb_down {
                            if self.shift_down {
                                self.input_state.camera_pan_ground[0] += dx;
                                self.input_state.camera_pan_ground[1] -= dy;
                            } else {
                                self.input_state.camera_pan_screen[0] += dx;
                                self.input_state.camera_pan_screen[1] -= dy;
                            }
                        }
                    }
                }

                winit::WindowEvent::MouseWheel { delta, .. } => match delta {
                    winit::MouseScrollDelta::PixelDelta(winit::dpi::LogicalPosition {
                        y, ..
                    }) => {
                        if !ui_captured_mouse {
                            match y.partial_cmp(&0.0) {
                                Some(Ordering::Greater) => self.input_state.camera_zoom_steps += 1,
                                Some(Ordering::Less) => self.input_state.camera_zoom_steps -= 1,
                                _ => (),
                            }
                        }
                    }

                    winit::MouseScrollDelta::LineDelta(_, y) => {
                        if !ui_captured_mouse {
                            match y.partial_cmp(&0.0) {
                                Some(Ordering::Greater) => self.input_state.camera_zoom_steps += 1,
                                Some(Ordering::Less) => self.input_state.camera_zoom_steps -= 1,
                                _ => (),
                            }
                        }
                    }
                },

                winit::WindowEvent::Resized(logical_size) => {
                    // Even if the window resized multiple times, only
                    // take the last one into account.
                    self.input_state.window_resized = Some(logical_size);
                }

                _ => (),
            }
        }
    }
}
