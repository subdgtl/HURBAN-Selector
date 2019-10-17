use std::cmp::Ordering;

#[derive(Debug, Clone, Copy, PartialEq, Default)]
pub struct InputState {
    pub tmp_submit_prog_and_run: bool,
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
                    self.input_state.close_requested = true;
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
                            self.input_state.close_requested = true;
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
                                self.input_state.camera_reset_viewport = true;
                            }
                            (
                                Some(winit::event::VirtualKeyCode::O),
                                winit::event::ElementState::Pressed,
                                &MODIFIERS_NONE,
                            ) => {
                                self.input_state.import_requested = true;
                            }
                            (
                                Some(winit::event::VirtualKeyCode::R),
                                winit::event::ElementState::Pressed,
                                &MODIFIERS_NONE,
                            ) => {
                                self.input_state.tmp_submit_prog_and_run = true;
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

                winit::event::WindowEvent::MouseWheel { delta, .. } => match delta {
                    winit::event::MouseScrollDelta::PixelDelta(winit::dpi::LogicalPosition {
                        y,
                        ..
                    }) => {
                        if !ui_captured_mouse {
                            match y.partial_cmp(&0.0) {
                                Some(Ordering::Greater) => self.input_state.camera_zoom_steps += 1,
                                Some(Ordering::Less) => self.input_state.camera_zoom_steps -= 1,
                                _ => (),
                            }
                        }
                    }

                    winit::event::MouseScrollDelta::LineDelta(_, y) => {
                        if !ui_captured_mouse {
                            match y.partial_cmp(&0.0) {
                                Some(Ordering::Greater) => self.input_state.camera_zoom_steps += 1,
                                Some(Ordering::Less) => self.input_state.camera_zoom_steps -= 1,
                                _ => (),
                            }
                        }
                    }
                },

                winit::event::WindowEvent::Resized(logical_size) => {
                    // Even if the window resized multiple times, only
                    // take the last one into account.
                    self.input_state.window_resized = Some(*logical_size);
                }

                _ => (),
            }
        }
    }
}
