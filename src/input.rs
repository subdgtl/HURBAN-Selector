use std::cmp::Ordering;

#[derive(Debug, Clone, Copy, PartialEq, Default)]
pub struct InputState {
    pub camera_pan_ground: [f32; 2],
    pub camera_pan_screen: [f32; 2],
    pub camera_rotate: [f32; 2],
    pub camera_zoom: f32,
    pub camera_zoom_steps: i32,
    pub camera_reset_viewport: bool,
    #[cfg(not(feature = "dist"))]
    pub debug_view_cycle: bool,
    pub close_requested: bool,
    pub open_screenshot_options: bool,
    pub window_resized: Option<winit::dpi::PhysicalSize<u32>>,
}

#[derive(Debug, Default)]
pub struct InputManager {
    lmb_down: bool,
    rmb_down: bool,
    modifiers: winit::event::ModifiersState,
    input_state: InputState,
    window_mouse_x: f64,
    window_mouse_y: f64,
}

impl InputManager {
    pub fn new() -> Self {
        Self {
            lmb_down: false,
            rmb_down: false,
            modifiers: winit::event::ModifiersState::empty(),
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
        match event {
            winit::event::Event::DeviceEvent { event, .. } => {
                if let winit::event::DeviceEvent::ModifiersChanged(modifiers) = event {
                    self.modifiers = *modifiers;
                }
            }

            winit::event::Event::WindowEvent { event, .. } => match event {
                winit::event::WindowEvent::CloseRequested => {
                    self.input_state.close_requested = true;
                }

                winit::event::WindowEvent::KeyboardInput { input, .. } => {
                    let winit::event::KeyboardInput {
                        virtual_keycode,
                        state,
                        ..
                    } = input;

                    // Respond to Cmd+Q unconditionally, even if GUI has focus
                    #[cfg(target_os = "macos")]
                    {
                        if self.modifiers == winit::event::ModifiersState::LOGO {
                            if let (
                                Some(winit::event::VirtualKeyCode::Q),
                                winit::event::ElementState::Pressed,
                            ) = (virtual_keycode, state)
                            {
                                self.input_state.close_requested = true;
                            };
                        }
                    }

                    // These events are responded to only when gui doesn't have
                    // focus and there are no active modifiers (we currently
                    // have no keyboard shortcuts with modifiers)
                    if !ui_captured_keyboard
                        && self.modifiers == winit::event::ModifiersState::empty()
                    {
                        match (virtual_keycode, state) {
                            (
                                Some(winit::event::VirtualKeyCode::A),
                                winit::event::ElementState::Pressed,
                            ) => {
                                self.input_state.camera_reset_viewport = true;
                            }
                            #[cfg(not(feature = "dist"))]
                            (
                                Some(winit::event::VirtualKeyCode::D),
                                winit::event::ElementState::Pressed,
                            ) => {
                                self.input_state.debug_view_cycle = true;
                            }
                            (
                                Some(winit::event::VirtualKeyCode::P),
                                winit::event::ElementState::Pressed,
                            ) => {
                                self.input_state.open_screenshot_options = true;
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
                            if self.modifiers.shift() {
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

                winit::event::WindowEvent::Resized(physical_size) => {
                    // Even if the window resized multiple times, only
                    // take the last one into account.
                    self.input_state.window_resized = Some(*physical_size);
                }

                _ => (),
            },

            _ => (),
        }
    }
}
