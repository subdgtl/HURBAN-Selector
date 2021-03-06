use std::cmp::Ordering;

#[derive(Debug, Clone, Copy, PartialEq, Default)]
pub struct InputState {
    pub camera_pan_ground: Option<([f32; 2], [f32; 2])>,
    pub camera_pan_screen: Option<([f32; 2], [f32; 2])>,
    pub camera_rotate: [f32; 2],
    pub camera_zoom: f32,
    pub camera_zoom_steps: i32,
    pub camera_reset_viewport: bool,
    #[cfg(not(feature = "dist"))]
    pub debug_view_cycle: bool,
    pub prog_run_requested: bool,
    pub prog_pop_requested: bool,
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
    mouse_x_frame_start: f64,
    mouse_y_frame_start: f64,
    mouse_x_frame_end: f64,
    mouse_y_frame_end: f64,
}

impl InputManager {
    pub fn new() -> Self {
        Self {
            lmb_down: false,
            rmb_down: false,
            modifiers: winit::event::ModifiersState::empty(),
            input_state: InputState::default(),
            mouse_x_frame_start: 0.0,
            mouse_y_frame_start: 0.0,
            mouse_x_frame_end: 0.0,
            mouse_y_frame_end: 0.0,
        }
    }

    pub fn input_state(&self) -> &InputState {
        &self.input_state
    }

    pub fn start_frame(&mut self) {
        self.input_state = InputState::default();

        self.mouse_x_frame_start = self.mouse_x_frame_end;
        self.mouse_y_frame_start = self.mouse_y_frame_end;
    }

    pub fn process_event<T>(
        &mut self,
        event: &winit::event::Event<T>,
        ui_captured_keyboard: bool,
        ui_captured_mouse: bool,
    ) {
        if let winit::event::Event::WindowEvent { event, .. } = event {
            match event {
                winit::event::WindowEvent::ModifiersChanged(modifiers) => {
                    self.modifiers = *modifiers;
                }

                winit::event::WindowEvent::CloseRequested => {
                    self.input_state.close_requested = true;
                }

                winit::event::WindowEvent::KeyboardInput { input, .. } => {
                    let winit::event::KeyboardInput {
                        virtual_keycode,
                        state,
                        ..
                    } = input;

                    // #justmacosthings
                    #[cfg(target_os = "macos")]
                    {
                        if self.modifiers == winit::event::ModifiersState::LOGO {
                            // Respond to Cmd+Q unconditionally, even if GUI has focus
                            if let (
                                Some(winit::event::VirtualKeyCode::Q),
                                winit::event::ElementState::Pressed,
                            ) = (virtual_keycode, state)
                            {
                                self.input_state.close_requested = true;
                            };

                            if !ui_captured_keyboard {
                                match (virtual_keycode, state) {
                                    // Cmd+Back is the same as Delete on some
                                    // macOS keyboards (at least the laptop
                                    // ones)
                                    (
                                        Some(winit::event::VirtualKeyCode::Back),
                                        winit::event::ElementState::Pressed,
                                    ) => {
                                        self.input_state.prog_pop_requested = true;
                                    }
                                    (_, _) => (),
                                }
                            }
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
                            (
                                Some(winit::event::VirtualKeyCode::Return),
                                winit::event::ElementState::Pressed,
                            ) => {
                                self.input_state.prog_run_requested = true;
                            }
                            (
                                Some(winit::event::VirtualKeyCode::NumpadEnter),
                                winit::event::ElementState::Pressed,
                            ) => {
                                self.input_state.prog_run_requested = true;
                            }
                            (
                                Some(winit::event::VirtualKeyCode::Delete),
                                winit::event::ElementState::Pressed,
                            ) => {
                                self.input_state.prog_pop_requested = true;
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
                    self.mouse_x_frame_end = position.x;
                    self.mouse_y_frame_end = position.y;

                    let dx = (self.mouse_x_frame_end - self.mouse_x_frame_start) as f32;
                    let dy = (self.mouse_y_frame_end - self.mouse_y_frame_start) as f32;

                    if !ui_captured_mouse {
                        if self.lmb_down && self.rmb_down {
                            self.input_state.camera_zoom = dy;
                        } else if self.lmb_down {
                            self.input_state.camera_rotate[0] = dx;
                            self.input_state.camera_rotate[1] = dy;
                        } else if self.rmb_down {
                            if self.modifiers.shift() {
                                self.input_state.camera_pan_ground = Some((
                                    [
                                        self.mouse_x_frame_start as f32,
                                        self.mouse_y_frame_start as f32,
                                    ],
                                    [self.mouse_x_frame_end as f32, self.mouse_y_frame_end as f32],
                                ));
                            } else {
                                self.input_state.camera_pan_screen = Some((
                                    [
                                        self.mouse_x_frame_start as f32,
                                        self.mouse_y_frame_start as f32,
                                    ],
                                    [self.mouse_x_frame_end as f32, self.mouse_y_frame_end as f32],
                                ));
                            }
                        }
                    }
                }

                winit::event::WindowEvent::MouseWheel { delta, .. } => match delta {
                    winit::event::MouseScrollDelta::PixelDelta(winit::dpi::PhysicalPosition {
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
            }
        }
    }
}
