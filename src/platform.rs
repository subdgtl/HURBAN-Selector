use std::cmp::Ordering;

use crate::app::Input;

#[derive(Debug, Default)]
pub struct InputManager {
    meta_down: bool,
    shift_down: bool,
    ctrl_down: bool,
    alt_down: bool,
    lmb_down: bool,
    rmb_down: bool,
    mouse_x: f64,
    mouse_y: f64,
    input: Input,
}

impl InputManager {
    pub fn new() -> Self {
        Self {
            meta_down: false,
            shift_down: false,
            ctrl_down: false,
            alt_down: false,
            lmb_down: false,
            rmb_down: false,
            mouse_x: 0.0,
            mouse_y: 0.0,
            input: Input::default(),
        }
    }

    pub fn input(&self) -> &Input {
        &self.input
    }

    pub fn start_frame(&mut self) {
        self.input = Input::default();

        self.input.meta_down = self.meta_down;
        self.input.shift_down = self.shift_down;
        self.input.ctrl_down = self.ctrl_down;
        self.input.alt_down = self.alt_down;

        self.input.lmb_down = self.lmb_down;
        self.input.rmb_down = self.rmb_down;
    }

    pub fn process_event<T>(&mut self, event: &winit::event::Event<T>) {
        if let winit::event::Event::WindowEvent { event, .. } = event {
            match event {
                winit::event::WindowEvent::CloseRequested => {
                    self.input.close_requested = true;
                }

                winit::event::WindowEvent::KeyboardInput { input, .. } => {
                    let winit::event::KeyboardInput {
                        virtual_keycode,
                        state,
                        ..
                    } = input;

                    match (virtual_keycode, state) {
                        (
                            Some(winit::event::VirtualKeyCode::LWin),
                            winit::event::ElementState::Pressed,
                        ) => {
                            self.meta_down = true;
                        }
                        (
                            Some(winit::event::VirtualKeyCode::LWin),
                            winit::event::ElementState::Released,
                        ) => {
                            self.meta_down = false;
                        }
                        (
                            Some(winit::event::VirtualKeyCode::RWin),
                            winit::event::ElementState::Pressed,
                        ) => {
                            self.meta_down = true;
                        }
                        (
                            Some(winit::event::VirtualKeyCode::RWin),
                            winit::event::ElementState::Released,
                        ) => {
                            self.meta_down = false;
                        }

                        (
                            Some(winit::event::VirtualKeyCode::LShift),
                            winit::event::ElementState::Pressed,
                        ) => {
                            self.shift_down = true;
                        }
                        (
                            Some(winit::event::VirtualKeyCode::LShift),
                            winit::event::ElementState::Released,
                        ) => {
                            self.shift_down = false;
                        }
                        (
                            Some(winit::event::VirtualKeyCode::RShift),
                            winit::event::ElementState::Pressed,
                        ) => {
                            self.shift_down = true;
                        }
                        (
                            Some(winit::event::VirtualKeyCode::RShift),
                            winit::event::ElementState::Released,
                        ) => {
                            self.shift_down = false;
                        }

                        (
                            Some(winit::event::VirtualKeyCode::LControl),
                            winit::event::ElementState::Pressed,
                        ) => {
                            self.ctrl_down = true;
                        }
                        (
                            Some(winit::event::VirtualKeyCode::LControl),
                            winit::event::ElementState::Released,
                        ) => {
                            self.ctrl_down = false;
                        }
                        (
                            Some(winit::event::VirtualKeyCode::RControl),
                            winit::event::ElementState::Pressed,
                        ) => {
                            self.ctrl_down = true;
                        }
                        (
                            Some(winit::event::VirtualKeyCode::RControl),
                            winit::event::ElementState::Released,
                        ) => {
                            self.ctrl_down = false;
                        }

                        (
                            Some(winit::event::VirtualKeyCode::LAlt),
                            winit::event::ElementState::Pressed,
                        ) => {
                            self.alt_down = true;
                        }
                        (
                            Some(winit::event::VirtualKeyCode::LAlt),
                            winit::event::ElementState::Released,
                        ) => {
                            self.alt_down = false;
                        }
                        (
                            Some(winit::event::VirtualKeyCode::RAlt),
                            winit::event::ElementState::Pressed,
                        ) => {
                            self.alt_down = true;
                        }
                        (
                            Some(winit::event::VirtualKeyCode::RAlt),
                            winit::event::ElementState::Released,
                        ) => {
                            self.alt_down = false;
                        }

                        (
                            Some(winit::event::VirtualKeyCode::A),
                            winit::event::ElementState::Pressed,
                        ) => {
                            self.input.key_a_pressed = true;
                        }
                        (
                            Some(winit::event::VirtualKeyCode::O),
                            winit::event::ElementState::Pressed,
                        ) => {
                            self.input.key_o_pressed = true;
                        }
                        (
                            Some(winit::event::VirtualKeyCode::Q),
                            winit::event::ElementState::Pressed,
                        ) => {
                            self.input.key_q_pressed = true;
                        }
                        _ => (),
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
                        _ => (),
                    }
                }

                winit::event::WindowEvent::CursorMoved { position, .. } => {
                    let x = position.x;
                    let y = position.y;
                    let x_prev = self.mouse_x;
                    let y_prev = self.mouse_y;
                    self.mouse_x = x;
                    self.mouse_y = y;

                    let dx = x - x_prev;
                    let dy = y - y_prev;

                    self.input.mouse_move[0] += dx;
                    self.input.mouse_move[1] += dy;
                }

                winit::event::WindowEvent::MouseWheel { delta, .. } => match delta {
                    winit::event::MouseScrollDelta::PixelDelta(winit::dpi::LogicalPosition {
                        y,
                        ..
                    }) => match y.partial_cmp(&0.0) {
                        Some(Ordering::Greater) => self.input.mouse_wheel += 1,
                        Some(Ordering::Less) => self.input.mouse_wheel -= 1,
                        _ => (),
                    },

                    winit::event::MouseScrollDelta::LineDelta(_, y) => match y.partial_cmp(&0.0) {
                        Some(Ordering::Greater) => self.input.mouse_wheel += 1,
                        Some(Ordering::Less) => self.input.mouse_wheel -= 1,
                        _ => (),
                    },
                },
                _ => (),
            }
        }

        self.input.meta_down = self.meta_down;
        self.input.shift_down = self.shift_down;
        self.input.ctrl_down = self.ctrl_down;
        self.input.alt_down = self.alt_down;

        self.input.lmb_down = self.lmb_down;
        self.input.rmb_down = self.rmb_down;
    }
}
