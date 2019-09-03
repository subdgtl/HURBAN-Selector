use std::cmp::Ordering;
use std::time::{Duration, Instant};

use imgui::{self, BackendFlags, ConfigFlags, Context, ImString, Io, Key, Ui};

use wgpu::winit::dpi::{LogicalPosition, LogicalSize};
use wgpu::winit::{
    DeviceEvent, ElementState, Event, KeyboardInput, MouseButton, MouseCursor, MouseScrollDelta,
    TouchPhase, VirtualKeyCode, Window, WindowEvent,
};

const MOUSE_DISABLE_DELAY: Duration = Duration::from_millis(100);

/// winit backend platform state
#[derive(Debug)]
pub struct WinitPlatform {
    hidpi_mode: ActiveHiDpiMode,
    hidpi_factor: f64,
    touch_pos: Option<(f64, f64)>,
    touch_last: Option<Instant>,
}

#[derive(Copy, Clone, Debug, PartialEq)]
enum ActiveHiDpiMode {
    Default,
    Rounded,
    Locked,
}

/// DPI factor handling mode.
///
/// Applications that use imgui-rs might want to customize the used DPI factor and not use
/// directly the value coming from winit.
///
/// **Note: if you use a mode other than default and the DPI factor is adjusted, winit and imgui-rs
/// will use different logical coordinates, so be careful if you pass around logical size or
/// position values.**
#[derive(Copy, Clone, Debug, PartialEq)]
pub enum HiDpiMode {
    /// The DPI factor from winit is used directly without adjustment
    Default,
    /// The DPI factor from winit is rounded to an integer value.
    ///
    /// This prevents the user interface from becoming blurry with non-integer scaling.
    Rounded,
    /// The DPI factor from winit is ignored, and the included value is used instead.
    ///
    /// This is useful if you want to force some DPI factor (e.g. 1.0) and not care about the value
    /// coming from winit.
    Locked(f64),
}

impl HiDpiMode {
    fn apply(&self, hidpi_factor: f64) -> (ActiveHiDpiMode, f64) {
        match *self {
            HiDpiMode::Default => (ActiveHiDpiMode::Default, hidpi_factor),
            HiDpiMode::Rounded => (ActiveHiDpiMode::Rounded, hidpi_factor.round()),
            HiDpiMode::Locked(value) => (ActiveHiDpiMode::Locked, value),
        }
    }
}

impl WinitPlatform {
    /// Initializes a winit platform instance and configures imgui.
    ///
    /// This function configures imgui-rs in the following ways:
    ///
    /// * backend flags are updated
    /// * keys are configured
    /// * platform name is set
    pub fn init(imgui: &mut Context) -> WinitPlatform {
        let io = imgui.io_mut();
        io.backend_flags.insert(BackendFlags::HAS_MOUSE_CURSORS);
        io.backend_flags.insert(BackendFlags::HAS_SET_MOUSE_POS);
        io[Key::Tab] = VirtualKeyCode::Tab as _;
        io[Key::LeftArrow] = VirtualKeyCode::Left as _;
        io[Key::RightArrow] = VirtualKeyCode::Right as _;
        io[Key::UpArrow] = VirtualKeyCode::Up as _;
        io[Key::DownArrow] = VirtualKeyCode::Down as _;
        io[Key::PageUp] = VirtualKeyCode::PageUp as _;
        io[Key::PageDown] = VirtualKeyCode::PageDown as _;
        io[Key::Home] = VirtualKeyCode::Home as _;
        io[Key::End] = VirtualKeyCode::End as _;
        io[Key::Insert] = VirtualKeyCode::Insert as _;
        io[Key::Delete] = VirtualKeyCode::Delete as _;
        io[Key::Backspace] = VirtualKeyCode::Back as _;
        io[Key::Space] = VirtualKeyCode::Space as _;
        io[Key::Enter] = VirtualKeyCode::Return as _;
        io[Key::Escape] = VirtualKeyCode::Escape as _;
        io[Key::A] = VirtualKeyCode::A as _;
        io[Key::C] = VirtualKeyCode::C as _;
        io[Key::V] = VirtualKeyCode::V as _;
        io[Key::X] = VirtualKeyCode::X as _;
        io[Key::Y] = VirtualKeyCode::Y as _;
        io[Key::Z] = VirtualKeyCode::Z as _;
        imgui.set_platform_name(Some(ImString::from(format!(
            "imgui-winit-support {}",
            env!("CARGO_PKG_VERSION")
        ))));
        WinitPlatform {
            hidpi_mode: ActiveHiDpiMode::Default,
            hidpi_factor: 1.0,
            touch_pos: None,
            touch_last: None,
        }
    }
    /// Attaches the platform instance to a winit window.
    ///
    /// This function configures imgui-rs in the following ways:
    ///
    /// * framebuffer scale (= DPI factor) is set
    /// * display size is set
    pub fn attach_window(&mut self, io: &mut Io, window: &Window, hidpi_mode: HiDpiMode) {
        let (hidpi_mode, hidpi_factor) = hidpi_mode.apply(window.get_hidpi_factor());
        self.hidpi_mode = hidpi_mode;
        self.hidpi_factor = hidpi_factor;
        io.display_framebuffer_scale = [hidpi_factor as f32, hidpi_factor as f32];
        if let Some(logical_size) = window.get_inner_size() {
            let logical_size = self.scale_size_from_winit(window, logical_size);
            io.display_size = [logical_size.width as f32, logical_size.height as f32];
        }
    }
    /// Returns the current DPI factor.
    ///
    /// The value might not be the same as the winit DPI factor (depends on the used DPI mode)
    pub fn hidpi_factor(&self) -> f64 {
        self.hidpi_factor
    }
    /// Scales a logical size coming from winit using the current DPI mode.
    ///
    /// This utility function is useful if you are using a DPI mode other than default, and want
    /// your application to use the same logical coordinates as imgui-rs.
    pub fn scale_size_from_winit(&self, window: &Window, logical_size: LogicalSize) -> LogicalSize {
        match self.hidpi_mode {
            ActiveHiDpiMode::Default => logical_size,
            _ => logical_size
                .to_physical(window.get_hidpi_factor())
                .to_logical(self.hidpi_factor),
        }
    }
    /// Scales a logical position coming from winit using the current DPI mode.
    ///
    /// This utility function is useful if you are using a DPI mode other than default, and want
    /// your application to use the same logical coordinates as imgui-rs.
    pub fn scale_pos_from_winit(
        &self,
        window: &Window,
        logical_pos: LogicalPosition,
    ) -> LogicalPosition {
        match self.hidpi_mode {
            ActiveHiDpiMode::Default => logical_pos,
            _ => logical_pos
                .to_physical(window.get_hidpi_factor())
                .to_logical(self.hidpi_factor),
        }
    }
    /// Scales a logical position for winit using the current DPI mode.
    ///
    /// This utility function is useful if you are using a DPI mode other than default, and want
    /// your application to use the same logical coordinates as imgui-rs.
    pub fn scale_pos_for_winit(
        &self,
        window: &Window,
        logical_pos: LogicalPosition,
    ) -> LogicalPosition {
        match self.hidpi_mode {
            ActiveHiDpiMode::Default => logical_pos,
            _ => logical_pos
                .to_physical(self.hidpi_factor)
                .to_logical(window.get_hidpi_factor()),
        }
    }
    /// Handles a winit event.
    ///
    /// This function performs the following actions (depends on the event):
    ///
    /// * window size / dpi factor changes are applied
    /// * keyboard state is updated
    /// * mouse state is updated
    pub fn handle_event(&mut self, io: &mut Io, window: &Window, event: &Event, time: Instant) {

        if let Some(touch_last) = self.touch_last {
            if time.duration_since(touch_last) > MOUSE_DISABLE_DELAY {
                self.touch_last = None;
            }
        }


        match *event {
            Event::WindowEvent {
                window_id,
                ref event,
            } if window_id == window.id() => {
                self.handle_window_event(io, window, event, time);
            }
            // Track key release events outside our window. If we don't do this,
            // we might never see the release event if some other window gets focus.
            Event::DeviceEvent {
                event:
                    DeviceEvent::Key(KeyboardInput {
                        state: ElementState::Released,
                        virtual_keycode: Some(key),
                        ..
                    }),
                ..
            } => {
                io.keys_down[key as usize] = false;
                match key {
                    VirtualKeyCode::LShift | VirtualKeyCode::RShift => io.key_shift = false,
                    VirtualKeyCode::LControl | VirtualKeyCode::RControl => io.key_ctrl = false,
                    VirtualKeyCode::LAlt | VirtualKeyCode::RAlt => io.key_alt = false,
                    VirtualKeyCode::LWin | VirtualKeyCode::RWin => io.key_super = false,
                    _ => (),
                }
            }
            _ => (),
        }
    }
    fn handle_window_event(&mut self, io: &mut Io, window: &Window, event: &WindowEvent, time: Instant) {
        match *event {
            WindowEvent::Resized(logical_size) => {
                let logical_size = self.scale_size_from_winit(window, logical_size);
                io.display_size = [logical_size.width as f32, logical_size.height as f32];
            }
            WindowEvent::HiDpiFactorChanged(scale) => {
                let hidpi_factor = match self.hidpi_mode {
                    ActiveHiDpiMode::Default => scale,
                    ActiveHiDpiMode::Rounded => scale.round(),
                    _ => return,
                };
                // Mouse position needs to be changed while we still have both the old and the new
                // values
                if io.mouse_pos[0].is_finite() && io.mouse_pos[1].is_finite() {
                    io.mouse_pos = [
                        io.mouse_pos[0] * (hidpi_factor / self.hidpi_factor) as f32,
                        io.mouse_pos[1] * (hidpi_factor / self.hidpi_factor) as f32,
                    ];
                }
                self.hidpi_factor = hidpi_factor;
                io.display_framebuffer_scale = [hidpi_factor as f32, hidpi_factor as f32];
                // Window size might change too if we are using DPI rounding
                if let Some(logical_size) = window.get_inner_size() {
                    let logical_size = self.scale_size_from_winit(window, logical_size);
                    io.display_size = [logical_size.width as f32, logical_size.height as f32];
                }
            }
            WindowEvent::KeyboardInput {
                input:
                    KeyboardInput {
                        virtual_keycode: Some(key),
                        state,
                        ..
                    },
                ..
            } => {
                let pressed = state == ElementState::Pressed;
                io.keys_down[key as usize] = pressed;
                match key {
                    VirtualKeyCode::LShift | VirtualKeyCode::RShift => io.key_shift = pressed,
                    VirtualKeyCode::LControl | VirtualKeyCode::RControl => io.key_ctrl = pressed,
                    VirtualKeyCode::LAlt | VirtualKeyCode::RAlt => io.key_alt = pressed,
                    VirtualKeyCode::LWin | VirtualKeyCode::RWin => io.key_super = pressed,
                    _ => (),
                }
            }
            WindowEvent::ReceivedCharacter(ch) => io.add_input_character(ch),
            WindowEvent::CursorMoved { position, .. } => {
                if self.touch_last.is_none() {
                    let position = self.scale_pos_from_winit(window, position);
                    println!("MOUSE POS: {:?}", position);
                    io.mouse_pos = [position.x as f32, position.y as f32];
                }
            }
            WindowEvent::MouseWheel {
                delta,
                phase: TouchPhase::Moved,
                ..
            } => match delta {
                MouseScrollDelta::LineDelta(h, v) => {
                    io.mouse_wheel_h = h;
                    io.mouse_wheel = v;
                }
                MouseScrollDelta::PixelDelta(pos) => {
                    match pos.x.partial_cmp(&0.0) {
                        Some(Ordering::Greater) => io.mouse_wheel_h += 1.0,
                        Some(Ordering::Less) => io.mouse_wheel_h -= 1.0,
                        _ => (),
                    }
                    match pos.y.partial_cmp(&0.0) {
                        Some(Ordering::Greater) => io.mouse_wheel += 1.0,
                        Some(Ordering::Less) => io.mouse_wheel -= 1.0,
                        _ => (),
                    }
                }
            },
            WindowEvent::MouseInput { state, button, .. } => {
                if self.touch_last.is_none() {
                    let pressed = state == ElementState::Pressed;

                    if pressed {
                        println!("MOUSE PRESSED {:?}", time);
                    } else {
                        println!("MOUSE RELEASED {:?}", time);
                    }

                    match button {
                        MouseButton::Left => io.mouse_down[0] = pressed,
                        MouseButton::Right => io.mouse_down[1] = pressed,
                        MouseButton::Middle => io.mouse_down[2] = pressed,
                        MouseButton::Other(idx @ 1...4) => io.mouse_down[idx as usize] = pressed,
                        _ => (),
                    }
                }
            }
            WindowEvent::Touch(touch) => {
                self.touch_last = Some(time);

                let monitor_physical_size = window.get_current_monitor().get_dimensions();
                let monitor_logical_size = monitor_physical_size.to_logical(self.hidpi_factor);

                if let Some(window_logical_position) = window.get_inner_position() {
                    if let Some(window_logical_size) = window.get_inner_size() {
                        let top_on_monitor = window_logical_position.y;
                        let left_on_monitor = window_logical_position.x;
                        let bottom_on_monitor = window_logical_position.y + window_logical_size.width;
                        let right_on_monitor = window_logical_position.x + window_logical_size.height;

                        let touch_position = self.scale_pos_from_winit(window, touch.location);

                        if touch_position.x >= left_on_monitor && touch_position.x < right_on_monitor
                            && touch_position.y >= top_on_monitor && touch_position.y < bottom_on_monitor {

                            println!("TOUCH POS: {:?} {:?}", touch_position, time);
                            let touch_position_wnd = LogicalPosition {
                                x: touch_position.x - window_logical_position.x,
                                y: touch_position.y - window_logical_position.y,
                            };

                            println!("WINDOW TOUCH POS: {:?} {:?}", touch_position_wnd, time);
                            io.mouse_pos = [touch_position_wnd.x as f32, touch_position_wnd.y as f32];
                            match touch.phase {
                                TouchPhase::Started => {
                                    io.mouse_down[0] = true;
                                    self.touch_pos = Some((touch_position_wnd.x, touch_position_wnd.y));
                                    println!("TOUCH PRESSED {:?}", time);
                                }
                                TouchPhase::Ended => {
                                    io.mouse_down[0] = false;
                                    self.touch_pos = None;
                                    println!("TOUCH RELEASED {:?}", time);
                                }
                                TouchPhase::Moved => {
                                    if io.mouse_down[0] {
                                        if let Some((x, y)) = self.touch_pos {
                                            let dpos = (touch_position_wnd.x - x, touch_position_wnd.y - y);
                                            self.touch_pos = Some((touch_position_wnd.x, touch_position_wnd.y));
                                            io.mouse_wheel_h += dpos.0 as f32 * 0.01;
                                            io.mouse_wheel += dpos.1 as f32 * 0.01;
                                        }
                                    }
                                }
                                _ => (),
                            }

                        } else {
                            println!("TOUCH POS OUTSIDE WINDOW");
                        }
                    }
                }
            }
            _ => (),
        }
    }
    /// Frame preparation callback.
    ///
    /// Call this before calling the imgui-rs context `frame` function.
    /// This function performs the following actions:
    ///
    /// * mouse cursor is repositioned (if requested by imgui-rs)
    pub fn prepare_frame(&self, io: &mut Io, window: &Window) -> Result<(), String> {
        if io.want_set_mouse_pos {
            let logical_pos = self.scale_pos_for_winit(
                window,
                LogicalPosition::new(f64::from(io.mouse_pos[0]), f64::from(io.mouse_pos[1])),
            );
            window.set_cursor_position(logical_pos)
        } else {
            Ok(())
        }
    }
    /// Render preparation callback.
    ///
    /// Call this before calling the imgui-rs UI `render_with`/`render` function.
    /// This function performs the following actions:
    ///
    /// * mouse cursor is changed and/or hidden (if requested by imgui-rs)
    pub fn prepare_render(&self, ui: &Ui, window: &Window) {
        let io = ui.io();
        if !io
            .config_flags
            .contains(ConfigFlags::NO_MOUSE_CURSOR_CHANGE)
        {
            match ui.mouse_cursor() {
                Some(mouse_cursor) if !io.mouse_draw_cursor => {
                    window.hide_cursor(false);
                    window.set_cursor(match mouse_cursor {
                        imgui::MouseCursor::Arrow => MouseCursor::Arrow,
                        imgui::MouseCursor::TextInput => MouseCursor::Text,
                        imgui::MouseCursor::ResizeAll => MouseCursor::Move,
                        imgui::MouseCursor::ResizeNS => MouseCursor::NsResize,
                        imgui::MouseCursor::ResizeEW => MouseCursor::EwResize,
                        imgui::MouseCursor::ResizeNESW => MouseCursor::NeswResize,
                        imgui::MouseCursor::ResizeNWSE => MouseCursor::NwseResize,
                        imgui::MouseCursor::Hand => MouseCursor::Hand,
                    });
                }
                _ => window.hide_cursor(true),
            }
        }
    }
}