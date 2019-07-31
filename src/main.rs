use tinyfiledialogs;
use wgpu::winit;

mod file;
mod obj;
mod scene;

fn main() {
    let mut event_loop = winit::EventsLoop::new();
    let _window = winit::Window::new(&event_loop).expect("Failed to create window.");
    let mut running = true;
    let mut scene = scene::Scene::new();

    while running {
        event_loop.poll_events(|event| {
            if let winit::Event::WindowEvent { event, .. } = event {
                match event {
                    winit::WindowEvent::CloseRequested => running = false,
                    winit::WindowEvent::Resized(logical_size) => {
                        dbg!(logical_size);
                    }
                    winit::WindowEvent::KeyboardInput {
                        input:
                            winit::KeyboardInput {
                                virtual_keycode: Some(code),
                                state: winit::ElementState::Pressed,
                                ..
                            },
                        ..
                    } => match code {
                        winit::VirtualKeyCode::O => {
                            if let Some(path) = tinyfiledialogs::open_file_dialog(
                                "Open",
                                "",
                                Some((&["*.obj"], "Wavefront (.obj)")),
                            ) {
                                let checksum = file::calculate_checksum(&path);

                                if scene.add_obj_contents(path, checksum).is_err() {
                                    tinyfiledialogs::message_box_ok(
                                        "Error",
                                        "The obj file is not valid.",
                                        tinyfiledialogs::MessageBoxIcon::Error,
                                    )
                                }
                            }
                        }
                        winit::VirtualKeyCode::S => {
                            let save_file: String;
                            match tinyfiledialogs::save_file_dialog("Save", "password.txt") {
                                Some(file_path) => save_file = file_path,
                                None => save_file = "null".to_string(),
                            }
                            dbg!(save_file);
                        }
                        _ => {}
                    },
                    _ => (),
                }
            }
        });
    }
}
