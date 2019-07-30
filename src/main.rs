use std::collections::HashMap;

use tinyfiledialogs;
use wgpu::winit;

mod file;
mod obj;

fn main() {
    let mut event_loop = winit::EventsLoop::new();
    let _window = winit::Window::new(&event_loop).expect("Failed to create window.");
    let mut running = true;
    let mut loaded_models = HashMap::new();

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
                            if let Some(file_path) = tinyfiledialogs::open_file_dialog(
                                "Open",
                                "",
                                Some((&["*.obj"], "Wavefront (.obj)")),
                            ) {
                                match file::load_obj(&file_path) {
                                    Ok((tobj_models, _)) => {
                                        let models = obj::tobj_to_internal(tobj_models);

                                        for model in models {
                                            let key =
                                                format!("{}-{}", file_path, model.name.clone());
                                            loaded_models.insert(key, model);
                                        }

                                        dbg!(&loaded_models);
                                    }
                                    Err(_) => tinyfiledialogs::message_box_ok(
                                        "Error",
                                        "The obj file is not valid.",
                                        tinyfiledialogs::MessageBoxIcon::Error,
                                    ),
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
