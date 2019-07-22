use wgpu::winit;

fn main() {
    let mut event_loop = winit::EventsLoop::new();
    let _window = winit::Window::new(&event_loop).expect("Failed to create window.");
    let mut running = true;

    while running {
        event_loop.poll_events(|event| {
            if let winit::Event::WindowEvent { event, .. } = event {
                match event {
                    winit::WindowEvent::CloseRequested => running = false,
                    winit::WindowEvent::Resized(logical_size) => {
                        dbg!(logical_size);
                    }
                    _ => (),
                }
            }
        });
    }
}
