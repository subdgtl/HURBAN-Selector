#[macro_export]
macro_rules! include_shader {
    ($name:expr) => {{
        include_bytes!(concat!(env!("OUT_DIR"), "/shaders/", $name))
    }};
}
