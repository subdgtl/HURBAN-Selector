use crate::convert::{clamp_cast_i32_to_u32, clamp_cast_u32_to_i32};

const DEFAULT_SPEED: f32 = 0.01;

#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Options {
    pub speed: f32,
    pub min: Option<u32>,
    pub max: Option<u32>,
}

impl Options {
    pub fn new() -> Self {
        Self::default()
    }

    #[allow(dead_code)]
    pub fn set_speed(&mut self, speed: f32) -> &mut Self {
        self.speed = speed;
        self
    }

    pub fn set_min(&mut self, min: Option<u32>) -> &mut Self {
        self.min = min;
        self
    }

    pub fn set_max(&mut self, max: Option<u32>) -> &mut Self {
        self.max = max;
        self
    }
}

impl Default for Options {
    fn default() -> Self {
        Self {
            speed: DEFAULT_SPEED,
            min: None,
            max: None,
        }
    }
}

pub fn drag_uint(
    ui: &imgui::Ui,
    label_and_id: &imgui::ImStr,
    value: &mut u32,
    options: &Options,
) -> bool {
    let mut int_value = clamp_cast_u32_to_i32(*value);
    let mut elem = ui.drag_int(label_and_id, &mut int_value).speed(options.speed);

    if let Some(min) = options.min {
        elem = elem.min(clamp_cast_u32_to_i32(min));
    } else {
        elem = elem.min(0_i32);
    }

    if let Some(max) = options.max {
        elem = elem.max(clamp_cast_u32_to_i32(max));
    } else {
        elem = elem.max(clamp_cast_u32_to_i32(u32::MAX));
    }

    let changed = elem.build();
    *value = clamp_cast_i32_to_u32(int_value);

    changed
}
