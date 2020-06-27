const DEFAULT_SPEED: f32 = 0.01;

#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Options {
    pub speed: f32,
    pub min: Option<i32>,
    pub max: Option<i32>,
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

    pub fn set_min(&mut self, min: Option<i32>) -> &mut Self {
        self.min = min;
        self
    }

    pub fn set_max(&mut self, max: Option<i32>) -> &mut Self {
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

pub fn drag_int(
    ui: &imgui::Ui,
    label_and_id: &imgui::ImStr,
    value: &mut i32,
    options: &Options,
) -> bool {
    let mut elem = ui.drag_int(label_and_id, value).speed(options.speed);

    if let Some(min) = options.min {
        elem = elem.min(min);
    } else {
        elem = elem.min(i32::MIN);
    }

    if let Some(max) = options.max {
        elem = elem.max(max);
    } else {
        elem = elem.max(i32::MAX);
    }

    elem.build()
}
