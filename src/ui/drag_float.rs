const DEFAULT_SPEED: f32 = 0.01;
const DEFAILT_POWER: f32 = 2.0;

#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Options {
    pub speed: f32,
    pub power: f32,
    pub min: Option<f32>,
    pub max: Option<f32>,
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

    #[allow(dead_code)]
    pub fn set_power(&mut self, power: f32) -> &mut Self {
        self.power = power;
        self
    }

    pub fn set_min(&mut self, min: Option<f32>) -> &mut Self {
        self.min = min;
        self
    }

    pub fn set_max(&mut self, max: Option<f32>) -> &mut Self {
        self.max = max;
        self
    }
}

impl Default for Options {
    fn default() -> Self {
        Self {
            speed: DEFAULT_SPEED,
            power: DEFAILT_POWER,
            min: None,
            max: None,
        }
    }
}

pub fn drag_float(
    ui: &imgui::Ui,
    label_and_id: &imgui::ImStr,
    value: &mut f32,
    options: &Options,
) -> bool {
    let mut elem = ui
        .drag_float(label_and_id, value)
        .speed(options.speed)
        .power(options.power);

    if let Some(min) = options.min {
        elem = elem.min(min);
    } else {
        elem = elem.min(f32::MIN);
    }

    if let Some(max) = options.max {
        elem = elem.max(max);
    } else {
        elem = elem.max(f32::MAX);
    }

    elem.build()
}
