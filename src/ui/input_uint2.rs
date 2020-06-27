use crate::convert::{clamp_cast_i32_to_u32, clamp_cast_u32_to_i32};

pub fn input_uint2(
    ui: &imgui::Ui,
    label_and_id: &imgui::ImStr,
    value: &mut [u32; 2],
) -> bool {
    let mut int_value = [
        clamp_cast_u32_to_i32(value[0]),
        clamp_cast_u32_to_i32(value[1]),
    ];
    let elem = ui.input_int2(label_and_id, &mut int_value);

    let changed = elem.build();
    *value = [
        clamp_cast_i32_to_u32(int_value[0]),
        clamp_cast_i32_to_u32(int_value[1]),
    ];

    changed
}
