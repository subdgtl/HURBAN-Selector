use std::path::Path;

use tobj;

#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Vertex {
    pub position: [f32; 3],
}

pub type Index = u16;

/// Loads file from `file_path` and converts it into `tobj` result. `file_path`
/// expects valid path string. It's not validated since the only input should
/// be from system file dialogs.
pub fn load_obj(file_path: &str) -> tobj::LoadResult {
    let path = Path::new(file_path);

    tobj::load_obj(&path)
}
