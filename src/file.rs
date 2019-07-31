use std::fs::File;
use std::io::prelude::*;
use std::io::BufReader;
use std::path::Path;

use crc32fast;
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

pub fn calculate_checksum(file_path: &str) -> u32 {
    let file = File::open(file_path).unwrap_or_else(|_| panic!("File {} to be loaded", file_path));
    let mut reader = BufReader::new(file);
    let mut contents = String::new();
    reader
        .read_to_string(&mut contents)
        .unwrap_or_else(|_| panic!("Read contents of file {} into string.", file_path));

    let bytes = contents.into_bytes();
    let mut hasher = crc32fast::Hasher::new();

    hasher.update(&bytes);
    hasher.finalize()
}
