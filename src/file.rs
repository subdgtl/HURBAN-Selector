use std::fs;

use crc32fast;

pub fn load_file_into_string(file_path: &str) -> String {
    fs::read_to_string(file_path).unwrap_or_else(|_| panic!("File {} should be loaded", file_path))
}

pub fn calculate_checksum(string: &str) -> u32 {
    let mut hasher = crc32fast::Hasher::new();

    hasher.update(string.as_bytes());
    hasher.finalize()
}
