use std::collections::hash_map::Entry;
use std::collections::HashMap;
use std::fs;
use std::io::ErrorKind;

use crate::file;
use crate::obj;

#[derive(Debug, PartialEq, Eq)]
pub enum ImporterError {
    FileNotFound,
    PermissionDenied,
    InvalidStructure,
    Other,
}

#[derive(Default)]
pub struct Scene {
    pub path_checksums: HashMap<String, u32>,
    pub checksum_paths: HashMap<u32, Vec<String>>,
    pub loaded_models: HashMap<String, Vec<obj::Model>>,
}

impl Scene {
    pub fn new() -> Self {
        Default::default()
    }

    pub fn add_obj_contents(&mut self, path: &str) -> Result<(), ImporterError> {
        let file_contents = self.load_file_to_string(path)?;
        let checksum = file::calculate_checksum(&file_contents);
        let is_identical_obj_loaded =
            self.checksum_paths.contains_key(&checksum) && !self.loaded_models.contains_key(path);

        if is_identical_obj_loaded {
            self.duplicate_obj_models(path, checksum);
        } else {
            match self.path_checksums.entry(path.to_string()) {
                Entry::Occupied(path_checksum) => {
                    if checksum != *path_checksum.get() {
                        match obj::obj_buf_into_tobj(&file_contents) {
                            Ok((tobj_models, _)) => {
                                let models = obj::tobj_to_internal(tobj_models);

                                self.loaded_models.insert(path.to_string(), models);
                            }
                            Err(_) => return Err(ImporterError::InvalidStructure),
                        }
                    }
                }
                Entry::Vacant(path_checksum) => match obj::obj_buf_into_tobj(&file_contents) {
                    Ok((tobj_models, _)) => {
                        dbg!(&tobj_models);
                        let models = obj::tobj_to_internal(tobj_models);

                        self.loaded_models.insert(path.to_string(), models);
                        path_checksum.insert(checksum);
                        self.checksum_paths.insert(checksum, vec![path.to_string()]);
                    }
                    Err(_) => return Err(ImporterError::InvalidStructure),
                },
            }
        }

        Ok(())
    }

    /// Loads contents of file to string. In case of any error, it is converted
    /// into `ImporterError`.
    fn load_file_to_string(&self, path: &str) -> Result<String, ImporterError> {
        match fs::read_to_string(path) {
            Ok(contents) => Ok(contents),
            Err(err) => match err.kind() {
                ErrorKind::NotFound => Err(ImporterError::FileNotFound),
                ErrorKind::PermissionDenied => Err(ImporterError::PermissionDenied),
                _ => Err(ImporterError::Other),
            },
        }
    }

    fn duplicate_obj_models(&mut self, path: &str, checksum: u32) {
        let duplicate_paths = self
            .checksum_paths
            .get_mut(&checksum)
            .expect("Checksum expected to be present in the scene");
        let duplicate_path = duplicate_paths[0].clone();
        let models_to_duplicate = self
            .loaded_models
            .get(&duplicate_path)
            .expect("Models for given path should be present in the scene");
        let cloned_models_to_duplicate = models_to_duplicate.clone();

        self.loaded_models
            .insert(path.to_string(), cloned_models_to_duplicate);
        duplicate_paths.push(path.to_string());
        self.path_checksums.insert(path.to_string(), checksum);
    }
}
