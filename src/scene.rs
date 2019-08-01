use std::collections::hash_map::Entry;
use std::collections::HashMap;

use tobj;

use crate::file;
use crate::obj;

pub struct Scene {
    pub path_checksums: HashMap<String, u32>,
    pub checksum_paths: HashMap<u32, Vec<String>>,
    pub loaded_models: HashMap<String, Vec<obj::Model>>,
}

impl Scene {
    pub fn new() -> Self {
        Self {
            path_checksums: HashMap::new(),
            checksum_paths: HashMap::new(),
            loaded_models: HashMap::new(),
        }
    }

    pub fn add_obj_contents(&mut self, path: String, checksum: u32) -> Result<(), tobj::LoadError> {
        let is_identical_obj_loaded =
            self.checksum_paths.contains_key(&checksum) && !self.loaded_models.contains_key(&path);

        if is_identical_obj_loaded {
            self.duplicate_obj_models(path, checksum);
        } else {
            match self.path_checksums.entry(path.clone()) {
                Entry::Occupied(path_checksum) => {
                    if checksum != *path_checksum.get() {
                        match file::load_obj(&path) {
                            Ok((tobj_models, _)) => {
                                let models = obj::tobj_to_internal(tobj_models);

                                self.loaded_models.insert(path, models);
                            }
                            Err(load_error) => return Err(load_error),
                        }
                    }
                }
                Entry::Vacant(path_checksum) => match file::load_obj(&path) {
                    Ok((tobj_models, _)) => {
                        let models = obj::tobj_to_internal(tobj_models);

                        self.loaded_models.insert(path.clone(), models);
                        path_checksum.insert(checksum);
                        self.checksum_paths.insert(checksum, vec![path.clone()]);
                    }
                    Err(load_error) => return Err(load_error),
                },
            }
        }

        Ok(())
    }

    fn duplicate_obj_models(&mut self, path: String, checksum: u32) {
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
            .insert(path.clone(), cloned_models_to_duplicate);
        duplicate_paths.push(path.clone());
        self.path_checksums.insert(path.clone(), checksum);
    }
}
