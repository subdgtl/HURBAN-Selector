use std::collections::hash_map::Entry;
use std::collections::HashMap;
use std::fs;
use std::io::ErrorKind;

use crc32fast;
use tobj;

use crate::viewport_renderer::{Index, Vertex};

#[derive(Debug, PartialEq, Eq)]
pub enum ImporterError {
    FileNotFound,
    PermissionDenied,
    InvalidStructure,
    Other,
}

#[derive(Debug, Clone, PartialEq)]
pub struct Model {
    pub name: String,
    pub vertices: Vec<Vertex>,
    pub indices: Vec<Index>,
}

/// `Importer` takes care of importing of obj files and caching of their
/// internal representations. It holds paths to files, their checksums and
/// parsed obj files.
#[derive(Debug, Default)]
pub struct Importer {
    pub path_checksums: HashMap<String, u32>,
    pub checksum_paths: HashMap<u32, Vec<String>>,
    pub loaded_models: HashMap<String, Vec<Model>>,
}

impl Importer {
    pub fn new() -> Self {
        Default::default()
    }

    /// Tries to import obj file from given `path`. If file was already
    /// imported and its checksum is identical, parsed models are returned
    /// from cache. If the checksum doesn't match, file is reimported and
    /// the old one is forgotten. If file wasn't imported, but its checksum
    /// matches other file, it is cloned. Brand new files are parsed and
    /// cached.
    pub fn import_obj(&mut self, path: &str) -> Result<Vec<Model>, ImporterError> {
        let file_contents = self.load_file_to_string(path)?;
        let checksum = calculate_checksum(&file_contents);
        let is_identical_obj_loaded =
            self.checksum_paths.contains_key(&checksum) && !self.loaded_models.contains_key(path);

        // Identical models imported from different path are duplicated.
        let models = if is_identical_obj_loaded {
            self.duplicate_models(checksum, path);

            self.loaded_models
                .get(path)
                .expect("Model should be present in cache")
                .clone()
        } else {
            match self.path_checksums.entry(path.to_string()) {
                // This file was already imported and we need to check its
                // checksum and either just return parsed models or reimport
                // and recache it.
                Entry::Occupied(mut path_checksum) => {
                    if checksum == *path_checksum.get() {
                        self.loaded_models
                            .get(path)
                            .expect("Model should be present in cache")
                            .clone()
                    } else {
                        match obj_buf_into_tobj(&file_contents) {
                            Ok((tobj_models, _)) => {
                                let models = tobj_to_internal(tobj_models);
                                let old_checksum = *path_checksum.get();

                                self.loaded_models.insert(path.to_string(), models.clone());
                                path_checksum.insert(checksum);
                                self.remove_checksum_path(old_checksum, &path);
                                self.add_checksum_path(checksum, &path);

                                models
                            }
                            Err(_) => return Err(ImporterError::InvalidStructure),
                        }
                    }
                }
                // This is a brand new file to parse.
                Entry::Vacant(path_checksum) => match obj_buf_into_tobj(&file_contents) {
                    Ok((tobj_models, _)) => {
                        let models = tobj_to_internal(tobj_models);

                        self.loaded_models.insert(path.to_string(), models.clone());
                        path_checksum.insert(checksum);
                        self.checksum_paths.insert(checksum, vec![path.to_string()]);

                        models
                    }
                    Err(_) => return Err(ImporterError::InvalidStructure),
                },
            }
        };

        Ok(models)
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

    fn duplicate_models(&mut self, checksum: u32, path: &str) {
        let duplicate_paths = self
            .checksum_paths
            .get_mut(&checksum)
            .expect("Checksum expected to be present in cache");
        let duplicate_path = duplicate_paths[0].clone();
        let models_to_duplicate = self
            .loaded_models
            .get(&duplicate_path)
            .expect("Models for given path should be present in cache");
        let cloned_models_to_duplicate = models_to_duplicate.clone();

        self.loaded_models
            .insert(path.to_string(), cloned_models_to_duplicate);
        duplicate_paths.push(path.to_string());
        self.path_checksums.insert(path.to_string(), checksum);
    }

    fn add_checksum_path(&mut self, checksum: u32, path: &str) {
        match self.checksum_paths.entry(checksum) {
            Entry::Occupied(mut checksum_paths) => {
                checksum_paths.get_mut().push(path.to_string());
            }
            Entry::Vacant(entry) => {
                entry.insert(vec![path.to_string()]);
            }
        };
    }

    fn remove_checksum_path(&mut self, checksum: u32, path: &str) {
        let checksum_paths = self
            .checksum_paths
            .get_mut(&checksum)
            .expect("Checksum should be present in cache");
        let pos = checksum_paths
            .iter()
            .position(|x| *x == *path)
            .expect("Path should be present in cached vector");
        checksum_paths.remove(pos);
    }
}

/// Converts contents of obj file into tobj representation. Materials are
/// ignored.
pub fn obj_buf_into_tobj(file_contents: &str) -> tobj::LoadResult {
    tobj::load_obj_buf(&mut file_contents.as_bytes(), |_| {
        Ok((vec![], HashMap::new()))
    })
}

/// Converts `tobj::Model` vector into vector of internal `Model` representations.
/// It expects valid `tobj::Model` representation, eg. number of positions
/// divisible by 3.
pub fn tobj_to_internal(tobj_models: Vec<tobj::Model>) -> Vec<Model> {
    let mut models = Vec::with_capacity(tobj_models.len());

    for model in tobj_models {
        let mut vertices = Vec::with_capacity(model.mesh.positions.len() / 3);

        for (index, _) in model.mesh.positions.iter().enumerate().step_by(3) {
            vertices.push(Vertex {
                position: [
                    model.mesh.positions[index],
                    model.mesh.positions[index + 1],
                    model.mesh.positions[index + 2],
                ],
            });
        }

        models.push(Model {
            name: model.name,
            vertices,
            indices: model.mesh.indices,
        });
    }

    models
}

pub fn calculate_checksum(string: &str) -> u32 {
    let mut hasher = crc32fast::Hasher::new();

    hasher.update(string.as_bytes());
    hasher.finalize()
}

#[cfg(test)]
mod tests {
    use super::*;

    fn create_tobj_model(indices: Vec<u32>, positions: Vec<f32>) -> tobj::Model {
        tobj::Model {
            name: String::from("Test model"),
            mesh: tobj::Mesh {
                indices,
                positions,
                material_id: None,
                normals: vec![],
                texcoords: vec![],
            },
        }
    }

    #[test]
    fn test_tobj_to_internal_returns_correct_representation_for_single_model() {
        let tobj_model = create_tobj_model(vec![1, 2], vec![6.0, 5.0, 4.0, 3.0, 2.0, 1.0]);
        let tobj_models = vec![tobj_model.clone()];
        let models = tobj_to_internal(tobj_models);

        assert_eq!(
            models,
            vec![Model {
                name: tobj_model.name,
                vertices: vec![
                    Vertex {
                        position: [6.0, 5.0, 4.0]
                    },
                    Vertex {
                        position: [3.0, 2.0, 1.0]
                    }
                ],
                indices: tobj_model.mesh.indices,
            }]
        );
    }

    #[test]
    fn test_tobj_to_internal_returns_correct_representation_for_multiple_models() {
        let tobj_model_1 = create_tobj_model(vec![1, 2], vec![6.0, 5.0, 4.0, 3.0, 2.0, 1.0]);
        let tobj_model_2 = create_tobj_model(vec![3, 4], vec![16.0, 15.0, 14.0, 13.0, 12.0, 11.0]);
        let tobj_models = vec![tobj_model_1.clone(), tobj_model_2.clone()];
        let models = tobj_to_internal(tobj_models);

        assert_eq!(
            models,
            vec![
                Model {
                    name: tobj_model_1.name,
                    vertices: vec![
                        Vertex {
                            position: [6.0, 5.0, 4.0]
                        },
                        Vertex {
                            position: [3.0, 2.0, 1.0]
                        }
                    ],
                    indices: tobj_model_1.mesh.indices,
                },
                Model {
                    name: tobj_model_2.name,
                    vertices: vec![
                        Vertex {
                            position: [16.0, 15.0, 14.0]
                        },
                        Vertex {
                            position: [13.0, 12.0, 11.0]
                        }
                    ],
                    indices: tobj_model_2.mesh.indices,
                }
            ]
        );
    }
}
