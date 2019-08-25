use std::collections::hash_map::Entry;
use std::collections::HashMap;
use std::error;
use std::fmt;
use std::fs;
use std::io::{self, Read};

use crc32fast;
use nalgebra::base::Vector3;
use nalgebra::geometry::Point3;
use tobj;

use crate::geometry::{Geometry, TriangleFace};

#[derive(Debug, PartialEq)]
pub enum ImporterError {
    FileNotFound,
    PermissionDenied,
    InvalidStructure,
    Other,
}

impl fmt::Display for ImporterError {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        match *self {
            ImporterError::FileNotFound => write!(f, "File was not found."),
            ImporterError::InvalidStructure => write!(f, "The obj file is not valid."),
            ImporterError::PermissionDenied => write!(f, "Permission denied."),
            ImporterError::Other => write!(f, "Unexpected error happened."),
        }
    }
}

impl error::Error for ImporterError {}

impl From<io::Error> for ImporterError {
    fn from(err: io::Error) -> Self {
        match err.kind() {
            io::ErrorKind::NotFound => ImporterError::FileNotFound,
            io::ErrorKind::PermissionDenied => ImporterError::PermissionDenied,
            _ => ImporterError::Other,
        }
    }
}

impl From<tobj::LoadError> for ImporterError {
    fn from(_err: tobj::LoadError) -> Self {
        ImporterError::InvalidStructure
    }
}

#[derive(Debug, Clone, PartialEq)]
pub struct Model {
    pub name: String,
    pub geometry: Geometry,
}

#[derive(Debug)]
pub struct FileMetadata {
    checksum: u32,
    last_modified: std::time::SystemTime,
}

/// `Importer` takes care of importing of obj files and caching of their
/// internal representations. It holds paths to files, their metadata
/// (checksums, timestamps) and models of parsed obj files.
#[derive(Debug, Default)]
pub struct Importer {
    path_metadata: HashMap<String, FileMetadata>,
    loaded_models: HashMap<u32, Vec<Model>>,
}

impl Importer {
    pub fn new() -> Self {
        Default::default()
    }

    /// Tries to import obj file from given `path`. If file was already imported
    /// and its timestamp is identical, parsed models are returned from cache.
    /// Otherwise, file is read, checksum calculated and cache is checked whether
    /// given file contents were already saved. If not, obj file is parsed and
    /// cached.
    pub fn import_obj(&mut self, path: &str) -> Result<Vec<Model>, ImporterError> {
        let mut file = fs::File::open(path)?;
        let file_modified = file.metadata().and_then(|metadata| metadata.modified())?;

        // If paths and timestamps match, we can just return cached models.
        if let Some(path_metadata) = self.path_metadata.get(path) {
            if path_metadata.last_modified == file_modified {
                return Ok(self
                    .loaded_models
                    .get(&path_metadata.checksum)
                    .expect("Should get loaded models by obj file's checksum")
                    .clone());
            }
        }

        let file_size = file.metadata().map(|m| m.len() as usize + 1).unwrap_or(0);
        let mut file_contents = Vec::with_capacity(file_size);
        file.read_to_end(&mut file_contents)?;
        let checksum = calculate_checksum(&file_contents);

        let models = match self.loaded_models.entry(checksum) {
            Entry::Occupied(loaded_model) => {
                self.path_metadata.insert(
                    path.to_string(),
                    FileMetadata {
                        checksum,
                        last_modified: file_modified,
                    },
                );

                loaded_model.get().clone()
            }
            Entry::Vacant(loaded_model) => {
                let (tobj_models, _) = obj_buf_into_tobj(&mut file_contents.as_slice())?;
                let models = tobj_to_internal(tobj_models);

                self.path_metadata.insert(
                    path.to_string(),
                    FileMetadata {
                        checksum,
                        last_modified: file_modified,
                    },
                );
                loaded_model.insert(models.clone());

                models
            }
        };

        Ok(models)
    }

    /// FIXME: This is a poor man's testing method for cache contents. It should
    /// be removed once cacher is removed from this structure and proper unit
    /// tests are written for it.
    pub fn is_cached(&self, path: &str, checksum: u32) -> bool {
        if let Some(path_metadata) = self.path_metadata.get(path) {
            return path_metadata.checksum == checksum
                && self.loaded_models.contains_key(&checksum);
        }

        false
    }
}

/// Converts contents of obj file into tobj representation. Materials are
/// ignored.
pub fn obj_buf_into_tobj(file_contents: &mut &[u8]) -> tobj::LoadResult {
    tobj::load_obj_buf(file_contents, |_| Ok((vec![], HashMap::new())))
}

/// Converts `tobj::Model` vector into vector of internal `Model` representations.
/// It expects valid `tobj::Model` representation, eg. number of positions
/// divisible by 3.
pub fn tobj_to_internal(tobj_models: Vec<tobj::Model>) -> Vec<Model> {
    let mut models = Vec::with_capacity(tobj_models.len());

    for model in tobj_models {
        let vertex_positions: Vec<_> = model
            .mesh
            .positions
            .chunks_exact(3)
            .map(|chunk| Point3::new(chunk[0], chunk[1], chunk[2]))
            .collect();

        let vertex_normals = if model.mesh.normals.is_empty() {
            None
        } else {
            let normals = model
                .mesh
                .normals
                .chunks_exact(3)
                .map(|chunk| Vector3::new(chunk[0], chunk[1], chunk[2]))
                .collect();

            Some(normals)
        };

        let faces = model
            .mesh
            .indices
            .chunks_exact(3)
            .map(|chunk| TriangleFace {
                vertices: (chunk[0], chunk[1], chunk[2]),
                normals: if vertex_normals.is_some() {
                    Some((chunk[0], chunk[1], chunk[2]))
                } else {
                    None
                },
            })
            .collect();

        let geometry = if let Some(vertex_normals) = vertex_normals {
            Geometry::from_triangle_faces_with_vertices_and_normals(
                faces,
                vertex_positions,
                vertex_normals,
            )
        } else {
            Geometry::from_triangle_faces_with_vertices(faces, vertex_positions)
        };

        models.push(Model {
            name: model.name,
            geometry,
        });
    }

    models
}

pub fn calculate_checksum(string: &[u8]) -> u32 {
    let mut hasher = crc32fast::Hasher::new();

    hasher.update(string);
    hasher.finalize()
}

#[cfg(test)]
mod tests {
    use super::*;

    fn create_tobj_model(indices: Vec<u32>, positions: Vec<f32>, normals: Vec<f32>) -> tobj::Model {
        tobj::Model {
            name: String::from("Test model"),
            mesh: tobj::Mesh {
                indices,
                positions,
                normals,
                texcoords: vec![],
                material_id: None,
            },
        }
    }

    #[test]
    fn test_tobj_to_internal_returns_correct_representation_for_single_model() {
        let tobj_model = create_tobj_model(
            vec![0, 1, 2],
            vec![6.0, 5.0, 4.0, 3.0, 2.0, 1.0, 0.0, 1.0, 2.0],
            vec![1.0, 0.0, 0.0, 1.0, 0.0, 0.0, 1.0, 0.0, 0.0],
        );
        let tobj_models = vec![tobj_model.clone()];
        let models = tobj_to_internal(tobj_models);

        assert_eq!(
            models,
            vec![Model {
                name: tobj_model.name,
                geometry: Geometry::from_triangle_faces_with_vertices_and_normals(
                    vec![TriangleFace {
                        vertices: (0, 1, 2),
                        normals: Some((0, 1, 2)),
                    }],
                    vec![
                        Point3::new(6.0, 5.0, 4.0),
                        Point3::new(3.0, 2.0, 1.0),
                        Point3::new(0.0, 1.0, 2.0),
                    ],
                    vec![
                        Vector3::new(1.0, 0.0, 0.0),
                        Vector3::new(1.0, 0.0, 0.0),
                        Vector3::new(1.0, 0.0, 0.0),
                    ],
                ),
            }]
        );
    }

    #[test]
    fn test_tobj_to_internal_returns_correct_representation_for_multiple_models() {
        let tobj_model_1 = create_tobj_model(
            vec![0, 1, 2],
            vec![6.0, 5.0, 4.0, 3.0, 2.0, 1.0, 0.0, 1.0, 2.0],
            vec![1.0, 0.0, 0.0, 1.0, 0.0, 0.0, 1.0, 0.0, 0.0],
        );
        let tobj_model_2 = create_tobj_model(
            vec![0, 1, 2],
            vec![16.0, 15.0, 14.0, 13.0, 12.0, 11.0, 10.0, 9.0, 8.0],
            vec![1.0, 0.0, 0.0, 1.0, 0.0, 0.0, 1.0, 0.0, 0.0],
        );
        let tobj_models = vec![tobj_model_1.clone(), tobj_model_2.clone()];
        let models = tobj_to_internal(tobj_models);

        assert_eq!(
            models,
            vec![
                Model {
                    name: tobj_model_1.name,
                    geometry: Geometry::from_triangle_faces_with_vertices_and_normals(
                        vec![TriangleFace {
                            vertices: (0, 1, 2),
                            normals: Some((0, 1, 2)),
                        }],
                        vec![
                            Point3::new(6.0, 5.0, 4.0),
                            Point3::new(3.0, 2.0, 1.0),
                            Point3::new(0.0, 1.0, 2.0),
                        ],
                        vec![
                            Vector3::new(1.0, 0.0, 0.0),
                            Vector3::new(1.0, 0.0, 0.0),
                            Vector3::new(1.0, 0.0, 0.0),
                        ],
                    ),
                },
                Model {
                    name: tobj_model_2.name,
                    geometry: Geometry::from_triangle_faces_with_vertices_and_normals(
                        vec![TriangleFace {
                            vertices: (0, 1, 2),
                            normals: Some((0, 1, 2)),
                        }],
                        vec![
                            Point3::new(16.0, 15.0, 14.0),
                            Point3::new(13.0, 12.0, 11.0),
                            Point3::new(10.0, 9.0, 8.0),
                        ],
                        vec![
                            Vector3::new(1.0, 0.0, 0.0),
                            Vector3::new(1.0, 0.0, 0.0),
                            Vector3::new(1.0, 0.0, 0.0),
                        ],
                    ),
                },
            ]
        );
    }
}
