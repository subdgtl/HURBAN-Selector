use std::collections::HashMap;
use std::error;
use std::fmt;
use std::fs;
use std::io::{self, Read};
use std::time::SystemTime;

use crc32fast;
#[cfg(test)]
use mockall::{automock, lazy_static, predicate};
use nalgebra::{Point3, Vector3};
use tobj;

use crate::mesh::{Mesh, NormalStrategy, TriangleFace};

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
    pub mesh: Mesh,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct FileMetadata {
    checksum: u32,
    last_modified: SystemTime,
}

pub type ImporterResult = Result<Vec<Model>, ImporterError>;

/// An interface for caching of obj files.
///
/// The source is expected to be file with path and FileMetadata.
#[cfg_attr(test, automock)]
pub trait ObjCache {
    /// Returns models if modified timestamp of file on given `path` didn't
    /// change.
    fn get_if_not_modified(&self, path: &str, modified: SystemTime) -> Option<Vec<Model>>;

    /// Returns models if checksum of source file is already cached.
    ///
    /// This means contents of file were cached, however it could have been
    /// from a different path. Therefore file path and metadata should be
    /// cached again.
    fn get_by_checksum(&self, checksum: u32) -> Option<Vec<Model>>;

    /// Sets given data in cache.
    ///
    /// Most likely there is going to be multiple caching structures present, so
    /// this method accepts all available data and picks only whatever makes
    /// sense to cache.
    fn set(&mut self, path: String, metadata: FileMetadata, models: &[Model]);
}

/// Cache with no size limits. Anything set is kept until the application exits.
#[derive(Debug, Default)]
pub struct EndlessCache {
    path_metadata: HashMap<String, FileMetadata>,
    loaded_models: HashMap<u32, Vec<Model>>,
}

impl ObjCache for EndlessCache {
    fn get_if_not_modified(&self, path: &str, modified: SystemTime) -> Option<Vec<Model>> {
        if let Some(path_metadata) = self.path_metadata.get(path) {
            if path_metadata.last_modified == modified {
                return Some(
                    self.loaded_models
                        .get(&path_metadata.checksum)
                        .expect("Should get loaded models by obj file's checksum")
                        .clone(),
                );
            }
        }

        None
    }

    fn get_by_checksum(&self, checksum: u32) -> Option<Vec<Model>> {
        self.loaded_models.get(&checksum).cloned()
    }

    fn set(&mut self, path: String, metadata: FileMetadata, models: &[Model]) {
        self.path_metadata.insert(path, metadata);
        self.loaded_models
            .entry(metadata.checksum)
            .or_insert_with(|| models.to_vec());
    }
}

/// `Importer` takes care of importing of obj files and caching of their
/// internal representations.
pub struct Importer<C: ObjCache> {
    cache: C,
}

impl<C: ObjCache> Importer<C> {
    pub fn new(cache: C) -> Self {
        Self { cache }
    }

    /// Tries to import obj file from given `path`. If file was already imported
    /// and its timestamp is identical, parsed models are returned from cache.
    /// Otherwise, file is read, checksum calculated and cache is checked whether
    /// given file contents were already saved. If not, obj file is parsed and
    /// cached.
    ///
    /// Empty models are filtered out of the result.
    pub fn import_obj(&mut self, path: &str) -> ImporterResult {
        let mut file = fs::File::open(path)?;
        let file_metadata = file.metadata().expect("Failed to load obj file metadata");
        let file_modified = file_metadata
            .modified()
            .expect("Failed to load modified timestamp of obj file");

        let models = match self.cache.get_if_not_modified(path, file_modified) {
            Some(models) => return Ok(models),
            None => {
                let file_size = file_metadata.len() as usize;
                // Allocate one extra byte so the buffer doesn't need to grow before the
                // final `read` call at the end of the file.
                let mut file_contents = Vec::with_capacity(file_size + 1);
                file.read_to_end(&mut file_contents)?;
                let checksum = calculate_checksum(&file_contents);

                let models = match self.cache.get_by_checksum(checksum) {
                    Some(models) => models.clone(),
                    None => {
                        let (mut tobj_models, _) =
                            obj_buf_into_tobj(&mut file_contents.as_slice())?;
                        let mut index = 0;

                        while index != tobj_models.len() {
                            if tobj_models[index].mesh.positions.is_empty() {
                                tobj_models.remove(index);
                            } else {
                                index += 1;
                            }
                        }

                        tobj_to_internal(tobj_models)
                    }
                };

                self.cache.set(
                    path.to_string(),
                    FileMetadata {
                        checksum,
                        last_modified: file_modified,
                    },
                    &models,
                );

                models
            }
        };

        Ok(models)
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

        let vertex_normals: Option<Vec<_>> = if model.mesh.normals.is_empty() {
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

        let faces_raw: Vec<(u32, u32, u32)> = model
            .mesh
            .indices
            .chunks_exact(3)
            .map(|chunk| (chunk[0], chunk[1], chunk[2]))
            .collect();

        let mesh = if let Some(vertex_normals) = vertex_normals {
            Mesh::from_triangle_faces_with_vertices_and_normals(
                faces_raw.into_iter().map(TriangleFace::from),
                vertex_positions,
                vertex_normals,
            )
        } else {
            Mesh::from_triangle_faces_with_vertices_and_computed_normals(
                faces_raw,
                vertex_positions,
                NormalStrategy::Sharp,
            )
        };

        models.push(Model {
            name: model.name,
            mesh,
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
    use std::time::Duration;

    use super::*;

    fn create_tobj_model(
        face_vertex_indices: Vec<u32>,
        positions: Vec<f32>,
        normals: Vec<f32>,
    ) -> tobj::Model {
        tobj::Model {
            name: String::from("Test model"),
            mesh: tobj::Mesh {
                indices: face_vertex_indices,
                positions,
                normals,
                texcoords: vec![],
                material_id: None,
            },
        }
    }

    fn triangle() -> tobj::Model {
        create_tobj_model(
            vec![0, 1, 2],
            vec![6.0, 5.0, 4.0, 3.0, 2.0, 1.0, 0.0, 1.0, 2.0],
            vec![0.0, 0.0, 1.0, 0.0, 0.0, 1.0, 0.0, 0.0, 1.0],
        )
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
                mesh: Mesh::from_triangle_faces_with_vertices_and_normals(
                    vec![TriangleFace::from_same_vertex_and_normal_index(0, 1, 2)],
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
                    mesh: Mesh::from_triangle_faces_with_vertices_and_normals(
                        vec![TriangleFace::from_same_vertex_and_normal_index(0, 1, 2)],
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
                    mesh: Mesh::from_triangle_faces_with_vertices_and_normals(
                        vec![TriangleFace::from_same_vertex_and_normal_index(0, 1, 2)],
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

    #[test]
    fn test_obj_cache_set_caches_new_path_with_metadata() {
        let mut cache = EndlessCache::default();
        let path = "/path/to/some.obj".to_string();
        let metadata = FileMetadata {
            checksum: 1u32,
            last_modified: SystemTime::now(),
        };

        cache.set(path.clone(), metadata, &[]);

        assert_eq!(cache.path_metadata.len(), 1);
        assert_eq!(
            cache
                .path_metadata
                .get(&path)
                .expect("Path should be present in cache"),
            &metadata
        );
    }

    #[test]
    fn test_obj_cache_set_overrides_existing_path_with_new_metadata() {
        let mut cache = EndlessCache::default();
        let path = "/path/to/some.obj".to_string();
        let metadata = FileMetadata {
            checksum: 1u32,
            last_modified: SystemTime::now(),
        };

        cache.set(path.clone(), metadata, &[]);

        let new_metadata = FileMetadata {
            checksum: 2u32,
            last_modified: SystemTime::now(),
        };

        cache.set(path.clone(), new_metadata, &[]);

        assert_eq!(cache.path_metadata.len(), 1);
        assert_eq!(
            cache
                .path_metadata
                .get(&path)
                .expect("Path should be present in cache"),
            &new_metadata
        );
    }

    #[test]
    fn test_obj_cache_set_caches_new_checksum_with_models() {
        let mut cache = EndlessCache::default();
        let path = "/path/to/some.obj".to_string();
        let checksum = 1u32;
        let metadata = FileMetadata {
            checksum,
            last_modified: SystemTime::now(),
        };
        let models = tobj_to_internal(vec![triangle()]);

        cache.set(path.clone(), metadata, &models);

        assert_eq!(cache.loaded_models.len(), 1);
        assert_eq!(
            cache
                .loaded_models
                .get(&checksum)
                .expect("Checksum should be present in cache"),
            &models
        );
    }

    #[test]
    fn test_obj_cache_set_does_not_override_checksum_with_different_models() {
        let mut cache = EndlessCache::default();
        let path = "/path/to/some.obj".to_string();
        let checksum = 1u32;
        let metadata = FileMetadata {
            checksum,
            last_modified: SystemTime::now(),
        };
        let models = tobj_to_internal(vec![triangle()]);

        cache.set(path.clone(), metadata, &models);

        let new_models = tobj_to_internal(vec![create_tobj_model(
            vec![0, 1, 2],
            vec![6.0, 5.0, 4.0, 3.0, 2.0, 1.0, 0.0, 1.0, 2.0],
            vec![0.0, 1.0, 1.0, 0.0, 1.0, 1.0, 0.0, 1.0, 1.0],
        )]);

        cache.set(path.clone(), metadata, &new_models);

        assert_eq!(cache.loaded_models.len(), 1);
        assert_eq!(
            cache
                .loaded_models
                .get(&checksum)
                .expect("Checksum should be present in cache"),
            &models
        );
    }

    #[test]
    fn test_obj_cache_get_if_not_modified_returns_models_if_timestamp_is_unchanged() {
        let mut cache = EndlessCache::default();
        let path = "/path/to/some.obj".to_string();
        let now = SystemTime::now();
        let metadata = FileMetadata {
            checksum: 1u32,
            last_modified: now,
        };
        let models = tobj_to_internal(vec![triangle()]);
        cache.set(path.clone(), metadata, &models);

        let loaded_models = cache.get_if_not_modified(&path, now);

        assert_eq!(loaded_models.expect("Models should be loaded"), models);
    }

    #[test]
    fn test_obj_cache_get_if_not_modified_returns_none_if_timestamp_changed() {
        let mut cache = EndlessCache::default();
        let path = "/path/to/some.obj".to_string();
        let now = SystemTime::now();
        let metadata = FileMetadata {
            checksum: 1u32,
            last_modified: now,
        };
        let models = tobj_to_internal(vec![triangle()]);
        cache.set(path.clone(), metadata, &models);

        let loaded_models = cache.get_if_not_modified(
            &path,
            now.checked_add(Duration::from_secs(1))
                .expect("Duration should be added"),
        );

        assert!(loaded_models.is_none());
    }

    #[test]
    fn test_obj_cache_get_by_checksum_returns_models_if_checksum_does_match() {
        let mut cache = EndlessCache::default();
        let path = "/path/to/some.obj".to_string();
        let checksum = 1u32;
        let metadata = FileMetadata {
            checksum,
            last_modified: SystemTime::now(),
        };
        let models = tobj_to_internal(vec![triangle()]);
        cache.set(path.clone(), metadata, &models);

        let loaded_models = cache.get_by_checksum(checksum);

        assert_eq!(loaded_models.expect("Models should be loaded"), models);
    }

    #[test]
    fn test_obj_cache_get_by_checksum_returns_none_if_checksum_does_not_match() {
        let mut cache = EndlessCache::default();
        let path = "/path/to/some.obj".to_string();
        let checksum = 1u32;
        let metadata = FileMetadata {
            checksum,
            last_modified: SystemTime::now(),
        };
        let models = tobj_to_internal(vec![triangle()]);
        cache.set(path.clone(), metadata, &models);

        let loaded_models = cache.get_by_checksum(checksum + 1);

        assert!(loaded_models.is_none());
    }

    // The following tests with mocked cache are technically integration tests
    // and they use fixture data. They're kept here to prevent complications
    // with automocks not being present in debug build, as it is built without
    // code marked as `cfg(test)`.
    //
    // FIXME: Ideal scenario would be if filesystem was mocked and passed into
    // this method as well. It'd become proper unit test that way.

    fn file_metadata(path: &str) -> FileMetadata {
        let mut file = fs::File::open(path).expect("Failed to open file");
        let file_metadata = file.metadata().expect("Failed to load obj file metadata");
        let last_modified = file_metadata
            .modified()
            .expect("Failed to load modified timestamp of obj file");
        let file_size = file_metadata.len() as usize;
        let mut file_contents = Vec::with_capacity(file_size + 1);
        file.read_to_end(&mut file_contents)
            .expect("Failed to read contents of file");
        let checksum = calculate_checksum(&file_contents);

        FileMetadata {
            checksum,
            last_modified,
        }
    }

    #[test]
    fn test_importer_import_obj_cache_sets_models_if_file_was_not_cached_before() {
        let mut cache = MockObjCache::new();
        cache
            .expect_get_if_not_modified()
            .returning(|_, _| None)
            .times(1);
        cache.expect_get_by_checksum().returning(|_| None).times(1);
        cache.expect_set().returning(|_, _, _| ()).times(1);

        let mut importer = Importer::new(cache);
        let path = "tests/fixtures/valid.obj";

        importer
            .import_obj(&path)
            .expect("Valid obj should be loaded");
    }

    #[test]
    fn test_importer_import_obj_cache_returns_unmodified_file() {
        let mut cache = MockObjCache::default();
        cache
            .expect_get_if_not_modified()
            .returning(|_, _| Some(vec![]))
            .times(1);
        cache.expect_get_by_checksum().returning(|_| None).times(0);
        cache.expect_set().returning(|_, _, _| ()).times(0);

        let mut importer = Importer::new(cache);
        let path = "tests/fixtures/valid.obj";

        importer
            .import_obj(&path)
            .expect("Valid obj should be loaded");
    }

    #[test]
    fn test_importer_import_obj_cache_returns_the_same_cached_data_from_different_file() {
        lazy_static! {
            static ref MODELS: Vec<Model> = vec![Model {
                name: "test".to_string(),
                mesh: Mesh::from_triangle_faces_with_vertices_and_normals(
                    vec![TriangleFace::from_same_vertex_and_normal_index(0, 1, 2)],
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
            }];
        }
        let path = "tests/fixtures/valid.obj";
        let file_metadata = file_metadata(&path);
        let mut cache = MockObjCache::new();
        cache
            .expect_get_if_not_modified()
            .returning(|_, _| None)
            .times(1);
        cache
            .expect_get_by_checksum()
            .returning(|_| Some(MODELS.to_vec()))
            .times(1);
        cache
            .expect_set()
            .with(
                predicate::eq(path.to_string()),
                predicate::eq(file_metadata),
                predicate::eq(&MODELS[..]),
            )
            .returning(|_, _, _| ())
            .times(1);

        let mut importer = Importer::new(cache);

        importer
            .import_obj(&path)
            .expect("Valid obj should be loaded");
    }
}
