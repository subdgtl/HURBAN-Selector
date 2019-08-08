use std::collections::{HashMap, HashSet};

use hurban_selector::file;
use hurban_selector::scene::{ImporterError, Scene};

#[test]
fn test_adds_valid_obj_file() {
    let mut scene = Scene::new();
    let path = "tests/fixtures/valid.obj".to_string();
    let file_contents = file::load_file_into_string(&path);
    let checksum = file::calculate_checksum(&file_contents);

    scene
        .add_obj_contents(&path)
        .expect("Valid object should be loaded");

    let mut expected_path_checksums = HashMap::new();
    expected_path_checksums.insert(path.clone(), checksum);

    let mut expected_checksum_paths = HashMap::new();
    expected_checksum_paths.insert(checksum, vec![path.clone()]);

    let loaded_models_paths: HashSet<_> = scene.loaded_models.keys().collect();
    let mut expected_loaded_models_paths: HashSet<&String> = HashSet::new();
    expected_loaded_models_paths.insert(&path);

    assert_eq!(scene.path_checksums, expected_path_checksums);
    assert_eq!(scene.checksum_paths, expected_checksum_paths);
    assert_eq!(loaded_models_paths, expected_loaded_models_paths);
}

#[test]
fn test_adds_two_different_valid_obj_files() {
    let mut scene = Scene::new();
    let path_1 = "tests/fixtures/valid.obj".to_string();
    let file_contents_1 = file::load_file_into_string(&path_1);
    let checksum_1 = file::calculate_checksum(&file_contents_1);
    let path_2 = "tests/fixtures/valid_2.obj".to_string();
    let file_contents_2 = file::load_file_into_string(&path_2);
    let checksum_2 = file::calculate_checksum(&file_contents_2);

    scene
        .add_obj_contents(&path_1)
        .expect("Valid object should be loaded");
    scene
        .add_obj_contents(&path_2)
        .expect("Valid object should be loaded");

    let mut expected_path_checksums = HashMap::new();
    expected_path_checksums.insert(path_1.clone(), checksum_1);
    expected_path_checksums.insert(path_2.clone(), checksum_2);

    let mut expected_checksum_paths = HashMap::new();
    expected_checksum_paths.insert(checksum_1, vec![path_1.clone()]);
    expected_checksum_paths.insert(checksum_2, vec![path_2.clone()]);

    let loaded_models_paths: HashSet<_> = scene.loaded_models.keys().collect();
    let mut expected_loaded_models_paths: HashSet<&String> = HashSet::new();
    expected_loaded_models_paths.insert(&path_1);
    expected_loaded_models_paths.insert(&path_2);

    assert_eq!(scene.path_checksums, expected_path_checksums);
    assert_eq!(scene.checksum_paths, expected_checksum_paths);
    assert_eq!(loaded_models_paths, expected_loaded_models_paths);
}

#[test]
fn test_does_not_add_invalid_obj_file() {
    let mut scene = Scene::new();
    let path = "tests/fixtures/invalid.obj".to_string();

    let error = scene
        .add_obj_contents(&path)
        .expect_err("Error should be thrown");

    assert_eq!(error, ImporterError::InvalidStructure);
    assert_eq!(scene.path_checksums, HashMap::new());
    assert_eq!(scene.checksum_paths, HashMap::new());
    assert_eq!(scene.loaded_models, HashMap::new());
}

#[test]
fn test_does_not_add_nonexistent_file() {
    let mut scene = Scene::new();
    let path = "tests/fixtures/wrong_path.obj".to_string();

    let error = scene
        .add_obj_contents(&path)
        .expect_err("Error should be thrown");

    assert_eq!(error, ImporterError::FileNotFound);
    assert_eq!(scene.path_checksums, HashMap::new());
    assert_eq!(scene.checksum_paths, HashMap::new());
    assert_eq!(scene.loaded_models, HashMap::new());
}

#[test]
fn test_does_not_add_the_same_obj_file_with_the_same_contents_twice() {
    let mut scene = Scene::new();
    let path = "tests/fixtures/valid.obj".to_string();
    let file_contents = file::load_file_into_string(&path);
    let checksum = file::calculate_checksum(&file_contents);

    scene
        .add_obj_contents(&path)
        .expect("Valid object should be loaded");
    scene
        .add_obj_contents(&path)
        .expect("Valid object should be loaded");

    let mut expected_path_checksums = HashMap::new();
    expected_path_checksums.insert(path.clone(), checksum);

    let mut expected_checksum_paths = HashMap::new();
    expected_checksum_paths.insert(checksum, vec![path.clone()]);

    let loaded_models_paths: HashSet<_> = scene.loaded_models.keys().collect();
    let mut expected_loaded_models_paths: HashSet<&String> = HashSet::new();
    expected_loaded_models_paths.insert(&path);

    assert_eq!(scene.path_checksums, expected_path_checksums);
    assert_eq!(scene.checksum_paths, expected_checksum_paths);
    assert_eq!(loaded_models_paths, expected_loaded_models_paths);
}

#[test]
fn test_adds_two_different_files_with_the_same_contents() {
    let mut scene = Scene::new();
    let path_1 = "tests/fixtures/valid.obj".to_string();
    let file_contents_1 = file::load_file_into_string(&path_1);
    let checksum_1 = file::calculate_checksum(&file_contents_1);
    let path_2 = "tests/fixtures/valid_copy.obj".to_string();
    let file_contents_2 = file::load_file_into_string(&path_2);
    let checksum_2 = file::calculate_checksum(&file_contents_2);

    scene
        .add_obj_contents(&path_1)
        .expect("Valid object should be loaded");
    scene
        .add_obj_contents(&path_2)
        .expect("Valid object should be loaded");

    let mut expected_path_checksums = HashMap::new();
    expected_path_checksums.insert(path_1.clone(), checksum_1);
    expected_path_checksums.insert(path_2.clone(), checksum_2);

    let mut expected_checksum_paths = HashMap::new();
    expected_checksum_paths.insert(checksum_1, vec![path_1.clone(), path_2.clone()]);

    let loaded_models_paths: HashSet<_> = scene.loaded_models.keys().collect();
    let mut expected_loaded_models_paths: HashSet<&String> = HashSet::new();
    expected_loaded_models_paths.insert(&path_1);
    expected_loaded_models_paths.insert(&path_2);

    assert_eq!(scene.path_checksums, expected_path_checksums);
    assert_eq!(scene.checksum_paths, expected_checksum_paths);
    assert_eq!(loaded_models_paths, expected_loaded_models_paths);
}
