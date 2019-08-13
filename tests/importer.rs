use std::fs;

use hurban_selector::importer::{self, Importer, ImporterError};

fn import_obj(path: &str) -> Vec<importer::Model> {
    let file_contents = fs::read(&path).expect("File should be read to bytes");
    let (tobj_models, _) =
        importer::obj_buf_into_tobj(&mut file_contents.as_slice()).expect("Obj should be parsed");

    importer::tobj_to_internal(tobj_models)
}

#[test]
fn test_caches_valid_obj_file() {
    let mut importer = Importer::new();
    let path = "tests/fixtures/valid.obj".to_string();
    let file_contents = fs::read(&path).expect("File should be read to bytes");
    let checksum = importer::calculate_checksum(&file_contents);

    importer
        .import_obj(&path)
        .expect("Valid object should be loaded");

    assert!(importer.is_cached(&path, checksum));
}

#[test]
fn test_returns_valid_obj_file() {
    let mut importer = Importer::new();
    let path = "tests/fixtures/valid.obj".to_string();

    let models = importer
        .import_obj(&path)
        .expect("Valid object should be loaded");
    let expected_models = import_obj(&path);

    assert_eq!(expected_models, models);
}

#[test]
fn test_caches_two_different_valid_obj_files() {
    let mut importer = Importer::new();
    let path_1 = "tests/fixtures/valid.obj".to_string();
    let file_contents_1 = fs::read(&path_1).expect("File should be read to bytes");
    let checksum_1 = importer::calculate_checksum(&file_contents_1);
    let path_2 = "tests/fixtures/valid_2.obj".to_string();
    let file_contents_2 = fs::read(&path_2).expect("File should be read to bytes");
    let checksum_2 = importer::calculate_checksum(&file_contents_2);

    importer
        .import_obj(&path_1)
        .expect("Valid object should be loaded");
    importer
        .import_obj(&path_2)
        .expect("Valid object should be loaded");

    assert!(importer.is_cached(&path_1, checksum_1));
    assert!(importer.is_cached(&path_2, checksum_2));
}

#[test]
fn test_returns_two_different_valid_obj_files() {
    let mut importer = Importer::new();
    let path_1 = "tests/fixtures/valid.obj".to_string();
    let path_2 = "tests/fixtures/valid_2.obj".to_string();

    let models_1 = importer
        .import_obj(&path_1)
        .expect("Valid object should be loaded");
    let models_2 = importer
        .import_obj(&path_2)
        .expect("Valid object should be loaded");
    let expected_models_1 = import_obj(&path_1);
    let expected_models_2 = import_obj(&path_2);

    assert_eq!(expected_models_1, models_1);
    assert_eq!(expected_models_2, models_2);
}

#[test]
fn test_does_not_cache_invalid_obj_file() {
    let mut importer = Importer::new();
    let path = "tests/fixtures/invalid.obj".to_string();
    let file_contents = fs::read(&path).expect("File should be read to bytes");
    let checksum = importer::calculate_checksum(&file_contents);

    importer
        .import_obj(&path)
        .expect_err("Error should be thrown");

    assert!(!importer.is_cached(&path, checksum));
}

#[test]
fn test_returns_error_when_importing_invalid_obj_file() {
    let mut importer = Importer::new();
    let path = "tests/fixtures/invalid.obj".to_string();

    let error = importer
        .import_obj(&path)
        .expect_err("Error should be thrown");

    assert_eq!(error, ImporterError::InvalidStructure);
}

#[test]
fn test_does_not_cache_nonexistent_file() {
    let mut importer = Importer::new();
    let path = "tests/fixtures/wrong_path.obj".to_string();

    importer
        .import_obj(&path)
        .expect_err("Error should be thrown");

    assert!(!importer.is_cached(&path, 0));
}

#[test]
fn test_returns_error_when_importing_nonexistent_file() {
    let mut importer = Importer::new();
    let path = "tests/fixtures/wrong_path.obj".to_string();

    let error = importer
        .import_obj(&path)
        .expect_err("Error should be thrown");

    assert_eq!(error, ImporterError::FileNotFound);
}

#[test]
fn test_does_not_cache_the_same_unchanged_obj_file_twice() {
    let mut importer = Importer::new();
    let path = "tests/fixtures/valid.obj".to_string();
    let file_contents = fs::read(&path).expect("File should be read to bytes");
    let checksum = importer::calculate_checksum(&file_contents);

    importer
        .import_obj(&path)
        .expect("Valid object should be loaded");
    importer
        .import_obj(&path)
        .expect("Valid object should be loaded");

    assert!(importer.is_cached(&path, checksum));
}

#[test]
fn test_returns_correct_models_when_importing_the_same_unchanged_file() {
    let mut importer = Importer::new();
    let path = "tests/fixtures/valid.obj".to_string();

    let models_1 = importer
        .import_obj(&path)
        .expect("Valid object should be loaded");
    let models_2 = importer
        .import_obj(&path)
        .expect("Valid object should be loaded");

    assert_eq!(models_1, models_2);
}

#[test]
fn test_caches_two_different_files_with_the_same_contents() {
    let mut importer = Importer::new();
    let path_1 = "tests/fixtures/valid.obj".to_string();
    let file_contents_1 = fs::read(&path_1).expect("File should be read to bytes");
    let checksum_1 = importer::calculate_checksum(&file_contents_1);
    let path_2 = "tests/fixtures/valid_copy.obj".to_string();
    let file_contents_2 = fs::read(&path_2).expect("File should be read to bytes");
    let checksum_2 = importer::calculate_checksum(&file_contents_2);

    importer
        .import_obj(&path_1)
        .expect("Valid object should be loaded");
    importer
        .import_obj(&path_2)
        .expect("Valid object should be loaded");

    assert!(importer.is_cached(&path_1, checksum_1));
    assert!(importer.is_cached(&path_2, checksum_2));
}

#[test]
fn test_returns_correct_models_when_importing_two_different_files_with_the_same_contents() {
    let mut importer = Importer::new();
    let path_1 = "tests/fixtures/valid.obj".to_string();
    let path_2 = "tests/fixtures/valid_copy.obj".to_string();

    let models_1 = importer
        .import_obj(&path_1)
        .expect("Valid object should be loaded");
    let models_2 = importer
        .import_obj(&path_2)
        .expect("Valid object should be loaded");

    assert_eq!(models_1, models_2);
}

#[test]
fn test_returns_error_when_invalid_unicode_character_is_encountered_in_obj_file() {
    let mut importer = Importer::new();

    let error = importer
        .import_obj(&"tests/fixtures/invalid_unicode.obj")
        .expect_err("Error should be thrown");

    assert_eq!(error, ImporterError::InvalidStructure);
}
