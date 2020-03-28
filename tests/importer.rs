use std::fs;

use hurban_selector::importer::{
    self, EndlessCache, Importer, ImporterError, InvalidStructureError,
};

fn import_obj(path: &str) -> Vec<importer::Model> {
    let file_contents = fs::read(&path).expect("File should be read to bytes");
    let (tobj_models, _) =
        importer::obj_buf_into_tobj(&mut file_contents.as_slice()).expect("Obj should be parsed");

    importer::tobj_to_internal(tobj_models)
        .expect("Failed to convert tobj representation to internal one.")
}

#[test]
fn test_importer_import_obj_returns_valid_obj_file() {
    let cache = EndlessCache::default();
    let mut importer = Importer::new(cache);
    let path = "tests/fixtures/valid.obj";

    let models = importer
        .import_obj(&path)
        .expect("Valid obj should be loaded");
    let expected_models = import_obj(&path);

    assert_eq!(expected_models, models);
}

#[test]
fn test_importer_import_obj_returns_two_different_valid_obj_files() {
    let cache = EndlessCache::default();
    let mut importer = Importer::new(cache);
    let path_1 = "tests/fixtures/valid.obj";
    let path_2 = "tests/fixtures/valid_2.obj";

    let models_1 = importer
        .import_obj(&path_1)
        .expect("Valid obj should be loaded");
    let models_2 = importer
        .import_obj(&path_2)
        .expect("Valid obj should be loaded");
    let expected_models_1 = import_obj(&path_1);
    let expected_models_2 = import_obj(&path_2);

    assert_eq!(expected_models_1, models_1);
    assert_eq!(expected_models_2, models_2);
}

#[test]
fn test_importer_import_obj_returns_error_when_importing_invalid_obj_file() {
    let cache = EndlessCache::default();
    let mut importer = Importer::new(cache);
    let path = "tests/fixtures/invalid.obj";

    let error = importer
        .import_obj(&path)
        .expect_err("Error should be thrown");

    assert_eq!(
        error,
        ImporterError::InvalidStructure(InvalidStructureError::ParsingError)
    );
}

#[test]
fn test_importer_import_obj_returns_error_when_importing_nonexistent_file() {
    let cache = EndlessCache::default();
    let mut importer = Importer::new(cache);
    let path = "tests/fixtures/wrong_path.obj";

    let error = importer
        .import_obj(&path)
        .expect_err("Error should be thrown");

    assert_eq!(error, ImporterError::FileNotFound);
}

#[test]
fn test_importer_import_obj_returns_correct_models_when_importing_the_same_unchanged_file() {
    let cache = EndlessCache::default();
    let mut importer = Importer::new(cache);
    let path = "tests/fixtures/valid.obj";

    let models_1 = importer
        .import_obj(&path)
        .expect("Valid obj should be loaded");
    let models_2 = importer
        .import_obj(&path)
        .expect("Valid obj should be loaded");

    assert_eq!(models_1, models_2);
}

#[test]
fn test_importer_import_obj_returns_correct_models_when_importing_two_different_files_with_the_same_contents(
) {
    let cache = EndlessCache::default();
    let mut importer = Importer::new(cache);
    let path_1 = "tests/fixtures/valid.obj";
    let path_2 = "tests/fixtures/valid_copy.obj";

    let models_1 = importer
        .import_obj(&path_1)
        .expect("Valid obj should be loaded");
    let models_2 = importer
        .import_obj(&path_2)
        .expect("Valid obj should be loaded");

    assert_eq!(models_1, models_2);
}

#[test]
fn test_importer_import_obj_returns_error_when_invalid_unicode_character_is_encountered_in_obj_file(
) {
    let cache = EndlessCache::default();
    let mut importer = Importer::new(cache);

    let error = importer
        .import_obj(&"tests/fixtures/invalid_unicode.obj")
        .expect_err("Error should be thrown");

    assert_eq!(
        error,
        ImporterError::InvalidStructure(InvalidStructureError::ParsingError)
    );
}

#[test]
fn test_importer_import_obj_returns_error_when_empty_model_is_encountered() {
    let cache = EndlessCache::default();
    let mut importer = Importer::new(cache);

    let error = importer
        .import_obj(&"tests/fixtures/empty_model.obj")
        .expect_err("Error should be thrown");

    assert_eq!(
        error,
        ImporterError::InvalidStructure(InvalidStructureError::BlankModel)
    );
}

#[test]
fn test_importer_import_obj_returns_error_when_duplicate_indices_are_encountered() {
    let cache = EndlessCache::default();
    let mut importer = Importer::new(cache);

    let error = importer
        .import_obj(&"tests/fixtures/invalid_geometry_duplicate_indices.obj")
        .expect_err("Error should be thrown");

    assert_eq!(
        error,
        ImporterError::InvalidStructure(InvalidStructureError::DuplicateIndices)
    );
}
