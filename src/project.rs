use std::error;
use std::fmt;
use std::fs::File;
use std::io::prelude::*;
use std::io::{self, BufReader};
use std::path::Path;

use ron;
use serde::Serialize;

use crate::interpreter::ast;

pub const DEFAULT_NEW_FILENAME: &str = "new_project.hurban";

pub const EXTENSION_DESCRIPTION: &str = "HURBAN selector project (.hurban)";
pub const EXTENSION_FILTER: &[&str] = &["*.hurban"];

#[derive(Debug, Clone, Copy)]
pub enum NextAction {
    Exit,
    NewProject,
    OpenProject,
}

#[derive(Debug, Default)]
pub struct ProjectStatus {
    pub path: Option<String>,
    pub error: Option<ProjectError>,
    pub new_requested: bool,
    pub open_requested: bool,
    pub changed_since_last_save: bool,
    pub prevent_overwrite_status: Option<NextAction>,
}

impl ProjectStatus {
    pub fn save(&mut self, path: &str) {
        self.path = Some(path.to_string());
        self.changed_since_last_save = false;
    }
}

#[derive(Debug, Clone)]
pub enum ProjectError {
    SerializingError(ron::ser::Error),
    DeserializingError(ron::de::Error),
    FileNotFound,
    PermissionDenied,
    UnexpectedError,
}

impl error::Error for ProjectError {}

impl fmt::Display for ProjectError {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        match self {
            ProjectError::SerializingError(err) => write!(
                f,
                "An error occurred while serializing project file: {}",
                err
            ),
            ProjectError::DeserializingError(err) => write!(
                f,
                "An error occurred while deserializing project file: {}",
                err
            ),
            ProjectError::FileNotFound => write!(f, "File was not found."),
            ProjectError::PermissionDenied => {
                write!(f, "Permission denied while accessing the file.")
            }
            ProjectError::UnexpectedError => write!(f, "An unexpected error occurred."),
        }
    }
}

impl From<ron::ser::Error> for ProjectError {
    fn from(err: ron::ser::Error) -> Self {
        ProjectError::SerializingError(err)
    }
}

impl From<ron::de::Error> for ProjectError {
    fn from(err: ron::de::Error) -> Self {
        ProjectError::DeserializingError(err)
    }
}

impl From<io::Error> for ProjectError {
    fn from(err: io::Error) -> Self {
        match err.kind() {
            io::ErrorKind::NotFound => ProjectError::FileNotFound,
            io::ErrorKind::PermissionDenied => ProjectError::PermissionDenied,
            _ => ProjectError::UnexpectedError,
        }
    }
}

#[derive(Debug, serde::Serialize, serde::Deserialize)]
pub struct Project {
    pub version: u32,
    pub stmts: Vec<ast::Stmt>,
}

pub fn save<P: AsRef<Path>>(path: P, project: Project) -> Result<(), ProjectError> {
    let pretty_config = ron::ser::PrettyConfig::default();
    let mut serializer = ron::ser::Serializer::new(Some(pretty_config), true);
    project.serialize(&mut serializer)?;

    let contents = serializer.into_output_string();
    let mut file = File::create(path)?;

    file.write_all(contents.as_bytes())?;

    Ok(())
}

pub fn open<P: AsRef<Path>>(path: P) -> Result<Project, ProjectError> {
    let file = File::open(path)?;
    let buf_reader = BufReader::new(file);
    let project = ron::de::from_reader(buf_reader)?;

    Ok(project)
}
