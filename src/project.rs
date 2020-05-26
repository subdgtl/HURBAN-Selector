use std::error;
use std::fmt;
use std::fs::File;
use std::io::prelude::*;
use std::io::{self, BufReader};
use std::path::{Path, PathBuf};

use serde::Serialize as _;

use crate::interpreter::ast;

pub const DEFAULT_NEW_FILENAME: &str = "new_project.hurban";

pub const EXTENSION: &str = "hurban";
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
    pub path: Option<PathBuf>,
    pub error: Option<ProjectError>,
    pub new_requested: bool,
    pub open_requested: bool,
    pub changed_since_last_save: bool,
    pub prevent_overwrite_status: Option<NextAction>,
}

impl ProjectStatus {
    pub fn save<P: AsRef<Path>>(&mut self, path: P) {
        self.path = Some(path.as_ref().to_path_buf());
        self.changed_since_last_save = false;
    }
}

#[derive(Debug, Clone)]
pub enum ProjectError {
    SerializeError(ron::error::Error),
    FileNotFound,
    PermissionDenied,
    UnexpectedError,
}

impl error::Error for ProjectError {}

impl fmt::Display for ProjectError {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        match self {
            ProjectError::SerializeError(err) => write!(
                f,
                "An error occurred while serializing or deserializing project file: {}",
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

impl From<ron::error::Error> for ProjectError {
    fn from(err: ron::error::Error) -> Self {
        ProjectError::SerializeError(err)
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

/// Saves project to given path. If this path does not contain valid project
/// extension, it is automatically added.
///
/// Returns `PathBuf` which can be different than original path if the project
/// extension was added.
pub fn save<P: AsRef<Path>>(path: P, project: Project) -> Result<PathBuf, ProjectError> {
    let mut path_buf = path.as_ref().to_path_buf();
    match path_buf.extension() {
        Some(extension) => {
            let extension = extension.to_string_lossy().into_owned();

            if extension != EXTENSION {
                path_buf.set_extension(format!("{}.{}", extension, EXTENSION));
            }
        }
        None => {
            path_buf.set_extension(EXTENSION);
        }
    }

    let mut output: Vec<u8> = Vec::new();

    let pretty_config = ron::ser::PrettyConfig::new()
        .with_indentor("  ".to_string())
        .with_new_line("\n".to_string())
        .with_separate_tuple_members(false)
        .with_enumerate_arrays(false);
    let mut serializer = ron::ser::Serializer::new(&mut output, Some(pretty_config), true)?;

    project.serialize(&mut serializer)?;

    let mut file = File::create(path_buf.as_path())?;
    file.write_all(&output)?;
    file.flush()?;

    Ok(path_buf)
}

pub fn open<P: AsRef<Path>>(path: P) -> Result<Project, ProjectError> {
    let file = File::open(path)?;
    let buf_reader = BufReader::new(file);
    let project = ron::de::from_reader(buf_reader)?;

    Ok(project)
}
