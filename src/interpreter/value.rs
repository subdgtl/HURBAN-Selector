use std::fmt;
use std::ops::Deref;
use std::sync::Arc;

use crate::convert::{cast_u32, cast_usize};
use crate::mesh::Mesh;

/// A type of a value.
///
/// Declared as inputs and outputs in functions. Checked dynamically
/// (and one day maybe also statically) by the interpreter.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Ty {
    Nil,
    Boolean,
    Int,
    Uint,
    Float,
    Float2,
    Float3,
    String,
    Mesh,
    MeshArray,
}

impl fmt::Display for Ty {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        match self {
            Ty::Nil => f.write_str("Nil"),
            Ty::Boolean => f.write_str("Boolean"),
            Ty::Int => f.write_str("Int"),
            Ty::Uint => f.write_str("Uint"),
            Ty::Float => f.write_str("Float"),
            Ty::Float2 => f.write_str("Float2"),
            Ty::Float3 => f.write_str("Float3"),
            Ty::String => f.write_str("String"),
            Ty::Mesh => f.write_str("Mesh"),
            Ty::MeshArray => f.write_str("MeshArray"),
        }
    }
}

/// A value as seen by the interpreter.
#[derive(Debug, Clone, PartialEq)]
pub enum Value {
    Nil,
    Boolean(bool),
    Int(i32),
    Uint(u32),
    Float(f32),
    Float2([f32; 2]),
    Float3([f32; 3]),
    String(String),
    Mesh(Arc<Mesh>),
    MeshArray(Arc<MeshArrayValue>),
}

impl Value {
    /// Get the value's type.
    pub fn ty(&self) -> Ty {
        match self {
            Value::Nil => Ty::Nil,
            Value::Boolean(_) => Ty::Boolean,
            Value::Int(_) => Ty::Int,
            Value::Uint(_) => Ty::Uint,
            Value::Float(_) => Ty::Float,
            Value::Float2(_) => Ty::Float2,
            Value::Float3(_) => Ty::Float3,
            Value::String(_) => Ty::String,
            Value::Mesh(_) => Ty::Mesh,
            Value::MeshArray(_) => Ty::MeshArray,
        }
    }

    /// Get the value if boolean, otherwise panic.
    ///
    /// # Panics
    /// This function panics when value is not a boolean.
    #[allow(dead_code)]
    pub fn unwrap_boolean(&self) -> bool {
        match self {
            Value::Boolean(boolean) => *boolean,
            _ => panic!("Value not boolean"),
        }
    }

    /// Get the value if int, otherwise panic.
    ///
    /// # Panics
    /// This function panics when value is not an int.
    #[allow(dead_code)]
    pub fn unwrap_int(&self) -> i32 {
        match self {
            Value::Int(int) => *int,
            _ => panic!("Value not int"),
        }
    }

    /// Get the value if uint, otherwise panic.
    ///
    /// # Panics
    /// This function panics when value is not an uint.
    pub fn unwrap_uint(&self) -> u32 {
        match self {
            Value::Uint(uint) => *uint,
            _ => panic!("Value not uint"),
        }
    }

    /// Get the value if float, otherwise panic.
    ///
    /// # Panics
    /// This function panics when value is not a float.
    pub fn unwrap_float(&self) -> f32 {
        match self {
            Value::Float(float) => *float,
            _ => panic!("Value not float"),
        }
    }

    /// Get the value if float2, otherwise panic.
    ///
    /// # Panics
    /// This function panics when value is not a float2.
    pub fn unwrap_float2(&self) -> [f32; 2] {
        match self {
            Value::Float2(float2) => *float2,
            _ => panic!("Value not float2"),
        }
    }

    /// Get the value if float3, otherwise panic.
    ///
    /// # Panics
    /// This function panics when value is not a float3.
    pub fn unwrap_float3(&self) -> [f32; 3] {
        match self {
            Value::Float3(float3) => *float3,
            _ => panic!("Value not float3"),
        }
    }

    /// Get the value if string, otherwise panic.
    ///
    /// # Panics
    /// This function panics when value is not a string.
    pub fn unwrap_string(&self) -> &str {
        match self {
            Value::String(string) => string,
            _ => panic!("Value not string"),
        }
    }

    /// Get the value if mesh, otherwise panic.
    ///
    /// # Panics
    /// This function panics when value is not a mesh.
    pub fn unwrap_mesh(&self) -> &Mesh {
        match self {
            Value::Mesh(mesh_ptr) => mesh_ptr,
            _ => panic!("Value not mesh"),
        }
    }

    /// Get the refcounted value if mesh, otherwise panic.
    ///
    /// # Panics
    /// This function panics when value is not a mesh.
    pub fn unwrap_refcounted_mesh(&self) -> Arc<Mesh> {
        match self {
            Value::Mesh(mesh_ptr) => Arc::clone(mesh_ptr),
            _ => panic!("Value not mesh"),
        }
    }

    /// Get the value if mesh array, otherwise panic.
    ///
    /// # Panics
    /// This function panics when value is not a mesh array.
    pub fn unwrap_mesh_array(&self) -> &MeshArrayValue {
        match self {
            Value::MeshArray(mesh_array_ptr) => mesh_array_ptr,
            _ => panic!("Value not mesh array"),
        }
    }
}

#[derive(Debug, Clone, PartialEq)]
pub struct MeshArrayValue(Vec<Arc<Mesh>>);

impl MeshArrayValue {
    pub fn new(meshes: Vec<Arc<Mesh>>) -> Self {
        Self(meshes)
    }

    pub fn get_refcounted(&self, index: u32) -> Option<Arc<Mesh>> {
        self.0.get(cast_usize(index)).map(Arc::clone)
    }

    pub fn len(&self) -> u32 {
        cast_u32(self.0.len())
    }

    pub fn is_empty(&self) -> bool {
        self.0.is_empty()
    }

    pub fn iter_refcounted<'a>(&'a self) -> impl Iterator<Item = Arc<Mesh>> + 'a {
        self.0.iter().cloned()
    }

    pub fn iter<'a>(&'a self) -> impl Iterator<Item = &'a Mesh> + 'a {
        self.0
            .iter()
            .map(|refcounted_mesh| Arc::deref(refcounted_mesh))
    }
}

impl fmt::Display for Value {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        match self {
            Value::Nil => f.write_str("<nil>"),
            Value::Boolean(boolean) => write!(f, "<boolean {}>", boolean),
            Value::Int(int) => write!(f, "<int {}>", int),
            Value::Uint(uint) => write!(f, "<uint {}>", uint),
            Value::Float(float) => write!(f, "<float {}>", float),
            Value::Float2(float2) => write!(f, "<float2 [{}, {}]>", float2[0], float2[1]),
            Value::Float3(float3) => {
                write!(f, "<float3 [{}, {}, {}]>", float3[0], float3[1], float3[2])
            }
            Value::String(string) => write!(f, "<string {}>", string),
            Value::Mesh(mesh) => {
                let vertex_count = mesh.vertices().len();
                let face_count = mesh.faces().len();

                write!(
                    f,
                    "<mesh (vertices: {}, faces: {})>",
                    vertex_count, face_count
                )
            }
            Value::MeshArray(mesh_array) => write!(f, "<mesh-array (size: {})>", mesh_array.len()),
        }
    }
}
