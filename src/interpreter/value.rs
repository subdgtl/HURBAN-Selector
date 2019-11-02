use std::fmt;
use std::sync::Arc;

use crate::geometry::Geometry;

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
    Float3,
    Geometry,
}

impl fmt::Display for Ty {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        match self {
            Ty::Nil => f.write_str("Nil"),
            Ty::Boolean => f.write_str("Boolean"),
            Ty::Int => f.write_str("Int"),
            Ty::Uint => f.write_str("Uint"),
            Ty::Float => f.write_str("Float"),
            Ty::Float3 => f.write_str("Float3"),
            Ty::Geometry => f.write_str("Geometry"),
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
    Float3([f32; 3]),
    Geometry(Arc<Geometry>),
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
            Value::Float3(_) => Ty::Float3,
            Value::Geometry(_) => Ty::Geometry,
        }
    }

    /// Get the value if boolean, otherwise `None`.
    ///
    /// Useful for getting a value of an optional parameter.
    #[allow(dead_code)]
    pub fn get_boolean(&self) -> Option<bool> {
        match self {
            Value::Boolean(boolean) => Some(*boolean),
            _ => None,
        }
    }

    /// Get the value if int, otherwise `None`.
    ///
    /// Useful for getting a value of an optional parameter.
    #[allow(dead_code)]
    pub fn get_int(&self) -> Option<i32> {
        match self {
            Value::Int(int) => Some(*int),
            _ => None,
        }
    }

    /// Get the value if uint, otherwise `None`.
    ///
    /// Useful for getting a value of an optional parameter.
    #[allow(dead_code)]
    pub fn get_uint(&self) -> Option<u32> {
        match self {
            Value::Uint(uint) => Some(*uint),
            _ => None,
        }
    }

    /// Get the value if float, otherwise `None`.
    ///
    /// Useful for getting a value of an optional parameter.
    #[allow(dead_code)]
    pub fn get_float(&self) -> Option<f32> {
        match self {
            Value::Float(float) => Some(*float),
            _ => None,
        }
    }

    /// Get the value if float3, otherwise `None`.
    ///
    /// Useful for getting a value of an optional parameter.
    pub fn get_float3(&self) -> Option<[f32; 3]> {
        match self {
            Value::Float3(flaot3) => Some(*flaot3),
            _ => None,
        }
    }

    /// Get the value if geometry, otherwise `None`.
    ///
    /// Useful for getting a value of an optional parameter.
    #[allow(dead_code)]
    pub fn get_geometry(&self) -> Option<&Geometry> {
        match self {
            Value::Geometry(geometry_ptr) => Some(geometry_ptr),
            _ => None,
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

    /// Get the value if float3, otherwise panic.
    ///
    /// # Panics
    /// This function panics when value is not a float3.
    #[allow(dead_code)]
    pub fn unwrap_float3(&self) -> [f32; 3] {
        match self {
            Value::Float3(float3) => *float3,
            _ => panic!("Value not float3"),
        }
    }

    /// Get the value if geometry, otherwise panic.
    ///
    /// # Panics
    /// This function panics when value is not a geometry.
    pub fn unwrap_geometry(&self) -> &Geometry {
        match self {
            Value::Geometry(geometry_ptr) => geometry_ptr,
            _ => panic!("Value not geometry"),
        }
    }

    /// Get the refcounted value if geometry, otherwise panic.
    ///
    /// # Panics
    /// This function panics when value is not a geometry.
    pub fn unwrap_refcounted_geometry(&self) -> Arc<Geometry> {
        match self {
            Value::Geometry(geometry_ptr) => Arc::clone(geometry_ptr),
            _ => panic!("Value not geometry"),
        }
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
            Value::Float3(float3) => {
                write!(f, "<float3 [{}, {}, {}]>", float3[0], float3[1], float3[2])
            }
            Value::Geometry(geometry) => {
                write!(f, "<geometry (vertices: {})>", geometry.vertices().len())
            }
        }
    }
}
