use bitflags::bitflags;

use super::{FuncError, LogMessage, Ty, Value};

/// Textual information about the function.
pub struct FuncInfo {
    /// The function's name.
    pub name: &'static str,
    /// The name of the function's return value.
    pub return_value_name: &'static str,
}

bitflags! {
    /// Information about the function behavior.
    ///
    /// Unset bits are always the safe default. Set bits may trigger
    /// interpreter optimizations. Incorrectly set bits may result in
    /// bugs.
    pub struct FuncFlags: u8 {
        /// The function is pure (referentially transparent), if its
        /// result only depends on its arguments. Pure functions may
        /// not be re-run by the interpreter, if their inputs did not
        /// change. This flag triggers optimizations. Setting it
        /// incorrectly might produce stale results.
        ///
        /// A pure func's log messages are not returned twice, if the
        /// func's result has been cached and not invalidated.
        const PURE = 0b_0000_0001;
    }
}

/// Information about a function parameter.
#[derive(Debug, Clone, PartialEq)]
pub struct ParamInfo {
    /// The parameter's name
    pub name: &'static str,

    /// Refinement of the parameter type. Can set additional
    /// constraints on the parameter's value, such as a default value
    /// or the value range.
    pub refinement: ParamRefinement,

    /// Whether the parameter is optional. The parameter value is
    /// allowed to have the type [`Nil`] in addition to its own type,
    /// if set to `true`.
    ///
    /// [`Nil`]: ../value.enum.Ty.html#variant.Nil
    pub optional: bool,
}

#[derive(Debug, Clone, Copy, PartialEq)]
pub enum ParamRefinement {
    #[allow(dead_code)]
    Boolean(BooleanParamRefinement),
    #[allow(dead_code)]
    Int(IntParamRefinement),
    Uint(UintParamRefinement),
    Float(FloatParamRefinement),
    Float2(Float2ParamRefinement),
    Float3(Float3ParamRefinement),
    String(StringParamRefinement),
    Mesh,
    MeshArray,
}

impl ParamRefinement {
    pub fn ty(&self) -> Ty {
        match self {
            Self::Boolean(_) => Ty::Boolean,
            Self::Int(_) => Ty::Int,
            Self::Uint(_) => Ty::Uint,
            Self::Float(_) => Ty::Float,
            Self::Float2(_) => Ty::Float2,
            Self::Float3(_) => Ty::Float3,
            Self::String(_) => Ty::String,
            Self::Mesh => Ty::Mesh,
            Self::MeshArray => Ty::MeshArray,
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub struct BooleanParamRefinement {
    pub default_value: bool,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub struct IntParamRefinement {
    pub default_value: Option<i32>,
    pub min_value: Option<i32>,
    pub max_value: Option<i32>,
}

impl IntParamRefinement {
    pub fn clamp(&self, value: i32) -> i32 {
        if let Some(min) = self.min_value {
            if value < min {
                return min;
            }
        }
        if let Some(max) = self.max_value {
            if value > max {
                return max;
            }
        }

        value
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub struct UintParamRefinement {
    pub default_value: Option<u32>,
    pub min_value: Option<u32>,
    pub max_value: Option<u32>,
}

impl UintParamRefinement {
    pub fn clamp(&self, value: u32) -> u32 {
        if let Some(min) = self.min_value {
            if value < min {
                return min;
            }
        }
        if let Some(max) = self.max_value {
            if value > max {
                return max;
            }
        }

        value
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Default)]
pub struct FloatParamRefinement {
    pub default_value: Option<f32>,
    pub min_value: Option<f32>,
    pub max_value: Option<f32>,
}

impl FloatParamRefinement {
    pub fn clamp(&self, value: f32) -> f32 {
        if let Some(min) = self.min_value {
            if value < min {
                return min;
            }
        }
        if let Some(max) = self.max_value {
            if value > max {
                return max;
            }
        }

        value
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Default)]
pub struct Float2ParamRefinement {
    pub default_value_x: Option<f32>,
    pub min_value_x: Option<f32>,
    pub max_value_x: Option<f32>,
    pub default_value_y: Option<f32>,
    pub min_value_y: Option<f32>,
    pub max_value_y: Option<f32>,
}

impl Float2ParamRefinement {
    pub fn clamp(&self, value: [f32; 2]) -> [f32; 2] {
        let x = if let Some(min_x) = self.min_value_x {
            if value[0] < min_x {
                min_x
            } else {
                value[0]
            }
        } else {
            value[0]
        };

        let y = if let Some(min_y) = self.min_value_y {
            if value[1] < min_y {
                min_y
            } else {
                value[1]
            }
        } else {
            value[1]
        };

        [x, y]
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Default)]
pub struct Float3ParamRefinement {
    pub default_value_x: Option<f32>,
    pub min_value_x: Option<f32>,
    pub max_value_x: Option<f32>,
    pub default_value_y: Option<f32>,
    pub min_value_y: Option<f32>,
    pub max_value_y: Option<f32>,
    pub default_value_z: Option<f32>,
    pub min_value_z: Option<f32>,
    pub max_value_z: Option<f32>,
}

impl Float3ParamRefinement {
    pub fn clamp(&self, value: [f32; 3]) -> [f32; 3] {
        let x = if let Some(min_x) = self.min_value_x {
            if value[0] < min_x {
                min_x
            } else {
                value[0]
            }
        } else {
            value[0]
        };

        let y = if let Some(min_y) = self.min_value_y {
            if value[1] < min_y {
                min_y
            } else {
                value[1]
            }
        } else {
            value[1]
        };

        let z = if let Some(min_z) = self.min_value_z {
            if value[2] < min_z {
                min_z
            } else {
                value[2]
            }
        } else {
            value[2]
        };

        [x, y, z]
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Default)]
pub struct StringParamRefinement {
    pub default_value: &'static str,
    pub file_path: bool,
    pub file_ext_filter: Option<(&'static [&'static str], &'static str)>,
}

/// An interface describing a function as seen by the interpreter.
///
/// Functions are pieces of callable code. They can receive parameters
/// and must produce a return value, even if [`Nil`].
///
/// [`Nil`]: ../value/enum.Ty.html#variant.Nil
pub trait Func {
    /// Textual information about the function, such as its name and
    /// the name of it's output value.
    fn info(&self) -> &FuncInfo {
        &FuncInfo {
            name: "<Unnamed operation>",
            return_value_name: "<Unnamed value>",
        }
    }

    /// Information about the function behaviour.
    ///
    /// See [`FuncFlags`] for more.
    ///
    /// [`FuncFlags`]: struct.FuncFlags.html
    fn flags(&self) -> FuncFlags;

    /// Information about the function's parameters.
    ///
    /// Used for static and dynamic typecheking. See [`ParamInfo`]
    /// for more.
    ///
    /// [`ParamInfo`]: struct.ParamInfo.html
    fn param_info(&self) -> &[ParamInfo];

    /// Information about the function's return type.
    ///
    /// Used for static and dynamic typecheking. See [`Ty`] for more.
    ///
    /// [`Ty`]: ../value/enum.Ty.html
    fn return_ty(&self) -> Ty;

    /// Call the function with arguments and receive the return value.
    ///
    /// A correct implementation's types provided in [`param_info`]
    /// and [`return_ty`] will match the types of values expected in
    /// and provided by this function.
    ///
    /// [`param_info`]: trait.Func.html#tymethod.param_info
    /// [`return_ty`]: trait.Func.html#tymethod.return_ty
    fn call(&mut self, args: &[Value], log: &mut dyn FnMut(LogMessage))
        -> Result<Value, FuncError>;
}
