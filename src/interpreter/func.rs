use bitflags::bitflags;

use super::{FuncError, Ty, Value};

bitflags! {
    /// Information about the function behaviour.
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
        const PURE = 0b_0000_0001;
    }
}

/// Information about a function parameter.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct ParamInfo {
    /// The type the parameter is required to have.
    pub ty: Ty,

    /// Whether the parameter is optional. The parameter value is
    /// allowed to have the type [`Nil`] in addition to its own type,
    /// if set to `true`.
    ///
    /// [`Nil`]: ../value.enum.Ty.html#variant.Nil
    pub optional: bool,
}

/// An interface describing a function as seen by the interpreter.
///
/// Functions are pieces of callable code. They can receive parameters
/// and must produce a return value, even if [`Nil`].
///
/// [`Nil`]: ../value/enum.Ty.html#variant.Nil
pub trait Func {
    /// Information about the function behaviour.
    ///
    /// See [`FuncFlags`] for more.
    ///
    /// [`FuncFlags`]: struct.FuncFlags.html
    fn flags(&self) -> FuncFlags {
        FuncFlags::empty()
    }

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
    fn call(&self, args: &[Value]) -> Result<Value, FuncError>;
}
