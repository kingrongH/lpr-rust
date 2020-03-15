use tensorflow::Status;
use rusttype::Error as FontError;

use std::error::Error;
use std::fmt;
use std::io::Error as IOError;

#[derive(Debug)]
pub struct LprError(LprErrorKind);

#[derive(Debug)]
pub enum LprErrorKind {
    IOError(IOError),
    TensorflowError(Status),
    FontError(FontError),
}

impl LprError {
    fn kind(&self) -> &LprErrorKind {
        &self.0
    }
}

impl<T> From<T> for LprError
where T:  Into<LprErrorKind>
{
    fn from(e: T) -> Self {
        Self(e.into())
    }
}

impl fmt::Display for LprError {

    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self.kind() {
            LprErrorKind::IOError(e) => e.fmt(f),
            LprErrorKind::TensorflowError(e) => e.fmt(f),
            LprErrorKind::FontError(e) => e.fmt(f),
        }
    }
}

impl Error for LprError {
    fn source(&self) -> Option<&(dyn Error + 'static)> {
        match self.kind() {
            LprErrorKind::IOError(e) => e.source(),
            LprErrorKind::FontError(e) => e.source(),
            LprErrorKind::TensorflowError(e) => e.source(),
        }
    }
}

impl From<IOError> for LprErrorKind {
    fn from(e: IOError) -> Self {
        Self::IOError(e)
    }
}

impl From<Status> for LprErrorKind {
    fn from(e: Status) -> Self {
        Self::TensorflowError(e)
    }
}

impl From<FontError> for LprErrorKind {
    fn from(e: FontError) -> Self {
        Self::FontError(e)
    }
}
