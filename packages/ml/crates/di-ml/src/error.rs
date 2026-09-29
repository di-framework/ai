use std::fmt;
use std::io;
use std::path::PathBuf;

/// Library error. CLI maps [`Error::Usage`] to exit status 2 and everything else to 3.
#[derive(Debug)]
pub enum Error {
    Usage(String),
    Fail(String),
    Io(io::Error),
}

impl Error {
    pub fn usage(msg: impl Into<String>) -> Self {
        Self::Usage(msg.into())
    }

    pub fn fail(msg: impl Into<String>) -> Self {
        Self::Fail(msg.into())
    }

    pub fn exit_code(&self) -> i32 {
        match self {
            Self::Usage(_) => 2,
            _ => 3,
        }
    }
}

impl fmt::Display for Error {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Usage(msg) | Self::Fail(msg) => f.write_str(msg),
            Self::Io(err) => write!(f, "{err}"),
        }
    }
}

impl std::error::Error for Error {
    fn source(&self) -> Option<&(dyn std::error::Error + 'static)> {
        match self {
            Self::Io(err) => Some(err),
            _ => None,
        }
    }
}

impl From<io::Error> for Error {
    fn from(err: io::Error) -> Self {
        Self::Io(err)
    }
}

pub type Result<T> = std::result::Result<T, Error>;

pub(crate) fn missing_file(path: PathBuf, hint: &str) -> Error {
    Error::usage(format!("{} is required. {hint}", path.display()))
}
