use serde::{Deserialize, Serialize};

use crate::nodes::{ConfigError, HandleRef, Id};

#[derive(Serialize, Deserialize, Clone, Debug)]
pub enum LintError {
    Cycle { edges: Vec<Id> },
    MismatchedTypes { edge: Id },
    UnInferrableTensorShape { handle: HandleRef },
    ConfigError(ConfigError),
}

#[derive(Serialize, Deserialize, Clone, Debug, Default)]
pub struct LintOutput {
    pub errors: Vec<LintError>,
}

impl LintOutput {
    pub fn push_error(&mut self, error: LintError) {
        self.errors.push(error);
    }

    pub fn has_blocking_errors(&self) -> bool {
        return !self.errors.is_empty();
    }
}
