use serde::{Deserialize, Serialize};

use crate::nodes::{Id, TensorShape};

#[derive(Serialize, Deserialize, Clone, Debug)]
pub enum LintError {
    Cycle { edges: Vec<Id> },
    MismatchedTypes { edge: Id },
    UnInferrableTensorShape { edge: Id },
}

#[derive(Serialize, Deserialize, Clone, Debug, Default)]
pub struct LintOutput {
    pub shapes: Vec<(Id, TensorShape)>,
    pub errors: Vec<LintError>,
}

impl LintOutput {
    fn push_shape(&mut self, id: Id, shape: TensorShape) {
        self.shapes.push((id, shape));
    }

    fn push_error(&mut self, error: LintError) {
        self.errors.push(error);
    }
}
