use serde::{Deserialize, Serialize};

use crate::nodes::{Config, HandleDef};

#[derive(Serialize, Deserialize, Clone)]
pub struct DummyNode;

impl Config for DummyNode {
    fn in_handles(&self) -> Vec<HandleDef> {
        vec![]
    }

    fn out_handles(&self) -> Vec<HandleDef> {
        vec![]
    }
}
