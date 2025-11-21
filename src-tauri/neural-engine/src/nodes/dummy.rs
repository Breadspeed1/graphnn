use serde::{Deserialize, Serialize};

use crate::nodes::{Config, DataType, HandleDef, Id};

#[derive(Serialize, Deserialize, Clone, Debug)]
pub struct DummyNode;

impl Config for DummyNode {
    fn in_handles(&self) -> Vec<HandleDef> {
        vec![HandleDef {
            id: Id::from("in_1"),
            dtype: DataType::Float,
        }]
    }

    fn out_handles(&self) -> Vec<HandleDef> {
        vec![HandleDef {
            id: Id::from("out_1"),
            dtype: DataType::Float,
        }]
    }
}
