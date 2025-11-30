use std::collections::HashMap;

use schemars::JsonSchema;
use serde::{Deserialize, Serialize};

use crate::nodes::{Config, ConfigError, DataType, HandleDef, Id, TensorShape};

#[derive(Serialize, Deserialize, Clone, Debug, Default, JsonSchema)]
pub struct DummyNode {
    pub fixed_output: Option<TensorShape>,
}

impl Config for DummyNode {
    fn in_handles(&self) -> Vec<HandleDef> {
        vec![HandleDef {
            id: Id::from("in_1"),
            dtype: DataType::FloatTensor,
        }]
    }

    fn out_handles(&self) -> Vec<HandleDef> {
        vec![HandleDef {
            id: Id::from("out_1"),
            dtype: DataType::FloatTensor,
        }]
    }

    fn infer_output_shapes(
        &self,
        input_shapes: &HashMap<Id, TensorShape>,
    ) -> Result<HashMap<Id, TensorShape>, ConfigError> {
        if let Some(shape) = &self.fixed_output {
            return Ok(HashMap::from([(Id::from("out_1"), shape.clone())]));
        }

        let in_shape = input_shapes.get(&Id::from("in_1"));
        Ok(HashMap::from([(
            Id::from("out_1"),
            in_shape
                .ok_or(ConfigError::MissingInputShape(Id::from("in_1")))?
                .clone(),
        )]))
    }
}
