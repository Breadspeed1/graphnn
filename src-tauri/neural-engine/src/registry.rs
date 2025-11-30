use std::sync::Arc;

use burn::config::Config;
use schemars::{JsonSchema, Schema, SchemaGenerator};
use serde::{Deserialize, Serialize};

#[derive(Serialize, Deserialize)]
pub struct RegistryEntry {
    schema: Schema,
    name: Arc<str>,
    description: Arc<str>,
}

impl RegistryEntry {
    pub fn for_config<C: JsonSchema + Config>(name: Arc<str>, description: Arc<str>) -> Self {
        Self {
            name,
            description,
            schema: SchemaGenerator::default().into_root_schema_for::<C>(),
        }
    }
}
