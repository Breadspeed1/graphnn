use burn::config::Config;
use schemars::{JsonSchema, Schema, SchemaGenerator};
use serde::{Deserialize, Serialize};

#[derive(Serialize, Deserialize)]
pub struct RegistryEntry {
    schema: Schema,
    name: String,
    description: String,
}

impl RegistryEntry {
    pub fn for_config<C: JsonSchema + Config>(name: String, description: String) -> Self {
        Self {
            name,
            description,
            schema: SchemaGenerator::default().into_root_schema_for::<C>(),
        }
    }
}
