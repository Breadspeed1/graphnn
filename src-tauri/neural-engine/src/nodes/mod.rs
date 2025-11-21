use std::{collections::HashMap, ops::Deref, rc::Rc};

use serde::{Deserialize, Serialize};

#[derive(Serialize, Deserialize)]
pub struct GraphBlueprint {
    pub nodes: HashMap<Id, NodeConfig>,
    pub edges: Vec<Edge>,
}

//TODO: make this an enum dispatch to a config trait
#[derive(Serialize, Deserialize, Clone)]
pub enum NodeConfig {}

#[derive(Serialize, Deserialize, Clone, PartialEq, Eq, Hash)]
pub struct Id(Rc<str>);

#[derive(Serialize, Deserialize, Clone)]
pub struct Edge {
    source: HandleRef,
    target: HandleRef,
    id: Id,
}

#[derive(Serialize, Deserialize, Clone)]
pub struct HandleRef {
    node_id: Id,
    handle_id: Id,
}

impl Deref for Id {
    type Target = Rc<str>;

    fn deref(&self) -> &Self::Target {
        &self.0
    }
}
