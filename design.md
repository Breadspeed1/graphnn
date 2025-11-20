# Design Documentation

## Computation graph structure

The working Rust state will be a hashmap of edges and vertices. Each vertex is a node and each edge is a directed IO connection between two nodes. This mimics the vertex/edge structure of svelte flow. Each node in working state (backend & ui) is just a configuration for a node. This node and edge list is then compiled to a burn module which can be trained/ran/whatever. Compilation errors (verification errors) can be emitted after the UI settles.