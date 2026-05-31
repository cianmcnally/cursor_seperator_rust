// ── mlp — Multi-task MLP training pipeline for Rust session recordings ─────
//
// Port of train_rust_recording_mlp.py to Rust using candle.
// See /memories/session/plan.md for architecture.

pub mod clips;
pub mod config;
pub mod io;
pub mod model;
pub mod train;
pub mod types;
