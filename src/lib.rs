// ── rust_cursor_bench library ─────────────────────────────────────────────────
//
// Shared library used by the main binary and the analysis binaries
// (model_input_viz, prediction_viz). Rerun is the single viewer, always built in.

pub mod coords;
pub mod cursor_action;
pub mod debug_viz;
pub mod mlp;
pub mod predictions;
pub mod recorder;
pub mod session_loader;
pub mod typing;
pub mod windows;
