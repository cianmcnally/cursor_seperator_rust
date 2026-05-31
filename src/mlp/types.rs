// ── mlp::types — Strongly-typed data structures for the entire training pipeline ──
//
// Every intermediate from session loading through to evaluation has a named
// struct with compile-time-checked field types.  No serde_json::Value or
// string-keyed dicts anywhere in the training data path.

use std::fmt;
use std::path::PathBuf;
use std::str::FromStr;

use crate::cursor_action::CursorActionKind;

// ── Action label ─────────────────────────────────────────────────────────────
// Wraps the existing CursorActionKind enum so it can round-trip through
// numeric indices (for .npz feature caches / classifier heads) without
// losing connection to the canonical Rust enum.

/// Typed action label that directly mirrors `CursorActionKind`.
/// The numeric index (`as_index`) is the classifier head's class index.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct ActionLabel(pub CursorActionKind);

impl ActionLabel {
    /// All seven canonical action labels in the fixed order used by the
    /// classifier head (and the Python pipeline after `sorted(keep)`).
    pub const ALL: [ActionLabel; 7] = [
        ActionLabel(CursorActionKind::Idle),
        ActionLabel(CursorActionKind::Move),
        ActionLabel(CursorActionKind::Click),
        ActionLabel(CursorActionKind::DoubleClick),
        ActionLabel(CursorActionKind::Drag),
        ActionLabel(CursorActionKind::Scroll),
        ActionLabel(CursorActionKind::Typing),
    ];

    /// Classifier-head index for this label.
    pub fn as_index(&self) -> usize {
        match self.0 {
            CursorActionKind::Idle => 0,
            CursorActionKind::Move => 1,
            CursorActionKind::Click => 2,
            CursorActionKind::DoubleClick => 3,
            CursorActionKind::Drag => 4,
            CursorActionKind::Scroll => 5,
            CursorActionKind::Typing => 6,
        }
    }

    /// Build an ActionLabel from a classifier-head index.
    pub fn from_index(idx: usize) -> Option<Self> {
        match idx {
            0 => Some(ActionLabel(CursorActionKind::Idle)),
            1 => Some(ActionLabel(CursorActionKind::Move)),
            2 => Some(ActionLabel(CursorActionKind::Click)),
            3 => Some(ActionLabel(CursorActionKind::DoubleClick)),
            4 => Some(ActionLabel(CursorActionKind::Drag)),
            5 => Some(ActionLabel(CursorActionKind::Scroll)),
            6 => Some(ActionLabel(CursorActionKind::Typing)),
            _ => None,
        }
    }

    /// Number of distinct action classes.
    pub const N_CLASSES: usize = 7;

    /// Lowercase string used in ndjson / session files.  Mirrors the Python
    /// pipeline's cursor_action strings.
    pub fn as_str(&self) -> &'static str {
        match self.0 {
            CursorActionKind::Idle => "idle",
            CursorActionKind::Move => "move",
            CursorActionKind::Click => "click",
            CursorActionKind::DoubleClick => "double_click",
            CursorActionKind::Drag => "drag",
            CursorActionKind::Scroll => "scroll",
            CursorActionKind::Typing => "typing",
        }
    }
}

impl FromStr for ActionLabel {
    type Err = String;
    fn from_str(s: &str) -> Result<Self, Self::Err> {
        match s {
            "idle" => Ok(ActionLabel(CursorActionKind::Idle)),
            "move" => Ok(ActionLabel(CursorActionKind::Move)),
            "click" | "SingleClick" | "JitterClick" => Ok(ActionLabel(CursorActionKind::Click)),
            "double_click" | "DoubleClick" => Ok(ActionLabel(CursorActionKind::DoubleClick)),
            "drag" => Ok(ActionLabel(CursorActionKind::Drag)),
            "scroll" => Ok(ActionLabel(CursorActionKind::Scroll)),
            "typing" => Ok(ActionLabel(CursorActionKind::Typing)),
            other => Err(format!("unknown action label: {other}")),
        }
    }
}

impl fmt::Display for ActionLabel {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{}", self.as_str())
    }
}

// ── Per-frame metadata (extracted from FrameRecord for training) ─────────────

/// The subset of a `FrameRecord` needed by the MLP training pipeline.
/// Intentionally narrow — only fields the model consumes.
#[derive(Debug, Clone)]
pub struct FrameMeta {
    /// Absolute path to the JPEG/PNG frame image.
    pub image_path: PathBuf,
    /// Normalised cursor centre [cx, cy] ∈ [0, 1]².
    pub cursor_pos_norm: [f32; 2],
    /// Action label for this frame.
    pub action: ActionLabel,
    /// Which session directory this frame belongs to (used for clip grouping).
    pub session_id: String,
}

// ── Labeled clip ─────────────────────────────────────────────────────────────

/// A T-frame temporal clip built from consecutive saved frames.
/// The label is taken from the *last* frame (matching Python pipeline).
#[derive(Debug, Clone)]
pub struct LabeledClip {
    /// Absolute paths to the T frame images in temporal order.
    pub frames: Vec<PathBuf>,
    /// Normalised cursor position from the last frame [cx, cy].
    pub cursor_pos: [f32; 2],
    /// Action label from the last frame.
    pub action: ActionLabel,
    /// Session directory name (for debugging / provenance).
    pub session_id: String,
}

// ── Feature cache (loaded from .npz) ─────────────────────────────────────────

/// Contents of the pre-extracted feature cache produced by the Python pipeline.
///
/// Shapes expected:
///   features    [N, D]        pooled V-JEPA features (float32)
///   pos        [N, 2]        normalised cursor positions (float32)
///   action_ids [N]           class indices 0..6 (int64)
#[derive(Debug, Clone)]
pub struct FeatureCache {
    /// Pooled V-JEPA feature vectors, shape [N_clips, D].
    pub features: Vec<Vec<f32>>,
    /// Normalised cursor positions, shape [N_clips, 2].
    pub pos: Vec<[f32; 2]>,
    /// Action class indices, length N_clips.
    pub action_ids: Vec<i32>,
    /// Action labels in classifier-head order (e.g. ["idle", "move", …]).
    pub action_labels: Vec<ActionLabel>,
}

impl FeatureCache {
    /// Number of clips in the cache.
    pub fn len(&self) -> usize {
        self.features.len()
    }

    /// Whether the cache is empty.
    pub fn is_empty(&self) -> bool {
        self.features.is_empty()
    }
}

// ── Evaluation metrics ───────────────────────────────────────────────────────

/// Per-split metrics computed after each evaluation pass.
#[derive(Debug, Clone, Default)]
pub struct EvalMetrics {
    /// Mean total loss (MSE + weighted CE).
    pub loss: f32,
    /// Root-mean-square position error (Euclidean distance in [0,1]²).
    pub pos_error: f32,
    /// Action classification accuracy (fraction correct).
    pub action_accuracy: f32,
}

// ── Training state ───────────────────────────────────────────────────────────

/// Tracks best-model selection and final results.
#[derive(Debug, Clone)]
pub struct TrainState {
    /// Lowest validation loss seen so far.
    pub best_val_loss: f32,
    /// Epoch at which best_val_loss was achieved (1-based).
    pub best_epoch: usize,
    /// Path to the saved weights file.
    pub weights_path: String,
}
