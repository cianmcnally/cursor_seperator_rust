// ── predictions — model output format and loader ─────────────────────────────
//
// Defines the predictions.ndjson format that a separate model_runner should emit.
// The recorder and visualizers do NOT depend on the model. This module only
// defines the data contract and provides a loader for the prediction_viz binary.
//
// Architecture:
//
//   record           → writes canonical session (+ debug.rrd)
//   model_input_viz  → reads saved session → debug_model_input.rrd
//
//   model_runner     → reads saved session → writes predictions.ndjson
//   prediction_viz   → reads predictions.ndjson + session → debug_predictions.rrd
//
// predictions.ndjson format — one JSON object per line:
//
// {
//   "session_id": "session_20260530_114505",
//   "clip_id": "clip_0000",
//   "frame_idx": 42,
//   "clip_frame_offset": 3,
//   "predicted_action": "drag",
//   "action_probs": {
//     "idle": 0.05,
//     "move": 0.10,
//     "click": 0.05,
//     "double_click": 0.02,
//     "drag": 0.65,
//     "scroll": 0.08,
//     "typing": 0.05
//   },
//   "predicted_cursor": {"x": 100, "y": 200},
//   "cursor_heatmap_path": "heatmaps/clip_0000_f03_cursor.png",
//   "confidence": 0.87,
//   "ground_truth_action": "drag",
//   "ground_truth_cursor": {"x": 102, "y": 198},
//   "cursor_distance_px": 2.83,
//   "action_correct": true
// }

use serde::{Deserialize, Serialize};
use std::collections::HashMap;

// ── Prediction record ─────────────────────────────────────────────────────────

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PredictionRecord {
    pub session_id: String,
    pub clip_id: String,
    pub frame_idx: u64,
    #[serde(default)]
    pub clip_frame_offset: u64,

    // ── Model outputs ─────────────────────────────────────────────────
    pub predicted_action: String,
    pub action_probs: HashMap<String, f64>,
    pub predicted_cursor: CursorPoint,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub cursor_heatmap_path: Option<String>,
    pub confidence: f64,

    // ── Optional ground truth (for eval) ──────────────────────────────
    #[serde(skip_serializing_if = "Option::is_none")]
    pub ground_truth_action: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub ground_truth_cursor: Option<CursorPoint>,

    // ── Computed metrics ──────────────────────────────────────────────
    #[serde(skip_serializing_if = "Option::is_none")]
    pub cursor_distance_px: Option<f64>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub action_correct: Option<bool>,
}

#[derive(Debug, Clone, Copy, Serialize, Deserialize)]
pub struct CursorPoint {
    pub x: f64,
    pub y: f64,
}

// ── Prediction loader ─────────────────────────────────────────────────────────

/// Load predictions from a predictions.ndjson file.
pub fn load_predictions(path: &std::path::Path) -> Result<Vec<PredictionRecord>, String> {
    use std::io::{BufRead, BufReader};

    if !path.exists() {
        return Err(format!("predictions file not found: {}", path.display()));
    }

    let f = std::fs::File::open(path)
        .map_err(|e| format!("open {}: {e}", path.display()))?;
    let reader = BufReader::new(f);
    let mut out = Vec::new();

    for line in reader.lines() {
        let line = line.map_err(|e| format!("read line: {e}"))?;
        if line.trim().is_empty() { continue; }
        let rec: PredictionRecord = serde_json::from_str(&line)
            .map_err(|e| format!("parse predictions.ndjson: {e}"))?;
        out.push(rec);
    }

    Ok(out)
}

/// Compute evaluation metrics from predictions + ground truth.
pub fn compute_eval_metrics(predictions: &[PredictionRecord]) -> EvalSummary {
    let mut total = 0u64;
    let mut correct_actions = 0u64;
    let mut total_cursor_dist = 0.0f64;
    let mut cursor_samples = 0u64;
    let mut action_confusion: HashMap<String, HashMap<String, u64>> = HashMap::new();

    for p in predictions {
        total += 1;

        if let Some(correct) = p.action_correct {
            if correct {
                correct_actions += 1;
            }
        } else if let Some(gt) = &p.ground_truth_action {
            if gt == &p.predicted_action {
                correct_actions += 1;
            }
        }

        // Confusion matrix
        if let Some(gt) = &p.ground_truth_action {
            action_confusion
                .entry(gt.clone())
                .or_default()
                .entry(p.predicted_action.clone())
                .and_modify(|c| *c += 1)
                .or_insert(1);
        }

        if let Some(dist) = p.cursor_distance_px {
            total_cursor_dist += dist;
            cursor_samples += 1;
        }
    }

    let action_accuracy = if total > 0 {
        correct_actions as f64 / total as f64
    } else {
        0.0
    };

    let avg_cursor_dist = if cursor_samples > 0 {
        total_cursor_dist / cursor_samples as f64
    } else {
        0.0
    };

    EvalSummary {
        total_predictions: total,
        correct_actions,
        action_accuracy,
        avg_cursor_distance_px: avg_cursor_dist,
        action_confusion,
    }
}

#[derive(Debug, Clone, Serialize)]
pub struct EvalSummary {
    pub total_predictions: u64,
    pub correct_actions: u64,
    pub action_accuracy: f64,
    pub avg_cursor_distance_px: f64,
    pub action_confusion: HashMap<String, HashMap<String, u64>>,
}
