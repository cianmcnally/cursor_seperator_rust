// ── prediction_viz — visualizes model predictions + evaluation metrics ───────
//
// Usage:
//   cargo run --bin prediction_viz -- path/to/session
//
// If path/to/session/predictions.ndjson exists, it loads and visualizes
// model predictions alongside ground truth from the saved session.
//
// Output:
//   path/to/session/debug_predictions.rrd

use std::path::PathBuf;

use rust_cursor_bench::predictions::{load_predictions, compute_eval_metrics};
use rust_cursor_bench::session_loader::LoadedSession;

fn main() {
    let args: Vec<String> = std::env::args().collect();
    if args.len() < 2 {
        eprintln!("Usage: prediction_viz <session_dir>");
        std::process::exit(1);
    }

    let session_dir = PathBuf::from(&args[1]);
    if !session_dir.is_dir() {
        eprintln!("Not a directory: {}", session_dir.display());
        std::process::exit(1);
    }

    let pred_path = session_dir.join("predictions.ndjson");
    if !pred_path.exists() {
        eprintln!("No predictions.ndjson found in {}", session_dir.display());
        eprintln!();
        eprintln!("Expected format (one JSON per line):");
        eprintln!(r#"{{"session_id":"...","clip_id":"clip_0000","frame_idx":42,"clip_frame_offset":3,"predicted_action":"drag","action_probs":{{"idle":0.05,"drag":0.65,...}},"predicted_cursor":{{"x":100,"y":200}},"cursor_heatmap_path":"heatmaps/clip_0000_f03_cursor.png","confidence":0.87,"ground_truth_action":"drag","ground_truth_cursor":{{"x":102,"y":198}},"cursor_distance_px":2.83,"action_correct":true}}"#);
        eprintln!();
        eprintln!("Create this file from your model_runner, then re-run this command.");
        std::process::exit(0);
    }

    // ── Load predictions ────────────────────────────────────────────────
    let predictions = match load_predictions(&pred_path) {
        Ok(p) => {
            println!("Loaded {} predictions from {}", p.len(), pred_path.display());
            p
        }
        Err(e) => {
            eprintln!("Failed to load predictions: {e}");
            std::process::exit(1);
        }
    };

    // ── Load session for ground truth context ───────────────────────────
    let session = LoadedSession::load(&session_dir).ok();

    // ── Create Rerun output ─────────────────────────────────────────────
    let rrd_path = session_dir.join("debug_predictions.rrd");

    let rec = match rerun::RecordingStreamBuilder::new(
        session.as_ref().map(|s| s.meta.session_id.as_str()).unwrap_or("predictions")
    ).save(&rrd_path) {
        Ok(r) => r,
        Err(e) => {
            eprintln!("Failed to create Rerun stream: {e}");
            std::process::exit(1);
        }
    };

    println!("Writing: {}", rrd_path.display());

    // ── Log predictions ─────────────────────────────────────────────────
    for (i, pred) in predictions.iter().enumerate() {
        rec.set_time_sequence("frame_idx", pred.frame_idx as i64);
        rec.set_time_sequence("prediction_idx", i as i64);

        // ── Action probabilities as bar chart ──────────────────────────
        let action_names: Vec<String> = pred.action_probs.keys().cloned().collect();
        let _action_values: Vec<f64> = action_names.iter()
            .map(|k| pred.action_probs.get(k).copied().unwrap_or(0.0))
            .collect();

        // Log as text for now (Rerun bar chart support depends on version)
        let probs_text: String = pred.action_probs.iter()
            .map(|(k, v)| format!("{k}: {v:.3}"))
            .collect::<Vec<_>>()
            .join(", ");

        let _ = rec.log(
            "/model_output/action_probs",
            &rerun::TextLog::new(format!(
                "clip={} frame={}: {}",
                pred.clip_id, pred.frame_idx, probs_text
            )),
        );

        // ── Predicted action label ──────────────────────────────────────
        let _ = rec.log(
            "/model_output/predicted_action",
            &rerun::TextLog::new(format!(
                "clip={} frame={}: predicted={} confidence={:.3}",
                pred.clip_id, pred.frame_idx,
                pred.predicted_action, pred.confidence
            )),
        );

        // ── Predicted cursor ────────────────────────────────────────────
        let cx = pred.predicted_cursor.x as f32;
        let cy = pred.predicted_cursor.y as f32;

        let _ = rec.log(
            "/model_output/predicted_cursor",
            &rerun::Points2D::new([(cx, cy)])
                .with_labels([format!("pred: ({:.0},{:.0})", cx, cy)])
                .with_radii([6.0]),
        );

        // ── Ground truth cursor (if available) ──────────────────────────
        if let Some(gt) = &pred.ground_truth_cursor {
            let gx = gt.x as f32;
            let gy = gt.y as f32;

            let _ = rec.log(
                "/model_output/predicted_cursor",
                &rerun::Points2D::new([(gx, gy)])
                    .with_labels([format!("gt: ({:.0},{:.0})", gx, gy)])
                    .with_radii([4.0]),
            );
        }

        // ── Confidence ──────────────────────────────────────────────────
        let _ = rec.log(
            "/model_output/confidence",
            &rerun::Scalars::new([pred.confidence]),
        );

        // ── Cursor heatmap path reference ───────────────────────────────
        if let Some(ref path) = pred.cursor_heatmap_path {
            let _ = rec.log(
                "/model_output/cursor_heatmap",
                &rerun::TextLog::new(format!("heatmap: {path}")),
            );
        }

        // ── Evaluation metrics ──────────────────────────────────────────
        if let Some(correct) = pred.action_correct {
            let _ = rec.log(
                "/eval/action_correct",
                &rerun::Scalars::new([if correct { 1.0f64 } else { 0.0 }]),
            );
        }

        if let Some(dist) = pred.cursor_distance_px {
            let _ = rec.log(
                "/eval/cursor_distance_px",
                &rerun::Scalars::new([dist]),
            );
        }
    }

    // ── Evaluation summary ──────────────────────────────────────────────
    let summary = compute_eval_metrics(&predictions);

    rec.set_time_sequence("frame_idx", 0);

    let _ = rec.log(
        "/eval/frame_accuracy",
        &rerun::Scalars::new([summary.action_accuracy]),
    );

    let _ = rec.log(
        "/eval/cursor_iou",
        &rerun::Scalars::new([summary.avg_cursor_distance_px]),
    );

    // Action confusion matrix as text
    for (gt, preds) in &summary.action_confusion {
        for (pred, count) in preds {
            let _ = rec.log(
                "/eval/action_confusion",
                &rerun::TextLog::new(format!("gt={gt} pred={pred} count={count}")),
            );
        }
    }

    // Summary
    let _ = rec.log(
        "/eval/warnings",
        &rerun::TextLog::new(format!(
            "Summary: {} predictions, accuracy={:.1}%, avg_cursor_dist={:.1}px",
            summary.total_predictions,
            summary.action_accuracy * 100.0,
            summary.avg_cursor_distance_px,
        )),
    );

    // ── Flush ───────────────────────────────────────────────────────────
    let _ = rec.flush_blocking();

    println!();
    println!("┌───────────────────────────────────────────────────────────┐");
    println!("│ EVALUATION SUMMARY                                        │");
    println!("├───────────────────────────────────────────────────────────┤");
    println!("│ Total predictions:  {:>5}                                │", summary.total_predictions);
    println!("│ Action accuracy:    {:>5.1}%                              │", summary.action_accuracy * 100.0);
    println!("│ Avg cursor dist:    {:>5.1} px                            │", summary.avg_cursor_distance_px);
    println!("└───────────────────────────────────────────────────────────┘");

    if !summary.action_confusion.is_empty() {
        println!();
        println!("Action confusion matrix:");
        for (gt, preds) in &summary.action_confusion {
            for (pred, count) in preds {
                println!("  {gt:>12} → {pred:<12} : {count}");
            }
        }
    }

    println!();
    println!("Open with: rerun {}", rrd_path.display());
}
