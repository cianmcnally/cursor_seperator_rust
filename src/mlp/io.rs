// ── mlp::io — Load .npz feature cache, save config.json ─────────────────────
//
// Reads pre-extracted features from the Python pipeline's .npz output and
// converts them to a strongly-typed FeatureCache.  Also handles writing the
// training config + results to rust_mlp_config.json.

use std::fs;
use std::io::{BufReader, Read};
use std::path::Path;

use crate::mlp::config::FullConfig;
use crate::mlp::types::{ActionLabel, FeatureCache, TrainState};

// ── Load .npz feature cache ─────────────────────────────────────────────────

/// Load a .npz file produced by the Python `extract_features()` function.
///
/// Expected arrays:
///   `X`          — [N, D] float32  pooled V-JEPA features
///   `pos`        — [N, 2] float32  normalised cursor positions
///   `action_ids` — [N]    int64    class indices
pub fn load_feature_cache(npz_path: &Path) -> Result<FeatureCache, String> {
    if !npz_path.exists() {
        return Err(format!(
            "Feature cache not found: {}\n\
             Run the Python feature extraction first, e.g.:\n\
             \n  python train_rust_recording_mlp.py \\\n    \
             --recording-dir recordings/session_* \\\n    \
             --out-dir .artifacts/rust_mlp \\\n    \
             --epochs 0\n\
             \nThen re-run this binary with the same --out-dir.",
            npz_path.display()
        ));
    }

    let file = fs::File::open(npz_path)
        .map_err(|e| format!("open {}: {e}", npz_path.display()))?;
    let reader = BufReader::new(file);

    let mut archive = zip::ZipArchive::new(reader)
        .map_err(|e| format!("read .npz as zip: {e}"))?;

    let mut x_data: Option<Vec<f32>> = None;
    let mut x_shape: Option<Vec<usize>> = None;
    let mut pos_data: Option<Vec<f32>> = None;
    let mut pos_shape: Option<Vec<usize>> = None;
    let mut act_data: Option<Vec<i64>> = None;

    for i in 0..archive.len() {
        let mut entry = archive.by_index(i).map_err(|e| format!("zip entry {i}: {e}"))?;
        let name = entry.name().to_string();

        // Read all bytes
        let mut buf = Vec::new();
        entry.read_to_end(&mut buf).map_err(|e| format!("read {name}: {e}"))?;

        let npy = npyz::NpyFile::new(&buf[..])
            .map_err(|e| format!("parse {name} as npy: {e}"))?;

        let shape: Vec<usize> = npy.shape().iter().map(|&s| s as usize).collect();

        match name.as_str() {
            "X.npy" => {
                x_shape = Some(shape.clone());
                x_data = Some(npy.into_vec::<f32>().map_err(|e| format!("X: {e}"))?);
            }
            "pos.npy" => {
                pos_shape = Some(shape.clone());
                pos_data = Some(npy.into_vec::<f32>().map_err(|e| format!("pos: {e}"))?);
            }
            "action_ids.npy" => {
                act_data = Some(npy.into_vec::<i64>().map_err(|e| format!("action_ids: {e}"))?);
            }
            _ => {
                eprintln!("  [note] skipping unknown .npz entry: {name}");
            }
        }
    }

    let x_shape = x_shape.ok_or("missing X.npy in .npz")?;
    let pos_shape = pos_shape.ok_or("missing pos.npy in .npz")?;
    let x_flat = x_data.ok_or("missing X data")?;
    let pos_flat = pos_data.ok_or("missing pos data")?;
    let act_flat_i64 = act_data.ok_or("missing action_ids data")?;

    if x_shape.len() != 2 {
        return Err(format!("expected X shape [N,D], got {x_shape:?}"));
    }
    if pos_shape.len() != 2 || pos_shape[1] != 2 {
        return Err(format!("expected pos shape [N,2], got {pos_shape:?}"));
    }

    let n = x_shape[0];
    let d = x_shape[1];

    if pos_shape[0] != n {
        return Err(format!(
            "X has {n} samples but pos has {} samples",
            pos_shape[0]
        ));
    }
    if act_flat_i64.len() != n {
        return Err(format!(
            "X has {n} samples but action_ids has {} entries",
            act_flat_i64.len()
        ));
    }

    // Convert action_ids from i64 → i32
    let action_ids: Vec<i32> = act_flat_i64.iter().map(|&a| a as i32).collect();

    // Reshape X into Vec<Vec<f32>>
    let features: Vec<Vec<f32>> = (0..n)
        .map(|i| x_flat[i * d..(i + 1) * d].to_vec())
        .collect();

    // Reshape pos into Vec<[f32; 2]>
    let pos: Vec<[f32; 2]> = (0..n)
        .map(|i| [pos_flat[i * 2], pos_flat[i * 2 + 1]])
        .collect();

    // Determine which action labels are present
    let n_classes = action_ids.iter().max().map(|&m| m as usize + 1).unwrap_or(7);
    let action_labels: Vec<ActionLabel> = (0..n_classes)
        .filter_map(|i| ActionLabel::from_index(i))
        .collect();

    eprintln!(
        "Loaded feature cache: {}  X={:?}  pos={:?}  actions={}",
        npz_path.file_name().unwrap_or_default().to_string_lossy(),
        x_shape,
        pos_shape,
        action_ids.len(),
    );

    Ok(FeatureCache {
        features,
        pos,
        action_ids,
        action_labels,
    })
}

// ── Save config.json ─────────────────────────────────────────────────────────

/// Write the full training config + results to `rust_mlp_config.json`.
pub fn save_config(
    config: &FullConfig,
    state: &TrainState,
    action_labels: &[ActionLabel],
) -> Result<(), String> {
    let config_path = config.config_abs();

    if let Some(parent) = config_path.parent() {
        fs::create_dir_all(parent)
            .map_err(|e| format!("create {}: {e}", parent.display()))?;
    }

    // Build a serializable config matching the Python output structure
    let output = serde_json::json!({
        "model_name":  config.data.model_name,
        "actions":     action_labels.iter().map(|a| a.to_string()).collect::<Vec<_>>(),
        "clip_len":    config.data.clip_len,
        "in_dim":      config.mlp.in_dim,
        "hidden_dim":  config.mlp.hidden_dim,
        "n_actions":   config.mlp.n_actions,
        "pos_weight":  config.train.pos_weight,
        "best_val_loss": state.best_val_loss,
        "best_epoch":  state.best_epoch,
        "weights":     config.weights_file,
        "cache":       config.cache_path.to_string_lossy(),
    });

    let json_str = serde_json::to_string_pretty(&output)
        .map_err(|e| format!("serialize config: {e}"))?;

    fs::write(&config_path, json_str)
        .map_err(|e| format!("write {}: {e}", config_path.display()))?;

    eprintln!("Config:  {}", config_path.display());
    Ok(())
}
