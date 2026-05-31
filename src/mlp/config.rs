// ── mlp::config — Typed configuration structs for the MLP training pipeline ──

use serde::{Deserialize, Serialize};
use std::path::PathBuf;

// ── Model hyper-parameters ───────────────────────────────────────────────────

/// Architecture hyper-parameters for the multi-task MLP.
/// Derived from data at runtime (in_dim, n_actions) — not hard-coded.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MlpConfig {
    /// Dimensionality of the input V-JEPA feature vectors.
    pub in_dim: usize,
    /// Hidden-layer width for both FC layers.
    pub hidden_dim: usize,
    /// Number of action classes (typically 7).
    pub n_actions: usize,
}

// ── Training hyper-parameters ────────────────────────────────────────────────

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TrainConfig {
    /// Number of full passes over the training set.
    pub epochs: usize,
    /// Mini-batch size.
    pub batch_size: usize,
    /// AdamW learning rate.
    pub lr: f64,
    /// AdamW weight decay.
    pub weight_decay: f64,
    /// Weight of the position (MSE) loss relative to the action (CE) loss.
    pub pos_weight: f64,
    /// Fraction of data held out for validation (stratified per class).
    pub val_frac: f32,
    /// RNG seed for reproducibility.
    pub seed: u64,
    /// Drop action classes with fewer than this many samples.
    pub min_class_count: usize,
    /// Compute device: "cpu" or "metal".
    #[serde(default = "default_device")]
    pub device: String,
    /// Stop training if val loss doesn't improve for this many epochs (0 = disabled).
    #[serde(default)]
    pub early_stopping_patience: usize,
    /// Log metrics every N epochs (1 = every epoch).
    #[serde(default = "default_log_interval")]
    pub log_interval: usize,
}

fn default_device() -> String {
    "cpu".into()
}

fn default_log_interval() -> usize {
    10
}

// ── Data loading configuration ───────────────────────────────────────────────

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DataConfig {
    /// One or more session recording directories (accepts globs via shell).
    pub recording_dirs: Vec<PathBuf>,
    /// Number of frames per temporal clip.  0 → use model's native T.
    pub clip_len: usize,
    /// Stride between clip start indices.  0 → clip_len / 2.
    pub clip_stride: usize,
    /// V-JEPA model name (for cache path and compatibility metadata).
    pub model_name: String,
    /// If true, force re-extraction even if a cached .npz exists.
    #[serde(default)]
    pub force_extract: bool,
    /// Merge JitterClick into SingleClick (matching Python default).
    #[serde(default = "default_true")]
    pub merge_jitter: bool,
}

fn default_true() -> bool {
    true
}

// ── Full configuration ───────────────────────────────────────────────────────

/// Aggregated configuration written to `rust_mlp_config.json`.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct FullConfig {
    /// Output directory for weights and config.
    pub out_dir: PathBuf,
    /// Model architecture parameters.
    pub mlp: MlpConfig,
    /// Training hyper-parameters.
    pub train: TrainConfig,
    /// Data loading parameters.
    pub data: DataConfig,
    /// Path to the .npz feature cache (relative to out_dir).
    #[serde(default)]
    pub cache_path: PathBuf,
    /// Best validation loss achieved (set after training).
    #[serde(default)]
    pub best_val_loss: f64,
    /// Epoch at which best_val_loss was achieved (1-based).
    #[serde(default)]
    pub best_epoch: usize,
    /// Action labels in classifier-head order.
    #[serde(default)]
    pub action_labels: Vec<String>,
    /// Relative path to the saved weights file.
    #[serde(default)]
    pub weights_file: String,
}

impl FullConfig {
    /// Canonical name for the feature cache file.
    pub fn feature_cache_filename(&self) -> String {
        format!(
            "features_{}_T{}.npz",
            self.data.model_name, self.data.clip_len
        )
    }

    /// Absolute path to the feature cache on disk.
    pub fn cache_abs(&self) -> PathBuf {
        self.out_dir.join(&self.cache_path)
    }

    /// Absolute path where model weights will be saved.
    pub fn weights_abs(&self) -> PathBuf {
        self.out_dir.join(&self.weights_file)
    }

    /// Absolute path for the config JSON.
    pub fn config_abs(&self) -> PathBuf {
        self.out_dir.join("rust_mlp_config.json")
    }
}
