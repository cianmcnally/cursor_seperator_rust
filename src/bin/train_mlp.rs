// ── train_mlp — CLI entry point for multi-task MLP training ─────────────────
//
// Port of train_rust_recording_mlp.py.
//
// Usage:
//   cargo run --bin train_mlp -- \
//     --recording-dir recordings/session_20260531_111423 \
//     --out-dir .artifacts/rust_mlp
//
// Multiple sessions:
//   cargo run --bin train_mlp -- \
//     --recording-dir recordings/session_* \
//     --out-dir .artifacts/rust_mlp_big

use std::path::PathBuf;

use clap::Parser;

use rust_cursor_bench::mlp::clips::{build_clips, load_all_sessions};
use rust_cursor_bench::mlp::config::{DataConfig, FullConfig, MlpConfig, TrainConfig};
use rust_cursor_bench::mlp::io::{load_feature_cache, save_config};
use rust_cursor_bench::mlp::train::train;

// ── CLI ──────────────────────────────────────────────────────────────────────

#[derive(Parser)]
#[command(
    name = "train_mlp",
    about = "Train a multi-task MLP on Rust session recordings (position regression + action classification)"
)]
struct Args {
    /// One or more session recording directories.
    #[arg(short = 'd', long, required = true, num_args = 1..)]
    recording_dir: Vec<PathBuf>,

    /// Output directory for weights, config, and feature cache.
    #[arg(short = 'o', long, default_value = ".artifacts/rust_mlp")]
    out_dir: PathBuf,

    /// V-JEPA model name (for cache path metadata).
    #[arg(long, default_value = "vjepa2_1_vit_base_384")]
    model: String,

    /// Frames per clip. 0 = use 16 (matching Python default).
    #[arg(long, default_value = "16")]
    clip_len: usize,

    /// Clip stride. 0 = clip_len / 2.
    #[arg(long, default_value = "0")]
    clip_stride: usize,

    /// Number of training epochs.
    #[arg(long, default_value = "80")]
    epochs: usize,

    /// Mini-batch size.
    #[arg(short = 'b', long, default_value = "16")]
    batch_size: usize,

    /// Hidden dimension of the MLP.
    #[arg(long, default_value = "256")]
    hidden_dim: usize,

    /// AdamW learning rate.
    #[arg(long, default_value = "0.0003")]
    lr: f64,

    /// AdamW weight decay.
    #[arg(long, default_value = "0.05")]
    weight_decay: f64,

    /// Weight of position loss relative to action loss.
    #[arg(long, default_value = "1.0")]
    pos_weight: f64,

    /// Fraction of data for validation (stratified).
    #[arg(long, default_value = "0.2")]
    val_frac: f32,

    /// RNG seed.
    #[arg(long, default_value = "7")]
    seed: u64,

    /// Compute device: "cpu" or "metal".
    #[arg(long, default_value = "cpu")]
    device: String,

    /// Stop if val loss doesn't improve for N eval intervals (0 = disabled).
    #[arg(long, default_value = "0")]
    early_stopping_patience: usize,

    /// Log metrics every N epochs.
    #[arg(long, default_value = "10")]
    log_interval: usize,

    /// Drop action classes with fewer than this many frames.
    #[arg(long, default_value = "5")]
    min_class: usize,

    /// Keep JitterClick as a separate class (default: merge into Click).
    #[arg(long)]
    no_merge_jitter: bool,

    /// Force re-extraction even if cached .npz exists.
    #[arg(long)]
    force_extract: bool,
}

// ── Main ─────────────────────────────────────────────────────────────────────

fn main() {
    let args = Args::parse();

    let stride = if args.clip_stride > 0 {
        args.clip_stride
    } else {
        (args.clip_len / 2).max(1)
    };

    // ── Load sessions ────────────────────────────────────────────────────
    eprintln!("=== Loading sessions ===");
    let (frames, action_labels) = match load_all_sessions(
        &args.recording_dir,
        args.min_class,
    ) {
        Ok(r) => r,
        Err(e) => {
            eprintln!("Error loading sessions: {e}");
            std::process::exit(1);
        }
    };

    // ── Build clips ──────────────────────────────────────────────────────
    eprintln!();
    eprintln!("=== Building clips (T={}, stride={}) ===", args.clip_len, stride);
    let clips = build_clips(&frames, args.clip_len, stride);
    eprintln!("Built {} clips", clips.len());

    if clips.is_empty() {
        eprintln!("No clips could be built. Try a shorter --clip-len or fewer --min-class.");
        std::process::exit(1);
    }

    // Print clip action distribution
    let mut clip_counts: std::collections::HashMap<String, usize> = std::collections::HashMap::new();
    for c in &clips {
        *clip_counts.entry(c.action.to_string()).or_insert(0) += 1;
    }
    eprintln!("Clip action counts: {clip_counts:?}");
    eprintln!("Actions: {:?}", action_labels.iter().map(|a| a.to_string()).collect::<Vec<_>>());

    // ── Build config ─────────────────────────────────────────────────────
    let n_actions = action_labels.len();

    let full_config = FullConfig {
        out_dir: args.out_dir.clone(),
        mlp: MlpConfig {
            in_dim: 0, // set after loading features
            hidden_dim: args.hidden_dim,
            n_actions,
        },
        train: TrainConfig {
            epochs: args.epochs,
            batch_size: args.batch_size,
            lr: args.lr,
            weight_decay: args.weight_decay,
            pos_weight: args.pos_weight,
            val_frac: args.val_frac,
            seed: args.seed,
            min_class_count: args.min_class,
            device: args.device,
            early_stopping_patience: args.early_stopping_patience,
            log_interval: args.log_interval,
        },
        data: DataConfig {
            recording_dirs: args.recording_dir.clone(),
            clip_len: args.clip_len,
            clip_stride: args.clip_stride,
            model_name: args.model.clone(),
            force_extract: args.force_extract,
            merge_jitter: !args.no_merge_jitter,
        },
        cache_path: PathBuf::from(full_config_builder_feature_cache_filename(
            &args.model,
            args.clip_len,
        )),
        best_val_loss: 0.0,
        best_epoch: 0,
        action_labels: action_labels.iter().map(|a| a.to_string()).collect(),
        weights_file: "rust_mlp_weights.safetensors".into(),
    };

    // Ensure out_dir exists
    let _ = std::fs::create_dir_all(&args.out_dir);

    // ── Load feature cache ───────────────────────────────────────────────
    let cache_abs = args.out_dir.join(&full_config.cache_path);
    eprintln!();
    eprintln!("=== Loading features from {} ===", cache_abs.display());

    let features = match load_feature_cache(&cache_abs) {
        Ok(f) => f,
        Err(e) => {
            eprintln!("{e}");
            eprintln!();
            eprintln!("Hint: extract features with the Python pipeline first, then re-run.");
            std::process::exit(1);
        }
    };

    // Update in_dim from loaded features
    let in_dim = features.features[0].len();
    let full_config = FullConfig {
        mlp: MlpConfig {
            in_dim,
            ..full_config.mlp
        },
        ..full_config
    };

    eprintln!("Feature dim: {in_dim}");
    eprintln!("Clips in cache: {}  (clips built: {})", features.len(), clips.len());

    if features.len() != clips.len() {
        eprintln!(
            "  [warn] clip count mismatch: cache has {} but {} clips were built this run.",
            features.len(),
            clips.len()
        );
        eprintln!(
            "  This is expected if you changed --clip-len or sessions. \
             Re-extract features with the Python pipeline if needed."
        );
    }

    // ── Train ────────────────────────────────────────────────────────────
    eprintln!();
    eprintln!("=== Training ===");

    let state = match train(&full_config, &features) {
        Ok(s) => s,
        Err(e) => {
            eprintln!("Training error: {e}");
            std::process::exit(1);
        }
    };

    // ── Save config ──────────────────────────────────────────────────────
    if let Err(e) = save_config(&full_config, &state, &features.action_labels) {
        eprintln!("Error saving config: {e}");
        std::process::exit(1);
    }

    eprintln!();
    eprintln!("=== Done ===");
    eprintln!("Weights: {}", args.out_dir.join("rust_mlp_weights.safetensors").display());
    eprintln!("Config:  {}", args.out_dir.join("rust_mlp_config.json").display());
}

fn full_config_builder_feature_cache_filename(model: &str, clip_len: usize) -> String {
    format!("features_{}_T{}.npz", model, clip_len)
}
