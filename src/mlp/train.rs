// ── mlp::train — Training loop for the multi-task MLP ────────────────────────
//
// Stratified train/val split, class weights, AdamW optimization,
// periodic evaluation, and best-model tracking.

use candle_core::{Device, Tensor};
use candle_nn::{AdamW, Optimizer, ParamsAdamW};

use crate::mlp::config::FullConfig;
use crate::mlp::model::{combined_loss, MultiTaskMlp, save_weights};
use crate::mlp::types::{EvalMetrics, FeatureCache, TrainState};

// ── Stratified split ─────────────────────────────────────────────────────────

/// Split indices stratified by class label.
pub fn stratified_split(
    action_ids: &[i32],
    n_classes: usize,
    val_frac: f32,
    seed: u64,
) -> (Vec<usize>, Vec<usize>) {
    use rand::prelude::*;
    use rand::rngs::StdRng;

    let mut rng = StdRng::seed_from_u64(seed);
    let mut train_idx: Vec<usize> = Vec::new();
    let mut val_idx: Vec<usize> = Vec::new();

    for cls in 0..n_classes {
        let mut idx: Vec<usize> = action_ids
            .iter()
            .enumerate()
            .filter(|(_, &a)| a as usize == cls)
            .map(|(i, _)| i)
            .collect();

        idx.shuffle(&mut rng);

        let n_val = if idx.len() >= 5 {
            ((idx.len() as f32) * val_frac).round().max(1.0) as usize
        } else {
            0
        };

        val_idx.extend(&idx[..n_val]);
        train_idx.extend(&idx[n_val..]);
    }

    (train_idx, val_idx)
}

// ── Class weights ────────────────────────────────────────────────────────────

/// Inverse-frequency class weights, normalized so mean = 1.
pub fn class_weights_tensor(
    action_ids: &[i32],
    n_classes: usize,
    device: &Device,
) -> Result<Tensor, candle_core::Error> {
    let mut counts = vec![0usize; n_classes];
    for &a in action_ids {
        if (a as usize) < n_classes {
            counts[a as usize] += 1;
        }
    }
    let mut weights: Vec<f32> = counts
        .iter()
        .map(|&c| if c > 0 { 1.0 / (c as f32) } else { 0.0 })
        .collect();
    let mean = weights.iter().sum::<f32>() / (n_classes as f32);
    if mean > 0.0 {
        for w in &mut weights {
            *w /= mean;
        }
    }
    Tensor::from_vec(weights, n_classes, device)
}

// ── Batch builder ────────────────────────────────────────────────────────────

fn build_batch(
    features: &[Vec<f32>],
    pos: &[[f32; 2]],
    action_ids: &[i32],
    indices: &[usize],
    device: &Device,
) -> Result<(Tensor, Tensor, Tensor), candle_core::Error> {
    let b = indices.len();
    let d = features[0].len();

    let mut x_data = vec![0.0f32; b * d];
    let mut pos_data = vec![0.0f32; b * 2];
    let mut act_data = vec![0i32; b];

    for (j, &idx) in indices.iter().enumerate() {
        for k in 0..d {
            x_data[j * d + k] = features[idx][k];
        }
        pos_data[j * 2] = pos[idx][0];
        pos_data[j * 2 + 1] = pos[idx][1];
        act_data[j] = action_ids[idx];
    }

    let xb = Tensor::from_vec(x_data, (b, d), device)?;
    let pb = Tensor::from_vec(pos_data, (b, 2), device)?;
    let ab = Tensor::from_vec(act_data, b, device)?;

    Ok((xb, pb, ab))
}

// ── Evaluation ───────────────────────────────────────────────────────────────

/// Evaluate model on the given indices.
pub fn evaluate(
    model: &MultiTaskMlp,
    features: &[Vec<f32>],
    pos: &[[f32; 2]],
    action_ids: &[i32],
    indices: &[usize],
    class_weights: &Tensor,
    pos_weight: f64,
    batch_size: usize,
    device: &Device,
) -> Result<EvalMetrics, String> {
    if indices.is_empty() {
        return Ok(EvalMetrics::default());
    }

    let n = indices.len();
    let mut total_loss = 0.0f32;
    let mut total_samples = 0usize;
    let mut pos_err_sum = 0.0f64;
    let mut correct = 0usize;

    for start in (0..n).step_by(batch_size) {
        let end = (start + batch_size).min(n);
        let batch_indices = &indices[start..end];
        let b = batch_indices.len();

        let (xb, pb, ab) = build_batch(features, pos, action_ids, batch_indices, device)
            .map_err(|e| format!("eval batch: {e}"))?;

        let (pred_pos, logits) = model.forward(&xb).map_err(|e| format!("forward: {e}"))?;

        let loss = combined_loss(&pred_pos, &pb, &logits, &ab, class_weights, pos_weight)
            .map_err(|e| format!("loss: {e}"))?;
        let loss_val: f32 = loss.to_scalar::<f32>().map_err(|e| format!("scalar: {e}"))?;

        total_loss += loss_val * (b as f32);
        total_samples += b;

        let pred_pos_data = pred_pos.to_vec2::<f32>().map_err(|e| format!("pred_pos: {e}"))?;
        let logits_data = logits.to_vec2::<f32>().map_err(|e| format!("logits: {e}"))?;

        for (j, &global_idx) in batch_indices.iter().enumerate() {
            let dx = (pred_pos_data[j][0] - pos[global_idx][0]) as f64;
            let dy = (pred_pos_data[j][1] - pos[global_idx][1]) as f64;
            pos_err_sum += (dx * dx + dy * dy).sqrt();

            let pred_class = argmax(&logits_data[j]);
            if pred_class == action_ids[global_idx] as usize {
                correct += 1;
            }
        }
    }

    Ok(EvalMetrics {
        loss: total_loss / (total_samples as f32),
        pos_error: (pos_err_sum / (total_samples as f64)) as f32,
        action_accuracy: (correct as f32) / (total_samples as f32),
    })
}

fn argmax(v: &[f32]) -> usize {
    v.iter()
        .enumerate()
        .max_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal))
        .map(|(i, _)| i)
        .unwrap_or(0)
}

// ── Main training loop ───────────────────────────────────────────────────────

pub fn train(config: &FullConfig, features: &FeatureCache) -> Result<TrainState, String> {
    use rand::prelude::*;
    use rand::rngs::StdRng;

    let device = match config.train.device.as_str() {
        "metal" => Device::new_metal(0).unwrap_or_else(|e| {
            eprintln!("  [warn] Metal not available ({e}), falling back to CPU");
            Device::Cpu
        }),
        _ => Device::Cpu,
    };

    let n = features.len();
    let d = features.features[0].len();
    let n_classes = config.mlp.n_actions;

    eprintln!();
    eprintln!(
        "Training: n={n}  d={d}  n_classes={n_classes}  hidden_dim={}  device={device:?}",
        config.mlp.hidden_dim
    );
    eprintln!(
        "  epochs={}  batch_size={}  lr={}  wd={}  pos_weight={}",
        config.train.epochs, config.train.batch_size, config.train.lr, config.train.weight_decay, config.train.pos_weight
    );
    if config.train.early_stopping_patience > 0 {
        eprintln!("  early_stopping_patience={}", config.train.early_stopping_patience);
    }

    // ── Stratified split ─────────────────────────────────────────────────
    let (train_idx, val_idx) = stratified_split(
        &features.action_ids,
        n_classes,
        config.train.val_frac,
        config.train.seed,
    );
    eprintln!("  train={}  val={}", train_idx.len(), val_idx.len());

    // ── Class weights ────────────────────────────────────────────────────
    let cw = class_weights_tensor(&features.action_ids, n_classes, &device)
        .map_err(|e| format!("class weights: {e}"))?;

    // ── Build model & optimizer ──────────────────────────────────────────
    let (varmap, vb) = crate::mlp::model::new_varmap(&device);
    let model = MultiTaskMlp::new(d, config.mlp.hidden_dim, n_classes, vb)
        .map_err(|e| format!("create model: {e}"))?;

    let adamw_params = ParamsAdamW {
        lr: config.train.lr,
        weight_decay: config.train.weight_decay,
        ..Default::default()
    };
    let mut opt = AdamW::new(varmap.all_vars(), adamw_params)
        .map_err(|e| format!("create optimizer: {e}"))?;

    // ── RNG for shuffling ───────────────────────────────────────────────
    let mut rng = StdRng::seed_from_u64(config.train.seed);
    let mut best_val_loss = f64::INFINITY;
    let mut best_epoch: usize = 0;
    let mut patience_counter: usize = 0;
    let log_interval = config.train.log_interval.max(1);

    // ── Epoch loop ──────────────────────────────────────────────────────
    for epoch in 1..=config.train.epochs {
        // Shuffle training indices
        let mut epoch_idx = train_idx.clone();
        epoch_idx.shuffle(&mut rng);

        let n_train = epoch_idx.len();

        for start in (0..n_train).step_by(config.train.batch_size) {
            let end = (start + config.train.batch_size).min(n_train);
            let batch_indices = &epoch_idx[start..end];

            let (xb, pb, ab) = build_batch(
                &features.features,
                &features.pos,
                &features.action_ids,
                batch_indices,
                &device,
            )
            .map_err(|e| format!("train batch: {e}"))?;

            let (pred_pos, logits) = model.forward(&xb).map_err(|e| format!("forward: {e}"))?;

            let loss = combined_loss(
                &pred_pos,
                &pb,
                &logits,
                &ab,
                &cw,
                config.train.pos_weight,
            )
            .map_err(|e| format!("loss: {e}"))?;

            opt.backward_step(&loss).map_err(|e| format!("backward_step: {e}"))?;
        }

        // ── Evaluation & logging ─────────────────────────────────────
        let do_eval = epoch == 1 || epoch % log_interval == 0 || epoch == config.train.epochs;

        if do_eval {
            let train_metrics = evaluate(
                &model,
                &features.features,
                &features.pos,
                &features.action_ids,
                &train_idx,
                &cw,
                config.train.pos_weight,
                config.train.batch_size,
                &device,
            )?;

            let val_metrics = evaluate(
                &model,
                &features.features,
                &features.pos,
                &features.action_ids,
                &val_idx,
                &cw,
                config.train.pos_weight,
                config.train.batch_size,
                &device,
            )?;

            eprintln!(
                "epoch {:03} | train loss={:.4} pos_err={:.4} acc={:.3} | val loss={:.4} pos_err={:.4} acc={:.3}",
                epoch,
                train_metrics.loss,
                train_metrics.pos_error,
                train_metrics.action_accuracy,
                val_metrics.loss,
                val_metrics.pos_error,
                val_metrics.action_accuracy,
            );

            let improved = (val_metrics.loss as f64) < best_val_loss;
            if improved {
                best_val_loss = val_metrics.loss as f64;
                best_epoch = epoch;
                patience_counter = 0;

                let weights_path = config.weights_abs();
                if let Some(parent) = weights_path.parent() {
                    let _ = std::fs::create_dir_all(parent);
                }
                save_weights(&varmap, &weights_path)
                    .map_err(|e| format!("save best weights: {e}"))?;
            } else {
                patience_counter += 1;
            }
        }

        // ── Periodic checkpoint (every epoch, for crash recovery) ────
        {
            let latest_path = config.out_dir.join("rust_mlp_weights_latest.safetensors");
            if let Some(parent) = latest_path.parent() {
                let _ = std::fs::create_dir_all(parent);
            }
            let _ = save_weights(&varmap, &latest_path);
        }

        // ── Early stopping ───────────────────────────────────────────
        if config.train.early_stopping_patience > 0
            && patience_counter >= config.train.early_stopping_patience
        {
            eprintln!(
                "Early stopping at epoch {epoch}: no improvement for {} epochs",
                patience_counter
            );
            break;
        }
    }

    if best_epoch == 0 {
        let weights_path = config.weights_abs();
        if let Some(parent) = weights_path.parent() {
            let _ = std::fs::create_dir_all(parent);
        }
        save_weights(&varmap, &weights_path).map_err(|e| format!("save weights: {e}"))?;
        best_epoch = config.train.epochs;
    }

    eprintln!();
    eprintln!("Best epoch {best_epoch} val_loss={:.4}", best_val_loss);
    eprintln!("Weights: {}", config.weights_abs().display());

    Ok(TrainState {
        best_val_loss: best_val_loss as f32,
        best_epoch,
        weights_path: config.weights_file.clone(),
    })
}
