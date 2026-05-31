// ── mlp::model — Multi-task MLP for cursor position regression + action classification ──
//
// Architecture (matching Python MultiTaskMLP):
//   input [B, D]
//     → fc1 (gelu)       [B, hidden_dim]
//     → fc2 (gelu)       [B, hidden_dim]
//     → pos_head         [B, 2]        — regression
//     → action_head      [B, n_actions] — classification logits

use candle_core::{Device, Result, Tensor, D};
use candle_nn::{Linear, VarBuilder, VarMap};

// ── Multi-task MLP ───────────────────────────────────────────────────────────

pub struct MultiTaskMlp {
    fc1: Linear,
    fc2: Linear,
    pos_head: Linear,
    action_head: Linear,
}

impl MultiTaskMlp {
    /// Create a new multi-task MLP.
    ///
    /// * `in_dim` — input feature dimensionality (from V-JEPA pooling)
    /// * `hidden_dim` — width of both hidden layers
    /// * `n_actions` — number of action classes
    /// * `vb` — VarBuilder for parameter initialization
    pub fn new(in_dim: usize, hidden_dim: usize, n_actions: usize, vb: VarBuilder) -> Result<Self> {
        let fc1 = candle_nn::linear(in_dim, hidden_dim, vb.pp("fc1"))?;
        let fc2 = candle_nn::linear(hidden_dim, hidden_dim, vb.pp("fc2"))?;
        let pos_head = candle_nn::linear(hidden_dim, 2, vb.pp("pos_head"))?;
        let action_head = candle_nn::linear(hidden_dim, n_actions, vb.pp("action_head"))?;
        Ok(Self {
            fc1,
            fc2,
            pos_head,
            action_head,
        })
    }

    /// Forward pass.
    ///
    /// * `x` — input tensor [B, in_dim]
    ///
    /// Returns `(pos_pred [B, 2], action_logits [B, n_actions])`.
    pub fn forward(&self, x: &Tensor) -> Result<(Tensor, Tensor)> {
        let h = x.apply(&self.fc1)?.gelu_erf()?;
        let h = h.apply(&self.fc2)?.gelu_erf()?;
        let pos = h.apply(&self.pos_head)?;
        let action = h.apply(&self.action_head)?;
        Ok((pos, action))
    }
}

// ── Loss functions ───────────────────────────────────────────────────────────

/// Compute MSE loss for position regression.
pub fn position_mse(pred: &Tensor, target: &Tensor) -> Result<Tensor> {
    // pred: [B, 2], target: [B, 2]
    let diff = (pred - target)?;
    let sq = diff.sqr()?;
    sq.mean_all()
}

/// Compute weighted cross-entropy loss for action classification.
///
/// * `logits` — [B, n_classes]
/// * `targets` — [B] integer class indices
/// * `class_weights` — [n_classes] per-class weight
pub fn weighted_cross_entropy(
    logits: &Tensor,
    targets: &Tensor,
    class_weights: &Tensor,
) -> Result<Tensor> {
    // Gather weights for each sample's target class.
    // class_weights is 1D [n_classes], gather along dim=0.
    let targets_i64 = targets.to_dtype(candle_core::DType::I64)?;
    let weights_per_sample = class_weights.gather(&targets_i64, 0)?.unsqueeze(1)?; // [B, 1]

    // Log softmax over the class dimension
    let log_probs = candle_nn::ops::log_softmax(logits, D::Minus1)?;

    // Negative log likelihood per sample: gather the log-prob of the target class
    let nll = log_probs.gather(&targets_i64.unsqueeze(1)?, 1)?.neg()?; // [B, 1]

    // Apply weights and mean
    let weighted = (nll * &weights_per_sample)?;
    weighted.mean_all()
}

/// Combined loss: pos_weight * MSE + weighted_CE.
pub fn combined_loss(
    pos_pred: &Tensor,
    pos_target: &Tensor,
    action_logits: &Tensor,
    action_target: &Tensor,
    class_weights: &Tensor,
    pos_weight: f64,
) -> Result<Tensor> {
    let mse = position_mse(pos_pred, pos_target)?;
    let ce = weighted_cross_entropy(action_logits, action_target, class_weights)?;
    // mse is a scalar; scale it by pos_weight (also a scalar tensor)
    let pw = Tensor::new(pos_weight as f32, pos_pred.device())?;
    Ok((mse.mul(&pw)? + ce)?)
}

// ── Save / Load ──────────────────────────────────────────────────────────────

/// Save model weights to a safetensors file via VarMap.
pub fn save_weights(varmap: &VarMap, path: &std::path::Path) -> Result<()> {
    varmap.save(path)?;
    Ok(())
}

/// Load model weights from a safetensors file into a VarMap.
pub fn load_weights(varmap: &mut VarMap, path: &std::path::Path) -> Result<()> {
    varmap.load(path)?;
    Ok(())
}

/// Create a new VarMap and VarBuilder for training.
pub fn new_varmap(device: &Device) -> (VarMap, VarBuilder<'static>) {
    let varmap = VarMap::new();
    let vb = VarBuilder::from_varmap(&varmap, candle_core::DType::F32, device);
    (varmap, vb)
}
