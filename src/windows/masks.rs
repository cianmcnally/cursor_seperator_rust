use super::model::{WindowCategory, WindowLayer, WindowMaskRole};

// ── Focus selection ───────────────────────────────────────────────────────────

const MIN_FOCUS_W: i32 = 180;
const MIN_FOCUS_H: i32 = 120;

/// Find the frontmost real app window for `active_pid`.
/// Popup/overlay windows that happen to be z=0 are excluded.
pub fn choose_focused_root_window(
    windows: &[WindowLayer],
    active_pid: i32,
) -> Option<&WindowLayer> {
    windows
        .iter()
        .filter(|w| w.is_onscreen)
        .filter(|w| w.alpha >= 0.01)
        .filter(|w| w.owner_pid == active_pid)
        .filter(|w| w.cg_layer == 0)
        .filter(|w| matches!(w.category, WindowCategory::NormalAppWindow))
        .filter(|w| w.bounds_pixels.w >= MIN_FOCUS_W && w.bounds_pixels.h >= MIN_FOCUS_H)
        .min_by_key(|w| w.z_index)
}

/// Fallback when NSWorkspace PID is unavailable.
/// If frontmost window is a popup, walk down the z-stack to find the real app window.
pub fn choose_focus_without_nsworkspace(windows: &[WindowLayer]) -> Option<&WindowLayer> {
    let front = windows.first()?;
    let anchor_pid = front.owner_pid;

    if front.category.is_popup_like() || front.cg_layer != 0 {
        return windows
            .iter()
            .filter(|w| w.owner_pid == anchor_pid)
            .filter(|w| w.cg_layer == 0)
            .filter(|w| matches!(w.category, WindowCategory::NormalAppWindow))
            .filter(|w| w.bounds_pixels.w >= MIN_FOCUS_W && w.bounds_pixels.h >= MIN_FOCUS_H)
            .min_by_key(|w| w.z_index);
    }

    if matches!(front.category, WindowCategory::NormalAppWindow) && front.cg_layer == 0 {
        Some(front)
    } else {
        None
    }
}

// ── Role classification ───────────────────────────────────────────────────────

fn classify_mask_role(
    w: &WindowLayer,
    focused_id: Option<u32>,
    active_pid: i32,
) -> WindowMaskRole {
    if !w.is_onscreen || w.alpha < 0.01 {
        return WindowMaskRole::Ignore;
    }

    if Some(w.window_id) == focused_id {
        return WindowMaskRole::FocusedRoot;
    }

    if matches!(w.category, WindowCategory::NormalAppWindow) && w.cg_layer == 0 {
        return WindowMaskRole::UnfocusedRoot;
    }

    if w.owner_pid == active_pid && w.category.is_popup_like() {
        return WindowMaskRole::PopupOfFocused;
    }

    if w.category.is_popup_like() {
        return WindowMaskRole::PopupOfUnfocused;
    }

    WindowMaskRole::Occluder
}

/// Assign `mask_role` on every window. Call after `include_in_segmentation` is finalised.
/// `active_pid`: from NSWorkspace; pass `None` to use the z-stack heuristic.
pub fn assign_mask_roles(windows: &mut [WindowLayer], active_pid: Option<i32>) {
    // Determine focused window id.
    let focused_id = if let Some(pid) = active_pid {
        choose_focused_root_window(windows, pid).map(|w| w.window_id)
    } else {
        choose_focus_without_nsworkspace(windows).map(|w| w.window_id)
    };

    // Resolve effective active_pid for popup ownership check.
    let effective_pid = active_pid.unwrap_or_else(|| {
        windows
            .iter()
            .find(|w| Some(w.window_id) == focused_id)
            .map(|w| w.owner_pid)
            .unwrap_or(-1)
    });

    for w in windows.iter_mut() {
        w.mask_role = if !w.include_in_segmentation {
            WindowMaskRole::Ignore
        } else {
            classify_mask_role(w, focused_id, effective_pid)
        };
    }
}
