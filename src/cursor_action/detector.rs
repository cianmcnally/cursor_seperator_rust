use std::collections::HashMap;
use std::sync::{Arc, Mutex, RwLock};
use std::time::Duration;

use crate::windows::WindowLayer;

use super::model::{CursorAction, CursorActionKind};
use super::tap::{MouseTapEvent, start_mouse_tap};

// ── Thresholds ────────────────────────────────────────────────────────────────
//
// 7-class model: idle / move / click / double_click / drag / scroll / typing.
// A press→release that never travels past DRAG_START_DIST is a `click`, regardless
// of how long it was held (held + jittery presses fold into `click`). A press that
// travels past it becomes a `drag`. No duration gate ⇒ no [250ms,400ms) dead-zone,
// and double-click works after any-length first press.

const DOUBLE_CLICK_MAX_NS:   u64 = 500_000_000;   // 500ms between clicks
const DOUBLE_CLICK_MAX_DIST: f64 = 8.0;           // px
const DRAG_START_DIST:       f64 = 6.0;           // px travel that turns a press into a drag
const DRAG_SAMPLE_MIN_DIST:  f64 = 2.0;           // px between sampled drag-path points
const DRAG_SAMPLE_MIN_NS:    u64 = 8_000_000;     // 8ms between sampled drag-path points
const MOVE_SAMPLE_MIN_DIST:  f64 = 4.0;           // px before a move refreshes activity
const MOVE_SAMPLE_MIN_NS:    u64 = 16_000_000;    // 16ms before a move refreshes activity
/// How long a momentary action (click/double/scroll/move/drag) keeps labelling the
/// per-frame `current` class before it decays back to `idle`.
const ACTIVE_TTL_NS:         u64 = 200_000_000;   // 200ms

// ── Public API ────────────────────────────────────────────────────────────────

pub struct CursorActionArgs {
    pub scale:          f64,
    pub cap_origin_x:   f64,
    pub cap_origin_y:   f64,
    pub cap_width:      u32,
    pub cap_height:     u32,
    pub export_actions: bool,
    /// If set, every discrete action (click/double/drag/scroll) is also sent here.
    pub event_tx:       Option<std::sync::mpsc::Sender<super::model::CursorAction>>,
}

/// Snapshot readable from the capture thread. All-Copy so per-frame reads are cheap
/// (no Vec/VecDeque clones).
#[derive(Clone, Copy, PartialEq, Eq, Default)]
pub struct ActionSnapshot {
    /// The live per-frame label — always exactly one of the seven classes.
    pub current:       CursorActionKind,
    pub is_dragging:   bool,
    pub is_mouse_down: bool,
}

pub struct SharedActionState {
    snap:   Mutex<ActionSnapshot>,
    /// Cumulative per-kind counts since the detector started.
    pub session_counts: Arc<Mutex<HashMap<&'static str, u64>>>,
}

impl SharedActionState {
    fn new() -> Arc<Self> {
        Arc::new(Self {
            snap:           Mutex::new(ActionSnapshot::default()),
            session_counts: Arc::new(Mutex::new(HashMap::new())),
        })
    }

    pub fn snapshot(&self) -> ActionSnapshot {
        *self.snap.lock().unwrap()
    }

    fn write(&self, s: ActionSnapshot) {
        *self.snap.lock().unwrap() = s;
    }

    fn increment(&self, kind: &'static str) {
        if let Ok(mut c) = self.session_counts.lock() {
            *c.entry(kind).or_insert(0) += 1;
        }
    }
}

/// Returns `(state, tap_alive)`. Poll `tap_alive` shortly after startup: if it is
/// still false, Input Monitoring permission is denied and no mouse events will be
/// recorded (the session would be useless).
pub fn start_cursor_action_detector(
    windows: Arc<RwLock<Vec<WindowLayer>>>,
    args:    CursorActionArgs,
) -> (Arc<SharedActionState>, Arc<std::sync::atomic::AtomicBool>) {
    let state  = SharedActionState::new();
    let state2 = state.clone();
    let (tx, rx) = std::sync::mpsc::channel::<MouseTapEvent>();
    let tap_alive = start_mouse_tap(tx);
    std::thread::spawn(move || run_detector(rx, state2, windows, args));
    (state, tap_alive)
}

// ── Internal state ────────────────────────────────────────────────────────────

struct DetState {
    last_pos:             Option<(i32, i32)>,
    last_move_ts:         Option<u64>,

    mouse_down_pos:       Option<(i32, i32)>,
    mouse_down_ts:        Option<u64>,
    is_mouse_down:        bool,

    is_dragging:          bool,

    drag_path:            Vec<(i32, i32)>,
    // min_x, min_y, max_x, max_y
    drag_bbox_mm:         Option<(i32, i32, i32, i32)>,
    last_drag_sample_pos: Option<(i32, i32)>,
    last_drag_sample_ts:  Option<u64>,
    drag_window_id:       Option<u32>,
    drag_z_index:         Option<usize>,

    last_click_pos:       Option<(i32, i32)>,
    last_click_ts:        Option<u64>,

    /// Most recent momentary signal (move/scroll/click/double/drag) + when it fired.
    /// Drives the decaying per-frame `current` label.
    last_activity:        Option<(CursorActionKind, u64)>,
}

impl DetState {
    fn new() -> Self {
        Self {
            last_pos:             None,
            last_move_ts:         None,
            mouse_down_pos:       None,
            mouse_down_ts:        None,
            is_mouse_down:        false,
            is_dragging:          false,
            drag_path:            Vec::new(),
            drag_bbox_mm:         None,
            last_drag_sample_pos: None,
            last_drag_sample_ts:  None,
            drag_window_id:       None,
            drag_z_index:         None,
            last_click_pos:       None,
            last_click_ts:        None,
            last_activity:        None,
        }
    }

    fn update_bbox(&mut self, p: (i32, i32)) {
        match &mut self.drag_bbox_mm {
            None => self.drag_bbox_mm = Some((p.0, p.1, p.0, p.1)),
            Some((mn_x, mn_y, mx_x, mx_y)) => {
                *mn_x = (*mn_x).min(p.0);
                *mn_y = (*mn_y).min(p.1);
                *mx_x = (*mx_x).max(p.0);
                *mx_y = (*mx_y).max(p.1);
            }
        }
    }

    fn bbox_arr(&self, cap_w: u32, cap_h: u32) -> Option<[i32; 4]> {
        let (mn_x, mn_y, mx_x, mx_y) = self.drag_bbox_mm?;
        let pad = 12i32;
        let x = (mn_x - pad).max(0);
        let y = (mn_y - pad).max(0);
        let w = (mx_x - mn_x + 1 + pad * 2).min(cap_w as i32 - x).max(0);
        let h = (mx_y - mn_y + 1 + pad * 2).min(cap_h as i32 - y).max(0);
        Some([x, y, w, h])
    }

    /// The live per-frame label. Always one of the seven classes.
    fn current_kind(&self, now: u64) -> CursorActionKind {
        if self.is_dragging  { return CursorActionKind::Drag; }
        if self.is_mouse_down { return CursorActionKind::Click; }  // press in progress
        match self.last_activity {
            Some((k, ts)) if now.saturating_sub(ts) < ACTIVE_TTL_NS => k,
            _ => CursorActionKind::Idle,
        }
    }

    fn snapshot(&self, now: u64) -> ActionSnapshot {
        ActionSnapshot {
            current:       self.current_kind(now),
            is_dragging:   self.is_dragging,
            is_mouse_down: self.is_mouse_down,
        }
    }

    fn reset_drag(&mut self) {
        self.is_dragging          = false;
        self.drag_bbox_mm         = None;
        self.drag_path.clear();
        self.last_drag_sample_pos = None;
        self.last_drag_sample_ts  = None;
        self.drag_window_id       = None;
        self.drag_z_index         = None;
    }
}

// ── Helpers ───────────────────────────────────────────────────────────────────

#[inline]
fn dist(a: (i32, i32), b: (i32, i32)) -> f64 {
    let dx = (a.0 - b.0) as f64;
    let dy = (a.1 - b.1) as f64;
    (dx * dx + dy * dy).sqrt()
}

fn now_ns() -> u64 {
    std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .unwrap_or_default()
        .as_nanos() as u64
}

#[inline]
fn to_px(x_pts: f64, y_pts: f64, scale: f64, ox: f64, oy: f64) -> (i32, i32) {
    ((x_pts * scale - ox) as i32, (y_pts * scale - oy) as i32)
}

fn window_at(pos: (i32, i32), wins: &[WindowLayer]) -> (Option<u32>, Option<usize>) {
    for w in wins {
        let b = w.bounds_pixels;
        if pos.0 >= b.x && pos.0 < b.x + b.w && pos.1 >= b.y && pos.1 < b.y + b.h {
            return (Some(w.window_id), Some(w.z_index));
        }
    }
    (None, None)
}

fn export(a: &CursorAction) {
    if let Ok(s) = serde_json::to_string(a) {
        println!("{{\"timestamp_ns\":{},\"cursor_actions\":[{}]}}", a.timestamp_ns, s);
    }
}

/// Emit a discrete action: stdout (if enabled), recorder channel, session counter.
fn emit(a: &CursorAction, args: &CursorActionArgs, state: &SharedActionState) {
    if args.export_actions { export(a); }
    if let Some(ref tx) = args.event_tx { let _ = tx.send(a.clone()); }
    state.increment(super::action_kind_str(a.kind));
}

// ── Detector loop ─────────────────────────────────────────────────────────────

fn run_detector(
    rx:      std::sync::mpsc::Receiver<MouseTapEvent>,
    state:   Arc<SharedActionState>,
    windows: Arc<RwLock<Vec<WindowLayer>>>,
    args:    CursorActionArgs,
) {
    let mut ds = DetState::new();
    let mut last_written = ActionSnapshot::default();
    state.write(last_written);

    loop {
        loop {
            match rx.try_recv() {
                Ok(ev) => {
                    let wins = windows.read().map(|g| g.clone()).unwrap_or_default();
                    process_event(&mut ds, ev, &args, &wins, &state);
                }
                Err(std::sync::mpsc::TryRecvError::Empty) => break,
                Err(_) => return,
            }
        }

        // Recompute the decaying per-frame label and publish only when it changes.
        let snap = ds.snapshot(now_ns());
        if snap != last_written {
            state.write(snap);
            last_written = snap;
        }

        std::thread::sleep(Duration::from_millis(4));
    }
}

// ── Event processing (state machine) ─────────────────────────────────────────

fn process_event(
    ds:    &mut DetState,
    ev:    MouseTapEvent,
    args:  &CursorActionArgs,
    wins:  &[WindowLayer],
    state: &SharedActionState,
) {
    let scale = args.scale;
    let ox    = args.cap_origin_x;
    let oy    = args.cap_origin_y;
    let cw    = args.cap_width;
    let ch    = args.cap_height;

    match ev {
        // ── Left mouse down ───────────────────────────────────────────────────
        MouseTapEvent::LeftMouseDown { ts_ns, x_pts, y_pts, .. } => {
            let pos = to_px(x_pts, y_pts, scale, ox, oy);
            ds.mouse_down_pos = Some(pos);
            ds.mouse_down_ts  = Some(ts_ns);
            ds.is_mouse_down  = true;
            ds.reset_drag();
            ds.last_pos = Some(pos);
            // current → Click (press in progress) via current_kind()
        }

        // ── Left mouse up ─────────────────────────────────────────────────────
        MouseTapEvent::LeftMouseUp { ts_ns, x_pts, y_pts, .. } => {
            let pos = to_px(x_pts, y_pts, scale, ox, oy);
            let (wid, zidx) = window_at(pos, wins);

            if ds.is_dragging {
                on_drag_end(ds, pos, ts_ns, wid, zidx, cw, ch, args, state);
                ds.last_activity = Some((CursorActionKind::Drag, ts_ns));
            } else if let (Some(down_pos), Some(down_ts)) =
                (ds.mouse_down_pos, ds.mouse_down_ts)
            {
                // Not a drag ⇒ a click, whatever its duration.
                let dur_ns = ts_ns.saturating_sub(down_ts);
                let moved  = dist(down_pos, pos);
                let kind   = on_click(ds, pos, down_pos, ts_ns, dur_ns, moved, wid, zidx, args, state);
                ds.last_activity = Some((kind, ts_ns));
            }

            ds.is_mouse_down = false;
            ds.reset_drag();
            ds.last_pos = Some(pos);
        }

        // ── Left mouse dragged ────────────────────────────────────────────────
        MouseTapEvent::LeftMouseDrag { ts_ns, x_pts, y_pts, .. } => {
            let pos = to_px(x_pts, y_pts, scale, ox, oy);
            if !ds.is_mouse_down {
                ds.last_pos = Some(pos);
                return;
            }
            let down_pos = match ds.mouse_down_pos { Some(p) => p, None => { ds.last_pos = Some(pos); return; } };
            let down_ts  = ds.mouse_down_ts.unwrap_or(ts_ns);

            if !ds.is_dragging && dist(down_pos, pos) > DRAG_START_DIST {
                // Begin a drag. No event yet — one `Drag` is emitted on release with
                // the full path, so events.ndjson gets one drag per gesture.
                let (wid, zidx) = window_at(down_pos, wins);
                ds.drag_window_id = wid;
                ds.drag_z_index   = zidx;
                ds.is_dragging    = true;

                ds.drag_path.push(down_pos);
                ds.update_bbox(down_pos);
                ds.last_drag_sample_pos = Some(down_pos);
                ds.last_drag_sample_ts  = Some(down_ts);
            }

            if ds.is_dragging {
                let should_sample = match (ds.last_drag_sample_pos, ds.last_drag_sample_ts) {
                    (Some(lp), Some(lt)) => {
                        dist(lp, pos) >= DRAG_SAMPLE_MIN_DIST
                            && ts_ns.saturating_sub(lt) >= DRAG_SAMPLE_MIN_NS
                    }
                    _ => true,
                };
                if should_sample {
                    ds.drag_path.push(pos);
                    ds.update_bbox(pos);
                    ds.last_drag_sample_pos = Some(pos);
                    ds.last_drag_sample_ts  = Some(ts_ns);
                }
            }

            ds.last_pos = Some(pos);
        }

        // ── Mouse moved (no button held) ──────────────────────────────────────
        MouseTapEvent::MouseMoved { ts_ns, x_pts, y_pts, .. } => {
            let pos = to_px(x_pts, y_pts, scale, ox, oy);

            let moved_enough = match (ds.last_pos, ds.last_move_ts) {
                (Some(lp), Some(lt)) => {
                    dist(lp, pos) >= MOVE_SAMPLE_MIN_DIST
                        && ts_ns.saturating_sub(lt) >= MOVE_SAMPLE_MIN_NS
                }
                _ => true,
            };
            if moved_enough {
                // Drives the per-frame `move` label. Move is NOT pushed to the
                // recorder event stream (high frequency, low value); the per-frame
                // cursor.action already captures movement.
                ds.last_activity = Some((CursorActionKind::Move, ts_ns));
                state.increment("move");
                if args.export_actions {
                    let (wid, zidx) = window_at(pos, wins);
                    export(&CursorAction {
                        kind:           CursorActionKind::Move,
                        timestamp_ns:   ts_ns,
                        position:       pos,
                        start_position: ds.last_pos,
                        end_position:   None,
                        path:           Vec::new(),
                        bbox:           None,
                        duration_ms:    None,
                        distance_px:    ds.last_pos.map(|lp| dist(lp, pos)),
                        button:         None,
                        click_count:    None,
                        window_id:      wid,
                        z_index:        zidx,
                        confidence:     1.0,
                        source:         "mouse_events".into(),
                    });
                }
            }

            ds.last_pos     = Some(pos);
            ds.last_move_ts = Some(ts_ns);
        }

        // ── Scroll wheel ──────────────────────────────────────────────────────
        MouseTapEvent::ScrollWheel { ts_ns, x_pts, y_pts, delta_y, .. } => {
            let pos = to_px(x_pts, y_pts, scale, ox, oy);
            let (wid, zidx) = window_at(pos, wins);
            let a = CursorAction {
                kind:           CursorActionKind::Scroll,
                timestamp_ns:   ts_ns,
                position:       pos,
                start_position: None,
                end_position:   None,
                path:           Vec::new(),
                bbox:           None,
                duration_ms:    None,
                distance_px:    Some(delta_y.abs() as f64),
                button:         None,
                click_count:    None,
                window_id:      wid,
                z_index:        zidx,
                confidence:     1.0,
                source:         "mouse_events".into(),
            };
            emit(&a, args, state);
            ds.last_activity = Some((CursorActionKind::Scroll, ts_ns));
        }
    }
}

/// Classify a press→release as `click` or `double_click` and emit it.
/// Returns the kind so the caller can refresh the per-frame label.
/// Hold duration and jitter survive in `duration_ms` / `distance_px`, not as labels.
fn on_click(
    ds:       &mut DetState,
    pos:      (i32, i32),
    down_pos: (i32, i32),
    ts_ns:    u64,
    dur_ns:   u64,
    moved:    f64,
    wid:      Option<u32>,
    zidx:     Option<usize>,
    args:     &CursorActionArgs,
    state:    &SharedActionState,
) -> CursorActionKind {
    let is_double = ds.last_click_ts
        .zip(ds.last_click_pos)
        .map_or(false, |(lt, lp)| {
            ts_ns.saturating_sub(lt) <= DOUBLE_CLICK_MAX_NS
                && dist(lp, pos) <= DOUBLE_CLICK_MAX_DIST
        });

    let (kind, count) = if is_double {
        (CursorActionKind::DoubleClick, 2u8)
    } else {
        (CursorActionKind::Click, 1u8)
    };

    let a = CursorAction {
        kind,
        timestamp_ns:   ts_ns,
        position:       pos,
        start_position: Some(down_pos),
        end_position:   Some(pos),
        path:           Vec::new(),
        bbox:           Some([pos.0 - 8, pos.1 - 8, 16, 16]),
        duration_ms:    Some(dur_ns as f64 / 1_000_000.0),
        distance_px:    Some(moved),
        button:         Some("left".into()),
        click_count:    Some(count),
        window_id:      wid,
        z_index:        zidx,
        confidence:     0.99,
        source:         "mouse_events".into(),
    };
    emit(&a, args, state);

    // After a double-click, clear history so a 3rd click starts a fresh single.
    if is_double {
        ds.last_click_pos = None;
        ds.last_click_ts  = None;
    } else {
        ds.last_click_pos = Some(pos);
        ds.last_click_ts  = Some(ts_ns);
    }
    kind
}

/// Emit a single `drag` event carrying the full sampled path + bounding box.
fn on_drag_end(
    ds:    &mut DetState,
    pos:   (i32, i32),
    ts_ns: u64,
    wid:   Option<u32>,
    zidx:  Option<usize>,
    cw:    u32,
    ch:    u32,
    args:  &CursorActionArgs,
    state: &SharedActionState,
) {
    let down_pos  = ds.mouse_down_pos.unwrap_or(pos);
    let down_ts   = ds.mouse_down_ts.unwrap_or(ts_ns);
    let dur_ms    = ts_ns.saturating_sub(down_ts) as f64 / 1_000_000.0;
    let drag_dist = dist(down_pos, pos);
    let path      = std::mem::take(&mut ds.drag_path);
    let bbox      = ds.bbox_arr(cw, ch);

    let drag = CursorAction {
        kind:           CursorActionKind::Drag,
        timestamp_ns:   ts_ns,
        position:       pos,
        start_position: Some(down_pos),
        end_position:   Some(pos),
        path,
        bbox,
        duration_ms:    Some(dur_ms),
        distance_px:    Some(drag_dist),
        button:         Some("left".into()),
        click_count:    None,
        window_id:      wid.or(ds.drag_window_id),
        z_index:        zidx.or(ds.drag_z_index),
        confidence:     0.95,
        source:         "mouse_events".into(),
    };
    emit(&drag, args, state);
}
