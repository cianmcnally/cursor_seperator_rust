use std::collections::VecDeque;
use std::sync::{Arc, Mutex, RwLock};
use std::time::Duration;

use crate::windows::WindowLayer;

use super::model::{CursorAction, CursorActionKind};
use super::tap::{MouseTapEvent, start_mouse_tap};

// ── Thresholds ────────────────────────────────────────────────────────────────

const DOUBLE_CLICK_MAX_NS:   u64 = 500_000_000;   // 500ms
const DOUBLE_CLICK_MAX_DIST: f64 = 8.0;           // px
const DRAG_START_DIST:       f64 = 6.0;           // px
const CLICK_MAX_NS:          u64 = 250_000_000;   // 250ms
const CLICK_MAX_MOVEMENT:    f64 = 6.0;           // px
const HOLD_MIN_NS:           u64 = 400_000_000;   // 400ms
const DRAG_SAMPLE_MIN_DIST:  f64 = 2.0;           // px
const DRAG_SAMPLE_MIN_NS:    u64 = 8_000_000;     // 8ms
const MOVE_SAMPLE_MIN_DIST:  f64 = 4.0;           // px
const MOVE_SAMPLE_MIN_NS:    u64 = 16_000_000;    // 16ms
const RECENT_CAP:            usize = 64;

// ── Public API ────────────────────────────────────────────────────────────────

pub struct CursorActionArgs {
    pub scale:          f64,
    pub cap_origin_x:   f64,
    pub cap_origin_y:   f64,
    pub cap_width:      u32,
    pub cap_height:     u32,
    pub export_actions: bool,
}

/// Snapshot readable from the render thread.
#[derive(Clone, Default)]
pub struct ActionSnapshot {
    pub recent_actions: VecDeque<CursorAction>,
    pub drag_path:      Vec<(i32, i32)>,
    pub drag_bbox:      Option<[i32; 4]>,
    pub is_dragging:    bool,
    pub is_mouse_down:  bool,
    pub mouse_down_pos: Option<(i32, i32)>,
}

pub struct SharedActionState(Mutex<ActionSnapshot>);

impl SharedActionState {
    fn new() -> Arc<Self> {
        Arc::new(Self(Mutex::new(ActionSnapshot::default())))
    }

    pub fn snapshot(&self) -> ActionSnapshot {
        self.0.lock().unwrap().clone()
    }

    fn write(&self, s: ActionSnapshot) {
        *self.0.lock().unwrap() = s;
    }
}

pub fn start_cursor_action_detector(
    windows: Arc<RwLock<Vec<WindowLayer>>>,
    args:    CursorActionArgs,
) -> Arc<SharedActionState> {
    let state  = SharedActionState::new();
    let state2 = state.clone();
    let (tx, rx) = std::sync::mpsc::channel::<MouseTapEvent>();
    start_mouse_tap(tx);
    std::thread::spawn(move || run_detector(rx, state2, windows, args));
    state
}

// ── Internal state ────────────────────────────────────────────────────────────

struct DetState {
    last_pos:             Option<(i32, i32)>,
    last_move_ts:         Option<u64>,

    mouse_down_pos:       Option<(i32, i32)>,
    mouse_down_ts:        Option<u64>,
    is_mouse_down:        bool,

    is_dragging:          bool,
    hold_emitted:         bool,

    drag_path:            Vec<(i32, i32)>,
    // min_x, min_y, max_x, max_y
    drag_bbox_mm:         Option<(i32, i32, i32, i32)>,
    last_drag_sample_pos: Option<(i32, i32)>,
    last_drag_sample_ts:  Option<u64>,
    drag_window_id:       Option<u32>,
    drag_z_index:         Option<usize>,

    last_click_pos:       Option<(i32, i32)>,
    last_click_ts:        Option<u64>,

    recent_actions:       VecDeque<CursorAction>,
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
            hold_emitted:         false,
            drag_path:            Vec::new(),
            drag_bbox_mm:         None,
            last_drag_sample_pos: None,
            last_drag_sample_ts:  None,
            drag_window_id:       None,
            drag_z_index:         None,
            last_click_pos:       None,
            last_click_ts:        None,
            recent_actions:       VecDeque::with_capacity(RECENT_CAP),
        }
    }

    fn push(&mut self, a: CursorAction) {
        if self.recent_actions.len() >= RECENT_CAP {
            self.recent_actions.pop_front();
        }
        self.recent_actions.push_back(a);
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

    fn snapshot(&self, cap_w: u32, cap_h: u32) -> ActionSnapshot {
        ActionSnapshot {
            recent_actions: self.recent_actions.clone(),
            drag_path:      self.drag_path.clone(),
            drag_bbox:      if self.is_dragging { self.bbox_arr(cap_w, cap_h) } else { None },
            is_dragging:    self.is_dragging,
            is_mouse_down:  self.is_mouse_down,
            mouse_down_pos: self.mouse_down_pos,
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

// ── Detector loop ─────────────────────────────────────────────────────────────

fn run_detector(
    rx:      std::sync::mpsc::Receiver<MouseTapEvent>,
    state:   Arc<SharedActionState>,
    windows: Arc<RwLock<Vec<WindowLayer>>>,
    args:    CursorActionArgs,
) {
    let mut ds = DetState::new();

    loop {
        let mut changed = false;

        loop {
            match rx.try_recv() {
                Ok(ev) => {
                    let wins = windows.read().map(|g| g.clone()).unwrap_or_default();
                    process_event(&mut ds, ev, &args, &wins);
                    changed = true;
                }
                Err(std::sync::mpsc::TryRecvError::Empty) => break,
                Err(_) => return,
            }
        }

        // ClickAndHold: time-based detection in polling loop
        if ds.is_mouse_down && !ds.is_dragging && !ds.hold_emitted {
            if let (Some(dp), Some(dt)) = (ds.mouse_down_pos, ds.mouse_down_ts) {
                let now = now_ns();
                if now.saturating_sub(dt) >= HOLD_MIN_NS {
                    let wins = windows.read().map(|g| g.clone()).unwrap_or_default();
                    let (wid, zidx) = window_at(dp, &wins);
                    let a = CursorAction {
                        kind:           CursorActionKind::ClickAndHold,
                        timestamp_ns:   now,
                        position:       dp,
                        start_position: Some(dp),
                        end_position:   None,
                        path:           Vec::new(),
                        bbox:           None,
                        duration_ms:    Some(now.saturating_sub(dt) as f64 / 1_000_000.0),
                        distance_px:    Some(0.0),
                        button:         Some("left".into()),
                        click_count:    None,
                        window_id:      wid,
                        z_index:        zidx,
                        confidence:     0.95,
                        source:         "mouse_events".into(),
                    };
                    if args.export_actions { export(&a); }
                    ds.push(a);
                    ds.hold_emitted = true;
                    changed = true;
                }
            }
        }

        if changed {
            state.write(ds.snapshot(args.cap_width, args.cap_height));
        }

        std::thread::sleep(Duration::from_millis(4));
    }
}

// ── Event processing (state machine) ─────────────────────────────────────────

fn process_event(
    ds:   &mut DetState,
    ev:   MouseTapEvent,
    args: &CursorActionArgs,
    wins: &[WindowLayer],
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
            ds.hold_emitted   = false;
            ds.reset_drag();
            ds.last_pos = Some(pos);
        }

        // ── Left mouse up ─────────────────────────────────────────────────────
        MouseTapEvent::LeftMouseUp { ts_ns, x_pts, y_pts, .. } => {
            let pos = to_px(x_pts, y_pts, scale, ox, oy);
            let (wid, zidx) = window_at(pos, wins);

            if ds.is_dragging {
                on_drag_end(ds, pos, ts_ns, wid, zidx, cw, ch, args);
            } else if let (Some(down_pos), Some(down_ts)) =
                (ds.mouse_down_pos, ds.mouse_down_ts)
            {
                let dur_ns = ts_ns.saturating_sub(down_ts);
                let moved  = dist(down_pos, pos);
                if dur_ns <= CLICK_MAX_NS && moved <= CLICK_MAX_MOVEMENT {
                    on_click(ds, pos, down_pos, ts_ns, dur_ns, moved, wid, zidx, args);
                }
                // else: slow release without drag — hold was already emitted or missed
            }

            ds.is_mouse_down = false;
            ds.reset_drag();
            ds.last_pos = Some(pos);
        }

        // ── Left mouse dragged ────────────────────────────────────────────────
        MouseTapEvent::LeftMouseDrag { ts_ns, x_pts, y_pts, .. } => {
            let pos = to_px(x_pts, y_pts, scale, ox, oy);
            if !ds.is_mouse_down {
                // Spurious drag without tracked down
                ds.last_pos = Some(pos);
                return;
            }
            let down_pos = match ds.mouse_down_pos { Some(p) => p, None => { ds.last_pos = Some(pos); return; } };
            let down_ts  = ds.mouse_down_ts.unwrap_or(ts_ns);

            let dist_from_down = dist(down_pos, pos);

            if !ds.is_dragging && dist_from_down > DRAG_START_DIST {
                // DragStart
                let (wid, zidx) = window_at(down_pos, wins);
                ds.drag_window_id = wid;
                ds.drag_z_index   = zidx;
                ds.is_dragging    = true;

                ds.drag_path.push(down_pos);
                ds.update_bbox(down_pos);
                ds.last_drag_sample_pos = Some(down_pos);
                ds.last_drag_sample_ts  = Some(down_ts);

                let a = CursorAction {
                    kind:           CursorActionKind::DragStart,
                    timestamp_ns:   ts_ns,
                    position:       down_pos,
                    start_position: Some(down_pos),
                    end_position:   None,
                    path:           vec![down_pos],
                    bbox:           None,
                    duration_ms:    None,
                    distance_px:    None,
                    button:         Some("left".into()),
                    click_count:    None,
                    window_id:      wid,
                    z_index:        zidx,
                    confidence:     0.95,
                    source:         "mouse_events".into(),
                };
                if args.export_actions { export(&a); }
                ds.push(a);
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

                    if args.export_actions {
                        let bbox = ds.bbox_arr(cw, ch);
                        let a = CursorAction {
                            kind:           CursorActionKind::DragMove,
                            timestamp_ns:   ts_ns,
                            position:       pos,
                            start_position: Some(down_pos),
                            end_position:   None,
                            path:           vec![pos],   // current point only; full path in DragEnd
                            bbox,
                            duration_ms:    Some(ts_ns.saturating_sub(down_ts) as f64 / 1_000_000.0),
                            distance_px:    Some(dist_from_down),
                            button:         Some("left".into()),
                            click_count:    None,
                            window_id:      ds.drag_window_id,
                            z_index:        ds.drag_z_index,
                            confidence:     0.95,
                            source:         "mouse_events".into(),
                        };
                        export(&a);
                        // DragMove not added to recent_actions (high frequency)
                    }
                }
            }

            ds.last_pos = Some(pos);
        }

        // ── Mouse moved (no button held) ──────────────────────────────────────
        MouseTapEvent::MouseMoved { ts_ns, x_pts, y_pts, .. } => {
            let pos = to_px(x_pts, y_pts, scale, ox, oy);

            if args.export_actions {
                let should_sample = match (ds.last_pos, ds.last_move_ts) {
                    (Some(lp), Some(lt)) => {
                        dist(lp, pos) >= MOVE_SAMPLE_MIN_DIST
                            && ts_ns.saturating_sub(lt) >= MOVE_SAMPLE_MIN_NS
                    }
                    _ => true,
                };
                if should_sample {
                    let (wid, zidx) = window_at(pos, wins);
                    let a = CursorAction {
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
                    };
                    export(&a);
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
            if args.export_actions { export(&a); }
            ds.push(a);
        }
    }
}

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
) {
    let is_double = ds.last_click_ts
        .zip(ds.last_click_pos)
        .map_or(false, |(lt, lp)| {
            ts_ns.saturating_sub(lt) <= DOUBLE_CLICK_MAX_NS
                && dist(lp, pos) <= DOUBLE_CLICK_MAX_DIST
        });

    let (kind, count) = if is_double {
        (CursorActionKind::DoubleClick, 2u8)
    } else {
        (CursorActionKind::SingleClick, 1u8)
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
    if args.export_actions { export(&a); }
    ds.push(a);
    ds.last_click_pos = Some(pos);
    ds.last_click_ts  = Some(ts_ns);
}

fn on_drag_end(
    ds:    &mut DetState,
    pos:   (i32, i32),
    ts_ns: u64,
    wid:   Option<u32>,
    zidx:  Option<usize>,
    cw:    u32,
    ch:    u32,
    args:  &CursorActionArgs,
) {
    let down_pos  = ds.mouse_down_pos.unwrap_or(pos);
    let down_ts   = ds.mouse_down_ts.unwrap_or(ts_ns);
    let dur_ms    = ts_ns.saturating_sub(down_ts) as f64 / 1_000_000.0;
    let drag_dist = dist(down_pos, pos);
    let path      = std::mem::take(&mut ds.drag_path);
    let bbox      = ds.bbox_arr(cw, ch);

    let drag_end = CursorAction {
        kind:           CursorActionKind::DragEnd,
        timestamp_ns:   ts_ns,
        position:       pos,
        start_position: Some(down_pos),
        end_position:   Some(pos),
        path:           path.clone(),
        bbox,
        duration_ms:    Some(dur_ms),
        distance_px:    Some(drag_dist),
        button:         Some("left".into()),
        click_count:    None,
        window_id:      wid,
        z_index:        zidx,
        confidence:     0.95,
        source:         "mouse_events".into(),
    };
    if args.export_actions { export(&drag_end); }
    ds.push(drag_end);

    // DragSelect: horizontal-dominant drags may be text selections
    let (h_ext, v_ext) = match ds.drag_bbox_mm {
        Some((mn_x, mn_y, mx_x, mx_y)) => ((mx_x - mn_x) as f64, (mx_y - mn_y) as f64),
        None => (0.0, 0.0),
    };
    if h_ext > v_ext && h_ext > 20.0 {
        let sel = CursorAction {
            kind:           CursorActionKind::DragSelect,
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
            window_id:      wid,
            z_index:        zidx,
            confidence:     0.40,   // low: no visual validation yet
            source:         "mouse_events".into(),
        };
        if args.export_actions { export(&sel); }
        ds.push(sel);
    }
}
