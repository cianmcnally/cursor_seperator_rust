/// rust_cursor_bench — screen capture → cursor-action + window-segmentation dataset.
///
/// Rerun is the single viewer; there are no on-screen panels.
///
///   cargo run                       # live: stream to a spawned Rerun viewer
///   cargo run -- --record-session   # record: dataset + recordings/<id>/debug.rrd
///   rerun recordings/<id>/debug.rrd # review a recorded session
///
/// Per frame the cursor action is exactly one of: idle, move, click, double_click,
/// drag, scroll, typing. In Rerun, toggle the /windows/<id>/full_mask layers to
/// stack window masks and confirm they register on the captured frame.
///
/// Flags:
///   --record-session          write the dataset to recordings/<id>/
///   --session-dir <path>      output root (default: recordings)
///   --max-seconds <n>         stop after n seconds (0 = unlimited)
///   --no-save-frames          skip frame JPEGs
///   --no-save-masks           skip mask PNGs
///   --frame-save-size WxH     downsample saved frames (default 960x540)
///   --native-res              save frames at native resolution
///   --include-self            include this process's own windows
///   --show-system-ui          include Dock / menu-bar windows
///   --normal-windows-only     only layer-0 windows
///   --dump-window-list        print the window list once and exit
///   --dump-screens            print screen geometry and exit
///   --export-cursor-actions   also print cursor-action JSON to stdout
mod coords;
mod cursor_action;
mod debug_viz;
pub mod predictions;
mod recorder;
mod scstream;
mod session_loader;
mod typing;
mod windows;

use std::ffi::c_void;
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::{Arc, RwLock};
use std::time::{Duration, Instant};

// ── Graceful shutdown on Ctrl+C / SIGTERM ─────────────────────────────────────
// Without this, Ctrl+C kills the process before the recorder flushes its NDJSON
// BufWriters, so the last buffered frame records are lost (frames.ndjson ends up
// short of the JPEGs on disk). The handler just flips a flag; the capture loop
// sees it, breaks, and runs the normal graceful-shutdown/flush path.
static STOP: AtomicBool = AtomicBool::new(false);

extern "C" {
    fn signal(sig: i32, handler: extern "C" fn(i32)) -> usize;
}
const SIGINT: i32 = 2;
const SIGTERM: i32 = 15;

extern "C" fn handle_stop_signal(_sig: i32) {
    STOP.store(true, Ordering::SeqCst);
}

fn install_signal_handlers() {
    unsafe {
        signal(SIGINT, handle_stop_signal);
        signal(SIGTERM, handle_stop_signal);
    }
}

use objc::rc::autoreleasepool;
use objc::runtime::Object;
use objc::{class, msg_send, sel, sel_impl};

// --------------------------------------------------------------------------
// CoreGraphics / CoreFoundation FFI
// --------------------------------------------------------------------------
#[repr(C)] #[derive(Copy, Clone)] struct CGPoint { x: f64, y: f64 }
#[repr(C)] #[derive(Copy, Clone)] struct CGSize  { width: f64, height: f64 }
#[repr(C)] #[derive(Copy, Clone)] struct CGRect  { origin: CGPoint, size: CGSize }
#[repr(C)] #[derive(Copy, Clone)] struct NSPoint { x: f64, y: f64 }
#[repr(C)] #[derive(Copy, Clone)] struct NSSize  { width: f64, height: f64 }

// kCGBitmapByteOrder32Little | kCGImageAlphaPremultipliedFirst → BGRA in memory
const BGRA_BITMAP_INFO: u32 = 0x2000 | 2;

#[link(name = "CoreGraphics", kind = "framework")]
extern "C" {
    fn CGEventCreate(src: *const c_void) -> *mut c_void;
    fn CGEventGetLocation(evt: *mut c_void) -> CGPoint;
    fn CFRelease(cf: *const c_void);
    fn CGColorSpaceCreateDeviceRGB() -> *mut c_void;
    fn CGColorSpaceRelease(cs: *mut c_void);
    fn CGBitmapContextCreate(
        data: *mut c_void, w: usize, h: usize, bpc: usize,
        bpr: usize, cs: *mut c_void, bi: u32,
    ) -> *mut c_void;
    fn CGContextRelease(ctx: *mut c_void);
    fn CGContextDrawImage(ctx: *mut c_void, rect: CGRect, img: *mut c_void);
    fn CGImageGetWidth(img: *mut c_void) -> usize;
    fn CGImageGetHeight(img: *mut c_void) -> usize;
}

fn get_mouse_pos() -> (f64, f64) {
    unsafe {
        let e = CGEventCreate(std::ptr::null());
        let p = CGEventGetLocation(e);
        CFRelease(e as _);
        (p.x, p.y)
    }
}

/// Backing scale factor from NSScreen.mainScreen (2.0 on Retina).
fn get_backing_scale() -> f64 {
    autoreleasepool(|| unsafe {
        let screen: *mut Object = msg_send![class!(NSScreen), mainScreen];
        if screen.is_null() { return 2.0; }
        let scale: f64 = msg_send![screen, backingScaleFactor];
        if scale <= 0.0 { 2.0 } else { scale }
    })
}

/// Cursor bounding box in capture-pixel coordinates.
/// `cap_origin_*` are the capture region origin in pixels (0 for full-display).
fn cursor_rect_px(
    mx: f64, my: f64,
    sprite: &CursorSprite,
    cap_origin_x: f64,
    cap_origin_y: f64,
    scale: f64,
) -> windows::model::RectI {
    let ax = sprite.ax;
    let x  = ((mx * scale - cap_origin_x) - sprite.hot_x * ax * scale).round() as i32;
    let y  = ((my * scale - cap_origin_y) - sprite.hot_y * ax * scale).round() as i32;
    let w  = (sprite.pts_w * ax * scale).round().max(1.0) as i32;
    let h  = (sprite.pts_h * ax * scale).round().max(1.0) as i32;
    windows::model::RectI { x, y, w, h }
}

// --------------------------------------------------------------------------
// Config
// --------------------------------------------------------------------------
/// true  → capture REGION_W x REGION_H pixels from (REGION_X, REGION_Y)
/// false → capture full display at native pixel resolution
const USE_REGION: bool = false;

const REGION_X: u32 = 0;
const REGION_Y: u32 = 0;
const REGION_W: u32 = 1280;
const REGION_H: u32 = 720;
const TARGET_FPS: u64 = 5;

// --------------------------------------------------------------------------
// FrameBundle — single snapshot per frame loop iteration
// --------------------------------------------------------------------------
/// All sensor data captured at one point in time.
/// Built once per frame and handed to the pipeline thread (Rerun + optional disk).
/// `pixels` and `windows` are Arc-shared so feeding the ring + the pipeline costs
/// one pointer clone, not a 33 MB deep copy.
struct FrameBundle {
    frame_index:   u64,
    timestamp_ns:  u64,
    pixels:        Arc<Vec<u8>>,
    width:         usize,
    height:        usize,
    bytes_per_row: usize,
    windows:       Arc<Vec<windows::WindowLayer>>,
    cursor_pos_px: (i32, i32),
    cursor_bbox:   Option<[i32; 4]>,
    action_snap:   cursor_action::ActionSnapshot,
    typing_result: typing::TypingDetectorResult,
    cap_w:         u32,
    cap_h:         u32,
    /// Derived from captured windows list — never a live AX query.
    focused_window_id: Option<u32>,
}

/// Extract the focused window ID from the captured window list.
/// The window sampler already assigns mask_role via assign_mask_roles(),
/// so we can read FocusedRoot from the already-sampled data.
fn focused_window_id_from_windows(windows: &[windows::WindowLayer]) -> Option<u32> {
    windows
        .iter()
        .find(|w| w.mask_role == windows::WindowMaskRole::FocusedRoot)
        .map(|w| w.window_id)
}


// --------------------------------------------------------------------------
// Cursor sprite — image data only, no position. Refreshed every ~500ms.
// --------------------------------------------------------------------------
struct CursorSprite {
    pixels: Vec<u8>,           // BGRA premultiplied
    img_w:  usize,
    img_h:  usize,
    hot_x:  f64,               // hotspot in pts, top-left-of-image origin
    hot_y:  f64,
    pts_w:  f64,               // NSImage display size in points
    pts_h:  f64,
    ax:     f64,               // accessibility cursor scale
}

fn load_cursor_sprite() -> Option<CursorSprite> {
    autoreleasepool(|| unsafe {
        let cursor: *mut Object = {
            let sys: *mut Object = msg_send![class!(NSCursor), currentSystemCursor];
            if sys.is_null() { msg_send![class!(NSCursor), arrowCursor] } else { sys }
        };
        if cursor.is_null() { return None; }

        let nsimage: *mut Object = msg_send![cursor, image];
        if nsimage.is_null() { return None; }

        let cgimg: *mut c_void = msg_send![
            nsimage,
            CGImageForProposedRect: (std::ptr::null::<NSPoint>() as *mut NSPoint)
            context: (std::ptr::null::<Object>() as *mut Object)
            hints: (std::ptr::null::<Object>() as *mut Object)
        ];
        if cgimg.is_null() { return None; }

        let img_w = CGImageGetWidth(cgimg);
        let img_h = CGImageGetHeight(cgimg);
        if img_w == 0 || img_h == 0 { return None; }

        let ns_size: NSSize = msg_send![nsimage, size];
        let hot: NSPoint    = msg_send![cursor, hotSpot];

        let ax: f64 = {
            let suite: *mut Object = msg_send![
                class!(NSString),
                stringWithUTF8String: b"com.apple.universalaccess\0".as_ptr()
            ];
            let ud: *mut Object = {
                let alloc: *mut Object = msg_send![class!(NSUserDefaults), alloc];
                let init: *mut Object = msg_send![alloc, initWithSuiteName: suite];
                let _: () = msg_send![init, autorelease];
                init
            };
            if ud.is_null() {
                1.0
            } else {
                let key: *mut Object = msg_send![
                    class!(NSString),
                    stringWithUTF8String: b"mouseDriverCursorSize\0".as_ptr()
                ];
                let val: *mut Object = msg_send![ud, objectForKey: key];
                if val.is_null() { 1.0 } else { msg_send![val, doubleValue] }
            }
        };

        let mut pixels = vec![0u8; img_w * img_h * 4];
        let cs  = CGColorSpaceCreateDeviceRGB();
        let ctx = CGBitmapContextCreate(
            pixels.as_mut_ptr() as *mut c_void,
            img_w, img_h, 8, img_w * 4, cs, BGRA_BITMAP_INFO,
        );
        if ctx.is_null() { CGColorSpaceRelease(cs); return None; }
        CGContextDrawImage(ctx, CGRect {
            origin: CGPoint { x: 0.0, y: 0.0 },
            size:   CGSize  { width: img_w as f64, height: img_h as f64 },
        }, cgimg);
        CGContextRelease(ctx);
        CGColorSpaceRelease(cs);

        Some(CursorSprite {
            pixels, img_w, img_h,
            hot_x: hot.x, hot_y: hot.y,
            pts_w: ns_size.width, pts_h: ns_size.height,
            ax,
        })
    })
}

// --------------------------------------------------------------------------
// CLI args
// --------------------------------------------------------------------------
#[allow(dead_code)]
struct WindowArgs {
    enabled:          bool,
    show_overlay:     bool,
    show_stack:       bool,
    composite_mask:   bool,
    include_self:     bool,
    show_system_ui:   bool,
    normal_only:      bool,
    dump_list:        bool,
    dump_screens:     bool,
    debug_coords:     bool,
}

impl WindowArgs {
    fn parse() -> Self {
        let args: Vec<String> = std::env::args().collect();
        let has = |flag: &str| args.iter().any(|a| a == flag);
        let dump_list  = has("--dump-window-list");
        let dump_scr   = has("--dump-screens");
        let enabled    = !has("--no-window-layers") || dump_list || dump_scr;
        Self {
            enabled,
            show_overlay:   has("--show-window-overlay"),
            show_stack:     has("--show-window-stack"),
            composite_mask: has("--composite-window-mask"),
            include_self:   has("--include-self"),
            show_system_ui: has("--show-system-ui"),
            normal_only:    has("--normal-windows-only"),
            dump_list,
            dump_screens:   dump_scr,
            debug_coords:   has("--debug-coords"),
        }
    }
}

struct TypingFlags {
    show_debug:  bool,
    show_roi:    bool,
    show_diff:   bool,
    typing_only: bool,
}

impl TypingFlags {
    fn parse() -> Self {
        let args: Vec<String> = std::env::args().collect();
        let has = |flag: &str| args.iter().any(|a| a == flag);
        Self {
            show_debug:  has("--show-typing-debug"),
            show_roi:    has("--show-typing-roi"),
            show_diff:   has("--show-typing-diff"),
            typing_only: has("--typing-only"),
        }
    }
    fn any_active(&self) -> bool {
        self.show_debug || self.show_roi || self.show_diff || self.typing_only
    }
}

struct CursorFlags {
    show_actions:     bool,
    show_drag_path:   bool,
    show_click_marks: bool,
    export_actions:   bool,
}

impl CursorFlags {
    fn parse() -> Self {
        let args: Vec<String> = std::env::args().collect();
        let has = |flag: &str| args.iter().any(|a| a == flag);
        Self {
            show_actions:     has("--show-cursor-actions"),
            show_drag_path:   has("--show-drag-path"),
            show_click_marks: has("--show-click-markers"),
            export_actions:   has("--export-cursor-actions"),
        }
    }
    fn any_visible(&self) -> bool {
        self.show_actions || self.show_drag_path || self.show_click_marks
    }
}

// --------------------------------------------------------------------------
// Recording CLI args — parsed and owned by main, passed to recorder module
// --------------------------------------------------------------------------
// These flags are re-exported from recorder::RecordArgs for discoverability.
//   --record-session          start a recording session
//   --session-dir <path>      root directory (default: recordings)
//   --no-save-frames          disable frame PNG output (on by default)
//   --no-save-masks           disable label/mask PNG output (on by default)
//   --no-save-overlays        disable debug overlay PNG output (on by default)
//   --dump-session-json       print session.json to stdout and exit
//   --max-seconds <n>         stop recording after n seconds (0 = unlimited)

fn main() {
    install_signal_handlers();

    let wargs  = WindowArgs::parse();
    let tflags = TypingFlags::parse();
    let cflags = CursorFlags::parse();
    let rargs  = recorder::RecordArgs::parse();

    println!("Starting ScreenCaptureKit stream (requires Screen Recording permission)…");

    let source = if USE_REGION {
        scstream::CaptureSource::Region { x: REGION_X, y: REGION_Y, w: REGION_W, h: REGION_H }
    } else {
        scstream::CaptureSource::FullDisplay
    };
    let info = scstream::start_capture(source, TARGET_FPS as u32);

    // Capture-region origin in pixels (0,0 for full-display) + Retina scale.
    let backing_scale   = get_backing_scale();
    let cap_origin_x_px = info.origin_x as f64;
    let cap_origin_y_px = info.origin_y as f64;

    // ── Window sampler thread ────────────────────────────────────────────────
    // Shared state: latest window list (front-to-back) + timing snapshot.
    let shared_windows: Arc<RwLock<Vec<windows::WindowLayer>>> =
        Arc::new(RwLock::new(Vec::new()));
    let shared_timings: Arc<RwLock<windows::WindowTimings>> =
        Arc::new(RwLock::new(windows::WindowTimings::default()));

    // Window sampler always runs — cutout panels need it regardless of flags.
    {
        let sw = shared_windows.clone();
        let st = shared_timings.clone();
        let cap_w_px  = info.width  as u32;
        let cap_h_px  = info.height as u32;
        let origin_x  = info.origin_x as f64;
        let origin_y  = info.origin_y as f64;
        let inc_self  = wargs.include_self;
        let sys_ui    = wargs.show_system_ui;
        let norm_only = wargs.normal_only;

        std::thread::spawn(move || {
            let desktop = windows::DesktopGeometry::from_capture(
                cap_w_px, cap_h_px, origin_x, origin_y,
            );
            let mut sampler = windows::WindowSampler::new(inc_self, sys_ui, norm_only);
            loop {
                let t0 = Instant::now();
                let layers = sampler.sample(&desktop);
                let sample_ms = t0.elapsed().as_secs_f64() * 1000.0;
                let seg_count = layers.iter().filter(|w| w.include_in_segmentation).count();
                let raw_count = layers.len();
                if let Ok(mut g) = sw.write() { *g = layers; }
                if let Ok(mut g) = st.write() {
                    g.window_sample_ms          = sample_ms;
                    g.raw_window_count          = raw_count;
                    g.segmentation_window_count = seg_count;
                }
                std::thread::sleep(Duration::from_millis(100));
            }
        });
    }

    // --dump-session-json: print session metadata and exit.
    if rargs.dump_session_json {
        let session = recorder::session::SessionMeta::new(
            info.width as u32, info.height as u32,
            info.origin_x as f64, info.origin_y as f64,
        );
        println!("{}", serde_json::to_string_pretty(&session).unwrap_or_default());
        return;
    }

    // ── Pipeline: capture → Rerun (+ optional disk). Always on. ──────────────
    // Live: streams to a spawned Rerun viewer. --record-session: also writes the
    // dataset + a debug.rrd. Started before the detectors so we can hand them the
    // event/key channels — live runs log events/keys to Rerun too.
    let (recorder_handle, session_dir) = match recorder::start_recorder(
        &rargs,
        info.width as u32, info.height as u32,
        info.origin_x as f64, info.origin_y as f64,
    ) {
        Ok(v) => v,
        Err(e) => { eprintln!("[pipeline] failed to start: {e}"); return; }
    };
    match &session_dir {
        Some(dir) => println!("[recorder] session → {}", dir.display()),
        None      => println!("[viewer] live preview — pass --record-session to save a dataset"),
    }

    // ── Frame ring buffer + typing detector ─────────────────────────────────
    let frame_ring = typing::FrameRingBuffer::new(10);
    let typing_state = typing::start_typing_detector(
        frame_ring.clone(),
        typing::TypingArgs {
            show_diff:    tflags.show_diff,
            scale:        backing_scale,
            cap_origin_x: cap_origin_x_px,
            cap_origin_y: cap_origin_y_px,
            key_tx:       Some(recorder_handle.key_sender()),
        },
    );
    if tflags.any_active() {
        println!("Typing detector started (requires Input Monitoring permission).");
    }

    // ── Cursor-action detector ───────────────────────────────────────────────
    let (action_state, tap_alive) = cursor_action::start_cursor_action_detector(
        shared_windows.clone(),
        cursor_action::CursorActionArgs {
            scale:          backing_scale,
            cap_origin_x:   cap_origin_x_px,
            cap_origin_y:   cap_origin_y_px,
            cap_width:      info.width  as u32,
            cap_height:     info.height as u32,
            export_actions: cflags.export_actions,
            event_tx:       Some(recorder_handle.event_sender()),
        },
    );
    if cflags.any_visible() || cflags.export_actions {
        println!("Cursor-action detector started (requires Input Monitoring permission).");
    }

    // Give the tap a moment to create, then verify it came up. A recording with no
    // mouse events is worthless — fail loud rather than silently produce a dud.
    std::thread::sleep(Duration::from_millis(300));
    if !tap_alive.load(Ordering::Relaxed) {
        eprintln!("\n========================================================================");
        eprintln!(" INPUT MONITORING PERMISSION DENIED — mouse event tap failed to start.");
        eprintln!(" This session would record 0 mouse events (clicks/drags/scrolls).");
        eprintln!(" Grant it in System Settings → Privacy & Security → Input Monitoring,");
        eprintln!(" then restart. (Keyboard typing detection needs the same permission.)");
        eprintln!("========================================================================\n");
        if rargs.enabled {
            eprintln!("[recorder] aborting recording — fix permission first.");
            recorder_handle.shutdown();
            return;
        }
    }

    // --dump-window-list: sample once, print, exit.
    if wargs.dump_list {
        std::thread::sleep(Duration::from_millis(300));
        if let Ok(g) = shared_windows.read() {
            windows::dump_window_list(&g);
        }
        return;
    }

    // --dump-screens: print screen geometry, exit.
    if wargs.dump_screens {
        let (sw, sh) = scstream::main_display_pixels();
        println!("Main display: {}×{} px  capture: {}×{} px  origin: ({},{})",
            sw, sh, info.width, info.height, info.origin_x, info.origin_y);
        return;
    }

    // ── Unified capture loop ─────────────────────────────────────────────────
    // One path for live and record. Snapshot all sensor state into a FrameBundle,
    // hand it to the pipeline thread (Rerun always; disk when recording), pace to
    // TARGET_FPS. No on-screen panels — Rerun is the only viewer.
    let frame_budget = Duration::from_micros(1_000_000 / TARGET_FPS);
    let mut local_frame = scstream::FrameData {
        pixels:        vec![0u8; info.width * info.height * 4],
        width:         info.width,
        height:        info.height,
        bytes_per_row: info.width * 4,
        seq:           0,
        captured_at:   Instant::now(),
    };
    // Cursor sprite is cached and refreshed ~2.5×/s — the Obj-C FFI + bitmap build
    // is far too heavy to run every frame.
    let mut sprite: Option<CursorSprite> = load_cursor_sprite();
    let mut last_sprite_load = Instant::now();
    let mut last_print       = Instant::now();
    let mut last_ring_seq    = 0u64;
    let mut last_record_seq  = 0u64;
    let mut record_frame_idx = 0u64;
    let record_start         = Instant::now();

    println!("Running at {} fps. Ctrl+C to stop — view in Rerun.", TARGET_FPS);
    println!("Capture: {}×{} px", info.width, info.height);

    loop {
        if STOP.load(Ordering::Relaxed) {
            println!("\n[recorder] stop signal — flushing…");
            break;
        }
        if rargs.max_seconds > 0 && record_start.elapsed().as_secs() >= rargs.max_seconds {
            break;
        }
        let t0 = Instant::now();

        // ── Capture: lock SharedFrame, memcpy, release ──
        if let Ok(guard) = info.frame.try_lock() {
            if let Some(ref f) = *guard {
                if local_frame.pixels.len() < f.pixels.len() {
                    local_frame.pixels.resize(f.pixels.len(), 0);
                }
                local_frame.pixels[..f.pixels.len()].copy_from_slice(&f.pixels);
                local_frame.width         = f.width;
                local_frame.height        = f.height;
                local_frame.bytes_per_row = f.bytes_per_row;
                local_frame.seq           = f.seq;
                local_frame.captured_at   = f.captured_at;
            }
        }

        // ── Cursor: refresh cached sprite + read live position ──
        if last_sprite_load.elapsed() >= Duration::from_millis(400) {
            if let Some(s) = load_cursor_sprite() { sprite = Some(s); }
            last_sprite_load = Instant::now();
        }
        let (mx, my) = get_mouse_pos();

        if local_frame.width == 0 {
            std::thread::sleep(frame_budget);
            continue;
        }

        // ── Build the per-frame bundle (one pixel alloc, Arc-shared downstream) ──
        let used_len = local_frame.height * local_frame.bytes_per_row;
        let mx_px = ((mx * backing_scale) - cap_origin_x_px) as i32;
        let my_px = ((my * backing_scale) - cap_origin_y_px) as i32;
        let cursor_bbox = sprite.as_ref().map(|s| {
            let r = cursor_rect_px(mx, my, s, cap_origin_x_px, cap_origin_y_px, backing_scale);
            [r.x, r.y, r.w, r.h]
        });
        let bundle_windows = Arc::new(
            shared_windows.read().map(|g| g.clone()).unwrap_or_default()
        );
        let focused_window_id = focused_window_id_from_windows(&bundle_windows);
        let bundle = FrameBundle {
            frame_index:   record_frame_idx,
            timestamp_ns:  std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .unwrap_or_default()
                .as_nanos() as u64,
            pixels:        Arc::new(local_frame.pixels[..used_len].to_vec()),
            width:         local_frame.width,
            height:        local_frame.height,
            bytes_per_row: local_frame.bytes_per_row,
            windows:       bundle_windows,
            cursor_pos_px: (mx_px, my_px),
            cursor_bbox,
            action_snap:   action_state.snapshot(),
            typing_result: typing_state.read(),
            cap_w:         info.width as u32,
            cap_h:         info.height as u32,
            focused_window_id,
        };

        // ── Feed the typing ring (shares the pixel Arc; one window clone) ──
        if local_frame.seq != last_ring_seq {
            last_ring_seq = local_frame.seq;
            let cur_rect = sprite.as_ref().map(|s| {
                cursor_rect_px(mx, my, s, cap_origin_x_px, cap_origin_y_px, backing_scale)
            });
            frame_ring.push(typing::RingEntry {
                captured_at:   local_frame.captured_at,
                _timestamp_ns:  bundle.timestamp_ns,
                pixels:        Arc::clone(&bundle.pixels),
                width:         bundle.width,
                height:        bundle.height,
                bytes_per_row: bundle.bytes_per_row,
                cursor_rect:   cur_rect,
                windows:       (*bundle.windows).clone(),
            });
        }

        // ── Hand the frame to the pipeline (Rerun always; disk when recording) ──
        if local_frame.seq != last_record_seq {
            last_record_seq = local_frame.seq;
            recorder_handle.send_frame(recorder::FrameWriteTask {
                frame_index:                  bundle.frame_index,
                timestamp_ns:                 bundle.timestamp_ns,
                pixels:                       Arc::clone(&bundle.pixels),
                width:                        bundle.width,
                height:                       bundle.height,
                bytes_per_row:                bundle.bytes_per_row,
                windows:                      Arc::clone(&bundle.windows),
                cursor_pos_px:                bundle.cursor_pos_px,
                cursor_bbox:                  bundle.cursor_bbox,
                cursor_sprite:                sprite.as_ref().map(|s| Arc::new(s.pixels.clone())),
                cursor_sprite_w:              sprite.as_ref().map(|s| s.img_w).unwrap_or(0),
                cursor_sprite_h:              sprite.as_ref().map(|s| s.img_h).unwrap_or(0),
                action_snap:                  bundle.action_snap,
                typing_result:                bundle.typing_result.clone(),
                cap_w:                        bundle.cap_w,
                cap_h:                        bundle.cap_h,
                focused_window_id_at_capture: bundle.focused_window_id,
            });
            record_frame_idx += 1;
        }

        // ── Stats once per second ──
        if last_print.elapsed() >= Duration::from_secs(1) {
            println!("frame {:>6}  ({} fps target)  Ctrl+C to stop", record_frame_idx, TARGET_FPS);
            if let Ok(wt) = shared_timings.read() {
                println!("  windows: sample {:.2}ms  raw={} seg={}",
                    wt.window_sample_ms, wt.raw_window_count, wt.segmentation_window_count);
            }
            // Live cursor-event tally — if this stays empty while you click/drag,
            // the mouse tap is not delivering (Input Monitoring / tap problem).
            if let Ok(c) = action_state.session_counts.lock() {
                let clicks = c.get("click").copied().unwrap_or(0)
                    + c.get("double_click").copied().unwrap_or(0);
                let drags  = c.get("drag").copied().unwrap_or(0);
                let scrolls = c.get("scroll").copied().unwrap_or(0);
                let moves  = c.get("move").copied().unwrap_or(0);
                println!("  events: click={clicks} drag={drags} scroll={scrolls} move={moves}");
            }
            last_print = Instant::now();
        }

        // ── Pace ──
        let elapsed = t0.elapsed();
        if elapsed < frame_budget {
            std::thread::sleep(frame_budget - elapsed);
        }
    }

    recorder_handle.shutdown();
    if let Some(dir) = session_dir {
        println!("\n── Recording complete: {} frames ──", record_frame_idx);
        println!("View in Rerun:  rerun {}/debug.rrd", dir.display());
    }
}
