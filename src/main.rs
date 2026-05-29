/// Live 3-panel cursor composite benchmark — Rust.
/// Panels: Raw | Cursor-only | Composite
/// Stats printed to terminal every second.
///
///   source ~/.cargo/env && cargo run --release
mod scstream;

use std::collections::VecDeque;
use std::ffi::c_void;
use std::time::{Duration, Instant};

use minifb::{Key, Window, WindowOptions};
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
const PANEL_W:  usize = 640;
const PANEL_H:  usize = 360;
const TARGET_FPS: u64 = 30;


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
// Cursor scaling map cache — recomputed only when dimensions change
// --------------------------------------------------------------------------
struct ScaleCache {
    xmap: Vec<usize>,
    ymap: Vec<usize>,
    pw: usize, ph: usize,
    cw: usize, ch: usize,
}

impl ScaleCache {
    fn new() -> Self {
        Self { xmap: Vec::new(), ymap: Vec::new(), pw: 0, ph: 0, cw: 0, ch: 0 }
    }

    fn update(&mut self, pw: usize, ph: usize, cw: usize, ch: usize) {
        if pw == self.pw && ph == self.ph && cw == self.cw && ch == self.ch { return; }
        self.xmap = (0..pw).map(|x| x * cw / pw).collect();
        self.ymap = (0..ph).map(|y| y * ch / ph).collect();
        self.pw = pw; self.ph = ph; self.cw = cw; self.ch = ch;
    }
}

// --------------------------------------------------------------------------
// Composite cursor onto a u32 fb panel. Clips once, no per-pixel bounds check.
// --------------------------------------------------------------------------
fn composite_cursor_fb(
    fb: &mut [u32], fb_w: usize, panel_x: usize,
    panel_w: usize, panel_h: usize,
    sprite: &CursorSprite,
    pos_x: f64, pos_y: f64,
    region_x: f64, region_y: f64,
    region_w: f64, region_h: f64,
    sc: &mut ScaleCache,
) {
    let sx = panel_w as f64 / region_w;
    let sy = panel_h as f64 / region_h;
    let ax = sprite.ax;

    let tl_x = (pos_x - region_x - sprite.hot_x * ax) * sx;
    let tl_y = (pos_y - region_y - sprite.hot_y * ax) * sy;

    let pw = ((sprite.pts_w * ax * sx).round() as usize).max(1);
    let ph = ((sprite.pts_h * ax * sy).round() as usize).max(1);
    let p_tl_x = tl_x.round() as i32;
    let p_tl_y = tl_y.round() as i32;

    // Clip to panel bounds once
    let x0 = p_tl_x.max(0) as usize;
    let y0 = p_tl_y.max(0) as usize;
    let x1 = ((p_tl_x + pw as i32) as usize).min(panel_w);
    let y1 = ((p_tl_y + ph as i32) as usize).min(panel_h);
    if x0 >= x1 || y0 >= y1 { return; }

    sc.update(pw, ph, sprite.img_w, sprite.img_h);
    let cw = sprite.img_w;

    for dy in y0..y1 {
        let py  = (dy as i32 - p_tl_y) as usize;
        let cy  = sc.ymap[py];
        let row = dy * fb_w + panel_x;

        for dx in x0..x1 {
            let px = (dx as i32 - p_tl_x) as usize;
            let cx = sc.xmap[px];
            let ci = (cy * cw + cx) * 4;
            let a  = sprite.pixels[ci + 3] as u32;
            if a == 0 { continue; }
            let ia = 255 - a;

            // cursor pixels: BGRA = [ci+0, ci+1, ci+2, ci+3]
            let src_b = sprite.pixels[ci]     as u32;
            let src_g = sprite.pixels[ci + 1] as u32;
            let src_r = sprite.pixels[ci + 2] as u32;

            let dst = fb[row + dx];
            let dst_r = (dst >> 16) & 0xFF;
            let dst_g = (dst >>  8) & 0xFF;
            let dst_b =  dst        & 0xFF;

            // premultiplied over: out = src + dst*(1-alpha)
            let out_r = (src_r + dst_r * ia / 255).min(255);
            let out_g = (src_g + dst_g * ia / 255).min(255);
            let out_b = (src_b + dst_b * ia / 255).min(255);

            fb[row + dx] = 0xFF000000 | (out_r << 16) | (out_g << 8) | out_b;
        }
    }
}

// --------------------------------------------------------------------------
// Nearest-neighbour downsample from captured BGRA frame → u32 panel.
// Handles stride (bytes_per_row ≥ width*4) and arbitrary scale ratios.
// BGRA byte order: [si]=B [si+1]=G [si+2]=R [si+3]=A
// --------------------------------------------------------------------------
fn write_panel_bgra_scaled_to_fb(
    frame: &scstream::FrameData,
    fb: &mut [u32], fb_w: usize, panel_x: usize,
    panel_w: usize, panel_h: usize,
) {
    let src_w = frame.width;
    let src_h = frame.height;
    let bpr   = frame.bytes_per_row;
    if src_w == 0 || src_h == 0 { return; }
    for dy in 0..panel_h {
        let sy      = dy * src_h / panel_h;
        let fb_row  = dy * fb_w + panel_x;
        let src_row = sy * bpr;
        for dx in 0..panel_w {
            let sx = dx * src_w / panel_w;
            let si = src_row + sx * 4;
            fb[fb_row + dx] = 0xFF000000
                | ((frame.pixels[si + 2] as u32) << 16)
                | ((frame.pixels[si + 1] as u32) <<  8)
                |   frame.pixels[si]     as u32;
        }
    }
}

fn fill_panel_fb(fb: &mut [u32], fb_w: usize, panel_x: usize, dw: usize, dh: usize, color: u32) {
    for dy in 0..dh {
        let row = dy * fb_w + panel_x;
        fb[row..row + dw].fill(color);
    }
}

// --------------------------------------------------------------------------
// Perf — O(1) avg via rolling sum
// --------------------------------------------------------------------------
#[derive(Clone, Default)]
struct FrameStats {
    capture_ms:   f64,
    cursor_ms:    f64,
    composite_ms: f64,
    update_ms:    f64,  // update_with_buffer only
    sleep_ms:     f64,  // intentional frame-pace sleep
    work_ms:      f64,  // total minus sleep
    total_ms:     f64,
    frame_age_ms: f64,  // age of latest SCK frame at display time
}

struct PerfRing {
    buf:        VecDeque<FrameStats>,
    sum:        FrameStats,
    cap:        usize,
    fps_frames: u32,
    fps_timer:  Instant,
    pub fps:    f64,
}

impl PerfRing {
    fn new(cap: usize) -> Self {
        Self {
            buf: VecDeque::with_capacity(cap),
            sum: FrameStats::default(),
            cap, fps_frames: 0, fps_timer: Instant::now(), fps: 0.0,
        }
    }

    fn push(&mut self, s: FrameStats) {
        if self.buf.len() >= self.cap {
            let old = self.buf.pop_front().unwrap();
            self.sum.capture_ms   -= old.capture_ms;
            self.sum.cursor_ms    -= old.cursor_ms;
            self.sum.composite_ms -= old.composite_ms;
            self.sum.update_ms    -= old.update_ms;
            self.sum.sleep_ms     -= old.sleep_ms;
            self.sum.work_ms      -= old.work_ms;
            self.sum.total_ms     -= old.total_ms;
            self.sum.frame_age_ms -= old.frame_age_ms;
        }
        self.sum.capture_ms   += s.capture_ms;
        self.sum.cursor_ms    += s.cursor_ms;
        self.sum.composite_ms += s.composite_ms;
        self.sum.update_ms    += s.update_ms;
        self.sum.sleep_ms     += s.sleep_ms;
        self.sum.work_ms      += s.work_ms;
        self.sum.total_ms     += s.total_ms;
        self.sum.frame_age_ms += s.frame_age_ms;
        self.buf.push_back(s);

        self.fps_frames += 1;
        let e = self.fps_timer.elapsed().as_secs_f64();
        if e >= 0.5 {
            self.fps = self.fps_frames as f64 / e;
            self.fps_frames = 0;
            self.fps_timer = Instant::now();
        }
    }

    fn avg(&self) -> FrameStats {
        let n = self.buf.len() as f64;
        if n == 0.0 { return FrameStats::default(); }
        FrameStats {
            capture_ms:   self.sum.capture_ms   / n,
            cursor_ms:    self.sum.cursor_ms     / n,
            composite_ms: self.sum.composite_ms  / n,
            update_ms:    self.sum.update_ms     / n,
            sleep_ms:     self.sum.sleep_ms      / n,
            work_ms:      self.sum.work_ms       / n,
            total_ms:     self.sum.total_ms      / n,
            frame_age_ms: self.sum.frame_age_ms  / n,
        }
    }
}

// --------------------------------------------------------------------------
// 6x8 bitmap font — writes directly into fb with panel_x offset
// --------------------------------------------------------------------------
const FONT_W: usize = 6;
const FONT_H: usize = 8;
static FONT6X8: &[u8; 95 * 8] = include_bytes!("font6x8.bin");

fn stamp_char_fb(
    fb: &mut [u32], fb_w: usize, panel_x: usize,
    panel_w: usize, panel_h: usize,
    x0: usize, y0: usize, ch: char, color: u32,
) {
    let idx = (ch as usize).saturating_sub(0x20).min(94);
    for row in 0..FONT_H {
        let byte = FONT6X8[idx * FONT_H + row];
        for col in 0..FONT_W {
            if byte & (0x80 >> col) != 0 {
                let x = x0 + col;
                let y = y0 + row;
                if x < panel_w && y < panel_h {
                    fb[y * fb_w + panel_x + x] = color;
                }
            }
        }
    }
}

fn stamp_str_fb(
    fb: &mut [u32], fb_w: usize, panel_x: usize,
    panel_w: usize, panel_h: usize,
    x0: usize, y0: usize, s: &str, color: u32,
) {
    for (i, ch) in s.chars().enumerate() {
        stamp_char_fb(fb, fb_w, panel_x, panel_w, panel_h,
                      x0 + i * (FONT_W + 1), y0, ch, color);
    }
}

fn draw_stats_fb(
    fb: &mut [u32], fb_w: usize, panel_x: usize,
    panel_w: usize, panel_h: usize,
    s: &FrameStats, fps: f64,
) {
    let lines = [
        format!("FPS       {:5.1}", fps),
        format!("total     {:5.2}ms", s.total_ms),
        format!("work      {:5.2}ms", s.work_ms),
        format!("sleep     {:5.2}ms", s.sleep_ms),
        format!("frame age {:5.2}ms", s.frame_age_ms),
        format!("capture   {:5.2}ms", s.capture_ms),
        format!("cursor    {:5.2}ms", s.cursor_ms),
        format!("composite {:5.2}ms", s.composite_ms),
        format!("update    {:5.2}ms", s.update_ms),
    ];
    let box_w = 160usize;
    let box_h = lines.len() * (FONT_H + 2) + 6;
    let x0 = 6usize;
    let y0 = 6usize;
    for y in y0..y0 + box_h {
        let row = y * fb_w + panel_x;
        for x in x0..x0 + box_w {
            if y < panel_h && x < panel_w {
                fb[row + x] = 0xBB000000;
            }
        }
    }
    for (i, line) in lines.iter().enumerate() {
        stamp_str_fb(fb, fb_w, panel_x, panel_w, panel_h,
                     x0 + 4, y0 + 4 + i * (FONT_H + 2), line, 0xFFFFFF00);
    }
}

// --------------------------------------------------------------------------
// Main
// --------------------------------------------------------------------------
fn main() {
    println!("Starting ScreenCaptureKit stream (requires Screen Recording permission)…");

    let source = if USE_REGION {
        scstream::CaptureSource::Region { x: REGION_X, y: REGION_Y, w: REGION_W, h: REGION_H }
    } else {
        scstream::CaptureSource::FullDisplay
    };
    let info = scstream::start_capture(source, TARGET_FPS as u32);

    // Cursor compositing uses the real captured region's geometry
    let cap_origin_x = info.origin_x as f64;
    let cap_origin_y = info.origin_y as f64;
    let cap_w        = info.width  as f64;
    let cap_h        = info.height as f64;

    let total_w = PANEL_W * 3;
    let mut win = Window::new(
        "Cursor Bench  |  Rust  (SCK)",
        total_w, PANEL_H,
        WindowOptions { resize: false, ..Default::default() },
    ).expect("window");
    win.limit_update_rate(None);

    // Local copy of the latest SCK frame — sized for the capture resolution.
    // Avoids holding the SCK mutex during the (possibly slow) preview blit.
    let mut local_frame = scstream::FrameData {
        pixels:        vec![0u8; info.width * info.height * 4],
        width:         info.width,
        height:        info.height,
        bytes_per_row: info.width * 4,
        seq:           0,
        captured_at:   std::time::Instant::now(),
    };

    let mut fb         = vec![0u32; total_w * PANEL_H];
    let mut perf       = PerfRing::new(60);
    let mut sc         = ScaleCache::new();
    let mut last_print = Instant::now();

    println!("Running. ESC to quit. Stats printed every second.");
    println!("Capture: {}×{} pixels  Preview panels: {}×{}", info.width, info.height, PANEL_W, PANEL_H);

    while win.is_open() && !win.is_key_down(Key::Escape) {
        let t0 = Instant::now();

        // ── Capture: lock SharedFrame, memcpy pixels, release lock ────────
        // capture_ms = mutex contention + memcpy; SCK callback runs on bg_queue.
        let t = Instant::now();
        if let Ok(guard) = info.frame.try_lock() {
            if let Some(ref f) = *guard {
                // Grow local buffer if stride differs from expected
                if local_frame.pixels.len() < f.pixels.len() {
                    local_frame.pixels.resize(f.pixels.len(), 0);
                }
                local_frame.pixels[..f.pixels.len()].copy_from_slice(&f.pixels);
                local_frame.width        = f.width;
                local_frame.height       = f.height;
                local_frame.bytes_per_row = f.bytes_per_row;
                local_frame.seq          = f.seq;
                local_frame.captured_at   = f.captured_at;
            }
        }
        let capture_ms   = t.elapsed().as_secs_f64() * 1000.0;
        let frame_age_ms = local_frame.captured_at.elapsed().as_secs_f64() * 1000.0;

        // ── Cursor: full sprite + position every frame ───────────────────
        let t = Instant::now();
        let sprite = load_cursor_sprite();
        let (mx, my) = get_mouse_pos();
        let cursor_ms = t.elapsed().as_secs_f64() * 1000.0;

        // ── Build panels: downsample capture → PANEL_W×PANEL_H preview ───
        let t = Instant::now();

        // Panel 1: raw preview (nearest-neighbour downsample)
        write_panel_bgra_scaled_to_fb(&local_frame, &mut fb, total_w, 0, PANEL_W, PANEL_H);

        // Panel 2: cursor-only on dark background
        fill_panel_fb(&mut fb, total_w, PANEL_W, PANEL_W, PANEL_H, 0xFF0D0D0D);

        // Panel 3: composite preview background
        write_panel_bgra_scaled_to_fb(&local_frame, &mut fb, total_w, PANEL_W * 2, PANEL_W, PANEL_H);

        // Cursor overlay — map from real capture coords to panel pixels
        if let Some(ref s) = sprite {
            composite_cursor_fb(&mut fb, total_w, PANEL_W, PANEL_W, PANEL_H, s, mx, my,
                                 cap_origin_x, cap_origin_y, cap_w, cap_h, &mut sc);
            composite_cursor_fb(&mut fb, total_w, PANEL_W * 2, PANEL_W, PANEL_H, s, mx, my,
                                 cap_origin_x, cap_origin_y, cap_w, cap_h, &mut sc);
        }
        let composite_ms = t.elapsed().as_secs_f64() * 1000.0;

        // ── Display: stats overlay + update_with_buffer ───────────────────
        let avg = perf.avg();
        draw_stats_fb(&mut fb, total_w, PANEL_W * 2, PANEL_W, PANEL_H, &avg, perf.fps);

        let t = Instant::now();
        win.update_with_buffer(&fb, total_w, PANEL_H).unwrap();
        let update_ms = t.elapsed().as_secs_f64() * 1000.0;

        let work_ms = t0.elapsed().as_secs_f64() * 1000.0;

        // ── Manual frame pace ─────────────────────────────────────────────
        let frame_budget = Duration::from_micros(1_000_000 / TARGET_FPS);
        let elapsed      = t0.elapsed();
        if elapsed < frame_budget {
            std::thread::sleep(frame_budget - elapsed);
        }

        let total_ms = t0.elapsed().as_secs_f64() * 1000.0;
        let sleep_ms = total_ms - work_ms;

        perf.push(FrameStats {
            capture_ms, cursor_ms, composite_ms,
            update_ms, sleep_ms, work_ms, total_ms, frame_age_ms,
        });

        if last_print.elapsed() >= Duration::from_secs(1) {
            let a = perf.avg();
            println!(
                "fps {:5.1}  total {:5.1}ms  work {:5.1}ms  sleep {:5.1}ms  age {:5.1}ms  capture {:4.2}ms  cursor {:4.2}ms  composite {:4.2}ms  update {:4.2}ms",
                perf.fps, a.total_ms, a.work_ms, a.sleep_ms, a.frame_age_ms, a.capture_ms, a.cursor_ms, a.composite_ms, a.update_ms
            );
            last_print = Instant::now();
        }
    }
}
