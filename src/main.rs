/// Live 3-panel cursor composite benchmark — Rust.
/// Panels: Raw | Cursor-only | Composite
/// Stats printed to terminal every second.
///
///   source ~/.cargo/env && cargo run --release
use std::collections::VecDeque;
use std::ffi::c_void;
use std::time::{Duration, Instant};

use minifb::{Key, Window, WindowOptions};
use objc::rc::autoreleasepool;
use objc::runtime::Object;
use objc::{class, msg_send, sel, sel_impl};
use screenshots::Screen;

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

// --------------------------------------------------------------------------
// Config
// --------------------------------------------------------------------------
const REGION_X: u32 = 0;
const REGION_Y: u32 = 0;
const REGION_W: u32 = 1280;
const REGION_H: u32 = 720;
const PANEL_W:  usize = 640;
const PANEL_H:  usize = 360;
const TARGET_FPS: u64 = 30;

// --------------------------------------------------------------------------
// Cursor snap — real cursor image, size, hotspot from NSCursor
// --------------------------------------------------------------------------
struct CursorSnap {
    pos_x: f64, pos_y: f64,   // screen coords, top-left origin (CGEventGetLocation)
    pixels: Vec<u8>,           // BGRA premultiplied
    img_w: usize, img_h: usize,
    hot_x: f64, hot_y: f64,   // hotspot in pts, top-left-of-image origin
    pts_w: f64, pts_h: f64,   // NSImage display size in points
    ax: f64,                   // accessibility cursor scale
}

fn get_cursor_snap() -> Option<CursorSnap> {
    autoreleasepool(|| unsafe {
        let cursor: *mut Object = {
            let sys: *mut Object = msg_send![class!(NSCursor), currentSystemCursor];
            if sys.is_null() { msg_send![class!(NSCursor), arrowCursor] } else { sys }
        };
        if cursor.is_null() { return None; }

        let nsimage: *mut Object = msg_send![cursor, image];
        if nsimage.is_null() { return None; }

        // CGImage from NSImage (valid for autorelease pool lifetime)
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

        // Accessibility cursor scale from com.apple.universalaccess
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

        // Render CGImage → BGRA pixels
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

        // Mouse position (top-left origin — same as CGEventGetLocation)
        let ev  = CGEventCreate(std::ptr::null());
        let pos = CGEventGetLocation(ev);
        CFRelease(ev as _);

        Some(CursorSnap {
            pos_x: pos.x, pos_y: pos.y,
            pixels,
            img_w, img_h,
            hot_x: hot.x, hot_y: hot.y,
            pts_w: ns_size.width, pts_h: ns_size.height,
            ax,
        })
    })
}

// Composite cursor BGRA (premultiplied) onto a BGRA dst buffer.
// Handles hotspot offset, accessibility scale, and panel scaling.
fn composite_cursor(
    dst: &mut [u8], dw: usize, dh: usize,
    snap: &CursorSnap,
    region_x: f64, region_y: f64,
    region_w: f64, region_h: f64,
) {
    let sx = dw as f64 / region_w;
    let sy = dh as f64 / region_h;
    let ax = snap.ax;

    // Cursor tip in capture-region coords (top-left origin)
    let tip_x = snap.pos_x - region_x;
    let tip_y = snap.pos_y - region_y;

    // Top-left of cursor image in region coords
    let tl_x = tip_x - snap.hot_x * ax;
    let tl_y = tip_y - snap.hot_y * ax;

    // Cursor display size in panel pixels
    let pw = ((snap.pts_w * ax * sx).round() as usize).max(1);
    let ph = ((snap.pts_h * ax * sy).round() as usize).max(1);
    let p_tl_x = (tl_x * sx).round() as i32;
    let p_tl_y = (tl_y * sy).round() as i32;

    let cw = snap.img_w;
    let ch = snap.img_h;

    for py in 0..ph {
        let dy = p_tl_y + py as i32;
        if dy < 0 || dy >= dh as i32 { continue; }
        let cy = py * ch / ph;
        for px in 0..pw {
            let dx = p_tl_x + px as i32;
            if dx < 0 || dx >= dw as i32 { continue; }
            let cx = px * cw / pw;
            let ci = (cy * cw + cx) * 4;
            let di = (dy as usize * dw + dx as usize) * 4;
            let a  = snap.pixels[ci + 3] as u32;
            if a == 0 { continue; }
            let ia = 255 - a;
            // Premultiplied over: out = src + dst*(1-alpha)
            dst[di]     = (snap.pixels[ci]     as u32 + dst[di]     as u32 * ia / 255).min(255) as u8;
            dst[di + 1] = (snap.pixels[ci + 1] as u32 + dst[di + 1] as u32 * ia / 255).min(255) as u8;
            dst[di + 2] = (snap.pixels[ci + 2] as u32 + dst[di + 2] as u32 * ia / 255).min(255) as u8;
            dst[di + 3] = 255;
        }
    }
}

// --------------------------------------------------------------------------
// Perf
// --------------------------------------------------------------------------
#[derive(Clone, Default)]
struct FrameStats {
    capture_ms:   f64,
    cursor_ms:    f64,
    composite_ms: f64,
    display_ms:   f64,
    total_ms:     f64,
}

struct PerfRing {
    buf:        VecDeque<FrameStats>,
    cap:        usize,
    fps_frames: u32,
    fps_timer:  Instant,
    pub fps:    f64,
}

impl PerfRing {
    fn new(cap: usize) -> Self {
        Self { buf: VecDeque::new(), cap, fps_frames: 0, fps_timer: Instant::now(), fps: 0.0 }
    }
    fn push(&mut self, s: FrameStats) {
        if self.buf.len() >= self.cap { self.buf.pop_front(); }
        self.buf.push_back(s);
        self.fps_frames += 1;
        let e = self.fps_timer.elapsed().as_secs_f64();
        if e >= 0.5 { self.fps = self.fps_frames as f64 / e; self.fps_frames = 0; self.fps_timer = Instant::now(); }
    }
    fn avg(&self) -> FrameStats {
        if self.buf.is_empty() { return FrameStats::default(); }
        let n = self.buf.len() as f64;
        let mut s = FrameStats::default();
        for f in &self.buf {
            s.capture_ms   += f.capture_ms;   s.cursor_ms    += f.cursor_ms;
            s.composite_ms += f.composite_ms; s.display_ms   += f.display_ms;
            s.total_ms     += f.total_ms;
        }
        s.capture_ms /= n; s.cursor_ms /= n; s.composite_ms /= n;
        s.display_ms /= n; s.total_ms  /= n;
        s
    }
}

// --------------------------------------------------------------------------
// Image ops — BGRA, top-left origin
// --------------------------------------------------------------------------

fn downsample_nearest(src: &[u8], sw: usize, sh: usize, dw: usize, dh: usize) -> Vec<u8> {
    let mut out = vec![0u8; dw * dh * 4];
    for dy in 0..dh {
        let sy = dy * sh / dh;
        for dx in 0..dw {
            let sx = dx * sw / dw;
            let si = (sy * sw + sx) * 4;
            let di = (dy * dw + dx) * 4;
            out[di..di+4].copy_from_slice(&src[si..si+4]);
        }
    }
    out
}

/// BGRA → minifb u32 (0xAARRGGBB)
fn bgra_to_u32(bgra: &[u8]) -> Vec<u32> {
    bgra.chunks_exact(4).map(|p| {
        0xFF000000 | ((p[2] as u32) << 16) | ((p[1] as u32) << 8) | p[0] as u32
    }).collect()
}

// --------------------------------------------------------------------------
// Simple 6x8 bitmap font for stats overlay
// --------------------------------------------------------------------------
const FONT_W: usize = 6;
const FONT_H: usize = 8;
static FONT6X8: &[u8; 95 * 8] = include_bytes!("font6x8.bin");

fn stamp_char(buf: &mut [u32], bw: usize, bh: usize, x0: usize, y0: usize, ch: char, color: u32) {
    let idx = (ch as usize).saturating_sub(0x20).min(94);
    for row in 0..FONT_H {
        let byte = FONT6X8[idx * FONT_H + row];
        for col in 0..FONT_W {
            if byte & (0x80 >> col) != 0 {
                let x = x0 + col; let y = y0 + row;
                if x < bw && y < bh { buf[y * bw + x] = color; }
            }
        }
    }
}

fn stamp_str(buf: &mut [u32], bw: usize, bh: usize, x0: usize, y0: usize, s: &str, color: u32) {
    for (i, ch) in s.chars().enumerate() {
        stamp_char(buf, bw, bh, x0 + i * (FONT_W + 1), y0, ch, color);
    }
}

fn draw_stats(buf: &mut Vec<u32>, bw: usize, bh: usize, s: &FrameStats, fps: f64) {
    let lines = [
        format!("FPS       {:5.1}", fps),
        format!("total     {:5.2}ms", s.total_ms),
        format!("capture   {:5.2}ms", s.capture_ms),
        format!("cursor    {:5.2}ms", s.cursor_ms),
        format!("composite {:5.2}ms", s.composite_ms),
        format!("display   {:5.2}ms", s.display_ms),
    ];
    let box_w = 160usize; let box_h = lines.len() * (FONT_H + 2) + 6;
    let x0 = 6usize; let y0 = 6usize;
    for y in y0..y0+box_h { for x in x0..x0+box_w {
        if y < bh && x < bw { buf[y * bw + x] = 0xBB000000; }
    }}
    for (i, line) in lines.iter().enumerate() {
        stamp_str(buf, bw, bh, x0 + 4, y0 + 4 + i * (FONT_H + 2), line, 0xFFFFFF00);
    }
}

// --------------------------------------------------------------------------
// Main
// --------------------------------------------------------------------------
fn main() {
    let screens = Screen::all().expect("no screens");
    let screen  = screens.iter().find(|s| s.display_info.x == 0 && s.display_info.y == 0)
                         .unwrap_or(&screens[0]);

    let total_w = PANEL_W * 3;
    let mut win = Window::new(
        "Cursor Bench  |  Rust",
        total_w, PANEL_H,
        WindowOptions { resize: false, ..Default::default() },
    ).expect("window");
    win.limit_update_rate(Some(Duration::from_micros(1_000_000 / TARGET_FPS)));

    let mut perf = PerfRing::new(60);
    let mut last_print = Instant::now();
    let mut fb = vec![0u32; total_w * PANEL_H];

    println!("Running. ESC to quit. Stats printed every second.");

    while win.is_open() && !win.is_key_down(Key::Escape) {
        let t0 = Instant::now();

        // ── Capture ──────────────────────────────────────────────────────
        let t = Instant::now();
        let img = screen.capture_area(
            REGION_X as i32, REGION_Y as i32, REGION_W, REGION_H
        ).expect("capture");
        let raw_bgra: Vec<u8> = img.as_raw().chunks_exact(4).flat_map(|p| {
            [p[2], p[1], p[0], p[3]] // RGBA → BGRA
        }).collect();
        let capture_ms = t.elapsed().as_secs_f64() * 1000.0;

        // ── Cursor snap (real cursor image + hotspot + ax scale) ─────────
        let t = Instant::now();
        let snap = get_cursor_snap();
        let cursor_ms = t.elapsed().as_secs_f64() * 1000.0;

        // ── Build panels ─────────────────────────────────────────────────
        let t = Instant::now();

        // Panel 1: raw (no cursor)
        let raw_panel = downsample_nearest(&raw_bgra, REGION_W as usize, REGION_H as usize, PANEL_W, PANEL_H);

        // Panel 2: cursor on dark background
        let mut cur_panel = vec![13u8; PANEL_W * PANEL_H * 4]; // dark grey
        for i in (3..cur_panel.len()).step_by(4) { cur_panel[i] = 255; }

        // Panel 3: composite
        let mut comp_panel = downsample_nearest(&raw_bgra, REGION_W as usize, REGION_H as usize, PANEL_W, PANEL_H);

        if let Some(ref s) = snap {
            composite_cursor(&mut cur_panel,  PANEL_W, PANEL_H, s,
                             REGION_X as f64, REGION_Y as f64,
                             REGION_W as f64, REGION_H as f64);
            composite_cursor(&mut comp_panel, PANEL_W, PANEL_H, s,
                             REGION_X as f64, REGION_Y as f64,
                             REGION_W as f64, REGION_H as f64);
        }
        let composite_ms = t.elapsed().as_secs_f64() * 1000.0;

        // ── Display ───────────────────────────────────────────────────────
        let t = Instant::now();
        let raw_u32  = bgra_to_u32(&raw_panel);
        let cur_u32  = bgra_to_u32(&cur_panel);
        let mut comp_u32 = bgra_to_u32(&comp_panel);

        let avg = perf.avg();
        draw_stats(&mut comp_u32, PANEL_W, PANEL_H, &avg, perf.fps);

        for y in 0..PANEL_H {
            let d = y * total_w;
            let s = y * PANEL_W;
            fb[d..d+PANEL_W].copy_from_slice(&raw_u32[s..s+PANEL_W]);
            fb[d+PANEL_W..d+PANEL_W*2].copy_from_slice(&cur_u32[s..s+PANEL_W]);
            fb[d+PANEL_W*2..d+PANEL_W*3].copy_from_slice(&comp_u32[s..s+PANEL_W]);
        }
        win.update_with_buffer(&fb, total_w, PANEL_H).unwrap();
        let display_ms = t.elapsed().as_secs_f64() * 1000.0;

        let total_ms = t0.elapsed().as_secs_f64() * 1000.0;
        perf.push(FrameStats { capture_ms, cursor_ms, composite_ms, display_ms, total_ms });

        if last_print.elapsed() >= Duration::from_secs(1) {
            let a = perf.avg();
            println!(
                "fps {:5.1}  total {:5.1}ms  capture {:5.1}ms  cursor {:4.2}ms  composite {:4.2}ms  display {:5.1}ms",
                perf.fps, a.total_ms, a.capture_ms, a.cursor_ms, a.composite_ms, a.display_ms
            );
            last_print = Instant::now();
        }
    }
}
