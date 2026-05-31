use std::fs::File;
use std::io::Write as IoWrite;
use std::path::Path;
use std::time::{SystemTime, UNIX_EPOCH};

use objc::rc::autoreleasepool;
use objc::runtime::Object;
use objc::{class, msg_send, sel, sel_impl};
use serde::{Deserialize, Serialize};

#[repr(C)] #[derive(Copy, Clone)] struct NSPoint { x: f64, y: f64 }
#[repr(C)] #[derive(Copy, Clone)] struct NSSize  { width: f64, height: f64 }
#[repr(C)] #[derive(Copy, Clone)] struct NSRect  { origin: NSPoint, size: NSSize }

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DisplayInfo {
    pub display_id:   u32,
    /// Pixel rect [x, y, w, h] of this display in capture coordinate space.
    pub frame_px:     [i32; 4],
    /// Point rect [x, y, w, h] — macOS logical coordinates.
    pub frame_points: [f64; 4],
    pub scale:        f64,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SessionMeta {
    pub session_id:           String,
    pub started_at_unix_ns:   u64,
    pub capture_size_px:      [u32; 2],
    /// Bounding box of all displays: [x, y, w, h] in pixels.
    pub virtual_desktop_px:   [i32; 4],
    pub displays:             Vec<DisplayInfo>,
    pub classes:              Vec<String>,
    pub window_actions:       Vec<String>,
    pub focus_event_kinds:    Vec<String>,
    pub keyboard_event_kinds: Vec<String>,
    pub cursor_actions:       Vec<String>,
}

impl SessionMeta {
    pub fn new(cap_w: u32, cap_h: u32, _origin_x: f64, _origin_y: f64) -> Self {
        let ts_ns = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .unwrap_or_default()
            .as_nanos() as u64;
        let ts_secs = ts_ns / 1_000_000_000;
        let session_id = format!("session_{}", fmt_datetime(ts_secs));
        let displays = get_displays();
        let vd = virtual_desktop_px(&displays, cap_w, cap_h);
        Self {
            session_id,
            started_at_unix_ns: ts_ns,
            capture_size_px: [cap_w, cap_h],
            virtual_desktop_px: vd,
            displays,
            classes: vec!["window".into(), "cursor".into(), "keyboard".into()],
            window_actions: vec!["focused".into(), "unfocused".into()],
            focus_event_kinds:    vec!["FocusChange".into()],
            keyboard_event_kinds: vec!["KeyDown".into()],
            cursor_actions: vec![
                "idle".into(), "move".into(), "click".into(), "double_click".into(),
                "drag".into(), "scroll".into(), "typing".into(),
            ],
        }
    }

    pub fn write_json(&self, dir: &Path) -> std::io::Result<()> {
        let json = serde_json::to_string_pretty(self)
            .map_err(|e| std::io::Error::new(std::io::ErrorKind::Other, e))?;
        let mut f = File::create(dir.join("session.json"))?;
        f.write_all(json.as_bytes())?;
        Ok(())
    }
}

// ── Display enumeration ───────────────────────────────────────────────────────

pub fn get_displays() -> Vec<DisplayInfo> {
    autoreleasepool(|| unsafe {
        let screens: *mut Object = msg_send![class!(NSScreen), screens];
        if screens.is_null() { return vec![]; }
        let count: usize = msg_send![screens, count];
        let mut out = Vec::with_capacity(count);
        for i in 0..count {
            let screen: *mut Object = msg_send![screens, objectAtIndex: i];
            if screen.is_null() { continue; }
            let frame: NSRect = msg_send![screen, frame];
            let scale: f64    = msg_send![screen, backingScaleFactor];
            let scale = if scale <= 0.0 { 2.0 } else { scale };

            // CGDirectDisplayID via NSScreen deviceDescription[@"NSScreenNumber"]
            let desc: *mut Object = msg_send![screen, deviceDescription];
            let key: *mut Object = msg_send![
                class!(NSString),
                stringWithUTF8String: b"NSScreenNumber\0".as_ptr()
            ];
            let num: *mut Object = msg_send![desc, objectForKey: key];
            let display_id: u32 = if num.is_null() { i as u32 }
                                  else { msg_send![num, unsignedIntValue] };

            let pts = frame;
            out.push(DisplayInfo {
                display_id,
                frame_points: [
                    pts.origin.x, pts.origin.y,
                    pts.size.width, pts.size.height,
                ],
                frame_px: [
                    (pts.origin.x * scale) as i32,
                    (pts.origin.y * scale) as i32,
                    (pts.size.width  * scale) as i32,
                    (pts.size.height * scale) as i32,
                ],
                scale,
            });
        }
        out
    })
}

fn virtual_desktop_px(displays: &[DisplayInfo], cap_w: u32, cap_h: u32) -> [i32; 4] {
    if displays.is_empty() {
        return [0, 0, cap_w as i32, cap_h as i32];
    }
    let mut x0 = i32::MAX;
    let mut y0 = i32::MAX;
    let mut x1 = i32::MIN;
    let mut y1 = i32::MIN;
    for d in displays {
        let [dx, dy, dw, dh] = d.frame_px;
        x0 = x0.min(dx);
        y0 = y0.min(dy);
        x1 = x1.max(dx + dw);
        y1 = y1.max(dy + dh);
    }
    [x0, y0, x1 - x0, y1 - y0]
}

// ── Timestamp formatting (no external deps) ───────────────────────────────────

/// Format unix seconds as `YYYYMMDD_HHMMSS`.
fn fmt_datetime(secs: u64) -> String {
    let mut s = secs;
    let sec  = s % 60; s /= 60;
    let min  = s % 60; s /= 60;
    let hour = s % 24; s /= 24;
    let (y, m, d) = days_since_epoch_to_ymd(s);
    format!("{:04}{:02}{:02}_{:02}{:02}{:02}", y, m, d, hour, min, sec)
}

fn days_since_epoch_to_ymd(mut days: u64) -> (u64, u64, u64) {
    let mut year = 1970u64;
    loop {
        let n = if is_leap(year) { 366 } else { 365 };
        if days < n { break; }
        days -= n;
        year += 1;
    }
    let months: [u64; 12] = [31, if is_leap(year) { 29 } else { 28 },
                               31, 30, 31, 30, 31, 31, 30, 31, 30, 31];
    let mut month = 1u64;
    for &dm in &months {
        if days < dm { break; }
        days -= dm;
        month += 1;
    }
    (year, month, days + 1)
}

fn is_leap(y: u64) -> bool {
    (y % 4 == 0 && y % 100 != 0) || y % 400 == 0
}
