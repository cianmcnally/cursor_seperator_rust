use std::ffi::c_void;

use super::coords::DesktopGeometry;
use super::masks::assign_mask_roles;
use super::model::{RectF, WindowCategory, WindowLayer, WindowMaskRole};

// ── AppKit (NSWorkspace) ──────────────────────────────────────────────────────

#[link(name = "AppKit", kind = "framework")]
extern "C" {}

/// PID of the frontmost application via NSWorkspace.
/// Returns None if AppKit call fails (shouldn't happen in normal use).
pub fn get_active_pid() -> Option<i32> {
    use objc::{class, msg_send, sel, sel_impl};
    unsafe {
        let workspace: *mut objc::runtime::Object =
            msg_send![class!(NSWorkspace), sharedWorkspace];
        if workspace.is_null() {
            return None;
        }
        let front: *mut objc::runtime::Object = msg_send![workspace, frontmostApplication];
        if front.is_null() {
            return None;
        }
        let pid: i32 = msg_send![front, processIdentifier];
        Some(pid)
    }
}

// ── CoreFoundation / CoreGraphics FFI ─────────────────────────────────────────

#[link(name = "CoreFoundation", kind = "framework")]
extern "C" {
    fn CFRelease(cf: *const c_void);
    fn CFGetTypeID(cf: *const c_void) -> usize;

    fn CFArrayGetCount(array: *const c_void) -> isize;
    fn CFArrayGetValueAtIndex(array: *const c_void, idx: isize) -> *const c_void;

    fn CFDictionaryGetValue(dict: *const c_void, key: *const c_void) -> *const c_void;

    fn CFStringCreateWithCString(
        alloc: *const c_void,
        c_str: *const i8,
        encoding: u32,
    ) -> *mut c_void;
    fn CFStringGetCString(
        string: *const c_void,
        buf: *mut i8,
        buf_size: isize,
        encoding: u32,
    ) -> bool;
    fn CFStringGetTypeID() -> usize;

    fn CFNumberGetValue(number: *const c_void, the_type: i32, value_ptr: *mut c_void) -> bool;
    fn CFNumberGetTypeID() -> usize;

    fn CFBooleanGetValue(boolean: *const c_void) -> bool;
    fn CFBooleanGetTypeID() -> usize;
}

#[link(name = "CoreGraphics", kind = "framework")]
extern "C" {
    fn CGWindowListCopyWindowInfo(option: u32, relative_to_window: u32) -> *mut c_void;
}

// ── Constants ──────────────────────────────────────────────────────────────────

const CF_STRING_ENCODING_UTF8: u32 = 0x0800_0100;

const K_CG_WINDOW_LIST_OPTION_ON_SCREEN_ONLY: u32 = 1;
const K_CG_WINDOW_LIST_EXCLUDE_DESKTOP_ELEMENTS: u32 = 16;
const K_CG_NULL_WINDOW_ID: u32 = 0;

// CFNumberType values
const CF_NUMBER_SINT32_TYPE: i32 = 3;
const CF_NUMBER_SINT64_TYPE: i32 = 4;
const CF_NUMBER_FLOAT64_TYPE: i32 = 6;

// ── Key strings (lazily created and released per sample call) ─────────────────

/// Create a temporary CFStringRef from a null-terminated byte slice.
/// Caller must CFRelease the returned pointer.
unsafe fn cfstr(s: &[u8]) -> *mut c_void {
    CFStringCreateWithCString(
        std::ptr::null(),
        s.as_ptr() as *const i8,
        CF_STRING_ENCODING_UTF8,
    )
}

/// Release a list of CFStringRefs created with cfstr().
unsafe fn release_keys(keys: &[*mut c_void]) {
    for &k in keys {
        if !k.is_null() {
            CFRelease(k);
        }
    }
}

// ── CF value extractors ───────────────────────────────────────────────────────

/// Get a String from a CFStringRef held in a dictionary.  Does NOT release the value.
unsafe fn dict_get_string(dict: *const c_void, key: *const c_void) -> Option<String> {
    let val = CFDictionaryGetValue(dict, key);
    if val.is_null() { return None; }
    if CFGetTypeID(val) != CFStringGetTypeID() { return None; }
    let mut buf = [0i8; 1024];
    if CFStringGetCString(val, buf.as_mut_ptr(), buf.len() as isize, CF_STRING_ENCODING_UTF8) {
        let cstr = std::ffi::CStr::from_ptr(buf.as_ptr());
        cstr.to_str().ok().map(|s| s.to_owned())
    } else {
        None
    }
}

/// Get an i32 from a CFNumberRef held in a dictionary.
unsafe fn dict_get_i32(dict: *const c_void, key: *const c_void) -> Option<i32> {
    let val = CFDictionaryGetValue(dict, key);
    if val.is_null() { return None; }
    if CFGetTypeID(val) != CFNumberGetTypeID() { return None; }
    let mut out: i32 = 0;
    if CFNumberGetValue(val, CF_NUMBER_SINT32_TYPE, &mut out as *mut i32 as *mut c_void) {
        Some(out)
    } else {
        None
    }
}

/// Get an i64 from a CFNumberRef held in a dictionary.
unsafe fn dict_get_i64(dict: *const c_void, key: *const c_void) -> Option<i64> {
    let val = CFDictionaryGetValue(dict, key);
    if val.is_null() { return None; }
    if CFGetTypeID(val) != CFNumberGetTypeID() { return None; }
    let mut out: i64 = 0;
    if CFNumberGetValue(val, CF_NUMBER_SINT64_TYPE, &mut out as *mut i64 as *mut c_void) {
        Some(out)
    } else {
        None
    }
}

/// Get an f64 from a CFNumberRef held in a dictionary.
unsafe fn dict_get_f64(dict: *const c_void, key: *const c_void) -> Option<f64> {
    let val = CFDictionaryGetValue(dict, key);
    if val.is_null() { return None; }
    if CFGetTypeID(val) != CFNumberGetTypeID() { return None; }
    let mut out: f64 = 0.0;
    if CFNumberGetValue(val, CF_NUMBER_FLOAT64_TYPE, &mut out as *mut f64 as *mut c_void) {
        Some(out)
    } else {
        None
    }
}

/// Get a bool from a CFBooleanRef or CFNumberRef held in a dictionary.
unsafe fn dict_get_bool(dict: *const c_void, key: *const c_void) -> Option<bool> {
    let val = CFDictionaryGetValue(dict, key);
    if val.is_null() { return None; }
    let bool_tid = CFBooleanGetTypeID();
    let num_tid = CFNumberGetTypeID();
    let tid = CFGetTypeID(val);
    if tid == bool_tid {
        Some(CFBooleanGetValue(val))
    } else if tid == num_tid {
        let mut n: i32 = 0;
        CFNumberGetValue(val, CF_NUMBER_SINT32_TYPE, &mut n as *mut i32 as *mut c_void);
        Some(n != 0)
    } else {
        None
    }
}

/// Parse the kCGWindowBounds sub-dictionary: {X, Y, Width, Height} → RectF.
unsafe fn parse_bounds(bounds_dict: *const c_void) -> Option<RectF> {
    if bounds_dict.is_null() { return None; }

    let k_x = cfstr(b"X\0");
    let k_y = cfstr(b"Y\0");
    let k_w = cfstr(b"Width\0");
    let k_h = cfstr(b"Height\0");

    let x = dict_get_f64(bounds_dict, k_x);
    let y = dict_get_f64(bounds_dict, k_y);
    let w = dict_get_f64(bounds_dict, k_w);
    let h = dict_get_f64(bounds_dict, k_h);

    release_keys(&[k_x, k_y, k_w, k_h]);

    match (x, y, w, h) {
        (Some(x), Some(y), Some(w), Some(h)) => Some(RectF { x, y, w, h }),
        _ => None,
    }
}

// ── WindowSampler ─────────────────────────────────────────────────────────────

pub struct WindowSampler {
    pub include_self: bool,
    pub show_system_ui: bool,
    pub normal_windows_only: bool,
    pub last_layers: Vec<WindowLayer>,
    our_pid: i32,
}

impl WindowSampler {
    pub fn new(include_self: bool, show_system_ui: bool, normal_windows_only: bool) -> Self {
        Self {
            include_self,
            show_system_ui,
            normal_windows_only,
            last_layers: Vec::new(),
            our_pid: std::process::id() as i32,
        }
    }

    /// Sample all on-screen windows and return a front-to-back vec (z_index 0 = frontmost).
    /// Also populates `self.last_layers` with the filtered segmentation list.
    pub fn sample(&mut self, desktop: &DesktopGeometry) -> Vec<WindowLayer> {
        let mut windows = unsafe { self.sample_raw(desktop) };

        // Popup windows must have a layer-0 sibling from the same process already
        // in segmentation. Without this, transient app-owned overlays (e.g. a
        // screenshot tool that creates a floating panel with no main window visible)
        // appear as large coloured squares in the mask.
        let layer0_pids: std::collections::HashSet<i32> = windows.iter()
            .filter(|w| w.include_in_segmentation && w.cg_layer == 0)
            .map(|w| w.owner_pid)
            .collect();
        for w in &mut windows {
            if w.include_in_segmentation && w.category.is_popup_like() {
                if !layer0_pids.contains(&w.owner_pid) {
                    w.include_in_segmentation = false;
                }
            }
        }

        let active_pid = get_active_pid();
        assign_mask_roles(&mut windows, active_pid);

        self.last_layers = windows
            .iter()
            .filter(|w| w.include_in_segmentation)
            .cloned()
            .collect();
        windows
    }

    unsafe fn sample_raw(&self, desktop: &DesktopGeometry) -> Vec<WindowLayer> {
        let option = K_CG_WINDOW_LIST_OPTION_ON_SCREEN_ONLY
            | K_CG_WINDOW_LIST_EXCLUDE_DESKTOP_ELEMENTS;

        let array = CGWindowListCopyWindowInfo(option, K_CG_NULL_WINDOW_ID);
        if array.is_null() {
            eprintln!("[windows] CGWindowListCopyWindowInfo returned null — Screen Recording permission needed?");
            return Vec::new();
        }

        let count = CFArrayGetCount(array);
        let mut out = Vec::with_capacity(count as usize);

        // Build key strings once per sample call.
        let k_number     = cfstr(b"kCGWindowNumber\0");
        let k_owner_pid  = cfstr(b"kCGWindowOwnerPID\0");
        let k_owner_name = cfstr(b"kCGWindowOwnerName\0");
        let k_name       = cfstr(b"kCGWindowName\0");
        let k_bounds     = cfstr(b"kCGWindowBounds\0");
        let k_layer      = cfstr(b"kCGWindowLayer\0");
        let k_alpha      = cfstr(b"kCGWindowAlpha\0");
        let k_onscreen   = cfstr(b"kCGWindowIsOnscreen\0");
        let k_sharing    = cfstr(b"kCGWindowSharingState\0");
        let k_store      = cfstr(b"kCGWindowStoreType\0");
        let k_memory     = cfstr(b"kCGWindowMemoryUsage\0");

        for i in 0..count {
            let dict = CFArrayGetValueAtIndex(array, i);
            if dict.is_null() { continue; }

            let window_id   = dict_get_i32(dict, k_number).unwrap_or(0) as u32;
            let owner_pid   = dict_get_i32(dict, k_owner_pid).unwrap_or(0);
            let owner_name  = dict_get_string(dict, k_owner_name).unwrap_or_else(|| String::from("?"));
            let window_name = dict_get_string(dict, k_name);
            let cg_layer    = dict_get_i32(dict, k_layer).unwrap_or(0);
            let alpha       = dict_get_f64(dict, k_alpha).unwrap_or(1.0);
            let is_onscreen = dict_get_bool(dict, k_onscreen).unwrap_or(true);
            let sharing_state = dict_get_i32(dict, k_sharing);
            let store_type    = dict_get_i32(dict, k_store);
            let memory_usage  = dict_get_i64(dict, k_memory);

            let bounds_pts = {
                let bdict = CFDictionaryGetValue(dict, k_bounds);
                parse_bounds(bdict).unwrap_or_default()
            };
            let bounds_px = desktop.window_rect_points_to_pixels(bounds_pts);

            let category = WindowCategory::from_layer_and_owner(cg_layer, &owner_name);

            // Exclude windows completely outside the capture area.
            // This handles windows on other displays in a multi-monitor setup.
            let cap_w = desktop.capture_width_px as i32;
            let cap_h = desktop.capture_height_px as i32;
            let outside_capture = bounds_px.x >= cap_w
                || bounds_px.y >= cap_h
                || bounds_px.x + bounds_px.w <= 0
                || bounds_px.y + bounds_px.h <= 0;

            // Exclude macOS screenshot and capture-tool UIs from segmentation.
            let is_capture_tool = matches!(
                owner_name.as_str(),
                "screencaptureui" | "Screenshot"
            );

            let include = !outside_capture
                && !is_capture_tool
                && self.should_include(is_onscreen, alpha, &bounds_px, cg_layer, owner_pid, &category);

            out.push(WindowLayer {
                window_id,
                z_index: i as usize,
                cg_layer,
                owner_pid,
                owner_name,
                window_name,
                bounds_points: bounds_pts,
                bounds_pixels: bounds_px,
                alpha,
                is_onscreen,
                sharing_state,
                store_type,
                memory_usage,
                category,
                include_in_segmentation: include,
                mask_role: WindowMaskRole::default(),
            });
        }

        release_keys(&[
            k_number, k_owner_pid, k_owner_name, k_name, k_bounds,
            k_layer, k_alpha, k_onscreen, k_sharing, k_store, k_memory,
        ]);

        CFRelease(array);
        out
    }

    fn should_include(
        &self,
        is_onscreen: bool,
        alpha: f64,
        px: &super::model::RectI,
        cg_layer: i32,
        pid: i32,
        category: &WindowCategory,
    ) -> bool {
        if !is_onscreen { return false; }
        if alpha < 0.01 { return false; }
        if px.w < 40 || px.h < 40 { return false; }
        if pid == self.our_pid && !self.include_self { return false; }

        match category {
            WindowCategory::Desktop => return false,
            WindowCategory::TinyJunk => return false,
            WindowCategory::SystemUi | WindowCategory::MenuBar | WindowCategory::Dock => {
                if !self.show_system_ui { return false; }
            }
            _ => {}
        }

        if self.normal_windows_only && cg_layer != 0 && !category.is_popup_like() {
            return false;
        }

        true
    }
}

/// Print a formatted window stack to stdout for --dump-window-list.
pub fn dump_window_list(windows: &[WindowLayer]) {
    println!("── Window stack ({} windows) ──────────────────────────────", windows.len());
    for w in windows {
        let seg = if w.include_in_segmentation { "SEG" } else { "   " };
        let name = w.window_name.as_deref().unwrap_or("");
        println!(
            "  z={:<3} id={:<6} lyr={:<4} pid={:<6} α={:.2} {}×{}@({},{}) [{:?}] {} [{:?}] {}  {}",
            w.z_index, w.window_id, w.cg_layer, w.owner_pid, w.alpha,
            w.bounds_pixels.w, w.bounds_pixels.h,
            w.bounds_pixels.x, w.bounds_pixels.y,
            w.category,
            seg, w.mask_role, w.owner_name, name,
        );
    }
}


