use super::model::{RectF, RectI};
use objc::rc::autoreleasepool;
use objc::runtime::Object;
use objc::{class, msg_send, sel, sel_impl};

// Local CGRect for NSScreen calls — same layout as in main.rs / scstream.rs.
#[repr(C)]
#[derive(Copy, Clone)]
struct CGPoint { x: f64, y: f64 }
#[repr(C)]
#[derive(Copy, Clone)]
struct CGSize { width: f64, height: f64 }
#[repr(C)]
#[derive(Copy, Clone)]
struct CGRect { origin: CGPoint, size: CGSize }

/// Maps between CoreGraphics window-server coordinate space and capture-pixel space.
///
/// CGWindowListCopyWindowInfo returns bounds in **points**, top-left origin of the
/// primary display (same Y direction as the captured frame).  The only transform
/// needed is: multiply by the Retina scale factor.
///
/// If you are capturing a sub-region, set `global_origin_points` to the top-left of
/// that region in display-point coordinates.
pub struct DesktopGeometry {
    pub capture_width_px:  u32,
    pub capture_height_px: u32,
    /// Top-left of the captured region in display points.  (0,0) for full-display.
    pub global_origin_points: (f64, f64),
    pub scale_x: f64,
    pub scale_y: f64,
    /// Set true if CGWindow Y increases upward (it doesn't on macOS, but kept as a
    /// calibration escape hatch).
    pub flipped_y: bool,
}

impl DesktopGeometry {
    /// Build from a known capture size in pixels.  Reads the scale factor from
    /// NSScreen.mainScreen.backingScaleFactor.
    pub fn from_capture(
        capture_width_px: u32,
        capture_height_px: u32,
        origin_x_pts: f64,
        origin_y_pts: f64,
    ) -> Self {
        let scale = get_main_screen_scale();
        Self {
            capture_width_px,
            capture_height_px,
            global_origin_points: (origin_x_pts, origin_y_pts),
            scale_x: scale,
            scale_y: scale,
            flipped_y: false,
        }
    }

    /// Convert a window rect from CGWindow points to capture-pixel coordinates.
    pub fn window_rect_points_to_pixels(&self, rect: RectF) -> RectI {
        let ox = self.global_origin_points.0;
        let oy = self.global_origin_points.1;

        let px_x = ((rect.x - ox) * self.scale_x).round() as i32;
        let px_w = (rect.w * self.scale_x).round() as i32;
        let px_h = (rect.h * self.scale_y).round() as i32;

        let px_y = if self.flipped_y {
            let cap_h = self.capture_height_px as f64;
            (cap_h - (rect.y - oy) * self.scale_y - rect.h * self.scale_y).round() as i32
        } else {
            ((rect.y - oy) * self.scale_y).round() as i32
        };

        RectI { x: px_x, y: px_y, w: px_w, h: px_h }
    }

    /// Scale a pixel rect from capture space to a smaller display panel.
    /// cap_w / cap_h are the full capture dimensions; panel_w / panel_h are the panel.
    pub fn pixel_rect_to_panel(
        rect: RectI,
        cap_w: usize,
        cap_h: usize,
        panel_w: usize,
        panel_h: usize,
    ) -> RectI {
        if cap_w == 0 || cap_h == 0 { return RectI::default(); }
        let x = rect.x as i64 * panel_w as i64 / cap_w as i64;
        let y = rect.y as i64 * panel_h as i64 / cap_h as i64;
        let w = rect.w as i64 * panel_w as i64 / cap_w as i64;
        let h = rect.h as i64 * panel_h as i64 / cap_h as i64;
        RectI { x: x as i32, y: y as i32, w: w.max(1) as i32, h: h.max(1) as i32 }
    }
}

fn get_main_screen_scale() -> f64 {
    autoreleasepool(|| unsafe {
        let screen: *mut Object = msg_send![class!(NSScreen), mainScreen];
        if screen.is_null() { return 2.0; }
        let scale: f64 = msg_send![screen, backingScaleFactor];
        if scale <= 0.0 { 2.0 } else { scale }
    })
}
