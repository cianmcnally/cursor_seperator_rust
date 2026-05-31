// ── model_input_viz — visualizes what the model would actually receive ────────
//
// Usage:
//   cargo run --bin model_input_viz -- path/to/session
//
// Output:
//   path/to/session/debug_model_input.rrd
//
// This reads only saved session files and simulates the full preprocessing
// pipeline that a V-JEPA / V-JEPA 2.1 model would see.
//
// IMPORTANT: This does NOT host or run the model. It only visualizes the
// preprocessing stages to prove correctness. The actual model runs separately
// in a model_runner that reads the same session files.
//
// Preprocessing stages simulated:
//   1. Raw clip extraction (N consecutive frames)
//   2. Resize to 384×384
//   3. Letterbox / crop to 224×224 (V-JEPA) or keep at 384 (V-JEPA 2.1)
//   4. Token grid visualization (24×24 for V-JEPA 2.1)
//   5. Cursor mask at model resolution
//   6. Window label mask at model resolution
//   7. Action label alignment check
//   8. Tubelet frame pair visualization

use std::path::PathBuf;

use rust_cursor_bench::session_loader::LoadedSession;

// ── Model constants (V-JEPA 2.1) ──────────────────────────────────────────────

const MODEL_RES: u32 = 384;        // V-JEPA 2.1 input resolution
const TOKEN_GRID: u32 = 24;        // patch grid (384/16 = 24)
const CLIP_LENGTH: usize = 16;     // frames per clip
const TUBELET_SIZE: usize = 2;     // frames per tubelet

// ── Main ──────────────────────────────────────────────────────────────────────

fn main() {
    let args: Vec<String> = std::env::args().collect();
    if args.len() < 2 {
        eprintln!("Usage: model_input_viz <session_dir>");
        std::process::exit(1);
    }

    let session_dir = PathBuf::from(&args[1]);
    if !session_dir.is_dir() {
        eprintln!("Not a directory: {}", session_dir.display());
        std::process::exit(1);
    }

    println!("Loading session from: {}", session_dir.display());

    let session = match LoadedSession::load(&session_dir) {
        Ok(s) => s,
        Err(e) => {
            eprintln!("Failed to load session: {e}");
            std::process::exit(1);
        }
    };

    println!(
        "Session: {}  Frames: {}",
        session.meta.session_id,
        session.frames.len(),
    );

    // ── Create Rerun output ──────────────────────────────────────────────
    let rrd_path = session_dir.join("debug_model_input.rrd");

    let rec = match rerun::RecordingStreamBuilder::new(session.meta.session_id.as_str())
        .save(&rrd_path)
    {
        Ok(r) => r,
        Err(e) => {
            eprintln!("Failed to create Rerun stream: {e}");
            std::process::exit(1);
        }
    };

    println!("Writing: {}", rrd_path.display());

    let cap_w = session.meta.capture_size_px[0] as usize;
    let cap_h = session.meta.capture_size_px[1] as usize;

    let mut total_warnings = 0u64;

    // ── Process clips ────────────────────────────────────────────────────
    let frames_with_images: Vec<_> = session.frames_with_images();
    let num_clips = frames_with_images.len().saturating_sub(CLIP_LENGTH - 1) / CLIP_LENGTH;

    println!("Frames with images: {}", frames_with_images.len());
    println!("Approximate clips:   {num_clips}");
    println!("CLIP_LENGTH:         {CLIP_LENGTH}");
    println!("MODEL_RES:           {MODEL_RES}×{MODEL_RES}");
    println!("TOKEN_GRID:          {TOKEN_GRID}×{TOKEN_GRID}");

    for clip_idx in 0..num_clips.min(50) {
        // limit to 50 clips to keep .rrd manageable
        let start = clip_idx * CLIP_LENGTH;
        let clip_frames = &frames_with_images[start..start + CLIP_LENGTH];

        let clip_start_fi = clip_frames[0].frame_index;
        let clip_end_fi = clip_frames[CLIP_LENGTH - 1].frame_index;
        let _clip_id = format!("clip_{:04}", clip_idx);

        // ── Set timeline ────────────────────────────────────────────────
        rec.set_time_sequence("clip_idx", clip_idx as i64);
        rec.set_time_sequence("frame_idx", clip_start_fi as i64);

        // Log clip range as text
        let _ = rec.log(
            "/model_input/frame_indices_used",
            &rerun::TextLog::new(format!(
                "clip {clip_idx}: frames {clip_start_fi}..{clip_end_fi}"
            )),
        );

        // ── 1. Raw clip frames ──────────────────────────────────────────
        let mut raw_rgb_frames: Vec<Vec<u8>> = Vec::new();
        for (offset, frame) in clip_frames.iter().enumerate() {
            if let Some((pixels, w, h)) = session.load_frame_png(frame) {
                // Convert RGBA → RGB
                let rgb: Vec<u8> = pixels
                    .chunks_exact(4)
                    .flat_map(|px| [px[0], px[1], px[2]])
                    .collect();
                raw_rgb_frames.push(rgb.clone());

                rec.set_time_sequence("clip_frame_offset", offset as i64);
                let _ = rec.log(
                    "/model_input/raw_clip_frames",
                    &rerun::Image::from_rgb24(rgb, [w, h]),
                );

                // Check: frame dimensions
                let _ = rec.log(
                    "/checks/model_input_frame_hashes",
                    &rerun::TextLog::new(format!(
                        "clip{clip_idx}_f{offset}: {}×{}",
                        w, h
                    )),
                );
            } else {
                rec_log_warning(
                    &rec, clip_start_fi, 0, &mut total_warnings,
                    &format!("clip {clip_idx} frame {offset}: no image data"),
                );
            }
        }

        // ── 2. Resize to MODEL_RES ──────────────────────────────────────
        for (offset, rgb) in raw_rgb_frames.iter().enumerate() {
            let resized = nearest_resize_rgb(rgb, cap_w as u32, cap_h as u32, MODEL_RES, MODEL_RES);

            rec.set_time_sequence("clip_frame_offset", offset as i64);
            let _ = rec.log(
                "/model_input/resized_frames_384",
                &rerun::Image::from_rgb24(resized.clone(), [MODEL_RES, MODEL_RES]),
            );

            // Check resize scale
            let scale_x = MODEL_RES as f64 / cap_w as f64;
            let scale_y = MODEL_RES as f64 / cap_h as f64;
            let _ = rec.log(
                "/checks/resize_scale",
                &rerun::TextLog::new(format!(
                    "clip{clip_idx}_f{offset}: scale=({scale_x:.4}, {scale_y:.4})"
                )),
            );
            if (scale_x - scale_y).abs() > 0.001 {
                rec_log_warning(
                    &rec, clip_start_fi, 0, &mut total_warnings,
                    &format!(
                        "non-uniform scale: x={scale_x:.4} y={scale_y:.4}"
                    ),
                );
            }
        }

        // ── 3. Letterbox/crop visualization ─────────────────────────────
        // For V-JEPA 2.1, we keep 384×384 (no letterbox needed since resize is square).
        // For standard V-JEPA (224), we'd letterbox to 224×224.
        // We show both paths for clarity.
        for (offset, rgb) in raw_rgb_frames.iter().enumerate() {
            let resized = nearest_resize_rgb(rgb, cap_w as u32, cap_h as u32, MODEL_RES, MODEL_RES);

            // Letterbox to 224×224 (V-JEPA classic)
            let letterboxed = letterbox_rgb(&resized, MODEL_RES, MODEL_RES, 224, 224);
            rec.set_time_sequence("clip_frame_offset", offset as i64);
            let _ = rec.log(
                "/model_input/letterboxed_or_cropped_frames",
                &rerun::Image::from_rgb24(letterboxed.clone(), [224, 224]),
            );

            // Check letterbox padding
            let pad_h = (224.0 / MODEL_RES as f64 * MODEL_RES as f64) as u32;
            let _ = rec.log(
                "/checks/letterbox_padding",
                &rerun::TextLog::new(format!(
                    "clip{clip_idx}_f{offset}: 384→224 pad_h={pad_h}"
                )),
            );
        }

        // ── 4. Token grid visualization (24×24) ─────────────────────────
        for (offset, rgb) in raw_rgb_frames.iter().enumerate() {
            let resized = nearest_resize_rgb(rgb, cap_w as u32, cap_h as u32, MODEL_RES, MODEL_RES);
            let token_grid = build_token_grid_image(&resized, MODEL_RES, TOKEN_GRID);

            rec.set_time_sequence("clip_frame_offset", offset as i64);
            let _ = rec.log(
                "/model_input/token_grid_24x24",
                &rerun::Image::from_rgb24(token_grid, [MODEL_RES, MODEL_RES]),
            );
        }

        // ── 5. Cursor mask at model resolution ──────────────────────────
        for (offset, frame) in clip_frames.iter().enumerate() {
            if let Some(mask_8bit) = session.load_cursor_mask(frame) {
                let mask_resized = nearest_resize_gray(
                    &mask_8bit, cap_w as u32, cap_h as u32, MODEL_RES, MODEL_RES,
                );
                // Convert to RGB for visualization (white where cursor exists)
                let mask_rgb: Vec<u8> = mask_resized
                    .iter()
                    .flat_map(|&v| if v > 128 { [255u8, 0, 0] } else { [0u8, 0, 0] })
                    .collect();

                rec.set_time_sequence("clip_frame_offset", offset as i64);
                let _ = rec.log(
                    "/model_input/cursor_mask_384",
                    &rerun::Image::from_rgb24(mask_rgb, [MODEL_RES, MODEL_RES]),
                );

                // Check mask coordinate space — mask should be at cap_w×cap_h
                let expected_mask_pixels = cap_w * cap_h;
                let _ = rec.log(
                    "/checks/mask_coordinate_space",
                    &rerun::TextLog::new(format!(
                        "clip{clip_idx}_f{offset}: cursor_mask size={} expected={expected_mask_pixels}",
                        mask_8bit.len()
                    )),
                );
                if mask_8bit.len() != expected_mask_pixels {
                    rec_log_warning(
                        &rec, clip_start_fi, 0, &mut total_warnings,
                        &format!(
                            "cursor mask coordinate space mismatch: {} vs {expected_mask_pixels}",
                            mask_8bit.len()
                        ),
                    );
                }
            }
        }

        // ── 6. Window label mask at model resolution ────────────────────
        for (offset, frame) in clip_frames.iter().enumerate() {
            if let Some(mask_data) = session.load_windows_label_mask(frame) {
                // mask_data is 16-bit (2 bytes per pixel), visualize as 8-bit RGB
                let mask_16: Vec<u16> = mask_data
                    .chunks_exact(2)
                    .map(|c| u16::from_le_bytes([c[0], c[1]]))
                    .collect();

                let mask_resized = nearest_resize_u16(
                    &mask_16, cap_w as u32, cap_h as u32, MODEL_RES, MODEL_RES,
                );

                // Colorize: distinct color per label
                let mask_rgb: Vec<u8> = mask_resized
                    .iter()
                    .flat_map(|&label| label_to_rgb(label))
                    .collect();

                rec.set_time_sequence("clip_frame_offset", offset as i64);
                let _ = rec.log(
                    "/model_input/window_mask_384",
                    &rerun::Image::from_rgb24(mask_rgb, [MODEL_RES, MODEL_RES]),
                );
            }
        }

        // ── 7. Action label ─────────────────────────────────────────────
        for (offset, frame) in clip_frames.iter().enumerate() {
            let action = &frame.cursor.action;

            rec.set_time_sequence("clip_frame_offset", offset as i64);
            let _ = rec.log(
                "/model_input/action_label",
                &rerun::TextLog::new(format!(
                    "clip{clip_idx}_f{offset}: action={action}"
                )),
            );

            // Check: does the action timestamp fall within this frame's window?
            // Events near this frame index
            let frame_fi = frame.frame_index;
            let relevant_events: Vec<_> = session.events_raw
                .iter()
                .filter(|v| {
                    let efi = v.get("frame_index").and_then(|f| f.as_u64()).unwrap_or(0);
                    efi == frame_fi
                })
                .collect();

            if relevant_events.is_empty() && action != "idle" {
                rec_log_warning(
                    &rec, frame_fi, 0, &mut total_warnings,
                    &format!(
                        "frame {frame_fi} action='{action}' but no event at this frame_index"
                    ),
                );
            }

            let _ = rec.log(
                "/checks/label_frame_offset",
                &rerun::TextLog::new(format!(
                    "clip{clip_idx}_f{offset}: fi={frame_fi} action={action} events_at_frame={}",
                    relevant_events.len()
                )),
            );
        }

        // ── 8. Tubelet frame pairs ──────────────────────────────────────
        for t in 0..(CLIP_LENGTH / TUBELET_SIZE) {
            let f0_idx = t * TUBELET_SIZE;
            let f1_idx = f0_idx + 1;

            if f0_idx < raw_rgb_frames.len() && f1_idx < raw_rgb_frames.len() {
                let f0 = &raw_rgb_frames[f0_idx];
                let f1 = &raw_rgb_frames[f1_idx];

                // Side-by-side tubelet pair
                let pair_rgb = side_by_side_rgb(
                    f0, cap_w as u32, cap_h as u32,
                    f1, cap_w as u32, cap_h as u32,
                );

                rec.set_time_sequence("tubelet_idx", t as i64);
                let _ = rec.log(
                    "/model_input/tubelet_frame_pairs",
                    &rerun::Image::from_rgb24(pair_rgb, [cap_w as u32 * 2, cap_h as u32]),
                );
            }
        }

        // ── Clip start/end checks ───────────────────────────────────────
        let _ = rec.log(
            "/checks/clip_start_frame",
            &rerun::Scalars::new([clip_start_fi as f64]),
        );
        let _ = rec.log(
            "/checks/clip_end_frame",
            &rerun::Scalars::new([clip_end_fi as f64]),
        );
    }

    // ── Source check ────────────────────────────────────────────────────
    rec.set_time_sequence("frame_idx", 0);
    let _ = rec.log(
        "/checks/model_input_source",
        &rerun::TextLog::new(format!(
            "source=reloaded_from_disk session={}",
            session.meta.session_id
        )),
    );

    // ── Flush ───────────────────────────────────────────────────────────
    let _ = rec.flush_blocking();

    println!(
        "Done. Processed {} clips, {} warnings.",
        num_clips.min(50), total_warnings
    );
    println!("Open with: rerun {}", rrd_path.display());
    println!();
    println!("┌─────────────────────────────────────────────────────────────┐");
    println!("│ TRUST CHECKLIST                                             │");
    println!("├─────────────────────────────────────────────────────────────┤");
    println!("│ 1. Open debug.rrd — verify recording-time data              │");
    println!("│ 2. Open debug_replay.rrd — verify reloaded data matches     │");
    println!("│ 3. Open debug_model_input.rrd — verify preprocessing        │");
    println!("│ 4. Check /model_input/cursor_mask_384 coords are in [0,384) │");
    println!("│ 5. Check /model_input/window_mask_384 same space            │");
    println!("│ 6. Check /checks/label_frame_offset for t vs t+1 bugs       │");
    println!("│ 7. Check tubelet pairs use consecutive frames               │");
    println!("│ 8. Check resize is uniform (same x,y scale)                 │");
    println!("└─────────────────────────────────────────────────────────────┘");
}

// ── Image processing helpers ──────────────────────────────────────────────────

fn nearest_resize_rgb(
    src: &[u8], src_w: u32, src_h: u32, dst_w: u32, dst_h: u32,
) -> Vec<u8> {
    let mut out = Vec::with_capacity((dst_w * dst_h * 3) as usize);
    for dy in 0..dst_h {
        let sy = (dy as usize * src_h as usize / dst_h as usize).min(src_h as usize - 1);
        let src_row = sy * src_w as usize * 3;
        for dx in 0..dst_w {
            let sx = (dx as usize * src_w as usize / dst_w as usize).min(src_w as usize - 1);
            let si = src_row + sx * 3;
            out.push(src[si]);
            out.push(src[si + 1]);
            out.push(src[si + 2]);
        }
    }
    out
}

fn nearest_resize_gray(
    src: &[u8], src_w: u32, src_h: u32, dst_w: u32, dst_h: u32,
) -> Vec<u8> {
    let mut out = Vec::with_capacity((dst_w * dst_h) as usize);
    for dy in 0..dst_h {
        let sy = (dy as usize * src_h as usize / dst_h as usize).min(src_h as usize - 1);
        let src_row = sy * src_w as usize;
        for dx in 0..dst_w {
            let sx = (dx as usize * src_w as usize / dst_w as usize).min(src_w as usize - 1);
            out.push(src[src_row + sx]);
        }
    }
    out
}

fn nearest_resize_u16(
    src: &[u16], src_w: u32, src_h: u32, dst_w: u32, dst_h: u32,
) -> Vec<u16> {
    let mut out = Vec::with_capacity((dst_w * dst_h) as usize);
    for dy in 0..dst_h {
        let sy = (dy as usize * src_h as usize / dst_h as usize).min(src_h as usize - 1);
        let src_row = sy * src_w as usize;
        for dx in 0..dst_w {
            let sx = (dx as usize * src_w as usize / dst_w as usize).min(src_w as usize - 1);
            out.push(src[src_row + sx]);
        }
    }
    out
}

fn letterbox_rgb(
    src: &[u8], src_w: u32, src_h: u32, dst_w: u32, dst_h: u32,
) -> Vec<u8> {
    let scale = (dst_w as f64 / src_w as f64).min(dst_h as f64 / src_h as f64);
    let new_w = (src_w as f64 * scale) as u32;
    let new_h = (src_h as f64 * scale) as u32;
    let offset_x = (dst_w - new_w) / 2;
    let offset_y = (dst_h - new_h) / 2;

    let resized = nearest_resize_rgb(src, src_w, src_h, new_w, new_h);
    let mut out = vec![0u8; (dst_w * dst_h * 3) as usize];

    for y in 0..new_h {
        let src_row = y as usize * new_w as usize * 3;
        let dst_row = (offset_y + y) as usize * dst_w as usize * 3;
        for x in 0..new_w {
            let si = src_row + x as usize * 3;
            let di = dst_row + (offset_x + x) as usize * 3;
            out[di] = resized[si];
            out[di + 1] = resized[si + 1];
            out[di + 2] = resized[si + 2];
        }
    }

    out
}

fn build_token_grid_image(
    rgb: &[u8], img_size: u32, grid: u32,
) -> Vec<u8> {
    let patch_size = img_size / grid;
    let mut out = rgb.to_vec(); // start with the image

    // Overlay grid lines
    let _grid_color = [0u8, 255, 0]; // green grid
    for i in 1..grid {
        // Horizontal line
        let y = i * patch_size;
        for x in 0..img_size {
            let idx = (y as usize * img_size as usize + x as usize) * 3;
            if idx + 2 < out.len() {
                out[idx] = out[idx].saturating_add(64).min(255);
                out[idx + 1] = 255;
                out[idx + 2] = out[idx + 2].saturating_add(64).min(255);
            }
        }
        // Vertical line
        let x = i * patch_size;
        for y in 0..img_size {
            let idx = (y as usize * img_size as usize + x as usize) * 3;
            if idx + 2 < out.len() {
                out[idx] = out[idx].saturating_add(64).min(255);
                out[idx + 1] = 255;
                out[idx + 2] = out[idx + 2].saturating_add(64).min(255);
            }
        }
    }

    out
}

fn side_by_side_rgb(
    left: &[u8], lw: u32, lh: u32,
    right: &[u8], rw: u32, _rh: u32,
) -> Vec<u8> {
    let total_w = lw + rw;
    let h = lh.min(_rh);
    let mut out = vec![0u8; (total_w * h * 3) as usize];

    for y in 0..h {
        let dst_row = y as usize * total_w as usize * 3;
        let l_row = y as usize * lw as usize * 3;
        let r_row = y as usize * rw as usize * 3;

        // Left side
        for x in 0..lw {
            let si = l_row + x as usize * 3;
            let di = dst_row + x as usize * 3;
            out[di] = left[si];
            out[di + 1] = left[si + 1];
            out[di + 2] = left[si + 2];
        }
        // Right side (with 2px separator)
        let sep = 2;
        for x in sep..rw {
            let si = r_row + x as usize * 3;
            let di = dst_row + (lw as usize + x as usize) * 3;
            out[di] = right[si];
            out[di + 1] = right[si + 1];
            out[di + 2] = right[si + 2];
        }
        // Separator line
        for x in lw..lw + sep {
            let di = dst_row + x as usize * 3;
            if di + 2 < out.len() {
                out[di] = 255;
                out[di + 1] = 0;
                out[di + 2] = 0;
            }
        }
    }

    out
}

fn label_to_rgb(label: u16) -> [u8; 3] {
    if label == 0 { return [0, 0, 0]; }
    let mut h = (label as u32).wrapping_mul(0x9e37_79b9);
    h ^= h >> 16;
    h = h.wrapping_mul(0x85eb_ca6b);
    h ^= h >> 13;
    h = h.wrapping_mul(0xc2b2_ae35);
    h ^= h >> 16;
    [
        0x40 | ((h & 0x7F) as u8),
        0x40 | (((h >> 8) & 0x7F) as u8),
        0x40 | (((h >> 16) & 0x7F) as u8),
    ]
}

fn rec_log_warning(
    rec: &rerun::RecordingStream,
    fi: u64,
    ts: u64,
    counter: &mut u64,
    msg: &str,
) {
    rec.set_time_sequence("frame_idx", fi as i64);
    rec.set_timestamp_nanos_since_epoch("capture_time", ts as i64);
    let _ = rec.log("/checks/warnings", &rerun::TextLog::new(msg));
    *counter += 1;
}
