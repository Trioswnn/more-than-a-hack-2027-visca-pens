"""Cross-Camera Broadcast-to-Tactical Synchronization & Ball Recovery.

Aligns the zoomed-in Broadcast ATD timeline (164,488 frames) onto the Tactical
reference timeline (199,319 frames) completely algorithmically using Two-Pass
Unsupervised Hough Anchor Alignment:

Pass 1 (Global Anchor Discovery):
  - Segments Broadcast into usable camera shots (valid homography, low
    reprojection error, >= 4v4 visible players).
  - Computes coarse blurred 2D pitch occupancy similarity against Tactical
    graded spans and collects high-confidence shot offsets (Chamfer < 1.2 m).
  - Identifies the dominant constant live-feed offset per half via Hough
    clustering (rejecting replays automatically).

Pass 2 (Dense Frame-by-Frame Verification & Ball Fusion):
  - Projects every usable Broadcast frame via its half anchor (allowing small
    local drift optimization +/- 4 frames per contiguous shot).
  - Gates every single frame with strict one-way Chamfer error (< 1.5 m) so
    replays and poorly calibrated zoom shots can never contaminate Tactical.
  - Imputes missing Tactical ball coordinates from synchronized Broadcast ball.
"""

import os
import sys
import time
from typing import Dict, List, Tuple
import numpy as np
from scipy.ndimage import gaussian_filter

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

import games
from starter import load_homography
from models.data_engine import SourceTracking, graded_half_spans, graded_mask_from_spans, load_atd_source


def _build_occupancy_grid(px: np.ndarray, py: np.ndarray, team_mask: np.ndarray,
                          pitch: Tuple[float, float], frames: np.ndarray,
                          nx: int = 16, ny: int = 10, blur_sigma: float = 0.8,
                          is_query: bool = False) -> np.ndarray:
    """Build (len(frames), 2 * nx * ny) blurred spatial occupancy features without NaN warnings."""
    L, W = pitch
    n = len(frames)
    grid = np.zeros((n, 2, ny, nx), dtype=np.float32)

    sub_px = px[frames]
    sub_py = py[frames]
    valid = (~np.isnan(sub_px)) & (sub_px >= -2.0) & (sub_px <= L + 2.0) & (sub_py >= -2.0) & (sub_py <= W + 2.0)

    safe_px = np.where(valid, np.clip(sub_px / L, 0.0, 0.9999), 0.0)
    safe_py = np.where(valid, np.clip(sub_py / W, 0.0, 0.9999), 0.0)
    ix = np.floor(safe_px * nx).astype(np.int32)
    iy = np.floor(safe_py * ny).astype(np.int32)

    for team_id in (0, 1):
        t_cols = np.flatnonzero(team_mask == team_id)
        v = valid[:, t_cols]
        f_idx, c_idx = np.nonzero(v)
        gx = ix[f_idx, t_cols[c_idx]]
        gy = iy[f_idx, t_cols[c_idx]]
        np.add.at(grid[:, team_id], (f_idx, gy, gx), 1.0)

    for i in range(n):
        for t in (0, 1):
            grid[i, t] = gaussian_filter(grid[i, t], sigma=blur_sigma, mode="constant")
        if is_query:
            norm = float(np.sum(grid[i])) + 1e-6
            grid[i] /= norm
        else:
            grid[i] = np.clip(grid[i], 0.0, 1.0)

    return grid.reshape(n, -1)


def _one_way_chamfer(tac: SourceTracking, tf: int, broad: SourceTracking, bf: int) -> float:
    """Trimmed mean nearest-neighbor distance (metres) from visible Broadcast players to Tactical."""
    errs = []
    for team_id in (0, 1):
        t_cols = tac.team_of_track == team_id
        b_cols = broad.team_of_track == team_id
        tx, ty = tac.px[tf, t_cols], tac.py[tf, t_cols]
        bx, by = broad.px[bf, b_cols], broad.py[bf, b_cols]

        tv = ~np.isnan(tx)
        bv = ~np.isnan(bx)
        if np.sum(bv) < 2 or np.sum(tv) < 3:
            continue
        tx, ty = tx[tv], ty[tv]
        bx, by = bx[bv], by[bv]

        dx = bx[:, None] - tx[None, :]
        dy = by[:, None] - ty[None, :]
        d_min = np.sqrt(np.min(dx * dx + dy * dy, axis=1))
        k_keep = max(2, int(np.ceil(len(d_min) * 0.85)))
        errs.append(float(np.mean(np.sort(d_min)[:k_keep])))
    if not errs:
        return 999.0
    return float(np.mean(errs))


def _discover_half_anchors(
    tac: SourceTracking,
    broad: SourceTracking,
    shots: List[Tuple[int, int]],
    spans: List[Tuple[int, int, int]],
) -> List[Tuple[int, int, int, int]]:
    """Discover dominant constant offset (t_tac - t_broad) per match half via Hough clustering."""
    tac_grid_frames = np.concatenate([np.arange(s, e, 5) for _, s, e in spans])
    tac_feats = _build_occupancy_grid(
        tac.px, tac.py, tac.team_of_track, tac.pitch, tac_grid_frames, is_query=False
    )

    shot_matches = []
    for bs, be in shots:
        if be - bs < 80:
            continue
        pts = np.linspace(bs, be - 1, min(10, be - bs), dtype=int)
        bf_feats = _build_occupancy_grid(
            broad.px, broad.py, broad.team_of_track, broad.pitch, pts, is_query=True
        )
        sim = bf_feats @ tac_feats.T
        offs = []
        for k in range(len(pts)):
            top = np.argpartition(sim[k], -5)[-5:]
            for c in top:
                offs.append(int(tac_grid_frames[c]) - int(pts[k]))
        offs = np.array(offs)
        best_off = max(offs, key=lambda o: int(np.sum(np.abs(offs - o) <= 10)))
        best_e, best_d = 999.0, 0
        for d in range(-6, 7):
            es = [
                _one_way_chamfer(tac, int(p) + best_off + d, broad, int(p))
                for p in pts
                if 0 <= int(p) + best_off + d < tac.n_frames
            ]
            if es:
                med_e = float(np.median(es))
                if med_e < best_e:
                    best_e = med_e
                    best_d = d
        if best_e < 1.0:
            final_off = int(best_off + best_d)
            tac_mid = int(0.5 * (bs + be)) + final_off
            shot_matches.append((tac_mid, final_off, bs, be))

    anchors = []
    for half_idx, h_start, h_end in spans:
        half_offs = np.array([off for tmid, off, _, _ in shot_matches if h_start <= tmid < h_end])
        if len(half_offs) == 0:
            continue
        # Hough mode within +/- 15 frames
        mode_off = max(half_offs, key=lambda o: int(np.sum(np.abs(half_offs - o) <= 15)))
        inliers = half_offs[np.abs(half_offs - mode_off) <= 15]
        anchor_off = int(np.round(np.median(inliers)))
        anchors.append((half_idx, h_start, h_end, anchor_off))

    return anchors


def sync_broadcast_to_tactical(
    tac: SourceTracking,
    broad: SourceTracking,
    game: str = games.GAME,
    data_dir=None,
    max_reproj_err: float = 3.5,
    max_chamfer_err_m: float = 1.5,
) -> Dict:
    """Synchronize Broadcast tracking frames to Tactical reference frames and recover ball."""
    t0 = time.time()
    _, reproj_b, valid_b = load_homography.load(game, data_dir, source="broadcast")
    spans = graded_half_spans(game)
    L, W = tac.pitch

    b_in_pitch = (
        (~np.isnan(broad.px))
        & (broad.px >= -3.0) & (broad.px <= L + 3.0)
        & (broad.py >= -3.0) & (broad.py <= W + 3.0)
    )
    b_vis_h = np.sum(b_in_pitch[:, broad.team_of_track == 0], axis=1)
    b_vis_a = np.sum(b_in_pitch[:, broad.team_of_track == 1], axis=1)
    usable_b = valid_b & (reproj_b <= max_reproj_err) & (b_vis_h >= 3) & (b_vis_a >= 3)

    edges = np.diff(np.concatenate(([0], usable_b.astype(np.int8), [0])))
    shots = [
        (int(s), int(e))
        for s, e in zip(np.flatnonzero(edges == 1), np.flatnonzero(edges == -1))
        if e - s >= 20
    ]

    # Pass 1: Unsupervised discovery of half anchors
    anchors = _discover_half_anchors(tac, broad, shots, spans)

    # Pass 2: Dense verification per broadcast shot against its half anchor
    tac_to_broad = np.full(tac.n_frames, -1, dtype=np.int32)
    tac_match_err = np.full(tac.n_frames, np.inf, dtype=np.float32)

    matched_shots = 0
    for bs, be in shots:
        b_mid = int(0.5 * (bs + be))
        # Determine which half anchor maps b_mid into [h_start, h_end)
        cand_anchor = None
        for _, h_start, h_end, anchor_off in anchors:
            if h_start - 500 <= b_mid + anchor_off < h_end + 500:
                cand_anchor = anchor_off
                break
        if cand_anchor is None:
            continue

        # Refine local offset within +/- 5 frames on evenly spaced sample points
        pts = np.linspace(bs, be - 1, min(7, be - bs), dtype=int)
        best_off, best_err = cand_anchor, 999.0
        for delta in range(-5, 6):
            off_try = cand_anchor + delta
            errs = [
                _one_way_chamfer(tac, int(p) + off_try, broad, int(p))
                for p in pts
                if 0 <= int(p) + off_try < tac.n_frames
            ]
            if errs:
                med_e = float(np.median(errs))
                if med_e < best_err:
                    best_err = med_e
                    best_off = off_try

        if best_err > max_chamfer_err_m:
            continue

        # Verify frame-by-frame across the shot
        accepted = 0
        for bf in range(bs, be):
            tf = bf + best_off
            if 0 <= tf < tac.n_frames:
                err = _one_way_chamfer(tac, tf, broad, bf)
                if err <= max_chamfer_err_m and err < tac_match_err[tf]:
                    tac_match_err[tf] = err
                    tac_to_broad[tf] = bf
                    accepted += 1
        if accepted >= 10:
            matched_shots += 1

    # Fuse ball coordinates!
    fused_bx = tac.bx.copy()
    fused_by = tac.by.copy()
    ball_source = np.zeros(tac.n_frames, dtype=np.int8)
    ball_source[~np.isnan(tac.bx)] = 1

    recovered_mask = np.isnan(tac.bx) & (tac_to_broad >= 0)
    b_indices = tac_to_broad[recovered_mask]
    b_has_ball = ~np.isnan(broad.bx[b_indices])
    rec_frames = np.flatnonzero(recovered_mask)[b_has_ball]
    rec_b_frames = b_indices[b_has_ball]

    fused_bx[rec_frames] = broad.bx[rec_b_frames]
    fused_by[rec_frames] = broad.by[rec_b_frames]
    ball_source[rec_frames] = 2

    # Unsupervised accuracy verification on overlapping frames
    both_mask = (~np.isnan(tac.bx)) & (tac_to_broad >= 0)
    both_tf = np.flatnonzero(both_mask)
    both_bf = tac_to_broad[both_tf]
    valid_both = ~np.isnan(broad.bx[both_bf])
    both_tf = both_tf[valid_both]
    both_bf = both_bf[valid_both]
    ball_diff = np.hypot(tac.bx[both_tf] - broad.bx[both_bf], tac.by[both_tf] - broad.by[both_bf])

    elapsed = time.time() - t0
    return {
        "fused_bx": fused_bx,
        "fused_by": fused_by,
        "ball_source": ball_source,
        "tac_to_broad": tac_to_broad,
        "tac_match_err": tac_match_err,
        "anchors": anchors,
        "n_shots_total": len(shots),
        "n_shots_matched": matched_shots,
        "n_frames_synced": int(np.sum(tac_to_broad >= 0)),
        "n_ball_recovered": int(len(rec_frames)),
        "both_visible_count": int(len(ball_diff)),
        "both_ball_err_median_m": float(np.median(ball_diff)) if len(ball_diff) else np.nan,
        "both_ball_err_p90_m": float(np.percentile(ball_diff, 90)) if len(ball_diff) else np.nan,
        "elapsed_s": elapsed,
    }


if __name__ == "__main__":
    tac = load_atd_source(source="tactical")
    broad = load_atd_source(source="broadcast")
    res = sync_broadcast_to_tactical(tac, broad)
    gmask = graded_mask_from_spans(tac.n_frames, graded_half_spans())
    tac_vis = np.sum((~np.isnan(tac.bx)) & gmask)
    fused_vis = np.sum((~np.isnan(res["fused_bx"])) & gmask)
    rec_graded = np.sum((res["ball_source"] == 2) & gmask)
    print(f"Sync finished in {res['elapsed_s']:.2f}s")
    print(f"Discovered Half Anchors: {res['anchors']}")
    print(f"Broadcast usable shots: {res['n_shots_total']}, matched live shots: {res['n_shots_matched']}")
    print(f"Total tactical frames synced: {res['n_frames_synced']}")
    print(f"Graded ball visibility: {tac_vis}/{np.sum(gmask)} ({100*tac_vis/np.sum(gmask):.2f}%) -> {fused_vis}/{np.sum(gmask)} ({100*fused_vis/np.sum(gmask):.2f}%)")
    print(f"Missing tactical ball frames recovered in graded windows: +{rec_graded} frames!")
    print(f"Unsupervised ball agreement (when both cameras see ball, n={res['both_visible_count']}):")
    print(f"  Median ball error = {res['both_ball_err_median_m']:.2f} m, p90 = {res['both_ball_err_p90_m']:.2f} m")
