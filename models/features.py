"""Physics-Informed Kinematic, Ball-Control, and Smart-Tagging Feature Engine.

Constructs per-frame features over the entire 199,319-frame Tactical timeline
without any reliance on or tuning against the 5-minute public ground truth.

Core Physics & Domain Principles:
1. Alive vs. Dead Separation: Missing ball coordinates (NaN) do NOT mean Dead Ball.
   Stoppages are characterized by collective player slowdowns (walking pace),
   out-of-bounds balls, post-goal celebrations, and pre-restart set piece setups.
2. Controlled Possession vs. Ball Flight: A change of possession requires *control*,
   not a ball flying past a defender at 18 m/s. We compute ball velocity, vector
   acceleration (impulse spikes at kicks/touches), and player-ball proximity/speed
   alignment.
3. Smart Tagging Changepoints: Each SET PIECES tag spans around a restart. We locate
   the exact kinematic restart moment inside each tag window and attach weak priors
   from high-precision tactical phases (BUILD UP, PROGRESSION, LONG BALL, etc.).
"""

import os
import sys
from dataclasses import dataclass
from typing import Tuple
import numpy as np
from scipy.ndimage import maximum_filter1d, uniform_filter1d

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

import games
from starter import load_smart_tagging
from models.data_engine import (
    SourceTracking,
    graded_half_spans,
    graded_mask_from_spans,
    remove_ball_glitches,
)


@dataclass
class MatchFeatures:
    n_frames: int
    graded_mask: np.ndarray
    half_spans: list

    # Cleaned & interpolated ball state
    bx: np.ndarray
    by: np.ndarray
    b_vis: np.ndarray            # Directly observed (tactical or synced broadcast)
    b_vis_or_interp: np.ndarray  # Includes short physical interpolations
    b_speed: np.ndarray          # m/s
    b_impulse: np.ndarray        # |dv| m/s across +/- 2 frames (touch/kick detector)
    b_oob: np.ndarray            # Ball outside pitch boundary
    b_edge_dist: np.ndarray      # Distance of ball to nearest pitch line (m)
    b_spd_roll_max: np.ndarray   # Rolling max on-pitch ball speed (+/- 25f)
    b_vis_roll: np.ndarray       # Rolling fraction on-pitch visible ball (+/- 25f)

    # Collective player kinematics (Alive vs Dead)
    p_mean_s: np.ndarray         # Mean visible player speed (m/s)
    p_p85_s: np.ndarray          # 85th percentile player speed (m/s)
    p_n_fast: np.ndarray         # Count of players > 3.0 m/s
    p_activity_smooth: np.ndarray  # Rolling +/- 12 frame mean speed
    p_past: np.ndarray           # Causal past 1s player mean speed
    p_fut: np.ndarray            # Anti-causal future 1s player mean speed
    p_decel: np.ndarray          # Collective acceleration (p_fut - p_past)

    # Ball-player control metrics
    d_home: np.ndarray           # Min distance to Home player (m)
    d_away: np.ndarray           # Min distance to Away player (m)
    d_min: np.ndarray            # min(d_home, d_away)
    nearest_h_speed: np.ndarray  # Speed of closest Home player (m/s)
    nearest_a_speed: np.ndarray  # Speed of closest Away player (m/s)
    rel_speed_h: np.ndarray      # ||v_ball - v_nearest_H|| (m/s)
    rel_speed_a: np.ndarray      # ||v_ball - v_nearest_A|| (m/s)
    h_density: np.ndarray        # Local Gaussian player density near ball (Home)
    a_density: np.ndarray        # Local Gaussian player density near ball (Away)
    pass_evt_h: np.ndarray       # Verified teammate-to-teammate pass completion (Home)
    pass_evt_a: np.ndarray       # Verified teammate-to-teammate pass completion (Away)

    # Smart Tagging & Structural Context
    hard_dead: np.ndarray        # Deterministic dead frames (H2 pre-kickoff, goal->kickoff)
    sp_dead_prior: np.ndarray    # Pre-restart set piece dead-ball prior [0, 1]
    sp_restart_team: np.ndarray  # -1=none, 0=Home, 1=Away at detected restart
    smart_alive_prior: np.ndarray  # Strong evidence of live play from phase tags
    smart_team_prior: np.ndarray   # Soft team score in [-1, +1] (+Home, -Away)


def _interpolate_short_gaps(bx: np.ndarray, by: np.ndarray, max_gap: int = 25,
                            fps: float = 25.0, max_speed: float = 40.0) -> Tuple[np.ndarray, np.ndarray]:
    """Physically plausible linear interpolation for short ball occlusions (<= 1.0s)."""
    ix = bx.copy()
    iy = by.copy()
    vis_idx = np.flatnonzero(~np.isnan(bx))
    if len(vis_idx) < 2:
        return ix, iy

    for k in range(len(vis_idx) - 1):
        t0, t1 = int(vis_idx[k]), int(vis_idx[k + 1])
        gap = t1 - t0 - 1
        if 1 <= gap <= max_gap:
            dt = (t1 - t0) / fps
            dist = float(np.hypot(bx[t1] - bx[t0], by[t1] - by[t0]))
            if dist / dt <= max_speed:
                alpha = np.linspace(0.0, 1.0, t1 - t0 + 1)[1:-1]
                ix[t0 + 1 : t1] = bx[t0] + alpha * (bx[t1] - bx[t0])
                iy[t0 + 1 : t1] = by[t0] + alpha * (by[t1] - by[t0])
    return ix, iy


def _interpolate_ball_drag_gaps(
    bx: np.ndarray,
    by: np.ndarray,
    min_gap: int = 26,
    max_gap: int = 45,
    fps: float = 25.0,
    max_speed: float = 36.0,
) -> Tuple[np.ndarray, np.ndarray]:
    """Bridge 1.0..1.8s mid-air ball occlusions when pre/post velocity vectors align with chord."""
    ix, iy = bx.copy(), by.copy()
    vis_idx = np.flatnonzero(~np.isnan(bx))
    if len(vis_idx) < 2:
        return ix, iy

    n = len(bx)
    for k in range(len(vis_idx) - 1):
        t0, t1 = int(vis_idx[k]), int(vis_idx[k + 1])
        gap = t1 - t0 - 1
        if min_gap <= gap <= max_gap:
            dt = (t1 - t0) / fps
            dx = float(bx[t1] - bx[t0])
            dy = float(by[t1] - by[t0])
            dist = float(np.hypot(dx, dy))
            if dist < 1.0 or (dist / dt) > max_speed:
                continue
            ok_dir = True
            ux, uy = dx / dist, dy / dist
            if t0 >= 3 and not np.isnan(bx[t0 - 3]):
                vx0 = (bx[t0] - bx[t0 - 3]) * (fps / 3.0)
                vy0 = (by[t0] - by[t0 - 3]) * (fps / 3.0)
                sp0 = float(np.hypot(vx0, vy0))
                if sp0 > 4.0 and (vx0 * ux + vy0 * uy) / sp0 < 0.45:
                    ok_dir = False
            if ok_dir and t1 + 3 < n and not np.isnan(bx[t1 + 3]):
                vx1 = (bx[t1 + 3] - bx[t1]) * (fps / 3.0)
                vy1 = (by[t1 + 3] - by[t1]) * (fps / 3.0)
                sp1 = float(np.hypot(vx1, vy1))
                if sp1 > 4.0 and (vx1 * ux + vy1 * uy) / sp1 < 0.45:
                    ok_dir = False
            if ok_dir:
                alpha = np.linspace(0.0, 1.0, t1 - t0 + 1)[1:-1]
                ix[t0 + 1 : t1] = bx[t0] + alpha * dx
                iy[t0 + 1 : t1] = by[t0] + alpha * dy
    return ix, iy


def _detect_gk_slots(tac: SourceTracking, spans: list) -> list:
    """Unsupervisedly identify goalkeeper track slots per half via extreme median X."""
    L, _ = tac.pitch
    gk_per_half = []
    for _, hs, he in spans:
        gks = []
        for col in range(tac.n_slots):
            xv = tac.px[hs:he, col]
            ok = ~np.isnan(xv)
            if int(np.sum(ok)) > 15000:
                mx = float(np.median(xv[ok]))
                if mx < 18.0 or mx > L - 18.0:
                    gks.append(col)
        gk_per_half.append(gks)
    return gk_per_half


def _stitch_team_tracklets(tac: SourceTracking, spans: list, max_gap: int = 45) -> np.ndarray:
    """Stitch fragmented same-team Tactical tracklets across short occlusions via linear extrapolation."""
    uid_arr = np.full((tac.n_frames, tac.n_slots), -1, dtype=np.int32)
    next_uid = 0

    for _, hs, he in spans:
        for team in (0, 1):
            cols = np.flatnonzero(tac.team_of_track == team)
            segs = []
            for c in cols:
                vis = ~np.isnan(tac.px[hs:he, c])
                padded = np.concatenate([[False], vis, [False]])
                starts = hs + np.flatnonzero(~padded[:-1] & padded[1:])
                ends = hs + np.flatnonzero(padded[:-1] & ~padded[1:])
                for s, e in zip(starts, ends):
                    if e - s >= 3:
                        segs.append([int(c), int(s), int(e)])

            segs.sort(key=lambda x: (x[1], x[2]))
            ended = []
            for c, s, e in segs:
                xs, ys = float(tac.px[s, c]), float(tac.py[s, c])
                k_b = min(e - 1, s + 6)
                dt_b = max(1, k_b - s)
                vxb = (float(tac.px[k_b, c]) - xs) / dt_b
                vyb = (float(tac.py[k_b, c]) - ys) / dt_b

                best_i, best_err = -1, 1e9
                for idx, (_, _, ee, uid_p, ex, ey, evx, evy) in enumerate(ended):
                    gap = s - ee
                    if gap < 0 or gap > max_gap:
                        continue
                    err_f = float(np.hypot(xs - (ex + evx * gap), ys - (ey + evy * gap)))
                    err_b = float(np.hypot(ex - (xs - vxb * gap), ey - (ys - vyb * gap)))
                    err = 0.5 * (err_f + err_b)
                    if err <= max(1.8, 0.26 * max(1, gap)) and err < best_err:
                        best_err, best_i = err, idx

                if best_i >= 0:
                    uid = ended.pop(best_i)[3]
                else:
                    uid = next_uid
                    next_uid += 1

                uid_arr[s:e, c] = uid
                ex, ey = float(tac.px[e - 1, c]), float(tac.py[e - 1, c])
                k_a = max(s, e - 1 - 6)
                dt_a = max(1, (e - 1) - k_a)
                evx = (ex - float(tac.px[k_a, c])) / dt_a
                evy = (ey - float(tac.py[k_a, c])) / dt_a
                ended.append((c, s, e, uid, ex, ey, evx, evy))

    return uid_arr


def _compute_team_vel_and_prox(
    px: np.ndarray, py: np.ndarray, ps: np.ndarray, bx: np.ndarray, by: np.ndarray,
    vx: np.ndarray, vy: np.ndarray, vis_frames: np.ndarray, cols: np.ndarray,
    uid_map: np.ndarray, fps: float, n: int,
):
    """Compute nearest/2nd-nearest distance, speed, vector relative speed, density, and carrier uid."""
    tx, ty, ts = px[:, cols], py[:, cols], ps[:, cols]
    tvx = np.full_like(tx, np.nan)
    tvy = np.full_like(ty, np.nan)
    v = (~np.isnan(tx[4:])) & (~np.isnan(tx[:-4]))
    tvx[2:-2][v] = (tx[4:][v] - tx[:-4][v]) * (fps / 4.0)
    tvy[2:-2][v] = (ty[4:][v] - ty[:-4][v]) * (fps / 4.0)

    d_min = np.full(n, np.nan, dtype=np.float32)
    n_spd = np.full(n, np.nan, dtype=np.float32)
    rel_spd = np.full(n, np.nan, dtype=np.float32)
    density = np.zeros(n, dtype=np.float32)
    carrier_uid = np.full(n, -1, dtype=np.int32)

    dx = tx[vis_frames] - bx[vis_frames, None]
    dy = ty[vis_frames] - by[vis_frames, None]
    dist = np.hypot(dx, dy)
    dist_safe = np.where(np.isnan(dist), 1e6, dist)
    idx = np.argmin(dist_safe, axis=1)
    min_d = dist_safe[np.arange(len(vis_frames)), idx]

    d_min[vis_frames] = np.where(min_d < 1e5, min_d, np.nan)
    n_spd[vis_frames] = ts[vis_frames, idx]
    rel_spd[vis_frames] = np.hypot(vx[vis_frames] - tvx[vis_frames, idx], vy[vis_frames] - tvy[vis_frames, idx])
    density[vis_frames] = np.sum(np.exp(-(dist_safe ** 2) / (2.0 * 8.0 ** 2)), axis=1)
    carrier_uid[vis_frames] = np.where(min_d < 2.2, uid_map[vis_frames, cols[idx]], -1)
    return d_min, n_spd, rel_spd, density, carrier_uid


def _fuse_broadcast_outfield(
    broad: SourceTracking, tac: SourceTracking, spans: list, gk_per_half: list,
    tac_to_broad: np.ndarray, tac_match_err: np.ndarray, vis_frames: np.ndarray,
    bx: np.ndarray, by: np.ndarray, vx: np.ndarray, vy: np.ndarray, fps: float,
    d_home: np.ndarray, d_away: np.ndarray, rel_speed_h: np.ndarray, rel_speed_a: np.ndarray,
) -> None:
    """Recover missing Tactical outfield carriers using tightly synced Broadcast observations."""
    gk_near = np.zeros(tac.n_frames, dtype=bool)
    for hi, (_, hs, he) in enumerate(spans):
        for gcol in gk_per_half[hi]:
            gd = np.hypot(tac.px[hs:he, gcol] - bx[hs:he], tac.py[hs:he, gcol] - by[hs:he])
            gk_near[hs:he] |= (~np.isnan(gd)) & (gd < 3.8)

    sync_vis = vis_frames[(tac_to_broad[vis_frames] >= 0) & (tac_match_err[vis_frames] <= 0.85) & (~gk_near[vis_frames])]
    if len(sync_vis) == 0:
        return
    bf_idx = tac_to_broad[sync_vis]

    b_prox = []
    b_rel = []
    for team in (0, 1):
        cols = np.flatnonzero(broad.team_of_track == team)
        bpx, bpy = broad.px[:, cols], broad.py[:, cols]
        bvx, bvy = np.full_like(bpx, np.nan), np.full_like(bpy, np.nan)
        ok = (~np.isnan(bpx[4:])) & (~np.isnan(bpx[:-4]))
        bvx[2:-2][ok] = (bpx[4:][ok] - bpx[:-4][ok]) * (fps / 4.0)
        bvy[2:-2][ok] = (bpy[4:][ok] - bpy[:-4][ok]) * (fps / 4.0)

        bd = np.hypot(bpx[bf_idx] - bx[sync_vis, None], bpy[bf_idx] - by[sync_vis, None])
        bd_s = np.where(np.isnan(bd), 1e6, bd)
        bi = np.argmin(bd_s, axis=1)
        bm = bd_s[np.arange(len(sync_vis)), bi]
        br = np.hypot(vx[sync_vis] - bvx[bf_idx, bi], vy[sync_vis] - bvy[bf_idx, bi])
        b_prox.append(bm)
        b_rel.append(br)

    bm_h, bm_a = b_prox
    brel_h, brel_a = b_rel
    td_h = np.where(~np.isnan(d_home[sync_vis]), d_home[sync_vis], 99.0)
    td_a = np.where(~np.isnan(d_away[sync_vis]), d_away[sync_vis], 99.0)
    miss = (np.minimum(td_h, td_a) > 2.4) & (np.minimum(bm_h, bm_a) < 2.2)

    rec_h = miss & (bm_h < 2.2) & (bm_h <= bm_a - 0.35)
    if np.any(rec_h):
        fh = sync_vis[rec_h]
        d_home[fh] = np.fmin(td_h[rec_h], bm_h[rec_h])
        rel_speed_h[fh] = np.where(~np.isnan(brel_h[rec_h]), brel_h[rec_h], rel_speed_h[fh])

    rec_a = miss & (bm_a < 2.2) & (bm_a <= bm_h - 0.35)
    if np.any(rec_a):
        fa = sync_vis[rec_a]
        d_away[fa] = np.fmin(td_a[rec_a], bm_a[rec_a])
        rel_speed_a[fa] = np.where(~np.isnan(brel_a[rec_a]), brel_a[rec_a], rel_speed_a[fa])


def _detect_verified_passes(
    spans: list, uid_h: np.ndarray, uid_a: np.ndarray,
    d_home: np.ndarray, d_away: np.ndarray, bx: np.ndarray, by: np.ndarray,
    b_vis: np.ndarray, b_oob: np.ndarray, b_speed: np.ndarray, b_impulse: np.ndarray,
) -> Tuple[np.ndarray, np.ndarray]:
    """Detect physically verified teammate-to-teammate pass completions per team."""
    n = len(bx)
    pass_h = np.zeros(n, dtype=np.float32)
    pass_a = np.zeros(n, dtype=np.float32)

    for uid_arr, d_tm, d_op, out_p in ((uid_h, d_home, d_away, pass_h), (uid_a, d_away, d_home, pass_a)):
        for _, hs, he in spans:
            last_uid, last_t, last_bx, last_by, consec, cur = -1, -999, np.nan, np.nan, 0, -1
            for t in range(hs, he):
                if b_oob[t] or not b_vis[t]:
                    continue
                dop = float(d_op[t]) if not np.isnan(d_op[t]) else 99.0
                dtm = float(d_tm[t]) if not np.isnan(d_tm[t]) else 99.0
                if dop < 1.6 and dop < dtm - 0.3:
                    last_uid, consec = -1, 0
                    continue
                u = int(uid_arr[t])
                bs = float(b_speed[t]) if not np.isnan(b_speed[t]) else 0.0
                bi = float(b_impulse[t]) if not np.isnan(b_impulse[t]) else 0.0
                if u >= 0 and dtm < 2.0 and dtm < dop - 0.25 and (bi >= 2.0 or bs < 10.0):
                    consec = consec + 1 if u == cur else 1
                    cur = u
                    if consec == 2:
                        if last_uid >= 0 and u != last_uid and 8 <= (t - last_t) <= 85:
                            if not np.isnan(last_bx) and np.hypot(bx[t] - last_bx, by[t] - last_by) >= 3.2:
                                out_p[t] = 1.0
                        last_uid, last_t = u, t
                        last_bx, last_by = float(bx[t]), float(by[t])
    return pass_h, pass_a


def extract_features(
    tac: SourceTracking,
    fused_bx: np.ndarray = None,
    fused_by: np.ndarray = None,
    game: str = games.GAME,
    data_dir=None,
    broad: SourceTracking = None,
    tac_to_broad: np.ndarray = None,
    tac_match_err: np.ndarray = None,
) -> MatchFeatures:
    """Compute full kinematic, multi-camera, pass-chain, and smart-tagging features."""
    n = tac.n_frames
    fps = float(games.GAMES[game]["fps"])
    L, W = tac.pitch
    spans = graded_half_spans(game)
    graded_mask = graded_mask_from_spans(n, spans)

    raw_bx = tac.bx if fused_bx is None else fused_bx
    raw_by = tac.by if fused_by is None else fused_by

    # 1. Clean ball glitches and interpolate short/medium occlusions (<= 1.8s)
    clean_bx, clean_by, _ = remove_ball_glitches(raw_bx, raw_by, fps=fps)
    b_vis_direct = ~np.isnan(clean_bx)
    bx, by = _interpolate_short_gaps(clean_bx, clean_by, max_gap=25, fps=fps)
    bx, by = _interpolate_ball_drag_gaps(bx, by, min_gap=26, max_gap=45, fps=fps)
    b_vis_interp = ~np.isnan(bx)

    # Smooth ball velocity via central difference over +/- 2 frames (0.16s window)
    vx = np.full(n, np.nan, dtype=np.float32)
    vy = np.full(n, np.nan, dtype=np.float32)
    valid_cd = b_vis_interp[4:] & b_vis_interp[:-4]
    vx[2:-2][valid_cd] = (bx[4:][valid_cd] - bx[:-4][valid_cd]) * (fps / 4.0)
    vy[2:-2][valid_cd] = (by[4:][valid_cd] - by[:-4][valid_cd]) * (fps / 4.0)

    # Fallback 1-frame difference where 4-frame span unavailable
    valid_1 = b_vis_interp[1:] & b_vis_interp[:-1]
    need_vx = np.isnan(vx[1:]) & valid_1
    vx[1:][need_vx] = (bx[1:][need_vx] - bx[:-1][need_vx]) * fps
    vy[1:][need_vx] = (by[1:][need_vx] - by[:-1][need_vx]) * fps

    b_speed = np.hypot(vx, vy)
    # Suppress non-physical jump speeds (> 42 m/s across occlusion edge boundaries)
    bad_spd = b_speed > 42.0
    b_speed[bad_spd] = np.nan
    vx[bad_spd] = np.nan
    vy[bad_spd] = np.nan
    # Impulse = magnitude of velocity vector change across +/- 2 frames (kicks/deflections/receptions)
    b_impulse = np.full(n, np.nan, dtype=np.float32)
    valid_imp = (~np.isnan(vx[4:])) & (~np.isnan(vx[:-4]))
    b_impulse[2:-2][valid_imp] = np.hypot(
        vx[4:][valid_imp] - vx[:-4][valid_imp],
        vy[4:][valid_imp] - vy[:-4][valid_imp],
    )

    # Physical out-of-bounds margin (0.6 m outside line avoids touchline noise)
    margin_m = 0.6
    b_oob = b_vis_interp & (
        (bx < -margin_m) | (bx > L + margin_m) | (by < -margin_m) | (by > W + margin_m)
    )
    b_edge_dist = np.where(
        b_vis_interp,
        np.minimum(np.minimum(bx, L - bx), np.minimum(by, W - by)),
        np.nan,
    ).astype(np.float32)

    on_pitch_vis = b_vis_interp & (~b_oob)
    b_spd_on_pitch = np.where(on_pitch_vis & (~np.isnan(b_speed)), b_speed, 0.0).astype(np.float32)
    b_spd_roll_max = np.zeros(n, dtype=np.float32)
    b_vis_roll = np.zeros(n, dtype=np.float32)
    for _, hs, he in spans:
        b_spd_roll_max[hs:he] = maximum_filter1d(b_spd_on_pitch[hs:he], size=51, mode="nearest")
        b_vis_roll[hs:he] = uniform_filter1d(on_pitch_vis[hs:he].astype(np.float32), size=51, mode="nearest")

    # 2. Collective player kinematics
    s_valid = ~np.isnan(tac.ps)
    ps_zero = np.where(s_valid, tac.ps, 0.0)
    s_cnt = np.maximum(1, np.sum(s_valid, axis=1))
    p_mean_s = (np.sum(ps_zero, axis=1) / s_cnt).astype(np.float32)
    p_n_fast = np.sum(tac.ps > 3.0, axis=1).astype(np.float32)

    p_p85_s = np.zeros(n, dtype=np.float32)
    for f in range(n):
        row = tac.ps[f]
        v = row[~np.isnan(row)]
        if len(v) > 0:
            p_p85_s[f] = np.percentile(v, 85)

    p_activity_smooth = np.zeros(n, dtype=np.float32)
    p_past = np.zeros(n, dtype=np.float32)
    p_fut = np.zeros(n, dtype=np.float32)
    for _, hs, he in spans:
        p_activity_smooth[hs:he] = uniform_filter1d(p_mean_s[hs:he], size=25, mode="nearest")
        p_past[hs:he] = uniform_filter1d(p_mean_s[hs:he], size=25, origin=12, mode="nearest")
        p_fut[hs:he] = uniform_filter1d(p_mean_s[hs:he], size=25, origin=-12, mode="nearest")
    p_decel = (p_fut - p_past).astype(np.float32)

    # 3. Stitched Player Tracklets, Outfield Proximity, and Broadcast Recovery
    h_cols = np.flatnonzero(tac.team_of_track == 0)
    a_cols = np.flatnonzero(tac.team_of_track == 1)
    gk_per_half = _detect_gk_slots(tac, spans)
    uid_map = _stitch_team_tracklets(tac, spans, max_gap=45)

    # Ignore far off-pitch bench/warm-up players (>2.5m outside line) when ball is in play
    tac_px, tac_py = tac.px.copy(), tac.py.copy()
    far_oob = (~b_oob)[:, None] & ((tac_px < -2.5) | (tac_px > L + 2.5) | (tac_py < -2.5) | (tac_py > W + 2.5))
    tac_px[far_oob] = np.nan
    tac_py[far_oob] = np.nan

    vis_frames = np.flatnonzero(b_vis_interp)
    d_home, nearest_h_speed, rel_speed_h, h_density, uid_h = _compute_team_vel_and_prox(
        tac_px, tac_py, tac.ps, bx, by, vx, vy, vis_frames, h_cols, uid_map, fps, n
    )
    d_away, nearest_a_speed, rel_speed_a, a_density, uid_a = _compute_team_vel_and_prox(
        tac_px, tac_py, tac.ps, bx, by, vx, vy, vis_frames, a_cols, uid_map, fps, n
    )

    if broad is not None and tac_to_broad is not None and tac_match_err is not None:
        _fuse_broadcast_outfield(
            broad, tac, spans, gk_per_half, tac_to_broad, tac_match_err, vis_frames,
            bx, by, vx, vy, fps, d_home, d_away, rel_speed_h, rel_speed_a,
        )

    pass_evt_h, pass_evt_a = _detect_verified_passes(
        spans, uid_h, uid_a, d_home, d_away, bx, by, b_vis_interp, b_oob, b_speed, b_impulse
    )
    d_min = np.fmin(d_home, d_away)

    # 4. Smart Tagging & Structural Context
    hard_dead = np.zeros(n, dtype=bool)
    sp_dead_prior = np.zeros(n, dtype=np.float32)
    sp_restart_team = np.full(n, -1, dtype=np.int8)
    smart_alive_prior = np.zeros(n, dtype=np.float32)
    smart_team_prior = np.zeros(n, dtype=np.float32)

    # Guaranteed 2nd-half pre-kickoff dead frames (FAQ.md & games.py)
    h2_start = spans[1][1]
    h2_ko = load_smart_tagging.KICKOFF_VIDEO_FRAMES["2nd Half"]
    hard_dead[h2_start:h2_ko] = True

    events = load_smart_tagging.load(game, data_dir, fps)

    # Post-goal celebrations -> kickoff after goal
    goals = [e for e in events if (e["code"] or "").strip().lower() == "goals"]
    ko_after_goals = [
        e for e in events
        if (e["code"] or "").strip().lower() == "set pieces"
        and (e["tags"].get("Type") or "").strip().lower() == "kick off (after goal)"
    ]
    for g in goals:
        gs = max(0, g["start_frame"])
        matches = [ko for ko in ko_after_goals if ko["start_frame"] >= gs - 250]
        if matches:
            ko = min(matches, key=lambda x: x["start_frame"])
            # Ball is dead during celebration until restart inside kickoff window
            ko_kick = _detect_set_piece_restart(
                ko["start_frame"], ko["end_frame"], b_vis_interp, b_speed, b_impulse, p_mean_s, d_min
            )
            hard_dead[gs:ko_kick] = True
        else:
            hard_dead[gs : min(n, g["end_frame"])] = True

    # Set Pieces (Corner Kick, Goal Kick, Free Kick, Throw-in, Kick Off)
    set_pieces = [e for e in events if (e["code"] or "").strip().lower() == "set pieces"]
    for sp in set_pieces:
        sp_type = (sp["tags"].get("Type") or "").strip().lower()
        sp_team_str = (sp["tags"].get("Team") or "").strip()
        s = max(0, sp["start_frame"] - 50)
        e = min(n, sp["end_frame"])
        if s >= e:
            continue

        # Locate exact kinematic restart frame within set piece window
        if sp_type == "kick off (start)":
            half_str = sp["tags"].get("Half")
            t_kick = load_smart_tagging.KICKOFF_VIDEO_FRAMES.get(half_str, sp["start_frame"])
        else:
            t_kick = _detect_set_piece_restart(
                sp["start_frame"], sp["end_frame"], b_vis_interp, b_speed, b_impulse, p_mean_s, d_min
            )
            # Frames inside set-piece setup before kick carry dead-ball prior
            pre_start = max(0, sp["start_frame"] - 25)
            if t_kick > pre_start:
                sp_dead_prior[pre_start:t_kick] = 1.0
            # Physics Law of the Game: if the ball went out of bounds within 18s (450f) before
            # this set-piece restart, play stayed dead continuously from OOB until t_kick.
            search_lo = max(0, t_kick - 450)
            oob_hits = np.flatnonzero(b_oob[search_lo:t_kick])
            if len(oob_hits) > 0:
                first_oob = search_lo + int(oob_hits[0])
                sp_dead_prior[first_oob:t_kick] = 1.0

        # Identify restarting team
        t_code = -1
        if sp_team_str == "Barcelona":
            t_code = 0
        elif sp_team_str == "Atletic Club":
            t_code = 1
        elif sp_type == "throw-in":
            # For throw-in, check nearest player right around t_kick
            win_s, win_e = max(0, t_kick - 15), min(n, t_kick + 15)
            dh_med = np.nanmedian(d_home[win_s:win_e]) if np.any(~np.isnan(d_home[win_s:win_e])) else 99.0
            da_med = np.nanmedian(d_away[win_s:win_e]) if np.any(~np.isnan(d_away[win_s:win_e])) else 99.0
            if min(dh_med, da_med) < 5.0:
                t_code = 0 if dh_med <= da_med else 1

        if t_code in (0, 1):
            sp_restart_team[t_kick : min(n, t_kick + 25)] = t_code

    # Tactical Phase Priors (soft evidence only)
    for ev in events:
        code = (ev["code"] or "").strip().upper()
        team = (ev["tags"].get("Team") or "").strip()
        s = max(0, ev["start_frame"])
        e = min(n, ev["end_frame"])
        if s >= e:
            continue

        # Alive evidence from dynamic play phases (trim last 15f near potential stoppage whistles)
        if code in (
            "BUILD UP", "PROGRESSION", "LONG BALL", "CREATING CHANCES",
            "ATTACKING TRANSITION", "DEFENSIVE TRANSITION",
            "DEFENDING IN ATTACKING THIRD", "DEFENDING IN MIDDLE THIRD", "DEFENDING IN DEFENSIVE THIRD",
        ):
            e_trim = max(s + 1, e - 15)
            smart_alive_prior[s:e_trim] = np.maximum(smart_alive_prior[s:e_trim], 0.8)

        # Team possession direction (defending tags imply opponent has ball!)
        sign = 0.0
        if team == "Barcelona":
            sign = 1.0
        elif team == "Atletic Club":
            sign = -1.0

        if sign != 0.0:
            if code in ("BUILD UP", "LONG BALL", "ATTACKING TRANSITION"):
                smart_team_prior[s:e] += sign * 0.72
            elif code in ("PROGRESSION", "CREATING CHANCES"):
                smart_team_prior[s:e] += sign * 0.45
            elif code == "DEFENSIVE TRANSITION":
                smart_team_prior[s:e] -= sign * 0.72
            elif code in ("DEFENDING IN ATTACKING THIRD", "DEFENDING IN DEFENSIVE THIRD", "DEFENDING IN MIDDLE THIRD"):
                w = 0.6 if "ATTACKING" in code else 0.45
                smart_team_prior[s:e] -= sign * w
            elif code in ("BALL IN FINAL THIRD", "BALL IN THE BOX"):
                smart_team_prior[s:e] += sign * 0.20

    smart_team_prior = np.clip(smart_team_prior, -1.0, 1.0)

    return MatchFeatures(
        n_frames=n,
        graded_mask=graded_mask,
        half_spans=spans,
        bx=bx,
        by=by,
        b_vis=b_vis_direct,
        b_vis_or_interp=b_vis_interp,
        b_speed=b_speed,
        b_impulse=b_impulse,
        b_oob=b_oob,
        b_edge_dist=b_edge_dist,
        b_spd_roll_max=b_spd_roll_max,
        b_vis_roll=b_vis_roll,
        p_mean_s=p_mean_s,
        p_p85_s=p_p85_s,
        p_n_fast=p_n_fast,
        p_activity_smooth=p_activity_smooth,
        p_past=p_past,
        p_fut=p_fut,
        p_decel=p_decel,
        d_home=d_home,
        d_away=d_away,
        d_min=d_min,
        nearest_h_speed=nearest_h_speed,
        nearest_a_speed=nearest_a_speed,
        rel_speed_h=rel_speed_h,
        rel_speed_a=rel_speed_a,
        h_density=h_density,
        a_density=a_density,
        pass_evt_h=pass_evt_h,
        pass_evt_a=pass_evt_a,
        hard_dead=hard_dead,
        sp_dead_prior=sp_dead_prior,
        sp_restart_team=sp_restart_team,
        smart_alive_prior=smart_alive_prior,
        smart_team_prior=smart_team_prior,
    )


def _detect_set_piece_restart(sp_start: int, sp_end: int, b_vis: np.ndarray,
                              b_speed: np.ndarray, b_impulse: np.ndarray,
                              p_mean_s: np.ndarray, d_min: np.ndarray) -> int:
    """Locate the kinematic restart kick/throw frame inside a SET PIECES tag window.

    A set piece tag starts several seconds before the kick. The restart occurs where
    either ball speed jumps from stationary/slow into fast flight, or a sharp impulse
    occurs with a player within kicking distance (< 2.5m).
    """
    n = len(b_vis)
    s = max(0, sp_start)
    e = min(n, sp_end)
    if s >= e:
        return s

    # Look for first clear acceleration/speed surge after ball was slow/missing
    bs_win = b_speed[s:e]
    imp_win = b_impulse[s:e]
    d_win = d_min[s:e]

    # Candidate restart: ball speed in [4.0, 40.0] m/s with impulse or player within kicking reach
    valid = ~np.isnan(bs_win)
    kick_cand = (
        valid
        & (bs_win >= 4.0)
        & (bs_win <= 40.0)
        & ((np.nan_to_num(imp_win, nan=0.0) >= 3.0) | (np.nan_to_num(d_win, nan=99.0) <= 3.5))
    )
    hits = np.flatnonzero(kick_cand)
    if len(hits) > 0:
        pm_win = p_mean_s[s:e]
        for idx in hits:
            # Check sustained ball motion in next 10 frames
            nxt = bs_win[idx : min(len(bs_win), idx + 10)]
            if np.sum(~np.isnan(nxt) & (nxt >= 2.5)) < 2:
                continue
            # Reject dead-ball spot-placement rolls where ball soon slows (< 3.5 m/s) while players stay still
            chk_a = min(len(bs_win), idx + 20)
            chk_b = min(len(bs_win), idx + 40)
            if chk_a < chk_b:
                fut_bs = bs_win[chk_a:chk_b]
                fut_pm = pm_win[chk_a:chk_b]
                vis_fut = ~np.isnan(fut_bs)
                if np.any(vis_fut) and np.nanmedian(fut_bs[vis_fut]) < 3.5 and np.mean(fut_pm) < 1.15:
                    continue
            return s + int(idx)
        return s + int(hits[0])

    # Fallback: midpoint-to-65% of set piece tag (since tags open ~3.5s early)
    return int(s + 0.45 * (e - s))
