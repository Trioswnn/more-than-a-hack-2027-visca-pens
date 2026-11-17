"""Vectorized Match Data Ingestion and Glitch Filtering Engine.

Parses Metrica FIFA/EPTS Automatic Tracking Data (ATD), homography metadata,
and SportsCode Smart Tagging events into dense, aligned NumPy arrays scaled to
real pitch dimensions in metres.
"""

import os
import sys
from dataclasses import dataclass
from typing import List, Tuple
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

import games
from starter import baseline, load_homography, load_smart_tagging

FIELD_SEP = ":"
GROUP_SEP = ";"
MISSING_CHAR = "N"


@dataclass
class SourceTracking:
    """Dense tracking arrays in metric pitch coordinates (metres)."""
    n_frames: int
    n_slots: int
    pitch: Tuple[float, float]
    team_of_track: np.ndarray  # (n_slots,) 0=Home, 1=Away
    px: np.ndarray             # (n_frames, n_slots) float32, metres
    py: np.ndarray             # (n_frames, n_slots) float32, metres
    ps: np.ndarray             # (n_frames, n_slots) float32, m/s
    bx: np.ndarray             # (n_frames,) float32, metres
    by: np.ndarray             # (n_frames,) float32, metres


def load_atd_source(game=games.GAME, data_dir=None, source="tactical") -> SourceTracking:
    """Stream ATD raw file into metric NumPy arrays."""
    meta_path, raw_path = games.require(["atd_meta", "atd_raw"], game, data_dir, source)
    team_of_track = baseline.team_of_each_track(meta_path)
    n_frames = games.source_frames(source, game)
    n_slots = len(team_of_track)
    length, width = games.GAMES[game]["pitch"]

    px = np.full((n_frames, n_slots), np.nan, dtype=np.float32)
    py = np.full((n_frames, n_slots), np.nan, dtype=np.float32)
    ps = np.full((n_frames, n_slots), np.nan, dtype=np.float32)
    bx = np.full(n_frames, np.nan, dtype=np.float32)
    by = np.full(n_frames, np.nan, dtype=np.float32)

    with open(raw_path, "r") as handle:
        for line in handle:
            parts = line.rstrip("\n").split(FIELD_SEP)
            if len(parts) != 3:
                continue
            fid = int(parts[0]) - 1
            if fid < 0 or fid >= n_frames:
                continue

            for slot, grp in enumerate(parts[1].split(GROUP_SEP)):
                if not grp or grp[0] == MISSING_CHAR:
                    continue
                c1 = grp.index(",")
                c2 = grp.index(",", c1 + 1)
                px[fid, slot] = float(grp[:c1]) * length
                py[fid, slot] = float(grp[c1 + 1 : c2]) * width
                s_str = grp[c2 + 1 :]
                if s_str and s_str[0] != MISSING_CHAR:
                    ps[fid, slot] = float(s_str)

            ball = parts[2]
            if ball and ball[0] != MISSING_CHAR:
                c1 = ball.index(",")
                c2 = ball.index(",", c1 + 1)
                bx[fid] = float(ball[:c1]) * length
                by[fid] = float(ball[c1 + 1 : c2]) * width

    return SourceTracking(
        n_frames=n_frames,
        n_slots=n_slots,
        pitch=(length, width),
        team_of_track=team_of_track,
        px=px,
        py=py,
        ps=ps,
        bx=bx,
        by=by,
    )


def remove_ball_glitches(bx: np.ndarray, by: np.ndarray, fps: float = 25.0,
                         max_speed_mps: float = 48.0) -> Tuple[np.ndarray, np.ndarray, int]:
    """Remove isolated tracking glitches where the ball teleports faster than physically possible.

    A powerful football shot peaks around ~35-40 m/s (~130-144 km/h). Single-frame or
    two-frame jumps exceeding max_speed_mps into an isolated cluster followed by snapping back
    are CV false positives (e.g. mistaking a white sock or sponsor logo for the ball).
    """
    cx = bx.copy()
    cy = by.copy()
    valid = ~np.isnan(cx)
    idx = np.flatnonzero(valid)
    if len(idx) < 3:
        return cx, cy, 0

    removed = 0
    # Iteratively remove isolated 1..4 point teleport islands (jump > max_speed in & out while direct span is normal)
    for k_len in (1, 2, 3, 4):
        while True:
            idx = np.flatnonzero(~np.isnan(cx))
            if len(idx) < k_len + 2:
                break
            pre = idx[: -k_len - 1]
            in_s = idx[1:-k_len]
            in_e = idx[k_len:-1]
            post = idx[k_len + 1 :]
            dt_in = np.maximum(1, in_s - pre) / fps
            dt_out = np.maximum(1, post - in_e) / fps
            dt_span = np.maximum(1, post - pre) / fps
            v_in = np.hypot(cx[in_s] - cx[pre], cy[in_s] - cy[pre]) / dt_in
            v_out = np.hypot(cx[post] - cx[in_e], cy[post] - cy[in_e]) / dt_out
            v_span = np.hypot(cx[post] - cx[pre], cy[post] - cy[pre]) / dt_span

            spikes = np.flatnonzero((v_in > max_speed_mps) & (v_out > max_speed_mps) & (v_span < max_speed_mps * 0.8))
            if len(spikes) == 0:
                break
            bad_mask = np.zeros(len(cx), dtype=bool)
            for sp in spikes:
                bad_mask[idx[sp + 1] : idx[sp + k_len] + 1] = True
            n_bad = int(np.sum(bad_mask & (~np.isnan(cx))))
            if n_bad == 0:
                break
            cx[bad_mask] = np.nan
            cy[bad_mask] = np.nan
            removed += n_bad

    return cx, cy, removed


def graded_half_spans(game=games.GAME) -> List[Tuple[int, int, int]]:
    """Return [(half, start_video_frame, stop_video_frame_exclusive)] from games.py."""
    return [(h, start, last + 1) for h, start, last in games.half_video_blocks(game)]


def graded_mask_from_spans(n_frames: int, spans: List[Tuple[int, int, int]]) -> np.ndarray:
    """Boolean mask True for all graded match frames."""
    mask = np.zeros(n_frames, dtype=bool)
    for _, start, stop in spans:
        mask[start:stop] = True
    return mask


if __name__ == "__main__":
    print("Testing data_engine on tactical and broadcast...")
    tac = load_atd_source(source="tactical")
    cx, cy, rem = remove_ball_glitches(tac.bx, tac.by)
    spans = graded_half_spans()
    gmask = graded_mask_from_spans(tac.n_frames, spans)
    vis_raw = np.sum(~np.isnan(tac.bx) & gmask)
    vis_clean = np.sum(~np.isnan(cx) & gmask)
    print(f"Tactical: {tac.n_frames} frames, {tac.n_slots} slots, spans={spans}")
    print(f"  Ball visible in graded: {vis_raw}/{np.sum(gmask)} ({100*vis_raw/np.sum(gmask):.2f}%), glitches removed: {rem} ({vis_clean} clean)")
    broad = load_atd_source(source="broadcast")
    print(f"Broadcast: {broad.n_frames} frames, {broad.n_slots} slots, ball vis={np.sum(~np.isnan(broad.bx))}")
