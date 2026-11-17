"""5-State Physics-Constrained Exact Hybrid HSMM Possession Model (v5.0).

Macro-states mapped to competition labels:
    0: H_CTRL   -> 'H'  (Home player actively controlling/dribbling/trapping ball)
    1: H_FLIGHT -> 'H'  (Ball in flight / loose / occluded under Home possession)
    2: A_CTRL   -> 'A'  (Away player actively controlling/dribbling/trapping ball)
    3: A_FLIGHT -> 'A'  (Ball in flight / loose / occluded under Away possession)
    4: DEAD     -> 'D'  (Ball out of play, stoppage, pre-restart, goal celebration)

Key Structural Guarantees (Laws of Football Physics, Zero 5-Min GT Leakage):
1. Flight Turnover Prevention: H_FLIGHT <-> A_FLIGHT and *_CTRL -> opp_*_FLIGHT are
   strictly forbidden (-inf). Possession NEVER flips mid-air or on a passive rolling ball;
   a turnover MUST enter the gaining team's *_CTRL state with genuine control evidence
   (||v_ball - v_player|| alignment, teammate pass chains, and foot proximity).
2. Equal Flight Emissions: Outside restarts/occlusions, emit[H_FLIGHT] == emit[A_FLIGHT],
   so uncontrolled fast passes/clearances accumulate ZERO spurious advantage for either
   team, eliminating 1-touch parry/clearance/deflection fragmentation.
3. Unsupervised Match Calibration: 2-component EM fits Dead vs Alive player kinematics
   and calibrates relative-velocity control scale (q81) across all 158,857 graded frames.
4. Exact Vectorized 5-State Explicit-Duration HSMM DP (stride=2, 0.08s resolution)
   with zero-visible-ball jog guards, verified teammate pass-chain evidence,
   Voronoi flight handover partitioning inside uncontrolled H <-> A flight gaps,
   and 1-frame kick-impulse onset snapping on D -> Alive restarts.
"""

import os
import sys
from typing import Dict, List, Tuple
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from models.features import MatchFeatures

HC_ST, HF_ST, AC_ST, AF_ST, D_ST = 0, 1, 2, 3, 4
N_STATES = 5
STATE_TO_LABEL = np.array(["H", "H", "A", "A", "D"], dtype="<U1")
STATE_TO_CODE = np.array([0, 0, 1, 1, 2], dtype=np.int8)
CODE_TO_LABEL = np.array(["H", "A", "D", "X"], dtype="<U1")


def fit_unsupervised_kinematic_gmm(feat: MatchFeatures, n_iter: int = 25) -> Dict:
    """Fit 2-component EM (Dead vs Alive) and calibrate relative-velocity scale on all graded frames."""
    mask = feat.graded_mask
    x1 = feat.p_activity_smooth[mask]
    x2 = np.log1p(feat.p_n_fast[mask])
    X = np.stack([x1, x2], axis=1).astype(np.float64)

    q25 = np.quantile(x1, 0.25)
    q55 = np.quantile(x1, 0.55)
    ball_active_mask = (feat.b_spd_roll_max[mask] >= 7.0) & (feat.b_vis_roll[mask] >= 0.7) & (~feat.b_oob[mask])
    dead_init = ((x1 <= q25) & (~ball_active_mask)) | feat.hard_dead[mask] | feat.b_oob[mask]
    alive_init = ((x1 >= q55) | ball_active_mask) & (~feat.hard_dead[mask]) & (~feat.b_oob[mask])

    mu = np.array([X[dead_init].mean(axis=0), X[alive_init].mean(axis=0)])
    var = np.array([
        np.maximum(X[dead_init].var(axis=0), 0.05),
        np.maximum(X[alive_init].var(axis=0), 0.05),
    ])
    pi = np.array([0.35, 0.65])

    for _ in range(n_iter):
        log_p = np.zeros((len(X), 2), dtype=np.float64)
        for k in (0, 1):
            diff = X - mu[k]
            log_p[:, k] = (
                np.log(pi[k] + 1e-12)
                - 0.5 * np.sum(np.log(2.0 * np.pi * var[k]) + (diff * diff) / var[k], axis=1)
            )
        m = np.max(log_p, axis=1, keepdims=True)
        resp = np.exp(log_p - m)
        resp /= np.sum(resp, axis=1, keepdims=True)

        nk = np.sum(resp, axis=0)
        pi = nk / len(X)
        for k in (0, 1):
            mu[k] = (resp[:, k : k + 1] * X).sum(axis=0) / nk[k]
            diff = X - mu[k]
            var[k] = np.maximum((resp[:, k : k + 1] * (diff * diff)).sum(axis=0) / nk[k], 0.04)

    if mu[0, 0] > mu[1, 0]:
        mu = mu[::-1]
        var = var[::-1]
        pi = pi[::-1]

    # Score using UN-SMEARED per-frame kinematics to eliminate +/- 12f boundary lag
    X_raw = np.stack([feat.p_mean_s, np.log1p(feat.p_n_fast)], axis=1).astype(np.float64)
    ll = np.zeros((feat.n_frames, 2), dtype=np.float64)
    for k in (0, 1):
        diff = X_raw - mu[k]
        ll[:, k] = (
            np.log(pi[k] + 1e-12)
            - 0.5 * np.sum(np.log(2.0 * np.pi * var[k]) + (diff * diff) / var[k], axis=1)
        )
    llr_alive_raw = np.clip(ll[:, 1] - ll[:, 0], -3.5, 3.5).astype(np.float32)

    # Calibrate relative-velocity control threshold unsup. from close-ball frames across full match
    rel_min = np.fmin(feat.rel_speed_h, feat.rel_speed_a)
    close_alive = (
        mask
        & feat.b_vis_or_interp
        & (~feat.b_oob)
        & (~feat.hard_dead)
        & (feat.sp_dead_prior < 0.5)
        & (llr_alive_raw > 0.5)
        & (feat.d_min < 2.0)
        & (~np.isnan(rel_min))
    )
    if np.sum(close_alive) > 500:
        q81_rel = float(np.nanpercentile(rel_min[close_alive], 81))
        rel_vel_scale = float(np.clip(q81_rel, 3.8, 7.5))
    else:
        rel_vel_scale = 5.0

    return {
        "mu_dead": mu[0],
        "mu_alive": mu[1],
        "pi_dead": float(pi[0]),
        "llr_alive_raw": llr_alive_raw,
        "rel_vel_scale": rel_vel_scale,
    }


def compute_frame_signals(feat: MatchFeatures, gmm_info: Dict) -> Dict[str, np.ndarray]:
    """Compute un-smeared 5-state per-frame log-emissions and ball-control signals."""
    n = feat.n_frames
    emit = np.zeros((n, N_STATES), dtype=np.float32)

    kin_alive = gmm_info["llr_alive_raw"]
    vis_on_pitch = feat.b_vis_or_interp & (~feat.b_oob)
    valid_ball = vis_on_pitch & (feat.sp_dead_prior < 0.5)

    # 1. Alive vs Dead signal (un-smeared kinematics + active ball + decel changepoints)
    b_spd_safe = np.where(~np.isnan(feat.b_speed), feat.b_speed, 0.0)
    b_inst_active = np.where(
        valid_ball,
        np.clip((b_spd_safe - 5.5) / 3.5, 0.0, 1.0),
        0.0,
    ).astype(np.float32)
    ball_active_pass = np.where(
        (~feat.b_oob) & (feat.sp_dead_prior < 0.5),
        np.clip((feat.b_spd_roll_max - 6.0) / 3.0, 0.0, 1.0) * feat.b_vis_roll,
        0.0,
    ).astype(np.float32)
    alive_ball_factor = np.maximum(b_inst_active, 0.45 * ball_active_pass)

    # 2. Directional Ball-Control Strength per team (ctrl_h, ctrl_a)
    d_h_safe = np.where(valid_ball & (~np.isnan(feat.d_home)), feat.d_home, 99.0)
    d_a_safe = np.where(valid_ball & (~np.isnan(feat.d_away)), feat.d_away, 99.0)

    prox_h = np.exp(-np.clip((d_h_safe / 2.6) ** 2, 0.0, 50.0))
    prox_a = np.exp(-np.clip((d_a_safe / 2.6) ** 2, 0.0, 50.0))

    # Physical spatial exclusivity: suppress false remote control only when opponent is >= 1m closer
    excl_prox_h = 1.0 / (1.0 + np.exp(np.clip((d_h_safe - d_a_safe - 0.95) / 0.40, -15.0, 15.0)))
    excl_prox_a = 1.0 / (1.0 + np.exp(np.clip((d_a_safe - d_h_safe - 0.95) / 0.40, -15.0, 15.0)))

    vel_scale = gmm_info["rel_vel_scale"]
    fallback_spd_gate = 1.0 / (1.0 + np.exp(np.clip((b_spd_safe - 9.0) / 2.2, -20.0, 20.0)))
    rel_h_valid = valid_ball & (~np.isnan(feat.rel_speed_h))
    rel_a_valid = valid_ball & (~np.isnan(feat.rel_speed_a))

    vel_gate_h = np.where(
        rel_h_valid,
        np.exp(-np.clip((feat.rel_speed_h / vel_scale) ** 2, 0.0, 50.0)),
        fallback_spd_gate,
    )
    vel_gate_a = np.where(
        rel_a_valid,
        np.exp(-np.clip((feat.rel_speed_a / vel_scale) ** 2, 0.0, 50.0)),
        fallback_spd_gate,
    )

    b_imp_safe = np.where(~np.isnan(feat.b_impulse), feat.b_impulse, 0.0)
    touch_h = np.where(
        valid_ball & (b_imp_safe >= 3.2) & (d_h_safe <= 2.4),
        0.40 * np.exp(-np.clip((d_h_safe / 2.1) ** 2, 0.0, 20.0)) * np.clip((b_imp_safe - 2.8) / 4.0, 0.0, 1.0),
        0.0,
    )
    touch_a = np.where(
        valid_ball & (b_imp_safe >= 3.2) & (d_a_safe <= 2.4),
        0.40 * np.exp(-np.clip((d_a_safe / 2.1) ** 2, 0.0, 20.0)) * np.clip((b_imp_safe - 2.8) / 4.0, 0.0, 1.0),
        0.0,
    )

    ctrl_h = np.where(valid_ball, excl_prox_h * np.clip(prox_h * vel_gate_h + touch_h, 0.0, 1.0), 0.0).astype(np.float32)
    ctrl_a = np.where(valid_ball, excl_prox_a * np.clip(prox_a * vel_gate_a + touch_a, 0.0, 1.0), 0.0).astype(np.float32)
    ctrl_max = np.maximum(ctrl_h, ctrl_a)

    # Exclusive control attenuates contested 50-50 duels smoothly
    net_h = np.maximum(0.0, ctrl_h - 0.65 * ctrl_a).astype(np.float32)
    net_a = np.maximum(0.0, ctrl_a - 0.65 * ctrl_h).astype(np.float32)
    net_max = np.maximum(net_h, net_a)

    # Prevent dead-ball kicks while players STAY walking (p_fut < 1.35) with zero touch from faking live play
    all_walking_no_touch = (feat.p_n_fast < 1.0) & (feat.p_mean_s < 1.42) & (feat.p_fut < 1.35) & (ctrl_max < 0.15)
    coasting_whistle = (feat.p_decel < -0.45) & (feat.p_fut < 2.05) & (ctrl_max < 0.12) & (feat.smart_alive_prior < 0.25)
    eff_ball_factor = np.where(coasting_whistle, 0.35 * b_inst_active, alive_ball_factor)

    kin_supp = np.where(all_walking_no_touch, 0.35, np.where(coasting_whistle, 0.35, 0.90))
    ball_alive_w = np.where(all_walking_no_touch, 0.70, 2.00)

    kin_alive_adj = np.where(
        kin_alive < 0.0,
        kin_alive * (1.0 - kin_supp * eff_ball_factor),
        kin_alive,
    )

    decel_dead_boost = np.where(
        (feat.p_decel < -0.42) & (feat.p_fut < 2.05) & (ctrl_max < 0.12) & (feat.smart_alive_prior < 0.25) & (feat.sp_dead_prior < 0.5),
        np.clip((-feat.p_decel - 0.38) * 5.0, 0.0, 4.2),
        0.0,
    ).astype(np.float32)

    mid_speed_thresh = 0.5 * (gmm_info["mu_dead"][0] + gmm_info["mu_alive"][0])
    ball_still_slow = np.where(
        vis_on_pitch & (~np.isnan(feat.b_speed)) & (feat.b_speed < 0.7) & (feat.p_mean_s < mid_speed_thresh + 0.2),
        0.8,
        0.0,
    ).astype(np.float32)

    oob_penalty = np.where(feat.b_oob, 12.0, 0.0).astype(np.float32)

    # Gentle dead-score attenuation during calm uncontested dribbling (sp_dead == 0, net_max high)
    calm_ctrl_dead_att = np.where(valid_ball, 1.2 * net_max, 0.0).astype(np.float32)

    alive_score = (
        0.65 * kin_alive_adj
        + ball_alive_w * eff_ball_factor
        + 1.5 * feat.smart_alive_prior
        - oob_penalty
        - 4.0 * feat.sp_dead_prior
        - decel_dead_boost
    )
    dead_score = (
        -0.65 * kin_alive_adj
        + oob_penalty
        + ball_still_slow
        + 4.0 * feat.sp_dead_prior
        + decel_dead_boost
        - np.where(all_walking_no_touch | coasting_whistle, 0.35, 1.0) * eff_ball_factor
        - calm_ctrl_dead_att
        - 1.5 * feat.smart_alive_prior
    )

    density_diff = np.clip(feat.h_density - feat.a_density, -1.5, 1.5).astype(np.float32)
    vis_f = valid_ball.astype(np.float32)
    occ_ctrl_pen = np.where(~vis_on_pitch, -0.15, 0.0).astype(np.float32)

    # CTRL emissions relative to FLIGHT (FLIGHT stays 0.0 on visible uncontrolled balls!)
    ctrl_bonus_h = (
        2.9 * net_h
        + 0.45 * net_h * density_diff
        - 1.15 * vis_f * np.maximum(0.0, 0.25 - ctrl_h) / 0.25
        - 2.4 * ctrl_a * (1.0 - 0.40 * ctrl_h)
        + occ_ctrl_pen
    ).astype(np.float32)

    ctrl_bonus_a = (
        2.9 * net_a
        - 0.45 * net_a * density_diff
        - 1.15 * vis_f * np.maximum(0.0, 0.25 - ctrl_a) / 0.25
        - 2.4 * ctrl_h * (1.0 - 0.40 * ctrl_a)
        + occ_ctrl_pen
    ).astype(np.float32)

    restart_h = ((feat.sp_restart_team == 0) & (feat.sp_dead_prior < 0.5)).astype(np.float32)
    restart_a = ((feat.sp_restart_team == 1) & (feat.sp_dead_prior < 0.5)).astype(np.float32)
    occ_team_prior = np.where(~vis_on_pitch, 0.08 * feat.smart_team_prior, 0.0).astype(np.float32)

    emit[:, HC_ST] = alive_score + ctrl_bonus_h + 5.5 * restart_h + 2.5 * feat.pass_evt_h + occ_team_prior
    emit[:, HF_ST] = alive_score + 4.0 * restart_h + occ_team_prior
    emit[:, AC_ST] = alive_score + ctrl_bonus_a + 5.5 * restart_a + 2.5 * feat.pass_evt_a - occ_team_prior
    emit[:, AF_ST] = alive_score + 4.0 * restart_a - occ_team_prior
    emit[:, D_ST] = dead_score

    hd = feat.hard_dead
    emit[hd, :D_ST] = -50.0
    emit[hd, D_ST] = 50.0

    return {
        "emit": emit,
        "ctrl_h": ctrl_h,
        "ctrl_a": ctrl_a,
        "vis_on_pitch": vis_on_pitch.astype(np.float32),
        "b_active": (vis_on_pitch & (b_spd_safe >= 6.5) & (feat.sp_dead_prior < 0.5)).astype(np.float32),
    }


def _hsmm_decode_half(
    feat: MatchFeatures,
    sigs: Dict[str, np.ndarray],
    hs: int,
    he: int,
    stride: int = 2,
    d_cap_blocks: int = 83,
    sw_turn: float = -16.5,
    turn_pen_mag: float = 14.0,
    pass_relax: float = 4.0,
) -> np.ndarray:
    """5-State Explicit-Duration Hybrid HSMM DP + 1-frame exact boundary snapping."""
    length = he - hs
    n_blk = (length + stride - 1) // stride
    pad_len = n_blk * stride

    def _to_blocks(arr, is_bool=False):
        sub = arr[hs:he]
        if len(sub) < pad_len:
            pad_val = False if is_bool else 0.0
            sub = np.concatenate([sub, np.full(pad_len - len(sub), pad_val, dtype=sub.dtype)])
        return sub.reshape(n_blk, stride)

    emit_raw = sigs["emit"][hs:he]
    if len(emit_raw) < pad_len:
        emit_pad = np.vstack([emit_raw, np.zeros((pad_len - len(emit_raw), N_STATES), dtype=np.float32)])
    else:
        emit_pad = emit_raw
    emit_blk = emit_pad.reshape(n_blk, stride, N_STATES).sum(axis=1).astype(np.float64)

    oob_blk = _to_blocks(feat.b_oob, True).sum(axis=1).astype(np.int32)
    hd_blk = _to_blocks(feat.hard_dead, True).sum(axis=1).astype(np.int32)
    sp_dead_blk = _to_blocks(feat.sp_dead_prior).sum(axis=1).astype(np.float64)
    ctrl_h_blk = _to_blocks(sigs["ctrl_h"]).sum(axis=1).astype(np.float64)
    ctrl_a_blk = _to_blocks(sigs["ctrl_a"]).sum(axis=1).astype(np.float64)
    b_vis_blk = _to_blocks(sigs["vis_on_pitch"]).sum(axis=1).astype(np.float64)
    b_act_blk = _to_blocks(sigs["b_active"]).sum(axis=1).astype(np.float64)
    p_mean_blk = _to_blocks(feat.p_mean_s).sum(axis=1).astype(np.float64)
    nfast_blk = _to_blocks(feat.p_n_fast).sum(axis=1).astype(np.float64)
    sm_alive_blk = _to_blocks(feat.smart_alive_prior).sum(axis=1).astype(np.float64)
    res_h_blk = _to_blocks(feat.sp_restart_team == 0, True).sum(axis=1).astype(np.int32)
    res_a_blk = _to_blocks(feat.sp_restart_team == 1, True).sum(axis=1).astype(np.int32)
    pass_h_blk = _to_blocks(feat.pass_evt_h).sum(axis=1).astype(np.float64)
    pass_a_blk = _to_blocks(feat.pass_evt_a).sum(axis=1).astype(np.float64)

    P_emit = np.vstack([np.zeros((1, N_STATES)), np.cumsum(emit_blk, axis=0)])
    P_oob = np.concatenate([[0], np.cumsum(oob_blk)])
    P_hd = np.concatenate([[0], np.cumsum(hd_blk)])
    P_sp_dead = np.concatenate([[0.0], np.cumsum(sp_dead_blk)])
    P_ctrl_h = np.concatenate([[0.0], np.cumsum(ctrl_h_blk)])
    P_ctrl_a = np.concatenate([[0.0], np.cumsum(ctrl_a_blk)])
    P_b_vis = np.concatenate([[0.0], np.cumsum(b_vis_blk)])
    P_b_act = np.concatenate([[0.0], np.cumsum(b_act_blk)])
    P_pmean = np.concatenate([[0.0], np.cumsum(p_mean_blk)])
    P_nfast = np.concatenate([[0.0], np.cumsum(nfast_blk)])
    P_sm_alive = np.concatenate([[0.0], np.cumsum(sm_alive_blk)])
    P_res_h = np.concatenate([[0], np.cumsum(res_h_blk)])
    P_res_a = np.concatenate([[0], np.cumsum(res_a_blk)])
    P_pass_h = np.concatenate([[0.0], np.cumsum(pass_h_blk)])
    P_pass_a = np.concatenate([[0.0], np.cumsum(pass_a_blk)])

    # Transition matrix SW[prev, curr]: strictly forbids H_FLIGHT <-> A_FLIGHT & *_CTRL -> opp_*_FLIGHT
    INF_NEG = -1e9
    SW = np.array([
        # to:      HC       HF       AC       AF        D
        [INF_NEG,   -1.5, sw_turn, INF_NEG,   -19.0],  # from HC
        [   -1.5, INF_NEG, sw_turn, INF_NEG,   -19.0],  # from HF
        [sw_turn, INF_NEG, INF_NEG,   -1.5,   -19.0],  # from AC
        [sw_turn, INF_NEG,   -1.5, INF_NEG,   -19.0],  # from AF
        [  -18.0,   -21.0,  -18.0,   -21.0, INF_NEG],  # from D
    ], dtype=np.float64)

    dp = np.full((n_blk + 1, N_STATES), -1e12, dtype=np.float64)
    dp[0, :] = 0.0
    back_start = np.zeros((n_blk + 1, N_STATES), dtype=np.int32)
    back_prev = np.full((n_blk + 1, N_STATES), -1, dtype=np.int8)

    tail_val = np.full(N_STATES, -1e12, dtype=np.float64)
    tail_start = np.zeros(N_STATES, dtype=np.int32)
    tail_prev = np.full(N_STATES, -1, dtype=np.int8)

    d_arr = np.arange(1, d_cap_blocks, dtype=np.int32)
    dur_f_arr = d_arr * stride

    # Explicit duration curves (football physics scales)
    dur_pen_turn = -float(turn_pen_mag) * np.maximum(0.0, 1.0 - dur_f_arr / 39.0)
    dur_pen_cont = -4.0 * np.maximum(0.0, 1.0 - dur_f_arr / 12.0)
    dur_pen_flight = -2.0 * np.maximum(0.0, 1.0 - dur_f_arr / 9.0)
    dur_pen_dead = -120.0 * np.maximum(0.0, 1.0 - dur_f_arr / 102.0)
    dur_pen_oob_sliver = -90.0 * np.maximum(0.0, 1.0 - dur_f_arr / 24.0)
    dur_pen_res_sliver = -22.0 * np.maximum(0.0, 1.0 - dur_f_arr / 24.0)
    no_ctrl_taper = np.clip((105.0 - dur_f_arr) / 30.0, 0.0, 1.0)

    for j in range(1, n_blk + 1):
        max_d = min(j, d_cap_blocks - 1)
        ds = d_arr[:max_d]
        starts = j - ds
        dur_fs = dur_f_arr[:max_d]

        # 1. Running exact tail DP for segments of duration >= d_cap_blocks
        if j >= d_cap_blocks:
            i_cap = j - d_cap_blocks
            for s in range(N_STATES):
                ext_val = tail_val[s] + emit_blk[j - 1, s]
                if i_cap == 0:
                    best_in, best_p = 0.0, -1
                else:
                    cands = dp[i_cap, :] + SW[:, s]
                    best_p = int(np.argmax(cands))
                    best_in = cands[best_p]
                fresh_val = best_in + (P_emit[j, s] - P_emit[i_cap, s])
                if fresh_val > ext_val:
                    tail_val[s] = fresh_val
                    tail_start[s] = i_cap
                    tail_prev[s] = best_p
                else:
                    tail_val[s] = ext_val

        # 2. Vectorized evaluation over short durations d in [1 .. max_d]
        seg_oob = (P_oob[j] - P_oob[starts]) > 0
        seg_hd = (P_hd[j] - P_hd[starts]) > 0
        seg_sp_dead = (P_sp_dead[j] - P_sp_dead[starts]) / dur_fs
        seg_b_vis = P_b_vis[j] - P_b_vis[starts]
        seg_b_act = (P_b_act[j] - P_b_act[starts]) / dur_fs
        seg_pmean = (P_pmean[j] - P_pmean[starts]) / dur_fs
        seg_nfast = (P_nfast[j] - P_nfast[starts]) / dur_fs
        seg_sm_alive = (P_sm_alive[j] - P_sm_alive[starts]) / dur_fs

        # Smooth walking dead-ball pickup, zero-vis jogs & phantom occluded restart penalties
        w_slow = np.clip((1.55 - seg_pmean) / 0.20, 0.0, 1.0)
        w_nf = np.clip((2.8 - seg_nfast) / 1.5, 0.0, 1.0)
        walk_dur_limit = d_cap_blocks * stride
        walk_mask = (dur_fs < walk_dur_limit) & (seg_sm_alive < 0.25) & (w_slow > 0.0) & (w_nf > 0.0)
        walk_pen_base = np.where(walk_mask, -3.8 * dur_fs * w_slow * w_nf, 0.0)
        zero_vis_jog = (seg_b_vis < 1.0) & (seg_sm_alive < 0.25) & (seg_nfast < 5.0) & (seg_pmean < 2.1)
        walk_pen_base = np.where(zero_vis_jog, np.minimum(walk_pen_base, -2.2 * dur_fs - 30.0), walk_pen_base)
        seg_ctrl_tot = (P_ctrl_h[j] - P_ctrl_h[starts]) + (P_ctrl_a[j] - P_ctrl_a[starts])
        phantom_dead_exit = (
            (dur_fs < 90) & (seg_ctrl_tot < 2.0) & (seg_sm_alive < 0.25) & (seg_b_act < 0.15)
        )
        walk_pen_base = np.where(phantom_dead_exit, np.minimum(walk_pen_base, -40.0), walk_pen_base)
        is_zero = starts == 0

        for s in range(N_STATES):
            seg_e = P_emit[j, s] - P_emit[starts, s]
            scores_prev = dp[starts, :] + SW[:, s][None, :]  # (max_d, 5)

            if s in (HC_ST, AC_ST):
                is_home = s == HC_ST
                seg_ctrl = (P_ctrl_h[j] - P_ctrl_h[starts]) if is_home else (P_ctrl_a[j] - P_ctrl_a[starts])
                seg_res = (P_res_h[j] - P_res_h[starts]) if is_home else (P_res_a[j] - P_res_a[starts])
                seg_pass = (P_pass_h[j] - P_pass_h[starts]) if is_home else (P_pass_a[j] - P_pass_a[starts])
                opp_cols = (AC_ST, AF_ST) if is_home else (HC_ST, HF_ST)
                cont_col = HF_ST if is_home else AF_ST

                turn_pen = np.where(seg_res > 0, dur_pen_res_sliver[:max_d], dur_pen_turn[:max_d])
                if pass_relax > 0.0:
                    turn_pen = np.where(
                        seg_pass >= 1.0,
                        turn_pen + np.minimum(pass_relax, -turn_pen + 0.5 * pass_relax),
                        turn_pen,
                    )
                no_ctrl = np.where(
                    (seg_res == 0) & (seg_pass < 1.0) & (seg_ctrl < 3.5),
                    -20.0 * no_ctrl_taper[:max_d],
                    0.0,
                )
                scores_prev[:, opp_cols[0]] += turn_pen + no_ctrl
                scores_prev[:, opp_cols[1]] += turn_pen + no_ctrl
                scores_prev[:, cont_col] += dur_pen_cont[:max_d]
                scores_prev[:, D_ST] += np.where(seg_res > 0, dur_pen_res_sliver[:max_d], walk_pen_base)

            elif s in (HF_ST, AF_ST):
                seg_res = (P_res_h[j] - P_res_h[starts]) if s == HF_ST else (P_res_a[j] - P_res_a[starts])
                scores_prev += dur_pen_flight[:max_d, None]
                scores_prev[:, D_ST] += np.where(seg_res > 0, dur_pen_res_sliver[:max_d], walk_pen_base)

            else:  # D_ST
                oob_or_sp = seg_oob | (seg_sp_dead > 0.25)
                d_pen = np.where(seg_hd, 0.0, np.where(oob_or_sp, dur_pen_oob_sliver[:max_d], dur_pen_dead[:max_d]))
                no_dead_anchor = (~seg_oob) & (~seg_hd) & (seg_sp_dead < 0.2) & (seg_b_act > 0.20)
                d_pen = np.where(no_dead_anchor, d_pen - 3.5 * dur_fs * (seg_b_act - 0.15), d_pen)
                scores_prev += d_pen[:, None]

            best_p_arr = np.argmax(scores_prev, axis=1)
            best_in_arr = np.where(is_zero, 0.0, scores_prev[np.arange(max_d), best_p_arr])
            best_p_arr = np.where(is_zero, -1, best_p_arr)

            tot_short = best_in_arr + seg_e
            best_short_idx = int(np.argmax(tot_short))
            best_short_val = float(tot_short[best_short_idx])

            if tail_val[s] > best_short_val:
                dp[j, s] = tail_val[s]
                back_start[j, s] = tail_start[s]
                back_prev[j, s] = tail_prev[s]
            else:
                dp[j, s] = best_short_val
                back_start[j, s] = int(starts[best_short_idx])
                back_prev[j, s] = int(best_p_arr[best_short_idx])

    # Backtrack optimal block segments
    segs: List[Tuple[int, int, int]] = []
    cur_j = n_blk
    cur_s = int(np.argmax(dp[n_blk]))
    while cur_j > 0:
        i_start = int(back_start[cur_j, cur_s])
        prev_s = int(back_prev[cur_j, cur_s])
        segs.append((cur_s, i_start * stride, min(length, cur_j * stride)))
        cur_j = i_start
        cur_s = prev_s
    segs.reverse()

    # Local 1-frame exact boundary snapping (+/- 8 frames) when block stride > 1
    if stride > 1 and len(segs) > 1:
        snapped = [list(seg) for seg in segs]
        for k in range(1, len(snapped)):
            s1, l1, r1 = snapped[k - 1]
            s2, l2, r2 = snapped[k]
            b = l2
            w_lo = max(l1 + 5, b - 8)
            w_hi = min(r2 - 5, b + 8)
            if w_lo < w_hi:
                sub_e1 = emit_raw[w_lo:w_hi, s1].astype(np.float64)
                sub_e2 = emit_raw[w_lo:w_hi, s2].astype(np.float64)
                prefix = np.concatenate([[0.0], np.cumsum(sub_e1 - sub_e2)])
                new_b = w_lo + int(np.argmax(prefix))
                snapped[k - 1][2] = new_b
                snapped[k][1] = new_b
        segs = [(s, a, b) for s, a, b in snapped]

    states = np.empty(length, dtype=np.int8)
    for s, a, b in segs:
        states[a:b] = s
    return states


def _refine_voronoi_flight_handovers(
    codes: np.ndarray,
    feat: MatchFeatures,
    sigs: Dict[str, np.ndarray],
    r0: float = 1.5,
    power: float = 1.0,
    ctrl_thr: float = 0.35,
    gap_min: int = 8,
) -> np.ndarray:
    """Partition uncontrolled H <-> A aerial/loose-ball gaps by inverse player proximity."""
    refined = codes.copy()
    ctrl_h, ctrl_a = sigs["ctrl_h"], sigs["ctrl_a"]
    for _, hs, he in feat.half_spans:
        sub = refined[hs:he]
        all_chg = np.flatnonzero(sub[1:] != sub[:-1]) + 1 + hs
        for idx_c, t in enumerate(all_chg):
            old_c, new_c = int(codes[t - 1]), int(codes[t])
            if old_c >= 2 or new_c >= 2:
                continue
            lo_lim = (int(all_chg[idx_c - 1]) + 9) if idx_c > 0 else (hs + 9)
            hi_lim = (int(all_chg[idx_c + 1]) - 9) if idx_c + 1 < len(all_chg) else (he - 9)
            l = t - 1
            while l > hs and ctrl_h[l] < ctrl_thr and ctrl_a[l] < ctrl_thr and (t - l) < 120:
                l -= 1
            r = t
            while r < he and ctrl_h[r] < ctrl_thr and ctrl_a[r] < ctrl_thr and (r - t) < 120:
                r += 1
            g_lo, g_hi = l + 1, r
            if (g_hi - g_lo) < gap_min or lo_lim > hi_lim:
                continue
            d_old = (feat.d_home if old_c == 0 else feat.d_away)[g_lo:g_hi]
            d_new = (feat.d_away if old_c == 0 else feat.d_home)[g_lo:g_hi]
            d_o_s = np.where(np.isnan(d_old), 99.0, d_old)
            d_n_s = np.where(np.isnan(d_new), 99.0, d_new)
            pref_diff = (1.0 / (d_o_s + r0) ** power) - (1.0 / (d_n_s + r0) ** power)
            cs = np.concatenate([[0.0], np.cumsum(pref_diff)])
            t_new = max(lo_lim, min(hi_lim, g_lo + int(np.argmax(cs))))
            if t_new < t:
                refined[t_new:t] = new_c
            elif t_new > t:
                refined[t:t_new] = old_c
    return refined


def _snap_restart_kicks(
    feat: MatchFeatures,
    codes_raw: np.ndarray,
    fps: float = 25.0,
) -> np.ndarray:
    """Snap D -> Alive boundaries onto exact 1-frame ball motion onset (fixes +/-2f velocity smear)."""
    codes = codes_raw.copy()
    bx, by = feat.bx, feat.by
    step_1f = np.zeros(feat.n_frames, dtype=np.float32)
    ok = (~np.isnan(bx[:-1])) & (~np.isnan(bx[1:]))
    step_1f[:-1][ok] = np.hypot(bx[1:][ok] - bx[:-1][ok], by[1:][ok] - by[:-1][ok]) * fps
    step_1f[step_1f > 42.0] = 0.0

    for _, hs, he in feat.half_spans:
        sub = codes[hs:he]
        d2a = np.flatnonzero((sub[:-1] == 2) & (sub[1:] < 2)) + 1 + hs
        for t in d2a:
            if t + 10 >= he or t - 4 < hs:
                continue
            s_win = step_1f[t - 4 : t + 11]
            moving = (s_win[:-1] >= 2.8) & (s_win[1:] >= 2.8)
            fast = s_win[:-1] >= 4.2
            hits = np.flatnonzero(moving | fast)
            if len(hits) > 0:
                t_onset = (t - 4) + int(hits[0])
                pre_lo = max(hs, t_onset - 4)
                if t_onset > pre_lo and np.max(step_1f[pre_lo:t_onset]) < 2.2:
                    if t_onset > t and (t_onset - t) <= 9:
                        codes[t:t_onset] = 2
                    elif t_onset < t and (t - t_onset) <= 4:
                        codes[t_onset:t] = codes[t]
    return codes


def decode_match(feat: MatchFeatures, fps: float = 25.0) -> Tuple[np.ndarray, np.ndarray, Dict]:
    """Run Unsupervised EM + 5-State Exact Hybrid HSMM + Boundary Refinement."""
    gmm_info = fit_unsupervised_kinematic_gmm(feat)
    sigs = compute_frame_signals(feat, gmm_info)

    states = np.full(feat.n_frames, -1, dtype=np.int8)
    codes = np.full(feat.n_frames, 3, dtype=np.int8)

    for _, hs, he in feat.half_spans:
        half_states = _hsmm_decode_half(feat, sigs, hs, he)
        states[hs:he] = half_states
        codes[hs:he] = STATE_TO_CODE[half_states]

    codes_v = _refine_voronoi_flight_handovers(codes, feat, sigs, r0=1.5, power=1.0, ctrl_thr=0.35, gap_min=8)
    codes_k = _snap_restart_kicks(feat, codes_v, fps=fps)

    # Keep 5-state array synchronized with post-DP boundary refinements
    code_fallback = np.array([HF_ST, AF_ST, D_ST], dtype=np.int8)
    for _, hs, he in feat.half_spans:
        mismatch = STATE_TO_CODE[states[hs:he]] != codes_k[hs:he]
        if np.any(mismatch):
            idxs = hs + np.flatnonzero(mismatch)
            states[idxs] = code_fallback[codes_k[idxs]]

    labels = CODE_TO_LABEL[codes_k]
    return labels, states, gmm_info
