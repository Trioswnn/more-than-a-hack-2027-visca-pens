"""End-to-End Championship Possession Pipeline (More Than a Hack 2027).

Reproducibly generates the full-match possession submission from raw Tactical
and Broadcast data without touching or tuning on any ground-truth file.

Usage:
    python models/run_pipeline.py
"""

import argparse
import os
import sys
import time
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

import games
from starter import baseline
from evaluation import possession_metrics as pm
from models.data_engine import load_atd_source, graded_half_spans, graded_mask_from_spans
from models.broadcast_sync import sync_broadcast_to_tactical
from models.features import extract_features
from models.state_model import decode_match


def print_full_match_audit(labels: np.ndarray, gmm_info: dict, game: str = games.GAME) -> None:
    """Print unsupervised sanity metrics across all 158,857 graded match frames."""
    spans = graded_half_spans(game)
    gmask = graded_mask_from_spans(len(labels), spans)
    fps = float(games.GAMES[game]["fps"])

    h_cnt = int(np.sum(labels[gmask] == "H"))
    a_cnt = int(np.sum(labels[gmask] == "A"))
    d_cnt = int(np.sum(labels[gmask] == "D"))
    x_cnt = int(np.sum(labels[gmask] == "X"))
    alive = h_cnt + a_cnt

    # Count spells across both halves independently
    codes = pm.label_codes(labels)
    runs = [(s, e) for _, s, e in spans]
    spells = pm.possession_spells(codes, runs)
    mean_dur_s = (alive / fps / len(spells)) if spells else 0.0

    print("\n=== Unsupervised Full-Match Audit (158,857 Graded Frames) ===")
    rel_sc = gmm_info.get("rel_vel_scale", 0.0)
    print(f"  EM Kinematic Modes : Dead mean={gmm_info['mu_dead'][0]:.2f} m/s, Alive mean={gmm_info['mu_alive'][0]:.2f} m/s | rel_scale={rel_sc:.2f} m/s")
    print(f"  Graded Distribution: H={100*h_cnt/np.sum(gmask):.1f}%  A={100*a_cnt/np.sum(gmask):.1f}%  D={100*d_cnt/np.sum(gmask):.1f}%  X={x_cnt}")
    if alive > 0:
        print(f"  Alive Possession   : Home {100*h_cnt/alive:.1f}% vs Away {100*a_cnt/alive:.1f}% ({alive/fps/60:.1f} min ball-in-play)")
    print(f"  Full-Match Spells  : {len(spells)} possessions, mean duration = {mean_dur_s:.1f} s")


def run(game: str = games.GAME, data_dir=None, name: str = "VISCA PENS", use_broadcast: bool = True):
    t_start = time.time()
    print(f"[1/5] Loading Tactical tracking data ({game}) ...")
    tac = load_atd_source(game=game, data_dir=data_dir, source="tactical")

    broad, fused_bx, fused_by, tac_to_broad, tac_match_err = None, None, None, None, None
    if use_broadcast:
        print("[2/5] Synchronizing Broadcast tracking & recovering occluded ball ...")
        broad = load_atd_source(game=game, data_dir=data_dir, source="broadcast")
        sync_res = sync_broadcast_to_tactical(tac, broad, game=game, data_dir=data_dir)
        fused_bx = sync_res["fused_bx"]
        fused_by = sync_res["fused_by"]
        tac_to_broad = sync_res["tac_to_broad"]
        tac_match_err = sync_res["tac_match_err"]
        print(f"      Matched {sync_res['n_shots_matched']}/{sync_res['n_shots_total']} shots; "
              f"recovered +{sync_res['n_ball_recovered']} ball frames "
              f"(median overlap err = {sync_res['both_ball_err_median_m']:.2f} m)")
    else:
        print("[2/5] Skipping Broadcast synchronization.")

    print("[3/5] Extracting physics, multi-camera, pass-chain, and Smart Tagging features ...")
    feat = extract_features(
        tac,
        fused_bx=fused_bx,
        fused_by=fused_by,
        game=game,
        data_dir=data_dir,
        broad=broad,
        tac_to_broad=tac_to_broad,
        tac_match_err=tac_match_err,
    )

    print("[4/5] Running Unsupervised EM + 5-State Exact Hybrid HSMM (v5.0) decoding ...")
    labels, _, gmm_info = decode_match(feat, fps=float(games.GAMES[game]["fps"]))
    print_full_match_audit(labels, gmm_info, game=game)

    sub_path = os.path.join(ROOT, "submissions", f"{name}.csv")
    eval_path = os.path.join(ROOT, "evaluation", "results", f"{name}.csv")

    print(f"\n[5/5] Writing & validating submission CSVs ...")
    baseline.write_submission(sub_path, labels)
    baseline.write_submission(eval_path, labels)

    table = pm.read_label_csv(sub_path)
    reason = pm.validate_submission(table, tac.n_frames)
    if reason:
        raise ValueError(f"Submission validation FAILED: {reason}")
    print(f"      PASSED validation: {sub_path} ({tac.n_frames} rows)")
    print(f"      Copied for scoring: {eval_path}")
    print(f"\n[DONE] Completed in {time.time() - t_start:.2f}s.")
    return sub_path, eval_path


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--name", default="VISCA PENS", help="Submission name")
    parser.add_argument("--no-broadcast", action="store_true", help="Disable broadcast ball recovery")
    args = parser.parse_args()
    run(name=args.name, use_broadcast=not args.no_broadcast)
