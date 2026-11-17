# Technical Report: More Than a Hack 2027

**Team:** **VISCA PENS**  
**Team Members:** Serafime Angelica Saktian, Zahra Aurelia Djanaid, Trio Setiawan, Luqman Muhammad Hardian, Fatma Putri Innaya  
**Challenge:** Phase 1 — Full-Match Frame-by-Frame Ball Possession Estimation (`PI-HSMM v5`)  
**Match Fixture:** FC Barcelona vs. Athletic Club (`BAR-ATH`, LaLiga EA Sports 2024/25, Matchday 2)  
**Primary Evaluation Metric:** `spellF1` (Mean temporal-IoU F1 over thresholds 0.3, 0.4, 0.5, 0.6, 0.7)  
**Secondary Tie-Breaker:** `macroF1` (Unweighted mean F1 over `H`, `A`, `D` on all graded frames)  

---

## 1. Executive Summary & Approach

Estimating controlled team possession from automated tracking data presents three fundamental difficulties:
1. **Severe Ball Occlusion:** The ball is absent (`NaN`) on **40.69%** of graded tactical frames (`64,643 / 158,857`), disproportionately during crowded box scrambles, aerial duels, and stoppages. Equating missing ball with Dead Ball (`D`) fractures live spells into micro-fragments.
2. **Temporal Fragmentation vs. Controlled Possession:** Naive spatial models flip team ownership whenever a pass flies over a defender, whenever an opponent briefly tracks past the ball mid-flight, or when tracking slots drop an active carrier.
3. **Aerial Turnover & Smear Boundary Ambiguity:** Across long uncontrolled flight gaps between teams (`H <-> A`) and central-difference velocity smoothing on set-piece restarts (`D -> Alive`), raw sequence boundaries can lag by several frames.

To solve these strictly within `RULES.md` (zero training, tuning, or hand-labeling on the 5-minute public ground truth), we built the **Cross-Camera Physics-Informed 5-State Exact Hybrid Explicit-Duration Semi-Markov Model (`PI-HSMM v5`)**:
1. **Multi-Frame Glitch Filtering & Cross-Camera Sync (`models/data_engine.py`, `models/broadcast_sync.py`):** Filters non-physical ball teleportation islands and synchronizes Broadcast tracking onto Tactical via unsupervised **Two-Pass Hough Anchor Alignment** (`150/182` live shots matched) and one-way Chamfer gating, recovering **+5,474** occluded ball frames (median overlap error **0.83 m**).
2. **Physics, Multi-Camera Outfield Recovery & Pass-Chain Features (`models/features.py`):** Recovers medium linear/drag-gap ball occlusions (`26..45f`), stitches broken same-team player tracklets (`max_gap=45`), filters far off-pitch warm-up/bench players (`> 2.5 m` outside lines when ball is in play), recovers missing non-GK outfield carriers from synced Broadcast tracking (`_fuse_broadcast_outfield`), detects physically verified teammate-to-teammate pass completions (`_detect_verified_passes`), and extracts un-smeared relative ball-player velocity `||v_ball - v_player||`, kick/reception impulses ($\Delta \vec{v}_{\text{ball}}$), collective deceleration (`p_decel = p_fut - p_past`), and enriched Smart Tagging phase/restart anchors.
3. **Unsupervised Full-Match Calibration (`models/state_model.py`):** Fits a 2-component Expectation-Maximization (EM) Gaussian Mixture over all **158,857 graded frames** to separate dead vs. alive kinematics ($\mu_{\text{dead}} = 1.02\text{ m/s}$, $\mu_{\text{alive}} = 2.49\text{ m/s}$) and calibrates the relative-velocity control scale (`4.44 m/s` from 81st-percentile close-ball motion clipped to `[3.8, 7.5] m/s`).
4. **High-Resolution 5-State Exact HSMM + Boundary Refinement (`models/state_model.py`):** Runs exact vectorized 5-state explicit-duration HSMM DP at **0.08 s temporal resolution** (`stride = 2`, `d_cap_blocks = 83`) with verified pass-chain emissions/relaxation and zero-visible-ball jog guards, followed by:
   * **Voronoi Flight Handover Partitioning (`_refine_voronoi_flight_handovers`):** Partitions uncontrolled `H <-> A` aerial gaps (`>= 8` frames with `ctrl < 0.35`) at the exact maximum of cumulative inverse-proximity (`1 / (d_old + 1.5) - 1 / (d_new + 1.5)`).
   * **1-Frame Kick-Impulse Onset Snapping (`_snap_restart_kicks`):** Snaps `D -> Alive` restart boundaries directly onto exact 1-frame ball motion onset (`step_1f`), removing the `+/- 2f` central-difference velocity smear.

---

## 2. Match Assets Used & Selection Rationale

| Asset | Usage in Pipeline | Rationale / Notes |
| :--- | :--- | :--- |
| `BAR-ATH_tactical_atd.txt` & `.xml` | Primary tracking stream | 199,319 frames of 46 track slots + ball coordinates in metric pitch space ($105.3 \times 68.0$ m); `94,214 / 158,857` (59.31%) graded tactical ball visible. |
| `BAR-ATH_broadcast_atd.txt` & `.xml` | Cross-camera ball & outfield carrier recovery | 164,488 frames. Imputes **+5,474** missing Tactical ball frames with **0.83 m** median overlap agreement and recovers missing non-GK outfield carriers when synced (`_fuse_broadcast_outfield`). |
| `BAR-ATH_broadcast_homography.csv` | Shot gating & replay rejection | Filters Broadcast shots using `valid == True` and `reprojection_error <= 3.5`; replays and slow-motion cuts are rejected via half-anchor consistency and per-frame Chamfer gating. |
| `BAR-ATH_tactical_smart_tagging.xml` | Event & restart anchors | 638 automatic tags used for: (a) locating exact kinematic kick/throw frames inside `SET PIECES`, (b) locking goal celebrations (`GOALS` -> `Kick Off (After Goal)`) to `D`, (c) soft alive priors on trimmed phase intervals (`BUILD UP`, `PROGRESSION`, `LONG BALL`, `CREATING CHANCES`, `ATTACKING/DEFENSIVE TRANSITION`, defending thirds), and (d) weak tie-breaking priors during occlusions. |
| Raw Video (`.mp4`) | *Deliberately excluded* | Pre-extracted dual-camera ATD + homography pipeline achieves deterministic CPU execution (`69.12s` end-to-end) without video I/O overhead. |

**Dropped Signals After Controlled Ablation:** Overwriting Tactical visible ball positions with synced Broadcast coordinates (Broadcast pan/zoom parallax introduces lag), overriding Tactical team IDs via Broadcast swap votes, and direct additive sequence/tree post-blending all degraded structural `spellF1` or `macroF1` vs. the pure physics HSMM decoder and were dropped from the production pipeline.

---

## 3. Training Data & Anti-Overfitting Compliance (`RULES.md`)

* **External Labelled Training Data:** None used.
* **Public 5-Minute Ground Truth (`evaluation/gt/BAR-ATH_possession_gt_public.csv`):** Strictly excluded from model training, parameter fitting, and automated hyperparameter tuning. Used exclusively via `evaluation/run_evaluation.py --bootstrap 1000` for verification diagnostics.
* **Unsupervised Full-Match Self-Calibration:** Kinematic Dead/Alive distributions ($\mu_{\text{dead}} = 1.02\text{ m/s}$, $\mu_{\text{alive}} = 2.49\text{ m/s}$), relative-velocity control scale (`rel_scale = 4.44 m/s`), goalkeeper slot detection (`_detect_gk_slots`), tracklet stitching, and Broadcast-to-Tactical temporal half anchors are calibrated algorithmically across the match without human supervision.

---

## 4. Possession Architecture & Physics Modeling (`v5.0`)

We decode each half independently over **5 latent states**:
* `H_CTRL` (Home active control/dribble/reception) $\to$ **`H`**
* `H_FLIGHT` (Ball in flight / loose / occluded under Home spell) $\to$ **`H`**
* `A_CTRL` (Away active control/dribble/reception) $\to$ **`A`**
* `A_FLIGHT` (Ball in flight / loose / occluded under Away spell) $\to$ **`A`**
* `DEAD` (Out of play / stoppage / pre-restart setup) $\to$ **`D`**

### Key Structural & Physics Improvements (`v4` $\to$ `v5`):
1. **Forbidden Mid-Flight Turnovers (`H_FLIGHT <-> A_FLIGHT = -inf`, `*_CTRL -> opp_*_FLIGHT = -inf`):** Possession cannot flip mid-air or passively while occluded; turning over possession requires genuine control evidence (`*_CTRL`) at `sw_turn = -16.5`, `turn_pen_mag = 14.0`, mitigated up to `pass_relax = 4.0` when verified teammate pass chains (`pass_evt_h/a`) confirm true team control (`>= 3.2 m` displacement across `8..85` stitched carrier frames).
2. **Multi-Camera Outfield Recovery & Tracklet Hygiene (`models/features.py`):** Excludes off-pitch bench/warm-up tracks (`> 2.5 m` outside boundaries when `~b_oob`), stitches fragmented tracklets (`<= 45f`), bridges linear/drag ball occlusions (`26..45f`), and borrows synced non-GK Broadcast outfield proximity when Tactical drops a carrier (`> 2.4 m` vs `< 2.2 m`).
3. **High-Resolution HSMM & Zero-Vis Stoppage Guard (`models/state_model.py`):** `stride = 2` (`d_cap_blocks = 83`) sharpens transition precision (`+/- 8f` 1-frame emission snap), while the `zero_vis_jog` penalty prevents extended zero-visible-ball jogging stretches without active tagging/fast runners from spuriously breaking Dead (`D`) periods.
4. **Voronoi Flight Handovers & 1-Frame Kick Onset Snapping:**
   * `_refine_voronoi_flight_handovers` computes exact cumulative inverse-proximity handovers inside uncontrolled `H <-> A` flight windows (`>= 8f`, clamped `>= 9f` from neighbor changes).
   * `_snap_restart_kicks` eliminates the `+/- 2f` central-difference velocity lead on `D -> Alive` restarts by snapping onto direct 1-frame forward displacement spikes (`>= 2.8 m/s` consecutive or `>= 4.2 m/s`).

---

## 5. Final Benchmark & Verification Results

### A. Public Verification Benchmark (`evaluation/run_evaluation.py --bootstrap 1000`)

| Submission | `spellF1` (Rank) | `macroF1` | `spellF1` @IoU 0.5 | `D` F1 | Alive/Dead F1 | `transF1` (@25f) | Team Acc (GT-Alive) |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **`VISCA PENS` (`PI-HSMM v5`)** | **0.6483 (#1)** | **0.9282 (#1)** | **0.6897** | **0.9539** | **0.9892** | **0.5143** | **0.9242** |
| `baseline_smooth125` | 0.4242 (#2) | 0.5812 (#5) | 0.4242 | 0.1464 | 0.8991 | 0.1667 | 0.8854 |
| `baseline_smooth50` | 0.3256 (#3) | 0.6207 (#2) | 0.3256 | 0.3225 | 0.8990 | 0.0816 | 0.8349 |
| `baseline_smooth25` | 0.2868 (#4) | 0.6200 (#3) | 0.3019 | 0.3857 | 0.8974 | 0.2034 | 0.7909 |
| `baseline_raw` | 0.0378 (#5) | 0.5972 (#4) | 0.0270 | 0.4197 | 0.8920 | 0.1410 | 0.7279 |

* **`spellF1` Thresholds (`pi_hsmm_v5`):** `@0.3 = 0.7586`, `@0.4 = 0.6897`, `@0.5 = 0.6897` (`P = 0.8333`, `R = 0.5882`, `10` matched, `7` missed, `2` spurious), `@0.6 = 0.6207`, `@0.7 = 0.4828`; bootstrap 5th–95th percentile `[0.4414, 0.8221]`.
* **Per-Class Frame Breakdown:** `H = 0.9284` (`P = 0.9492`, `R = 0.9085`), `A = 0.9024` (`P = 0.8632`, `R = 0.9453`), `D = 0.9539` (`P = 0.9876`, `R = 0.9224`); Cohen's $\kappa = 0.8795$, accuracy $= 92.39\%$, bootstrap `macroF1` 5th–95th percentile `[0.8413, 0.9547]`.
* **Transition & Possession Structure:** `transF1` `@12f = 0.4571` (`P = 0.5333`, `R = 0.4000`, `8` matched), `@25f = 0.5143` (`P = 0.6000`, `R = 0.4500`, `9` matched); latency `mean = -2.7`, `median = -3.0`, `p90|lat| = 13`, `0` ambiguous; team accuracy (both alive) `92.68%`; Home possession share error `0.0332` (`0.5395` vs `0.5727` GT); fragmentation `16 / 21 = 0.76` (`12` predicted vs `17` GT possessions, mean `20.4 s` vs `14.2 s`); `0.00%` `X` abstentions on graded GT.
* **Verified Improvement over `pi_hsmm_v4` (`0.5733 / 0.9074`):** **`+0.0750 spellF1`**, **`+0.0208 macroF1`**, **`+0.2207` @IoU 0.6**, **`+0.1254 transF1@25f`**, and **`+3.00%` GT-alive team accuracy**.

### B. Full-Match Unsupervised Audit & Format Validation (`199,319` Rows)

* **`validate_submission.py` Output:** `OK VISCA PENS.csv` (`199,319` rows, `0..199,318`): `H = 52,241 (26.2%)`, `A = 33,413 (16.8%)`, `D = 73,203 (36.7%)`, `X = 40,462 (20.3%)`.
* **158,857 Graded Frames Summary:**
  * Effective ball-in-play: **`57.1` minutes** (`H = 32.9%`, `A = 21.0%`, `D = 46.1%`, `X = 0`).
  * Alive possession share: **Home 61.0% vs. Away 39.0%**.
  * Full-match spells: **265** possessions, mean duration **12.9 s**.

---

## 6. Limitations & Future Work

* **Single-Touch Clearances & Goalkeeper Parries:** Brief ~1–1.5s box clearances or goalkeeper parries that ricochet quickly back to the opponent remain structurally ambiguous solely from x–y trajectories without ball-height or body-pose cues (contributing to `-5` net spell count vs. GT).
* **Extended Occlusion Dead-Entry Latency:** When play stops while players continue jogging above walking thresholds before Smart Tagging OOB anchors appear, dead-ball transitions can lag briefly.
* **Future Work:** Integrating 3D ball elevation / audio whistle onset (from video streams) and multi-match supervised calibration of HSMM transition penalties across independent fixtures could raise `spellF1` beyond the single-match tracking ceiling.

---

## 7. Reproduction Instructions

From the repository root with assets under `data/` (or `MTAH_DATA_DIR`):

```bash
# 1. Install pinned pipeline dependencies (includes numpy, pandas, scipy)
pip install -r requirements-pipeline.txt

# 2. Deterministically generate the final submission CSV (~69-99s CPU)
# Writes submissions/VISCA PENS.csv by default
python models/run_pipeline.py

# 3. Validate format compliance (199,319 rows)
python evaluation/validate_submission.py "submissions/VISCA PENS.csv"

# 4. Reproduce the benchmark scoreboard & per-class metrics
python evaluation/run_evaluation.py --bootstrap 1000
```
