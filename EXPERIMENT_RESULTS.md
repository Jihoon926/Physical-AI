# Pick-and-Place Experiment Results

Last updated: 2026-06-16

## Current Code State

- `student_policy.py`
  - Current approach threshold: `WAYPOINT_DISTANCE_THRESHOLDS[0] = 0.018`
  - No inference-time slowdown/action scaling
  - Keeps external tester input keys stable.
  - Computes internal derived features in `POLICY.forward`:
    - `hand_to_waypoint`, `hand_to_block`, `block_to_target`
    - `hand_to_block` in block-local yaw frame
    - block yaw, hand yaw, and yaw-error `sin/cos`
  - Uses stage-specific XY/Z waypoint transition checks.
- `student_trainer.py`
  - Includes optional waypoint-near action penalty:
    - `--action_penalty_weight`
    - `--action_penalty_radius`
  - Penalty weights are precomputed in dataset loading for speed.
  - Default run remains no action penalty.
- `02_submission_tester.ipynb`
  - Currently points to:
    - `CHECKPOINT_FILE = './student_internal_best.pth'`
    - `STATS_FILE = './student_internal_stats.npz'`
    - `MAX_ENVS = 50`
- Additional experiment files:
  - `student_policy_internal_rel.py` / `student_trainer_internal_rel.py`
    - Keeps external input keys at the base 53 dims.
    - Computes the original 9 relative features inside `POLICY.forward`.
  - `rnn_policy.py` / `rnn_trainer.py`
    - Uses a GRU stage selector with geometric waypoint positions.
    - Current rollout uses RNN stage prediction plus geometric fallback to avoid stage deadlock.
    - The stage-only run initializes and freezes the action head from `student_internal_rel_best.pth`.
  - `collect_dagger_failures.py`
    - Rolls out the current policy and saves failed episodes with IK expert relabels.
    - Latest failure dataset: `dagger_failures/rnn_transition_all250_20260615_232704` with 50 failed all-250 `open_env` episodes.
    - Latest random-train failure dataset: `dagger_failures/rnn_transition_random_train_dagger30_20260616_181740` with 30 failed randomized train-split episodes.
    - Latest unfiltered-env failure dataset: `dagger_failures/rnn_transition_unfiltered_env_dagger30_20260617_023356` with 30 failed unfiltered random-env episodes.
  - `rnn_transition_policy.py` / `rnn_transition_trainer.py`
    - Uses a GRU binary `advance/stay` transition head instead of predicting absolute waypoint stage.
    - Action head is initialized from `student_internal_rel_best.pth` and fine-tuned with train + DAgger data.
  - `render_policy_episode.py`
    - Renders a single policy rollout to mp4 and writes the rollout trace JSON.
  - `random_trajectory_generator.py`
    - Generates full expert trajectories in the same core npz format as the original training data.
    - Latest dataset: `random_trajectories_1000_20260615_232704`
    - Extra unseen test dataset: `random_trajectories_100_extra_20260616_214636`
    - Table XY size fixed at `0.25 x 0.35`; table height, block size/yaw/position, and target position randomized.
  - `random_env_generator.py`
    - Generates randomized environment-only `.npz` files without running or filtering by expert success.
    - Latest unfiltered test dataset: `random_envs_100_unfiltered_20260617_005245`
    - Latest fresh unfiltered eval dataset: `random_envs_100_unfiltered_eval_seed600000`

## Results

| ID | Checkpoint / Policy | Training Setting | Rollout Setting | Eval Set | Success | Rate | Mean Dist | Median Dist | Max Dist | Artifact | Notes |
|---:|---|---|---|---|---:|---:|---:|---:|---:|---|---|
| 1 | `student_best.pth` | 5 epoch, no action penalty | threshold `0.025` | first 20 `open_env` | 5/20 | 25.00% | 0.1283 | 0.1247 | 0.2663 | `02_submission_tester_executed.ipynb` | Early smoke test |
| 2 | `student_best.pth` | 5 epoch, no action penalty | threshold `0.025` | first 50 `open_env` | 15/50 | 30.00% | 0.1253 | 0.1306 | 0.2765 | `02_submission_tester_50_executed.ipynb` | Original 50-env baseline |
| 3 | `student_best.pth` | 5 epoch, no action penalty | threshold `0.018` + inference slowdown | first 50 `open_env` | 0/50 | 0.00% | 0.1633 | 0.1506 | 0.2728 | `02_submission_tester_50_after_waypoint_executed.ipynb` | Bad: action scaling breaks learned timing |
| 4 | `student_best.pth` | 5 epoch, no action penalty | threshold `0.012` | first 50 `open_env` | 14/50 | 28.00% | 0.1218 | 0.1256 | 0.2453 | `02_submission_tester_50_threshold_only_executed.ipynb` | Too strict; slower/stuck cases increase |
| 5 | `student_best.pth` | 5 epoch, no action penalty | threshold `0.018` | first 50 `open_env` | 21/50 | 42.00% | 0.1082 | 0.1014 | 0.2695 | `02_submission_tester_50_threshold_0018_executed.ipynb` | Best 50-env result so far |
| 6 | `student_penalty_best.pth` | 10 epoch, action penalty `0.05`, radius `0.08` | threshold `0.018` | first 50 `open_env` subset | 15/50 | 30.00% | 0.1249 | 0.1265 | 0.2687 | parsed from `02_submission_tester_penalty_full_executed.ipynb` | Penalty does not improve first-50 subset |
| 7 | `student_penalty_best.pth` | 10 epoch, action penalty `0.05`, radius `0.08` | threshold `0.018` | all 250 `open_env` | 81/250 | 32.40% | 0.1191 | 0.1235 | 0.3216 | `02_submission_tester_penalty_full_executed.ipynb` | Full-set measured result |
| 8 | `student_internal_best.pth` | 5 epoch, no action penalty, internal yaw/local-frame features | threshold `0.018` + XY/Z stage gates | first 50 `open_env` | 12/50 | 24.00% | 0.1324 | 0.1242 | 0.2942 | `02_submission_tester_internal_50_executed.ipynb`; `runs/rendered_videos/student_internal_50_success_127.mp4` | Worse than ID 5; internal features alone did not improve rollout |
| 9 | `student_internal_rel_best.pth` with `student_policy_internal_rel.py` | 10 epoch, no action penalty, original 9 relative features computed inside model | threshold `0.018` | all 250 `open_env` | 72/250 | 28.80% | 0.1283 | 0.1344 | 0.3324 | `runs/internal_rel_eval_full_20260615_033753.json` | Internalizing the original relative features preserves the tester input contract but underperforms the earlier penalty full-set result |
| 10 | `rnn_stage_best.pth` with `rnn_policy.py` | 10 epoch, GRU stage selector; action head initialized/frozen from `student_internal_rel_best.pth` | RNN stage prediction + geometric fallback | all 250 `open_env` | 113/250 | 45.20% | 0.1066 | 0.0910 | 0.3058 | `runs/rnn_stage_eval_full_20260615_064549_combined.json` | Best full-set result so far; pure randomly initialized RNN action head failed smoke rollout, so retained experiment isolates stage selection |
| 11 | `rnn_transition_dagger_best.pth` with `rnn_transition_policy.py` | 10 epoch, train 2000 + 28 DAgger failure trajectories; binary advance/stay GRU; action head initialized from `student_internal_rel_best.pth` and fine-tuned | advance/stay RNN + geometric fallback | first 50 `open_env` | 41/50 | 82.00% | 0.0752 | 0.0372 | 0.6615 | `runs/rnn_transition_dagger_eval50_20260615_185937.json`; `dagger_failures/rnn_stage_first50_20260615_170449` | Best first-50 result so far; failures: `106`, `111`, `114`, `115`, `119`, `132`, `134`, `139`, `141` |
| 12 | `rnn_transition_dagger_best.pth` with `rnn_transition_policy.py` | Same checkpoint as ID 11 | advance/stay RNN + geometric fallback | next 50 `open_env`, lexicographic sort index `50:100` | 40/50 | 80.00% | 0.0808 | 0.0394 | 0.8768 | `runs/rnn_transition_dagger_eval50_50_100_20260615_224045.json` | Held-out from DAgger collection slice; failures: `146`, `150`, `152`, `16`, `173`, `175`, `176`, `177`, `181`, `182` |
| 13 | `rnn_transition_dagger_best.pth` with `rnn_transition_policy.py` | Same checkpoint as ID 11; used as DAgger collection policy | advance/stay RNN + geometric fallback | all 250 `open_env` | 200/250 | 80.00% | 0.0734 | 0.0407 | 0.8768 | `dagger_failures/rnn_transition_all250_20260615_232704/summary.json` | Collected 50 failed episodes as relabeled DAgger trajectories; max failure `173.npz` |
| 14 | `rnn_transition_dagger_all250_best.pth` with `rnn_transition_policy.py` | 10 epoch, train 2000 + 50 all-250 DAgger failures; action head initialized from `student_internal_rel_best.pth` | advance/stay RNN + geometric fallback | 1000 randomized generated trajectories as envs | 176/1000 | 17.60% | 0.1279 | 0.1224 | 0.3019 | `runs/rnn_transition_dagger_all250_random1000_eval_20260616_024738_combined.json`; `random_trajectories_1000_20260615_232704` | Randomized distribution is much harder than `open_env`; table size fixed, table height/block/target randomized |
| 15 | `rnn_transition_random800_dagger50_best.pth` with `rnn_transition_policy.py` | 10 epoch, original train 2000 + random train 800 + all-250 DAgger failures 50; valid = original valid 250 + random valid 100 | advance/stay RNN + geometric fallback | random held-out 100, disjoint manifest split | 23/100 | 23.00% | 0.1204 | 0.1205 | 0.2509 | `runs/rnn_transition_random800_dagger50_heldout100_eval_20260616_170338_combined.json`; `splits/random_1000_20260616_split_summary.json` | Same held-out old model was 21/100, so random expert BC alone only slightly improved rollout |
| 16 | `rnn_transition_random800_dagger80_best.pth` with `rnn_transition_policy.py` | 10 epoch, original train 2000 + random train 800 + all-250 DAgger 50 + random-train DAgger 30; valid = original valid 250 + random valid 100; warm-start from ID 15 | advance/stay RNN + geometric fallback | same random held-out 100, disjoint manifest split | 81/100 | 81.00% | 0.0621 | 0.0416 | 0.2341 | `runs/rnn_transition_random800_dagger80_heldout100_eval_20260616_192410.json`; `dagger_failures/rnn_transition_random_train_dagger30_20260616_181740/summary.json` | DAgger was collected only from random train split, not held-out; closed-loop failure relabeling improved much more than adding random expert BC alone |
| 17 | `rnn_transition_random800_dagger80_best.pth` with `rnn_transition_policy.py` | Same checkpoint as ID 16 | advance/stay RNN + geometric fallback | newly generated random 100, seed 300000, never used for train/valid/DAgger | 81/100 | 81.00% | 0.0620 | 0.0416 | 0.2175 | `runs/rnn_transition_random800_dagger80_extra100_eval_20260616_215331.json`; `random_trajectories_100_extra_20260616_214636/summary.json` | Confirms ID 16 was not just lucky on one held-out split; new 100 had identical success rate |
| 18 | `rnn_transition_random800_dagger80_best.pth` with `rnn_transition_policy.py` | Same checkpoint as ID 16 | advance/stay RNN + geometric fallback | newly generated unfiltered env-only random 100, seed 500000, no expert-success filtering | 61/100 | 61.00% | 0.0854 | 0.0514 | 0.2359 | `runs/rnn_transition_random800_dagger80_unfiltered_env100_eval_20260617_005401.json`; `random_envs_100_unfiltered_20260617_005245/summary.json` | More realistic stress test than ID 17. The earlier 81/100 tests used only scenes where the scripted expert trajectory succeeded, which filtered out harder sampled scenes |
| 19 | `rnn_transition_random800_dagger80_best.pth` with `rnn_transition_policy.py` | Same checkpoint as ID 16 | advance/stay RNN + geometric fallback | fresh unfiltered env-only random 100, seed 600000 | 66/100 | 66.00% | 0.0828 | 0.0461 | 0.2813 | `runs/rnn_transition_random800_dagger80_eval_seed600000_unfiltered100_20260617_032657.json`; `random_envs_100_unfiltered_eval_seed600000/summary.json` | Baseline for testing the unfiltered DAgger fine-tune |
| 20 | `rnn_transition_random800_dagger110_ft_best.pth` with `rnn_transition_policy.py` | 5 epoch fine-tune from ID 16, original train 2000 + random train 800 + all-250 DAgger 50 + random-train DAgger 30 + unfiltered-env DAgger 30 | advance/stay RNN + geometric fallback | same fresh unfiltered env-only random 100, seed 600000 | 61/100 | 61.00% | 0.0823 | 0.0425 | 0.2788 | `runs/rnn_transition_random800_dagger110_ft_eval_seed600000_unfiltered100_20260617_042901.json`; `runs/rnn_transition_random800_dagger110_ft_seed600000_comparison.json`; videos `runs/failure_videos/dagger110_ft_seed600000_top*.mp4` | Not better than ID 19: fixed 7 old failures but regressed 12 old successes, net -5 successes. Keep ID 16 checkpoint as current best |

## Training Logs

Penalty model training:

```text
Epoch 1/10  - Train Loss: 0.0365266 - Valid Loss: 0.0023376
Epoch 2/10  - Train Loss: 0.0055421 - Valid Loss: 0.0015252
Epoch 3/10  - Train Loss: 0.0038076 - Valid Loss: 0.0012870
Epoch 4/10  - Train Loss: 0.0031631 - Valid Loss: 0.0011540
Epoch 5/10  - Train Loss: 0.0028348 - Valid Loss: 0.0010851
Epoch 6/10  - Train Loss: 0.0026503 - Valid Loss: 0.0010949
Epoch 7/10  - Train Loss: 0.0025363 - Valid Loss: 0.0010087
Epoch 8/10  - Train Loss: 0.0024564 - Valid Loss: 0.0010309
Epoch 9/10  - Train Loss: 0.0023948 - Valid Loss: 0.0009957
Epoch 10/10 - Train Loss: 0.0023421 - Valid Loss: 0.0009721
```

Observation: validation loss improved, but rollout success did not clearly improve. Offline BC loss is not a reliable proxy for task success here.

Internal-feature model training:

```text
Epoch 1/5 - Train Loss: 0.0356669 - Valid Loss: 0.0023672
Epoch 2/5 - Train Loss: 0.0056117 - Valid Loss: 0.0015537
Epoch 3/5 - Train Loss: 0.0038217 - Valid Loss: 0.0013067
Epoch 4/5 - Train Loss: 0.0031822 - Valid Loss: 0.0011774
Epoch 5/5 - Train Loss: 0.0028706 - Valid Loss: 0.0011179
```

Observation: validation loss improved normally, but first-50 rollout success dropped to 24%. The added features and stricter stage gates are not enough without better waypoint/grasp timing targets.

Internal-relative model training:

```text
Epoch 1/10  - Train Loss: 0.0365667 - Valid Loss: 0.0023394
Epoch 2/10  - Train Loss: 0.0055617 - Valid Loss: 0.0015323
Epoch 3/10  - Train Loss: 0.0038142 - Valid Loss: 0.0012784
Epoch 4/10  - Train Loss: 0.0031665 - Valid Loss: 0.0011460
Epoch 5/10  - Train Loss: 0.0028358 - Valid Loss: 0.0010892
Epoch 6/10  - Train Loss: 0.0026492 - Valid Loss: 0.0010802
Epoch 7/10  - Train Loss: 0.0025346 - Valid Loss: 0.0010158
Epoch 8/10  - Train Loss: 0.0024571 - Valid Loss: 0.0010072
Epoch 9/10  - Train Loss: 0.0023920 - Valid Loss: 0.0009886
Epoch 10/10 - Train Loss: 0.0023433 - Valid Loss: 0.0009626
```

Observation: validation loss is strong, but full-set rollout is 72/250. This confirms again that lower BC validation loss does not guarantee better closed-loop control.

RNN stage-only training:

```text
Epoch 1/10  - Train Loss: 0.0278140 - Valid Loss: 0.0130419 - Valid stage acc: 0.9351
Epoch 2/10  - Train Loss: 0.0115836 - Valid Loss: 0.0087972 - Valid stage acc: 0.9602
Epoch 3/10  - Train Loss: 0.0090650 - Valid Loss: 0.0073720 - Valid stage acc: 0.9666
Epoch 4/10  - Train Loss: 0.0078437 - Valid Loss: 0.0070852 - Valid stage acc: 0.9688
Epoch 5/10  - Train Loss: 0.0070830 - Valid Loss: 0.0058302 - Valid stage acc: 0.9749
Epoch 6/10  - Train Loss: 0.0065191 - Valid Loss: 0.0052617 - Valid stage acc: 0.9769
Epoch 7/10  - Train Loss: 0.0060888 - Valid Loss: 0.0049685 - Valid stage acc: 0.9786
Epoch 8/10  - Train Loss: 0.0057992 - Valid Loss: 0.0047892 - Valid stage acc: 0.9796
Epoch 9/10  - Train Loss: 0.0056163 - Valid Loss: 0.0046014 - Valid stage acc: 0.9803
Epoch 10/10 - Train Loss: 0.0055331 - Valid Loss: 0.0044639 - Valid stage acc: 0.9813
```

Observation: a pure randomly initialized RNN action head failed smoke rollout by staying at/near early stages. The retained RNN experiment therefore uses the successful MLP action head and trains only the RNN stage selector.

Transition RNN + DAgger training:

```text
DAgger collection: first 50 open_env with rnn_stage_best.pth
Saved failures: 28/50 into dagger_failures/rnn_stage_first50_20260615_170449
Collection policy success: 22/50

Epoch 1/10  - Train Loss: 0.0511803 - Valid Loss: 0.0121821 - Valid advance acc: 0.9605
Epoch 2/10  - Train Loss: 0.0255585 - Valid Loss: 0.0095448 - Valid advance acc: 0.9514
Epoch 3/10  - Train Loss: 0.0204391 - Valid Loss: 0.0089637 - Valid advance acc: 0.9621
Epoch 4/10  - Train Loss: 0.0179498 - Valid Loss: 0.0084404 - Valid advance acc: 0.9647
Epoch 5/10  - Train Loss: 0.0165996 - Valid Loss: 0.0083236 - Valid advance acc: 0.9676
Epoch 6/10  - Train Loss: 0.0157620 - Valid Loss: 0.0080773 - Valid advance acc: 0.9653
Epoch 7/10  - Train Loss: 0.0149119 - Valid Loss: 0.0079805 - Valid advance acc: 0.9677
Epoch 8/10  - Train Loss: 0.0143140 - Valid Loss: 0.0075911 - Valid advance acc: 0.9662
Epoch 9/10  - Train Loss: 0.0139436 - Valid Loss: 0.0087382 - Valid advance acc: 0.9672
Epoch 10/10 - Train Loss: 0.0133252 - Valid Loss: 0.0077690 - Valid advance acc: 0.9696
```

Observation: the binary transition model plus DAgger relabels substantially improved the first-50 rollout, from 22/50 for the collection policy to 41/50. The max failure (`141.npz`, 0.6615 m) is severe and should be inspected before running full 250.

Transition RNN + random expert split retraining:

```text
Split seed: 20260616
Random train/valid/heldout: 800/100/100, disjoint
Found 2850 training files (50 dagger, 800 extra train) and 350 validation files (100 extra valid).

Epoch 1/10  - Train Loss: 0.0235792 - Valid Loss: 0.0100556
Epoch 2/10  - Train Loss: 0.0139453 - Valid Loss: 0.0084116
Epoch 3/10  - Train Loss: 0.0126089 - Valid Loss: 0.0076650
Epoch 4/10  - Train Loss: 0.0119614 - Valid Loss: 0.0074996
Epoch 5/10  - Train Loss: 0.0115374 - Valid Loss: 0.0067299
Epoch 6/10  - Train Loss: 0.0112939 - Valid Loss: 0.0068022
Epoch 7/10  - Train Loss: 0.0112033 - Valid Loss: 0.0067399
Epoch 8/10  - Train Loss: 0.0109981 - Valid Loss: 0.0067310
Epoch 9/10  - Train Loss: 0.0109084 - Valid Loss: 0.0068267
Epoch 10/10 - Train Loss: 0.0108106 - Valid Loss: 0.0068008
```

Observation: best validation checkpoint came from epoch 5. The same held-out 100 split improved only from 21/100 to 23/100, so simple random expert BC is not enough. The next likely bottleneck is closed-loop drift/failure under the randomized distribution, which points toward collecting DAgger failures on the random held-out/valid distribution or improving the waypoint/grasp geometry.

Transition RNN + random-train DAgger retraining:

```text
DAgger collection: random train split with rnn_transition_random800_dagger50_best.pth
Saved failures: 30/43 into dagger_failures/rnn_transition_random_train_dagger30_20260616_181740
Collection policy success: 13/43

Found 2880 training files (50 dagger, 830 extra train) and 350 validation files (100 extra valid).
Warm-start: initialized 19 matching tensors from rnn_transition_random800_dagger50_best.pth

Epoch 1/10  - Train Loss: 0.0233025 - Valid Loss: 0.0072434
Epoch 2/10  - Train Loss: 0.0179497 - Valid Loss: 0.0070161
Epoch 3/10  - Train Loss: 0.0164044 - Valid Loss: 0.0068117
Epoch 4/10  - Train Loss: 0.0154575 - Valid Loss: 0.0072844
Epoch 5/10  - Train Loss: 0.0148276 - Valid Loss: 0.0076617
Epoch 6/10  - Train Loss: 0.0144286 - Valid Loss: 0.0067097
Epoch 7/10  - Train Loss: 0.0140373 - Valid Loss: 0.0066491
Epoch 8/10  - Train Loss: 0.0138535 - Valid Loss: 0.0069214
Epoch 9/10  - Train Loss: 0.0134921 - Valid Loss: 0.0067289
Epoch 10/10 - Train Loss: 0.0133224 - Valid Loss: 0.0070198
```

Observation: best validation checkpoint came from epoch 7. Validation worsens after that while train loss keeps falling, so `rnn_transition_random800_dagger80_best.pth` is the right checkpoint to evaluate. On the same disjoint random held-out 100 split, success improved from 23/100 to 81/100, showing that closed-loop DAgger relabels matter much more here than open-loop random expert BC alone.

## Failure Evidence

Worst full-set penalty failure:

- Env: `173.npz`
- Final distance: `0.3216 m`
- Final waypoint: `descend`
- Steps: `5000`
- Video: `runs/failure_videos/penalty_full_max_fail_173.mp4`

Transition RNN + DAgger first-50 max failure:

- Env: `141.npz`
- Final distance: `0.6615 m`
- Final waypoint: `retreat`
- Steps: `5000`
- Video: `runs/failure_videos/rnn_transition_dagger_fail_141.mp4`
- Trace: `runs/failure_videos/rnn_transition_dagger_fail_141.json`

Transition RNN + DAgger held-out next-50 max failure:

- Env: `173.npz`
- Final distance: `0.8768 m`
- Final waypoint: `move`
- Steps: `5000`
- Video: `runs/failure_videos/rnn_transition_dagger_holdout_fail_173.mp4`
- Trace: `runs/failure_videos/rnn_transition_dagger_holdout_fail_173.json`

## Current Analysis

### 1. Rotated Blocks Need Orientation Awareness

Some blocks start rotated. Current waypoint positions use only block center and target position. The policy receives `curr_bquat`, but the waypoint itself has no desired grasp yaw or block-face alignment target. The model must infer grasp orientation implicitly from demonstrations, which is fragile.

Likely fix:

- Add explicit orientation features:
  - block yaw `sin/cos`
  - hand yaw `sin/cos`
  - yaw error between hand/gripper and block face
  - `hand_to_block` expressed in the block local frame
- Optionally add `waypoint_yaw` or `grasp_yaw_error` as model inputs.
- Retrain after adding these features because `input_dim` changes.

### 2. Gripper Moves Before Robust Descent/Grasp

The current transition logic mostly uses Euclidean distance to waypoint plus a stage counter. It does not separately enforce:

- XY alignment over block
- Z descent completion
- actual gripper/finger closure
- contact or stable grasp before lift

Likely fix:

- Split transition checks by stage:
  - `approach -> descend`: require small XY error at high Z.
  - `descend -> grasp`: require both small XY error and small Z error for N consecutive steps.
  - `grasp -> lift`: require gripper command closed for N steps and, if possible, actual finger state/contact proxy.
- Avoid inference-time action slowdown unless the training target or generated data also reflects that slower control.

### 3. Action Penalty Was Not Enough

The waypoint-near action penalty improved validation loss but did not improve the first-50 rollout result. The issue is likely not just "too large action"; it is geometry and transition correctness.

## Next Experiments

| Priority | Experiment | Expected Benefit | Risk |
|---:|---|---|---|
| 1 | Add block/gripper yaw features and local-frame relative vectors | Better grasp alignment for rotated blocks | Requires retraining and stats/checkpoint separation |
| 2 | Stage-specific transition checks with XY/Z thresholds | Prevents diagonal descend and premature grasp/lift | May stall if thresholds are too strict |
| 3 | Add actual gripper/finger-state wait before `grasp -> lift` | Reduces lift before secure grasp | Need confirm qpos indices/contact proxy |
| 4 | Re-test `student_best.pth` on all 250 with threshold `0.018` | Direct comparison to penalty full result | Long runtime |
| 5 | Generate new demonstration data with orientation-aware waypoints | Best consistency between data and rollout | More expensive |

## Cleanup

Removed failed/interrupted or empty log files from aborted runs. Kept checkpoints, generated videos, and executed notebooks that contain reproducible result outputs.
