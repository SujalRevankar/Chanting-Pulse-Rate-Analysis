# Supplied results and their scope

One supplied face-recording output set is reproduced here as `results/face_example/`. No participant identity, input path, original video, ROI patch or checkpoint is published. These estimates cannot establish a population effect or causation. There is no simultaneous reference sensor in the chanting exports.

## Duration-weighted phase averages

| Face region | Silence 1 | Ha | Silence 2 | Ya | Ha minus Silence 1 | Ya minus Silence 2 |
|---|---:|---:|---:|---:|---:|---:|
| forehead | 77.54 | 89.21 | 86.67 | 87.84 | +11.67 | +1.17 |
| left_cheek | 77.68 | 84.67 | 86.67 | 88.09 | +6.98 | +1.42 |
| right_cheek | 77.54 | 88.62 | 86.28 | 88.28 | +11.09 | +2.00 |

All values above are estimated BPM. Ha's preceding silence and Ya's preceding silence are different baselines. Face-site disagreement and accepted coverage must be considered; the three regions are not independent participants.

## Accepted coverage

| Face region | Silence 1 | Ha | Silence 2 | Ya |
|---|---:|---:|---:|---:|
| forehead | 90.0% | 50.0% | 90.0% | 90.0% |
| left_cheek | 90.0% | 90.0% | 90.0% | 90.0% |
| right_cheek | 90.0% | 90.0% | 90.0% | 90.0% |

## Supplied UBFC evaluation

These are recalculated from the supplied `ubfc_test_windows.csv`, not a newly executed training run. Each ROI contains 6 attempted windows across 3 subject IDs, with 5 finite model/reference pairs. No saved checkpoint, role registry or split manifest was supplied, so separation from training cannot be independently verified here.

| ROI | Available paired windows | MAE (BPM) | RMSE (BPM) | Accepted / attempted |
|---|---:|---:|---:|---:|
| forehead | 5 | 0.527 | 0.735 | 66.7% |
| left_cheek | 5 | 0.439 | 0.520 | 83.3% |
| right_cheek | 5 | 0.352 | 0.481 | 83.3% |

Accepted-window metrics are also retained in `ubfc_metrics.csv`. Acceptance fraction uses all attempted windows as its denominator, whereas MAE/RMSE require finite paired estimates; their denominators differ. This small, quantized evaluation does not justify a general accuracy percentage, clinical claim or chanting-effect claim.

The animation uses these face phase means. They are summaries of usable windows, not instantaneous readings. It compresses each 60-second phase into 6 seconds, with no invented transition values. The person and mouth motion are illustrative; no actual chanting audio, participant video or facial tracking is shown.
