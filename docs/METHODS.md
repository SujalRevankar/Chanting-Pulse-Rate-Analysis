# Methods and facial regions

The custom CNN–LSTM estimates a pulse waveform; it does not classify speech or recognize Ha/Ya. Each RGB patch is 32×32. Three 2D convolution blocks produce 32 features per frame; a one-layer LSTM has 64 hidden units; a linear layer produces one waveform sample per frame. Training minimizes one minus waveform correlation with aligned contact PPG.

MediaPipe Face Mesh tracks one face. `src/roi.py` selects these landmark sets and fills their convex hulls. Left/right are code labels and may be reversed anatomically in mirrored footage.

| ROI | Landmark indices |
|---|---|
| forehead | 10,109,67,103,54,68,104,69,108,151,337,299,333,298,284,332,297,338 |
| left_cheek | 50,101,205,206,207,187,123,116,117,118,119,120 |
| right_cheek | 280,330,425,426,427,411,352,345,346,347,348,349 |

Out-of-frame or tiny regions are rejected. Pixels with channel minimum ≤8 or maximum ≥245 are excluded. At least 60% of the polygon must remain. Excluded crop pixels are filled with the valid-pixel mean before resizing. The ROI illustration is schematic: no landmark detector has been run on its fictional character and its overlay is not a measured tracking result.

At the default 30 Hz processing grid, waveform clips span 10 seconds. BPM is the dominant periodogram frequency in 0.7–3 Hz, multiplied by 60. The built-in POS comparator uses mean ROI RGB. POS agreement measures consistency, not physiological accuracy. The same spectral estimator is used for model and reference; discrete frequency bins can produce identical rounded BPM values.

Each 60-second phase excludes 3 seconds at each boundary. Analysis windows are nominally 30 seconds, with a final shorter window of at least 10 seconds. QC checks ROI coverage (≥95%), gaps (≤0.2 seconds), motion, brightness changes, spectral concentration (≥0.35), and model/POS difference (≤10 BPM). Values are heuristics, not clinical thresholds. Phase means are weighted by accepted duration; coverage is accepted duration / full phase duration. Multiple facial regions observe one person, not independent participants.

The repository edition disables palm extraction, EVM and raw ROI-video previews by default. Metadata/logs generated locally can contain input paths; review any future exports before publication. The original pipeline is offline video processing, not a live webcam application.
