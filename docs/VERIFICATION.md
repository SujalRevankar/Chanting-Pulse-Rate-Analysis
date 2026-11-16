# Verification of this repository edition

- Python syntax checked for all included modules and tests.
- Six signal/protocol/gap tests passed (`test_core.py`).
- Five reference parsing/alignment tests passed (`test_reference.py`).
- One sequence contract test passed for both HA_YA and YA_HA: correct labels, metadata and monkey-patch restoration.
- Recalculated every supplied face phase mean and accepted duration from the selected windows; paired differences match the supplied CSV.
- Recalculated all available/accepted UBFC MAE and RMSE rows from the supplied per-window table; values match.
- All local Markdown links resolve.
- MP4: H.264, 1200×720, 288 frames, 12 fps, 24 seconds, silent. Important phase states and ROI image visually inspected.
- Publication file selection excludes participant/dataset images, original videos, NPZ image caches, model weights, absolute personal path metadata and prediction logs. Source/config text checked for common credential and private-path patterns; no live credentials found. This is not an audit of prior Git history.

Not rerun: Torch training/resume, OpenCV decoding, MediaPipe tracking, full inference or the dependency-heavy supplied tests (`test_streaming`, `test_incremental`, `test_automations`, `smoke_pipeline`). Those dependencies, original recordings and model checkpoint are not present in this preparation environment. Existing tests are supplied for local use, not represented as passing here. The media is an illustration; it cannot validate the rPPG model.
