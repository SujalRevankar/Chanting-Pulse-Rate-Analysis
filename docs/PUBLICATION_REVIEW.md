# Publication review

Included: reviewed source edition, tests, generic configuration, face-only numerical exports, UBFC evaluation tables and fictional media.

Excluded: all NPZ caches (they contain ROI image patches), original participant/dataset videos and photographs, model checkpoints, local path metadata, prediction logs and unrelated palm runs. CSVs can still contain physiological information: only publish numerical outputs you are entitled to disclose.

Changes from uploaded source: face-only prediction; lazy hand-tracker initialization; previews and EVM disabled by default; rejected windows are NaN breaks in new timelines; selected-order metadata corrected; wrapper monkey patches restored after execution. These changes do not retrain the model or alter the supplied recorded estimates. Optional toolbox support remains, disabled by default. Its checkout is not bundled.

Verification is documented in `VERIFICATION.md`. Full acquisition, training and inference were not rerun: no source videos or checkpoint were supplied, and Torch/MediaPipe/OpenCV are absent in this preparation environment. A successful code syntax check is not a hardware/physiological validation.

The three facial-region masks are landmark convex hulls, not rectangles. The fictional illustration shows approximate locations only. Source code is the authority for exact landmark membership.
