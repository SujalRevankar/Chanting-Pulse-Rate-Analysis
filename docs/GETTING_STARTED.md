# Run locally

Use Python 3.11. Open a terminal inside the extracted repository folder.

```sh
python -m venv .venv
```

Windows Command Prompt: `.venv\Scripts\activate`

Linux/macOS: `source .venv/bin/activate`

```sh
python -m pip install -r requirements.txt
cd src
```

Obtain UBFC-rPPG Dataset 2 separately under its data terms. Each participant folder must contain `vid.avi` and `ground_truth.txt`. Keep all videos and compact ROI caches private.

Incremental training, one subject at a time:

```sh
python train_incremental.py --subject "D:/data/DATASET_2/subject1" --state "../incremental_training"
python train_incremental.py --subject "D:/data/DATASET_2/subject3" --state "../incremental_training"
```

Reserve different participants before inspecting results. Never change a participant's role or present the same person under another ID.

```sh
python train_incremental.py --subject "D:/data/DATASET_2/subject20" --role validation --state "../incremental_training"
python train_incremental.py --subject "D:/data/DATASET_2/subject22" --role test --state "../incremental_training"
```

IDs above are examples, not a claim about the supplied run's actual split. The same state directory must be reused. Later training does not automatically re-evaluate the final checkpoint on test subjects; read the original implementation before managing a final evaluation. The checkpoint and role registry were not supplied in this publication package.

Alternative full-dataset training with participant-level splitting:

```sh
python train_ubfc.py --ubfc "D:/data/DATASET_2" --output "../training_output" --no-toolbox
```

Predict a private, aligned, constant-frame-rate four-minute face recording:

```sh
python predict_sequence.py --order HA_YA --model "../incremental_training/cnn_lstm.pt" --input "D:/private/face_recording.mp4" --output "../predictions/face_run" --no-toolbox --no-evm
```

For Silence → Ya → Silence → Ha use `--order YA_HA`. These labels come from the command, not audio detection. Start recording at the first silence boundary. The wrapper allows approximately 240 ±2 seconds and analyzes the first 240 seconds. At least 15 source fps is required; acquisition near the model's 30 fps grid is preferable. Conversion of variable-frame-rate material must preserve real duration.

Outputs include `windows.csv`, `segments.csv`, `paired_changes.csv`, `video_summary.csv` and waveforms. Only trusted locally trained PyTorch checkpoints should be loaded: the current loader uses `weights_only=False`. A trained checkpoint is required; no pretrained weights are distributed here.

Run lightweight checks from repository root:

```sh
python -m compileall -q src
PYTHONPATH=src python -m unittest discover -s tests -p "test_core.py"
```

For Windows Command Prompt, replace the second line with `set PYTHONPATH=src` followed by `python -m unittest discover -s tests -p "test_core.py"`.
