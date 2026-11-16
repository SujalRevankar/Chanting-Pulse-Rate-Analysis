# Upload to GitHub

1. Create a repository, for example `Chanting-Face-rPPG-Analysis`. Choose public or private as you prefer. Do not add an automatic license that conflicts with this package's custom license.
2. Extract the ZIP on your computer. Open its `Chanting_Face_rPPG` folder.
3. Open your GitHub repository's main Code page. Use the small `+` beside “Go to file” (or “Add file”), then “Upload files”.
4. Drag the CONTENTS of `Chanting_Face_rPPG` onto the upload area: `src`, `tests`, `docs`, `results`, `assets`, `tools`, README.md, requirements.txt, LICENSE.txt, UPLOAD_STEPS.md and the hidden `.gitignore` file. Drag folders from Windows File Explorer, rather than opening them through the file picker.
5. Confirm GitHub shows paths such as `src/model.py` and `assets/chanting-demo.gif`. README.md must be at the repository root, not inside an extra wrapper folder.
6. Commit with message `Add face-only chanting rPPG research prototype`.
7. Check that the README displays its GIF, face illustration and chart. The MP4 is available through its download link; a Markdown MP4 link need not play inline.
8. Keep your future private recordings, NPZ caches, checkpoints and raw output folders outside the upload selection. Do not drag the original upload collection into GitHub.

Suggested repository description: Python research prototype exploring facial rPPG pulse-rate estimates across silence, Ha and Ya phases using a custom CNN–LSTM and POS comparison.
