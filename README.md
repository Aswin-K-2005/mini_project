**NeonGuard — Futuristic Profanity & Privacy Toolkit**

Fast local prototype for profanity detection, audible alerts, and video censoring.

NeonGuard lets you experiment with semantic profanity detection (embeddings), audible beeps for detected profanity, and multiple video censoring modes (faces and offensive gestures). It's built for local testing and prototyping — easy to extend and swap components.

Core pieces
- `audio_processor.py` — semantic profanity detector + beep generator and test harness
- `videoprocessor.py` — webcam demo: face detection (OpenCV), optional MediaPipe gestures, motion tracking, and blur/censor modes

Why this repo
- Quick to run locally — minimal external dependencies for core features
- Modular: replace the face/NSFW detector (NudeNet or other) with a small change
- Dual detection modes: semantic embeddings (preferred) and keyword fallback

Install (Windows — minimal)

```powershell
python -m pip install --upgrade pip
pip install numpy opencv-python
```

Recommended (full features)

```powershell
pip install pyaudio                  # plays the beep (or use alternate audio lib)
pip install sentence-transformers    # best results for semantic detection
pip install mediapipe                # optional: hand gesture detection
```

Quick start

- Run webcam demo:

```powershell
python videoprocessor.py
```

  - Choose a blur mode when prompted: `pixelation`, `heavy`, `extreme`, `black`
  - Press `q` to quit

- Run audio/text profanity demo:

```powershell
python audio_processor.py
```

  - Enter similarity threshold (default `0.7`). Lower = stricter detection.
  - The demo runs several phrases and plays a beep for detected profanity.

Key design notes
- Face detection: uses OpenCV Haar cascades as a fast local stand-in. Swap in NudeNet or other detectors for production use.
- Semantic detection: uses `sentence-transformers` if installed; otherwise reverts to keyword matching.
- Gesture detection: uses MediaPipe if available and disables gracefully when it isn't installed.

Configuration tips
- Tweak detection sensitivity via `SemanticProfanityDetector`'s similarity threshold.
- Add custom profanities with `SemanticProfanityDetector.add_custom_profanity([...])`.
- Change `blur_mode` in `VideoProcessorTestComplete` to control censor style.

Troubleshooting
- If `pyaudio` is hard to install on Windows, try `pipwin install pyaudio`.
- If the webcam fails to open, check drivers and that no other app is using the camera.

Ethics & privacy
- This repo is for research and prototyping only. Review legal and privacy requirements before any deployment.
- Semantic models can misclassify — validate with your data.

Next steps (pick one)
- I can add a `requirements.txt` for reproducible installs.
- I can add a short demo script that auto-switches blur modes.
- I can generate screenshots / a GIF showing blur levels.

Tell me which option you want and I'll implement it.
