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
pip install -r requirements.txt
```

Recommended (full features)

```powershell
pip install pyaudio                  # plays the beep (or use alternate audio lib)
pip install sentence-transformers    # best results for semantic detection
pip install "mediapipe<0.10.8"       # optional: hand gesture detection (required for current API)
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
- Video nudity detection: now prefers NudeNet. If unavailable, OpenCV Haar face detection is used as fallback for privacy masking.
- Semantic detection: uses `sentence-transformers` if installed; otherwise reverts to keyword matching.
- STT-first audio filtering: optional `faster-whisper` transcription with word timestamps enables beeping only on bad words while preserving safe words.
- Gesture detection: uses MediaPipe if available (current implementation expects `mediapipe<0.10.8` API) and disables gracefully when unavailable.

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


Selective word beep pipeline (new)
- Use `AudioTranscriber.transcribe_with_timestamps(audio_path)` to get text + word timing.
- Pass the payload to `AdvancedProfanityDetector.censor_transcript(...)` to replace only profane words.
- Apply `censor_audio_by_word_timestamps(...)` to beep only flagged word intervals in the waveform.


Production stream routing (camera + mic for other apps)
- `stream_filter.py` now captures raw webcam + microphone, applies filtering, and enforces bounded latency (drops stale packets instead of allowing cumulative delay growth).
- Virtual camera output: install `pyvirtualcam` and set `use_virtual_cam=True` in `StreamingFilterProduction`.
- Virtual microphone routing: set `audio_output_device_index` to a virtual cable output device (for example VB-CABLE output). In conferencing apps, choose the corresponding cable input as microphone.
- Latency controls:
  - `target_latency_ms`: desired steady-state delay.
  - `max_latency_ms`: hard upper bound; stale packets are dropped.
  - `sync_tolerance_ms`: A/V timestamp matching window.


Production hardening updates
- `stream_filter.py` now supports CLI configuration (`--target-latency-ms`, `--max-latency-ms`, `--sync-tolerance-ms`, `--use-virtual-cam`, `--audio-output-device-index`).
- Added `--list-audio-devices` helper for selecting virtual cable output devices safely.
- Added periodic JSONL health logging (`runtime_health.jsonl`) with FPS/latency/drop counters for long-run validation.
- Internal timing now uses monotonic clocks for stable sync calculations.

Suggested production acceptance targets
- p95 end-to-end latency <= 220ms
- p99 end-to-end latency <= 280ms
- sync offset p95 <= 40ms
- stale/drop rate <= 2% over 30+ minutes


Safe environment setup script
- Run `bash setup.bash` to create an isolated `.venv` and install dependencies without touching system Python.
- This reduces OS-level conflicts and keeps the runtime contained for production deployments.

Live stream profanity beep replacement
- `stream_filter.py` now supports live STT censoring so profane spoken words are replaced by beep in streamed audio output.
- New flags:
  - `--disable-live-stt-censor` (turn off live STT censoring)
  - `--stt-model-size` (default `base`)
  - `--stt-window-s` and `--stt-poll-interval-s` (latency/accuracy tuning)
  - `--stt-temp-wav` (temporary transcription audio file path)
