"""
stream_filter.py - production-oriented low-latency A/V filtering pipeline.
"""

import argparse
import io
import json
import os
import threading
import time
import wave
from collections import deque
from dataclasses import dataclass, asdict
from queue import Empty, Queue

import cv2
import numpy as np
import pyaudio

from audio_processor import AdvancedProfanityDetector, AudioTranscriber
from videoprocessor import VideoProcessorTestComplete

try:
    from vosk import Model as VoskModel, KaldiRecognizer as VoskRecognizer
    VOSK_AVAILABLE = True
except ImportError:
    VOSK_AVAILABLE = False

try:
    import pyvirtualcam
    VIRTUAL_CAM_AVAILABLE = True
except ImportError:
    VIRTUAL_CAM_AVAILABLE = False


@dataclass
class FilterConfig:
    blur_mode: str = 'pixelation'
    recall_mode: str = 'high'
    video_source: int = 0
    # Defaults tuned for STT word-level censoring reliability.
    target_latency_ms: float = 950.0
    max_latency_ms: float = 1400.0
    sync_tolerance_ms: float = 40.0
    use_virtual_cam: bool = False
    virtual_cam_device: str = 'NeonGuard Virtual Camera'
    audio_output_device_index: int | None = None
    health_log_path: str = 'runtime_health.jsonl'
    health_log_interval_s: float = 5.0
    enable_live_stt_censor: bool = True
    # 'base' is much more reliable for short profane words than 'tiny'.
    stt_model_size: str = 'base'
    # Smaller window + faster polling helps reduce "wait for sentence end".
    stt_window_s: float = 1.2
    stt_poll_interval_s: float = 0.5
    stt_temp_wav: str = 'live_stt_window.wav'
    stt_force_keyword_fallback: bool = True
    stt_vad_filter: bool = False
    stt_language: str | None = "en"
    # Extra hold so STT results arrive before playback.
    stt_alignment_delay_ms: float = 350.0
    video_detection_interval: int = 4
    enable_fast_audio_gate: bool = True
    # Lightweight safety net (still uses STT, but no word timestamps): forces an immediate beep.
    fast_gate_window_s: float = 0.8
    fast_gate_poll_interval_s: float = 0.2
    fast_gate_beep_ms: float = 420.0
    fast_gate_cooldown_ms: float = 450.0
    fast_gate_temp_wav: str = 'fast_gate_window.wav'
    enable_kws: bool = False
    kws_model_path: str = "vosk-model-small-en-us-0.15"
    kws_cooldown_ms: float = 500.0
    kws_beep_ms: float = 320.0
    audio_timeline_seconds: float = 6.0


@dataclass
class AVPacket:
    frame: np.ndarray
    audio: np.ndarray
    capture_ts: float
    metadata: dict


class AsyncDetectionWorker:
    def __init__(self, detector, queue_size=4):
        self.detector = detector
        self.input_queue = Queue(maxsize=queue_size)
        self.output_queue = Queue(maxsize=queue_size)
        self.running = False
        self.thread = None
        self.latencies = deque(maxlen=60)
        self.stats = {'queue_drops': 0, 'avg_latency_ms': 0.0, 'errors': 0}

    def start(self):
        self.running = True
        self.thread = threading.Thread(target=self._loop, daemon=True)
        self.thread.start()

    def stop(self):
        self.running = False
        if self.thread:
            self.thread.join(timeout=2)

    def submit_frame(self, frame, capture_ts):
        try:
            self.input_queue.put_nowait((frame.copy(), capture_ts))
        except Exception:
            self.stats['queue_drops'] += 1

    def latest_result(self):
        latest = None
        while True:
            try:
                latest = self.output_queue.get_nowait()
            except Empty:
                return latest

    def _loop(self):
        while self.running:
            try:
                frame, capture_ts = self.input_queue.get(timeout=0.1)
            except Empty:
                continue

            try:
                start = time.monotonic()
                processed, metadata = self.detector.process_frame(frame, capture_ts)
                latency_ms = (time.monotonic() - start) * 1000.0
                self.latencies.append(latency_ms)
                self.stats['avg_latency_ms'] = float(np.mean(self.latencies))
                self.output_queue.put_nowait({
                    'frame': processed,
                    'metadata': metadata,
                    'capture_ts': capture_ts,
                    'latency_ms': latency_ms,
                })
            except Exception:
                self.stats['errors'] += 1


class AudioDelayBuffer:
    def __init__(self, min_delay_ms=60.0, max_delay_ms=180.0):
        self.buffer = deque()
        self.min_delay_ms = min_delay_ms
        self.max_delay_ms = max_delay_ms
        self.target_delay_ms = min_delay_ms
        self.video_latency_samples = deque(maxlen=90)

    def update_video_latency(self, latency_ms):
        self.video_latency_samples.append(latency_ms)
        if len(self.video_latency_samples) >= 10:
            p95 = float(np.percentile(self.video_latency_samples, 95))
            self.target_delay_ms = float(np.clip(p95, self.min_delay_ms, self.max_delay_ms))

    def add(self, audio_chunk, capture_ts):
        self.buffer.append((audio_chunk, capture_ts))

    def pop_ready(self):
        if not self.buffer:
            return None
        chunk, ts = self.buffer[0]
        if (time.monotonic() - ts) * 1000.0 >= self.target_delay_ms:
            return self.buffer.popleft()
        return None

    def trim_older_than(self, max_age_ms):
        now = time.monotonic()
        while self.buffer and (now - self.buffer[0][1]) * 1000.0 > max_age_ms:
            self.buffer.popleft()


class AudioTimelineBuffer:
    """
    Stores audio chunks on the capture timestamp timeline and allows in-place overlays
    (beeps) before playback. This enables clean "replace bad word then play" behavior.
    """

    def __init__(self, sample_rate: int, max_seconds: float):
        self.sample_rate = int(sample_rate)
        self.max_seconds = float(max_seconds)
        self._chunks = deque()  # each item: {'ts': float, 'audio': np.ndarray(float32)}
        self._lock = threading.Lock()

    def add_chunk(self, audio: np.ndarray, ts: float):
        audio = audio.astype(np.float32, copy=False)
        if audio.ndim != 1:
            audio = audio.reshape(-1)
        with self._lock:
            self._chunks.append({'ts': float(ts), 'audio': audio.copy()})
            self._trim_locked()

    def overlay_beep(self, ev_start: float, ev_end: float, beep_wave: np.ndarray):
        if ev_end <= ev_start:
            return
        with self._lock:
            if not self._chunks:
                return
            for ch in self._chunks:
                ts0 = ch['ts']
                a = ch['audio']
                ts1 = ts0 + (len(a) / float(self.sample_rate))
                overlap_start = max(ts0, float(ev_start))
                overlap_end = min(ts1, float(ev_end))
                if overlap_end <= overlap_start:
                    continue
                i0 = int((overlap_start - ts0) * self.sample_rate)
                i1 = int((overlap_end - ts0) * self.sample_rate)
                i0 = max(0, min(i0, len(a)))
                i1 = max(0, min(i1, len(a)))
                if i1 <= i0:
                    continue
                seg_len = i1 - i0
                a[i0:i1] = beep_wave[:seg_len] if seg_len <= len(beep_wave) else np.resize(beep_wave, seg_len)

    def get_chunk(self, ts: float, n_samples: int) -> np.ndarray:
        ts = float(ts)
        n_samples = int(n_samples)
        if n_samples <= 0:
            return np.zeros(0, dtype=np.float32)

        out = np.zeros(n_samples, dtype=np.float32)
        with self._lock:
            if not self._chunks:
                return out
            req_end = ts + (n_samples / float(self.sample_rate))
            for ch in self._chunks:
                ts0 = ch['ts']
                a = ch['audio']
                ts1 = ts0 + (len(a) / float(self.sample_rate))
                overlap_start = max(ts0, ts)
                overlap_end = min(ts1, req_end)
                if overlap_end <= overlap_start:
                    continue
                src_i0 = int((overlap_start - ts0) * self.sample_rate)
                src_i1 = int((overlap_end - ts0) * self.sample_rate)
                dst_i0 = int((overlap_start - ts) * self.sample_rate)
                dst_i1 = dst_i0 + (src_i1 - src_i0)
                src_i0 = max(0, min(src_i0, len(a)))
                src_i1 = max(0, min(src_i1, len(a)))
                dst_i0 = max(0, min(dst_i0, len(out)))
                dst_i1 = max(0, min(dst_i1, len(out)))
                if dst_i1 > dst_i0 and src_i1 > src_i0:
                    out[dst_i0:dst_i1] = a[src_i0:src_i1]
            return out

    def _trim_locked(self):
        if not self._chunks:
            return
        newest = self._chunks[-1]
        newest_end = newest['ts'] + (len(newest['audio']) / float(self.sample_rate))
        keep_after = newest_end - self.max_seconds
        while self._chunks:
            ch = self._chunks[0]
            ch_end = ch['ts'] + (len(ch['audio']) / float(self.sample_rate))
            if ch_end >= keep_after:
                break
            self._chunks.popleft()


class StreamingFilterProduction:
    def __init__(self, config: FilterConfig):
        self.config = config
        self.use_virtual_cam = config.use_virtual_cam and VIRTUAL_CAM_AVAILABLE

        self.video_processor = VideoProcessorTestComplete(blur_mode=config.blur_mode)
        self.detection_worker = AsyncDetectionWorker(self.video_processor)
        self.audio_detector = AdvancedProfanityDetector(
            recall_mode=config.recall_mode,
            allow_mild_profanity=False,
        )
        self.audio_transcriber = AudioTranscriber(model_size=config.stt_model_size)
        self.last_fast_gate_at = 0.0
        self.fast_gate_queue = Queue(maxsize=1)
        self.fast_gate_beep_until = 0.0
        self.kws_audio_queue = Queue(maxsize=10)
        self.kws_model = None
        self.kws_recognizer = None
        self.kws_last_fire_ms = 0.0

        self.video_capture = cv2.VideoCapture(config.video_source, cv2.CAP_DSHOW)
        self.video_capture.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
        self.video_capture.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
        self.video_capture.set(cv2.CAP_PROP_FPS, 30)

        # Use int16 for output stability/compatibility; keep internal as float32 (-1..1).
        self.audio_in_format = pyaudio.paInt16
        self.audio_out_format = pyaudio.paInt16
        self.audio_channels = 1
        self.audio_rate = 44100
        # Align audio chunks with video cadence (~30fps) to reduce A/V pairing jitter.
        self.target_fps = 30.0
        self.audio_chunk = int(round(self.audio_rate / self.target_fps))  # ~1470 samples
        self.pyaudio = pyaudio.PyAudio()

        # Keep total end-to-end latency near target by budgeting audio delay below target.
        # Video frames are held to enforce the final constant latency.
        audio_budget_ms = max(0.0, float(config.target_latency_ms) - float(config.stt_alignment_delay_ms))
        audio_min_ms = max(10.0, audio_budget_ms * 0.75)
        audio_max_ms = max(audio_min_ms, audio_budget_ms * 1.25)
        self.audio_delay = AudioDelayBuffer(
            min_delay_ms=audio_min_ms,
            max_delay_ms=min(config.max_latency_ms, audio_max_ms),
        )

        self.video_queue = Queue(maxsize=24)
        self.audio_queue = Queue(maxsize=64)
        self.output_queue = Queue(maxsize=24)

        self.running = False
        self.stop_event = threading.Event()
        self.threads = []
        self.stats_lock = threading.Lock()
        self.beep_lock = threading.Lock()

        self.beep_sound = self._generate_beep()
        self.beep_until = 0.0
        self.beep_events = deque(maxlen=400)
        self.audio_history = deque(maxlen=max(8, int(self.audio_rate * config.stt_window_s / self.audio_chunk) + 4))
        self.last_stt_at = 0.0
        self.stt_queue = Queue(maxsize=2)
        self.audio_sync_cache = deque(maxlen=256)
        self.latest_audio_packet = None
        self.last_audio_write_ts = time.monotonic()
        self.pending_audio = deque()

        self.stats = {
            'capture_fps': 0.0,
            'output_fps': 0.0,
            'video_detection_ms': 0.0,
            'audio_delay_ms': 0.0,
            'sync_offset_ms': 0.0,
            'end_to_end_ms': 0.0,
            'frames_output': 0,
            'drops_video_queue': 0,
            'drops_audio_queue': 0,
            'drops_stale': 0,
            'sync_drops': 0,
            'profanity_events': 0,
            'live_stt_runs': 0,
            'live_stt_profanity_hits': 0,
            'fast_gate_runs': 0,
            'fast_gate_hits': 0,
            'kws_runs': 0,
            'kws_hits': 0,
            'runtime_errors': 0,
            'started_at': time.time(),
        }

        if self.config.enable_kws and VOSK_AVAILABLE:
            try:
                if os.path.isdir(self.config.kws_model_path):
                    self.kws_model = VoskModel(self.config.kws_model_path)
                    self.kws_recognizer = VoskRecognizer(self.kws_model, self.audio_rate)
                    # We only need text; keep words off for speed.
                    try:
                        self.kws_recognizer.SetWords(False)
                    except Exception:
                        pass
                    print("[OK] Vosk keyword spotting enabled")
                else:
                    print(f"[WARN] Vosk model folder not found: {self.config.kws_model_path}")
            except Exception as e:
                print(f"[WARN] Failed to init Vosk KWS: {e}")
                self.kws_model = None
                self.kws_recognizer = None
        elif self.config.enable_kws and (not VOSK_AVAILABLE):
            print("[WARN] Vosk not installed; KWS disabled (pip install vosk)")

    @staticmethod
    def list_audio_devices():
        pa = pyaudio.PyAudio()
        devices = []
        try:
            for idx in range(pa.get_device_count()):
                info = pa.get_device_info_by_index(idx)
                devices.append({
                    'index': idx,
                    'name': info.get('name'),
                    'maxInputChannels': int(info.get('maxInputChannels', 0)),
                    'maxOutputChannels': int(info.get('maxOutputChannels', 0)),
                })
        finally:
            pa.terminate()
        return devices

    def _safe_stat_inc(self, key, value=1):
        with self.stats_lock:
            self.stats[key] += value

    def _safe_stat_set(self, key, value):
        with self.stats_lock:
            self.stats[key] = value

    def _generate_beep(self, duration_ms=250, frequency=1000):
        t = np.linspace(0, duration_ms / 1000.0, int(self.audio_rate * duration_ms / 1000.0), endpoint=False)
        return (0.55 * np.sin(2 * np.pi * frequency * t)).astype(np.float32)

    def _bounded_put(self, queue_obj, item, drop_stat_key):
        try:
            queue_obj.put_nowait(item)
            return
        except Exception:
            pass
        try:
            queue_obj.get_nowait()
        except Empty:
            pass
        self._safe_stat_inc(drop_stat_key)
        try:
            queue_obj.put_nowait(item)
        except Exception:
            self._safe_stat_inc(drop_stat_key)

    def _audio_window_to_wav_bytes(self, samples):
        pcm = np.clip(samples, -1.0, 1.0)
        pcm_i16 = (pcm * 32767).astype(np.int16)
        buf = io.BytesIO()
        with wave.open(buf, 'wb') as wf:
            wf.setnchannels(1)
            wf.setsampwidth(2)
            wf.setframerate(self.audio_rate)
            wf.writeframes(pcm_i16.tobytes())
        buf.seek(0)
        return buf
    def _recent_audio_window(self, seconds):
        samples_needed = max(1, int(self.audio_rate * float(seconds)))
        if len(self.audio_history) < 2:
            return None, None, None

        chunks = []
        total = 0
        start_ts = None
        last_chunk = None
        last_ts = None

        for chunk, chunk_ts in reversed(self.audio_history):
            if last_chunk is None:
                last_chunk = chunk
                last_ts = chunk_ts
            chunks.append(chunk)
            total += len(chunk)
            start_ts = chunk_ts
            if total >= samples_needed:
                break

        minimum_samples = max(self.audio_chunk * 2, samples_needed // 2)
        if total < minimum_samples:
            return None, None, None

        window = np.concatenate(list(reversed(chunks)))
        if len(window) > samples_needed:
            trim = len(window) - samples_needed
            window = window[-samples_needed:]
            start_ts = float(start_ts) + (trim / float(self.audio_rate))

        end_ts = float(last_ts) + (len(last_chunk) / float(self.audio_rate))
        return window.copy(), float(start_ts), float(end_ts)

    def _schedule_live_stt_beep_events(self, now_mono):
        cfg = self.config
        if not cfg.enable_live_stt_censor or not self.audio_transcriber.available:
            return
        if now_mono - self.last_stt_at < cfg.stt_poll_interval_s:
            return
        window, start_ts, _ = self._recent_audio_window(cfg.stt_window_s)
        if window is None or window.size == 0:
            return

        self.last_stt_at = now_mono
        payload = {'window': window, 'start_ts': start_ts}
        try:
            self.stt_queue.put_nowait(payload)
        except Exception:
            # keep real-time path healthy; drop old STT work when overloaded
            try:
                self.stt_queue.get_nowait()
                self.stt_queue.put_nowait(payload)
            except Exception:
                pass

    def _schedule_fast_gate(self, now_mono):
        cfg = self.config
        if not cfg.enable_fast_audio_gate or not self.audio_transcriber.available:
            return
        if now_mono - self.last_fast_gate_at < cfg.fast_gate_poll_interval_s:
            return
        window, window_start_ts, window_end_ts = self._recent_audio_window(cfg.fast_gate_window_s)
        if window is None or window.size == 0:
            return

        payload = {
            'window': window,
            'window_start_ts': window_start_ts,
            'window_end_ts': window_end_ts,
        }
        self.last_fast_gate_at = now_mono
        try:
            self.fast_gate_queue.put_nowait(payload)
        except Exception:
            # drop older pending work; keep latest
            try:
                self.fast_gate_queue.get_nowait()
                self.fast_gate_queue.put_nowait(payload)
            except Exception:
                pass

    def _fast_gate_thread(self):
        cfg = self.config
        while not self.stop_event.is_set():
            try:
                payload = self.fast_gate_queue.get(timeout=0.2)
            except Empty:
                continue

            window = payload['window']
            window_start_ts = float(payload.get('window_start_ts', time.monotonic()))
            window_end_ts = float(payload.get('window_end_ts', window_start_ts))
            try:
                window_wav = self._audio_window_to_wav_bytes(window)
                transcript = self.audio_transcriber.transcribe_text(window_wav)
                text = (transcript.get('text', '') or '').strip()
                self._safe_stat_inc('fast_gate_runs')

                is_profane, _, _ = self.audio_detector.detect(text)
                if (not is_profane) and cfg.stt_force_keyword_fallback:
                    is_profane = self._segment_is_profane_fast(text)

                if is_profane:
                    # Cooldown to avoid constant beeping on long speech.
                    now_ms = time.monotonic() * 1000.0
                    if now_ms < self.fast_gate_beep_until + cfg.fast_gate_cooldown_ms:
                        continue
                    dur_s = float(cfg.fast_gate_beep_ms) / 1000.0
                    # Low-latency path: force-beep upcoming playback immediately.
                    # This avoids timing/overlap issues with scheduled events at low latency.
                    self.beep_until = max(self.beep_until, time.monotonic() + dur_s)
                    self.fast_gate_beep_until = now_ms
                    self._safe_stat_inc('fast_gate_hits')
            except Exception:
                self._safe_stat_inc('runtime_errors')
            finally:
                pass

    def _kws_thread(self):
        """Streaming keyword spotting using Vosk partial results."""
        if not (self.config.enable_kws and self.kws_recognizer is not None):
            return

        while not self.stop_event.is_set():
            try:
                chunk_bytes, chunk_ts = self.kws_audio_queue.get(timeout=0.2)
            except Empty:
                continue

            try:
                # Feed audio to recognizer
                self._safe_stat_inc('kws_runs')
                _accepted = self.kws_recognizer.AcceptWaveform(chunk_bytes)

                # Prefer partial for low latency, but also check final.
                partial = ''
                try:
                    payload = json.loads(self.kws_recognizer.PartialResult() or '{}')
                    partial = (payload.get('partial') or '').strip()
                except Exception:
                    partial = ''

                final_text = ''
                if _accepted:
                    try:
                        payload = json.loads(self.kws_recognizer.Result() or '{}')
                        final_text = (payload.get('text') or '').strip()
                    except Exception:
                        final_text = ''

                text = (partial or final_text).strip()
                if not text:
                    continue

                # Very fast check: severe words + phonetic fallback
                is_profane, _, _ = self.audio_detector.detect(text)
                if (not is_profane) and self.config.stt_force_keyword_fallback:
                    is_profane = self._segment_is_profane_fast(text)
                if not is_profane:
                    continue

                now_ms = time.monotonic() * 1000.0
                if now_ms - self.kws_last_fire_ms < float(self.config.kws_cooldown_ms):
                    continue

                dur_s = float(self.config.kws_beep_ms) / 1000.0
                with self.beep_lock:
                    self.beep_events.append((chunk_ts, chunk_ts + dur_s))
                self.kws_last_fire_ms = now_ms
                self._safe_stat_inc('kws_hits')
            except Exception:
                self._safe_stat_inc('runtime_errors')


    def _segment_is_profane_fast(self, text):
        """Aggressive fallback for live stream: keyword + phonetic checks."""
        if not text:
            return False

        normalized = text.lower()

        # Fast exact severe-word trigger
        severe_words = getattr(self.audio_detector, 'severe_words', [])
        for w in severe_words:
            if f" {w} " in f" {normalized} ":
                return True

        # Phonetic fallback from detector patterns
        try:
            if self.audio_detector.detect_phonetic(normalized):
                return True
        except Exception:
            pass

        return False

    def _stt_thread(self):
        cfg = self.config
        while not self.stop_event.is_set():
            try:
                payload = self.stt_queue.get(timeout=0.2)
            except Empty:
                continue

            window = payload['window']
            start_ts = payload['start_ts']
            try:
                window_wav = self._audio_window_to_wav_bytes(window)
                transcript = self.audio_transcriber.transcribe_with_timestamps(
                    window_wav,
                    vad_filter=bool(cfg.stt_vad_filter),
                    language=cfg.stt_language,
                )
                result = self.audio_detector.censor_transcript(transcript)
                self._safe_stat_inc('live_stt_runs')
                hit_count = 0
                # IMPORTANT: we HOLD audio before playback (stt_alignment_delay_ms).
                # So we must keep events that may be up to that hold duration "in the past",
                # otherwise we discard the exact word timings we wanted to beep.
                hold_s = float(cfg.stt_alignment_delay_ms) / 1000.0
                now_cutoff = time.monotonic() - (hold_s + 0.35)

                raw_segments = transcript.get('segments', [])
                censored_segments = result.get('segments', [])

                with self.beep_lock:
                    for idx, censored_seg in enumerate(censored_segments):
                        word_events = censored_seg.get('word_events', [])

                        # Primary path: beep exact profane word timings.
                        for word_event in word_events:
                            ev_start = start_ts + float(word_event['start'])
                            ev_end = start_ts + float(word_event['end'])
                            if ev_end > now_cutoff:
                                self.beep_events.append((ev_start, ev_end))
                                hit_count += 1

                        # Fallback path: if word timings are absent, evaluate ORIGINAL STT text.
                        if not word_events and idx < len(raw_segments):
                            raw_text = raw_segments[idx].get('text', '').strip()
                            is_profane, _, _ = self.audio_detector.detect(raw_text)

                            # Extra aggressive fallback for live censoring reliability
                            if (not is_profane) and self.config.stt_force_keyword_fallback:
                                is_profane = self._segment_is_profane_fast(raw_text)

                            if is_profane:
                                ev_start = start_ts + float(raw_segments[idx].get('start', 0.0))
                                ev_end = start_ts + float(raw_segments[idx].get('end', 0.0))
                                if ev_end > now_cutoff and ev_end > ev_start:
                                    self.beep_events.append((ev_start, ev_end))
                                    hit_count += 1

                if hit_count:
                    self._safe_stat_inc('live_stt_profanity_hits', hit_count)
                else:
                    # Emergency fallback: if transcript text is profane but no timings,
                    # schedule a short immediate beep so censorship is still audible.
                    full_text = (transcript.get('text', '') or '').strip()
                    if full_text and self._segment_is_profane_fast(full_text):
                        now_ts = time.monotonic()
                        with self.beep_lock:
                            self.beep_events.append((now_ts, now_ts + 0.28))
                        self._safe_stat_inc('live_stt_profanity_hits', 1)
            except Exception:
                self._safe_stat_inc('runtime_errors')
            finally:
                pass

    def _apply_scheduled_beeps(self, audio_chunk, chunk_ts):
        with self.beep_lock:
            if not self.beep_events:
                return audio_chunk
        out = audio_chunk.copy()
        chunk_end = chunk_ts + (len(out) / self.audio_rate)

        with self.beep_lock:
            while self.beep_events and self.beep_events[0][1] < chunk_ts - 0.3:
                self.beep_events.popleft()

            events_snapshot = list(self.beep_events)

        for ev_start, ev_end in events_snapshot:
            overlap_start = max(chunk_ts, ev_start)
            overlap_end = min(chunk_end, ev_end)
            if overlap_end <= overlap_start:
                continue
            i0 = int((overlap_start - chunk_ts) * self.audio_rate)
            i1 = int((overlap_end - chunk_ts) * self.audio_rate)
            i0 = max(0, min(i0, len(out)))
            i1 = max(0, min(i1, len(out)))
            if i1 <= i0:
                continue
            seg_len = i1 - i0
            out[i0:i1] = self.beep_sound[:seg_len] if seg_len <= len(self.beep_sound) else np.resize(self.beep_sound, seg_len)
        return out

    def _video_thread(self):
        frame_count = 0
        fps_start = time.monotonic()
        last_processed = None
        while not self.stop_event.is_set():
            ok, frame = self.video_capture.read()
            if not ok:
                continue
            ts = time.monotonic()
            frame_count += 1

            if frame_count % 30 == 0:
                elapsed = max(1e-6, time.monotonic() - fps_start)
                self._safe_stat_set('capture_fps', frame_count / elapsed)

            detect_every = max(1, int(self.config.video_detection_interval))
            if frame_count % detect_every == 0:
                self.detection_worker.submit_frame(frame, ts)

            result = self.detection_worker.latest_result()
            if result:
                last_processed = result
                self._safe_stat_set('video_detection_ms', result['latency_ms'])
                self.audio_delay.update_video_latency(result['latency_ms'])

            out_frame = last_processed['frame'] if last_processed else frame
            metadata = last_processed['metadata'] if last_processed else {}
            self._bounded_put(self.video_queue, {'frame': out_frame, 'ts': ts, 'metadata': metadata}, 'drops_video_queue')

    def _audio_thread(self):
        stream_in = self.pyaudio.open(
            format=self.audio_in_format,
            channels=self.audio_channels,
            rate=self.audio_rate,
            input=True,
            frames_per_buffer=self.audio_chunk,
        )
        try:
            while not self.stop_event.is_set():
                data = stream_in.read(self.audio_chunk, exception_on_overflow=False)
                audio_i16 = np.frombuffer(data, dtype=np.int16)
                audio = (audio_i16.astype(np.float32) / 32768.0).clip(-1.0, 1.0)
                ts = time.monotonic()

                # Feed raw int16 bytes to streaming KWS (Vosk) if enabled.
                if self.config.enable_kws and self.kws_recognizer is not None:
                    try:
                        self.kws_audio_queue.put_nowait((data, ts))
                    except Exception:
                        pass

                self.audio_delay.add(audio, ts)
                self.audio_delay.trim_older_than(self.config.max_latency_ms)
                self._safe_stat_set('audio_delay_ms', self.audio_delay.target_delay_ms)

                ready = self.audio_delay.pop_ready()
                if not ready:
                    continue

                delayed_audio, delayed_ts = ready
                # Keep a clean copy for STT; do NOT apply beeps yet.
                # Beeps are applied right before playback (after the hold),
                # so STT results that arrive during the hold can affect this chunk.
                self.audio_history.append((delayed_audio.copy(), delayed_ts))
                now_mono = time.monotonic()
                self._schedule_live_stt_beep_events(now_mono)
                self._schedule_fast_gate(now_mono)

                # Enqueue raw audio first; apply beeps later (after hold).
                packet = {'audio': delayed_audio, 'ts': delayed_ts}
                self.latest_audio_packet = packet

                # Hold audio briefly so STT beep events can arrive BEFORE playback.
                self.pending_audio.append(packet)
                hold_s = self.config.stt_alignment_delay_ms / 1000.0
                now_mono = time.monotonic()
                while self.pending_audio and (now_mono - self.pending_audio[0]['ts']) >= hold_s:
                    pkt = self.pending_audio.popleft()
                    out_audio = pkt['audio']

                    # manual beep trigger fallback (hotkey 'b')
                    if time.monotonic() < self.beep_until:
                        n = min(len(out_audio), len(self.beep_sound))
                        out_audio = out_audio.copy()
                        out_audio[:n] = self.beep_sound[:n]

                    # Apply scheduled STT beeps as late as possible so events can land on time.
                    out_audio = self._apply_scheduled_beeps(out_audio, pkt['ts'])
                    self._bounded_put(self.audio_queue, {'audio': out_audio, 'ts': pkt['ts']}, 'drops_audio_queue')
        finally:
            while self.pending_audio:
                pkt = self.pending_audio.popleft()
                out_audio = self._apply_scheduled_beeps(pkt['audio'], pkt['ts'])
                self._bounded_put(self.audio_queue, {'audio': out_audio, 'ts': pkt['ts']}, 'drops_audio_queue')
            stream_in.stop_stream()
            stream_in.close()

    def _sync_thread(self):
        # Buffer video frames and only release them once they've aged to the desired
        # end-to-end latency. This ensures the viewer sees stable, constant delay.
        video_hold = deque()

        while not self.stop_event.is_set():
            try:
                video_hold.append(self.video_queue.get(timeout=0.1))
            except Empty:
                pass

            if not video_hold:
                continue

            now_mono = time.monotonic()
            v_ts = video_hold[0]['ts']
            age_ms = (now_mono - v_ts) * 1000.0
            if age_ms < self.config.target_latency_ms:
                # Not old enough yet; keep buffering.
                time.sleep(0.001)
                continue

            # Drop extremely stale frames to avoid unbounded buffering.
            if age_ms > self.config.max_latency_ms:
                video_hold.popleft()
                self._safe_stat_inc('drops_stale')
                continue

            video_pkt = video_hold.popleft()
            v_ts = video_pkt['ts']

            e2e_ms = (time.monotonic() - v_ts) * 1000.0
            self._safe_stat_set('sync_offset_ms', 0.0)
            self._safe_stat_set('end_to_end_ms', e2e_ms)

            if e2e_ms > self.config.max_latency_ms:
                self._safe_stat_inc('drops_stale')
                continue

            matched_audio = None
            best_diff_ms = float('inf')

            for _ in range(12):
                try:
                    self.audio_sync_cache.append(self.audio_queue.get_nowait())
                except Empty:
                    break

            expected_audio_ts = v_ts
            cache_ttl_s = self.config.max_latency_ms / 1000.0
            while self.audio_sync_cache and self.audio_sync_cache[0]['ts'] < expected_audio_ts - cache_ttl_s:
                self.audio_sync_cache.popleft()

            best_idx = None
            for idx, pkt in enumerate(self.audio_sync_cache):
                diff_ms = abs((pkt['ts'] - expected_audio_ts) * 1000.0)
                if diff_ms < best_diff_ms:
                    best_diff_ms = diff_ms
                    matched_audio = pkt
                    best_idx = idx

            if matched_audio is None or best_diff_ms > self.config.sync_tolerance_ms:
                self._safe_stat_inc('sync_drops')
                fallback = self.latest_audio_packet
                if fallback is not None:
                    pkt = AVPacket(video_pkt['frame'], fallback['audio'], v_ts, video_pkt.get('metadata', {}))
                else:
                    silence = np.zeros(self.audio_chunk, dtype=np.float32)
                    pkt = AVPacket(video_pkt['frame'], silence, v_ts, video_pkt.get('metadata', {}))
                self._bounded_put(self.output_queue, pkt, 'drops_stale')
                continue

            if best_idx is not None:
                del self.audio_sync_cache[best_idx]

            self._safe_stat_set('sync_offset_ms', best_diff_ms)
            pkt = AVPacket(video_pkt['frame'], matched_audio['audio'], v_ts, video_pkt.get('metadata', {}))
            self._bounded_put(self.output_queue, pkt, 'drops_stale')

    def _health_thread(self):
        while not self.stop_event.is_set():
            snap = self.get_stats_snapshot()
            snap['timestamp'] = time.time()
            try:
                with open(self.config.health_log_path, 'a', encoding='utf-8') as f:
                    f.write(json.dumps(snap) + '\n')
            except Exception:
                self._safe_stat_inc('runtime_errors')
            self.stop_event.wait(self.config.health_log_interval_s)

    def _output_thread(self):
        kwargs = dict(
            format=self.audio_out_format,
            channels=self.audio_channels,
            rate=self.audio_rate,
            output=True,
            # Give PortAudio a bit more buffering to reduce stutter on Windows.
            frames_per_buffer=self.audio_chunk * 4,
        )
        if self.config.audio_output_device_index is not None:
            kwargs['output_device_index'] = self.config.audio_output_device_index

        audio_out = self.pyaudio.open(**kwargs)
        window_name = 'Production Streaming Filter'
        cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
        virtual_cam = None

        if self.use_virtual_cam:
            if not VIRTUAL_CAM_AVAILABLE:
                print('[WARN] pyvirtualcam not installed; virtual cam disabled.')
            else:
                virtual_cam = pyvirtualcam.Camera(
                    width=1280,
                    height=720,
                    fps=30,
                    device=self.config.virtual_cam_device,
                )
                print(f'[OK] Virtual camera active: {virtual_cam.device}')

        fps_count = 0
        start = time.monotonic()
        silence_i16 = (np.zeros(self.audio_chunk, dtype=np.int16)).tobytes()
        blank_frame = np.zeros((720, 1280, 3), dtype=np.uint8)
        try:
            while not self.stop_event.is_set():
                try:
                    pkt = self.output_queue.get(timeout=0.1)
                except Empty:
                    # Keep audio continuous even if video/sync stalls.
                    audio_out.write(silence_i16, exception_on_underflow=False)
                    # Pump the UI so the window stays responsive and visible.
                    frame = blank_frame.copy()
                    self._draw_stats_overlay(frame)
                    cv2.putText(frame, "Waiting for A/V packets...", (18, 340), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 200, 255), 2, cv2.LINE_AA)
                    cv2.imshow(window_name, frame)
                    cv2.waitKey(1)
                    continue

                # Write audio first to avoid UI/camera pacing starving audio.
                audio_chunk = np.clip(pkt.audio.astype(np.float32), -1.0, 1.0)
                audio_i16 = (audio_chunk * 32767.0).astype(np.int16)
                audio_out.write(audio_i16.tobytes(), exception_on_underflow=False)
                self.last_audio_write_ts = time.monotonic()

                frame = pkt.frame.copy()
                self._draw_stats_overlay(frame)
                cv2.imshow(window_name, frame)

                if virtual_cam is not None:
                    virtual_cam.send(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
                    # Avoid blocking sleep here; it can starve audio and cause stutter.

                fps_count += 1
                self._safe_stat_inc('frames_output')
                if fps_count % 30 == 0:
                    self._safe_stat_set('output_fps', fps_count / max(1e-6, time.monotonic() - start))

                key = cv2.waitKey(1) & 0xFF
                if key == ord('q'):
                    self.stop()
                elif key == ord('b'):
                    self.trigger_beep(0.5)
        finally:
            if virtual_cam is not None:
                virtual_cam.close()
            audio_out.stop_stream()
            audio_out.close()
            cv2.destroyAllWindows()

    def _draw_stats_overlay(self, frame):
        s = self.get_stats_snapshot()
        overlay = frame.copy()
        cv2.rectangle(overlay, (10, 10), (560, 304), (0, 0, 0), -1)
        cv2.addWeighted(overlay, 0.55, frame, 0.45, 0, frame)

        lines = [
            f"Capture FPS: {s['capture_fps']:.1f}",
            f"Output FPS: {s['output_fps']:.1f}",
            f"Video detection: {s['video_detection_ms']:.1f} ms",
            f"Audio delay target: {s['audio_delay_ms']:.1f} ms",
            f"Sync offset: {s['sync_offset_ms']:.1f} ms",
            f"End-to-end latency: {s['end_to_end_ms']:.1f} ms",
            f"Dropped stale: {s['drops_stale']} | Sync drops: {s['sync_drops']}",
            f"Queue drops V/A: {s['drops_video_queue']}/{s['drops_audio_queue']}",
            f"Live STT runs/hits: {s['live_stt_runs']}/{s['live_stt_profanity_hits']}",
            f"Fast gate runs/hits: {s.get('fast_gate_runs', 0)}/{s.get('fast_gate_hits', 0)}",
            f"KWS runs/hits: {s.get('kws_runs', 0)}/{s.get('kws_hits', 0)}",
            "Press 'b' test beep | 'q' quit",
        ]
        for i, line in enumerate(lines):
            color = (0, 255, 0) if i in (0, 1, 5) else (120, 200, 255)
            cv2.putText(frame, line, (18, 34 + 24 * i), cv2.FONT_HERSHEY_SIMPLEX, 0.55, color, 1, cv2.LINE_AA)

    def trigger_beep(self, duration=0.5):
        self.beep_until = time.monotonic() + duration
        self._safe_stat_inc('profanity_events')

    def get_stats_snapshot(self):
        with self.stats_lock:
            snap = dict(self.stats)
        snap['uptime_s'] = time.time() - snap['started_at']
        snap['config'] = asdict(self.config)
        return snap

    def start(self):
        self.running = True
        self.stop_event.clear()
        self.detection_worker.start()
        self.threads = [
            threading.Thread(target=self._video_thread, daemon=True),
            threading.Thread(target=self._audio_thread, daemon=True),
            threading.Thread(target=self._stt_thread, daemon=True),
            threading.Thread(target=self._fast_gate_thread, daemon=True),
            threading.Thread(target=self._kws_thread, daemon=True),
            threading.Thread(target=self._sync_thread, daemon=True),
            threading.Thread(target=self._output_thread, daemon=True),
            threading.Thread(target=self._health_thread, daemon=True),
        ]
        for th in self.threads:
            th.start()

    def stop(self):
        if not self.running:
            return
        self.running = False
        self.stop_event.set()
        self.detection_worker.stop()

        for th in self.threads:
            th.join(timeout=2)

        self.video_capture.release()
        self.video_processor.cleanup()
        self.pyaudio.terminate()

        print('\n' + '=' * 70)
        print('FINAL STATS')
        print('=' * 70)
        for k, v in self.get_stats_snapshot().items():
            if k != 'config':
                print(f'  {k}: {v}')
        print('=' * 70)

    def run(self):
        try:
            self.start()
            while self.running:
                time.sleep(0.1)
        except KeyboardInterrupt:
            pass
        finally:
            self.stop()


def parse_args():
    parser = argparse.ArgumentParser(description='NeonGuard production streaming filter')
    parser.add_argument('--video-source', type=int, default=0)
    parser.add_argument('--blur-mode', default='pixelation', choices=['pixelation', 'heavy', 'extreme', 'black'])
    parser.add_argument('--recall-mode', default='high', choices=['high', 'balanced', 'precision'])
    parser.add_argument('--target-latency-ms', type=float, default=FilterConfig.target_latency_ms)
    parser.add_argument('--max-latency-ms', type=float, default=FilterConfig.max_latency_ms)
    parser.add_argument('--sync-tolerance-ms', type=float, default=40.0)
    parser.add_argument('--use-virtual-cam', action='store_true')
    parser.add_argument('--virtual-cam-device', default='NeonGuard Virtual Camera')
    parser.add_argument('--audio-output-device-index', type=int, default=None)
    parser.add_argument('--health-log-path', default='runtime_health.jsonl')
    parser.add_argument('--health-log-interval-s', type=float, default=5.0)
    parser.add_argument('--disable-live-stt-censor', action='store_true')
    parser.add_argument('--stt-model-size', default=FilterConfig.stt_model_size)
    parser.add_argument('--stt-window-s', type=float, default=FilterConfig.stt_window_s)
    parser.add_argument('--stt-poll-interval-s', type=float, default=FilterConfig.stt_poll_interval_s)
    parser.add_argument('--stt-temp-wav', default='live_stt_window.wav')
    parser.add_argument('--disable-stt-force-keyword-fallback', action='store_true')
    parser.add_argument('--stt-alignment-delay-ms', type=float, default=FilterConfig.stt_alignment_delay_ms)
    parser.add_argument('--stt-vad-filter', action='store_true', help='Enable VAD filter (may miss short words)')
    parser.add_argument('--stt-language', default="en", help='Whisper language code (set empty for auto)')
    parser.add_argument('--video-detection-interval', type=int, default=FilterConfig.video_detection_interval)
    parser.add_argument('--disable-fast-audio-gate', action='store_true')
    parser.add_argument('--fast-gate-window-s', type=float, default=FilterConfig.fast_gate_window_s)
    parser.add_argument('--fast-gate-poll-interval-s', type=float, default=FilterConfig.fast_gate_poll_interval_s)
    parser.add_argument('--fast-gate-beep-ms', type=float, default=FilterConfig.fast_gate_beep_ms)
    parser.add_argument('--fast-gate-cooldown-ms', type=float, default=FilterConfig.fast_gate_cooldown_ms)
    parser.add_argument('--fast-gate-temp-wav', default='fast_gate_window.wav')
    parser.add_argument('--enable-kws', action='store_true')
    parser.add_argument('--kws-model-path', default="vosk-model-small-en-us-0.15")
    parser.add_argument('--kws-cooldown-ms', type=float, default=500.0)
    parser.add_argument('--kws-beep-ms', type=float, default=320.0)
    parser.add_argument('--list-audio-devices', action='store_true')
    return parser.parse_args()


if __name__ == '__main__':
    args = parse_args()

    if args.list_audio_devices:
        for d in StreamingFilterProduction.list_audio_devices():
            print(f"[{d['index']}] {d['name']} | in={d['maxInputChannels']} out={d['maxOutputChannels']}")
        raise SystemExit(0)

    cfg = FilterConfig(
        blur_mode=args.blur_mode,
        recall_mode=args.recall_mode,
        video_source=args.video_source,
        target_latency_ms=args.target_latency_ms,
        max_latency_ms=args.max_latency_ms,
        sync_tolerance_ms=args.sync_tolerance_ms,
        use_virtual_cam=args.use_virtual_cam,
        virtual_cam_device=args.virtual_cam_device,
        audio_output_device_index=args.audio_output_device_index,
        health_log_path=args.health_log_path,
        health_log_interval_s=args.health_log_interval_s,
        enable_live_stt_censor=not args.disable_live_stt_censor,
        stt_model_size=args.stt_model_size,
        stt_window_s=args.stt_window_s,
        stt_poll_interval_s=args.stt_poll_interval_s,
        stt_temp_wav=args.stt_temp_wav,
        stt_force_keyword_fallback=not args.disable_stt_force_keyword_fallback,
        stt_alignment_delay_ms=args.stt_alignment_delay_ms,
        stt_vad_filter=bool(args.stt_vad_filter),
        stt_language=(args.stt_language if args.stt_language else None),
        video_detection_interval=args.video_detection_interval,
        enable_fast_audio_gate=not args.disable_fast_audio_gate,
        fast_gate_window_s=args.fast_gate_window_s,
        fast_gate_poll_interval_s=args.fast_gate_poll_interval_s,
        fast_gate_beep_ms=args.fast_gate_beep_ms,
        fast_gate_cooldown_ms=args.fast_gate_cooldown_ms,
        fast_gate_temp_wav=args.fast_gate_temp_wav,
        enable_kws=args.enable_kws,
        kws_model_path=args.kws_model_path,
        kws_cooldown_ms=args.kws_cooldown_ms,
        kws_beep_ms=args.kws_beep_ms,
    )

    print('=' * 70)
    print('PRODUCTION STREAMING FILTER')
    print('=' * 70)
    print(f'config: {cfg}')

    app = StreamingFilterProduction(cfg)
    app.run()
