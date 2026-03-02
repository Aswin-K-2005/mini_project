"""
stream_filter.py - production-oriented low-latency A/V filtering pipeline.
"""

import argparse
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
    import pyvirtualcam
    VIRTUAL_CAM_AVAILABLE = True
except ImportError:
    VIRTUAL_CAM_AVAILABLE = False


@dataclass
class FilterConfig:
    blur_mode: str = 'heavy'
    recall_mode: str = 'high'
    video_source: int = 0
    target_latency_ms: float = 150.0
    max_latency_ms: float = 230.0
    sync_tolerance_ms: float = 40.0
    use_virtual_cam: bool = False
    virtual_cam_device: str = 'NeonGuard Virtual Camera'
    audio_output_device_index: int | None = None
    health_log_path: str = 'runtime_health.jsonl'
    health_log_interval_s: float = 5.0
    enable_live_stt_censor: bool = True
    stt_model_size: str = 'base'
    stt_window_s: float = 2.0
    stt_poll_interval_s: float = 0.8
    stt_temp_wav: str = 'live_stt_window.wav'


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

        self.video_capture = cv2.VideoCapture(config.video_source, cv2.CAP_DSHOW)
        self.video_capture.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
        self.video_capture.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
        self.video_capture.set(cv2.CAP_PROP_FPS, 30)

        self.audio_format = pyaudio.paFloat32
        self.audio_channels = 1
        self.audio_rate = 44100
        self.audio_chunk = 1024
        self.pyaudio = pyaudio.PyAudio()

        self.audio_delay = AudioDelayBuffer(
            min_delay_ms=max(40.0, config.target_latency_ms * 0.35),
            max_delay_ms=min(config.max_latency_ms, config.target_latency_ms * 1.1),
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
            'runtime_errors': 0,
            'started_at': time.time(),
        }

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
        return (0.35 * np.sin(2 * np.pi * frequency * t)).astype(np.float32)

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

    def _write_temp_wav(self, samples, path):
        pcm = np.clip(samples, -1.0, 1.0)
        pcm_i16 = (pcm * 32767).astype(np.int16)
        with wave.open(path, 'wb') as wf:
            wf.setnchannels(1)
            wf.setsampwidth(2)
            wf.setframerate(self.audio_rate)
            wf.writeframes(pcm_i16.tobytes())

    def _schedule_live_stt_beep_events(self, now_mono):
        cfg = self.config
        if not cfg.enable_live_stt_censor or not self.audio_transcriber.available:
            return
        if now_mono - self.last_stt_at < cfg.stt_poll_interval_s:
            return
        if len(self.audio_history) < 2:
            return

        self.last_stt_at = now_mono
        start_ts = self.audio_history[0][1]
        window = np.concatenate([a for a, _ in self.audio_history])
        if window.size == 0:
            return

        payload = {'window': window.copy(), 'start_ts': start_ts}
        try:
            self.stt_queue.put_nowait(payload)
        except Exception:
            # keep real-time path healthy; drop old STT work when overloaded
            try:
                self.stt_queue.get_nowait()
                self.stt_queue.put_nowait(payload)
            except Exception:
                pass

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
                self._write_temp_wav(window, cfg.stt_temp_wav)
                transcript = self.audio_transcriber.transcribe_with_timestamps(cfg.stt_temp_wav)
                result = self.audio_detector.censor_transcript(transcript)
                self._safe_stat_inc('live_stt_runs')
                with self.beep_lock:
                    for seg in result.get('segments', []):
                        for word_event in seg.get('word_events', []):
                            ev_start = start_ts + float(word_event['start'])
                            ev_end = start_ts + float(word_event['end'])
                            if ev_end > time.monotonic() - 1.0:
                                self.beep_events.append((ev_start, ev_end))
                                self._safe_stat_inc('live_stt_profanity_hits')
            except Exception:
                self._safe_stat_inc('runtime_errors')
            finally:
                if os.path.exists(cfg.stt_temp_wav):
                    try:
                        os.remove(cfg.stt_temp_wav)
                    except OSError:
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

            if frame_count % 2 == 0:
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
            format=self.audio_format,
            channels=self.audio_channels,
            rate=self.audio_rate,
            input=True,
            frames_per_buffer=self.audio_chunk,
        )
        try:
            while not self.stop_event.is_set():
                data = stream_in.read(self.audio_chunk, exception_on_overflow=False)
                audio = np.frombuffer(data, dtype=np.float32)
                ts = time.monotonic()

                self.audio_delay.add(audio, ts)
                self.audio_delay.trim_older_than(self.config.max_latency_ms)
                self._safe_stat_set('audio_delay_ms', self.audio_delay.target_delay_ms)

                ready = self.audio_delay.pop_ready()
                if not ready:
                    continue

                delayed_audio, delayed_ts = ready
                self.audio_history.append((delayed_audio.copy(), delayed_ts))
                self._schedule_live_stt_beep_events(time.monotonic())

                # manual beep trigger fallback
                if time.monotonic() < self.beep_until:
                    n = min(len(delayed_audio), len(self.beep_sound))
                    delayed_audio = delayed_audio.copy()
                    delayed_audio[:n] = self.beep_sound[:n]

                delayed_audio = self._apply_scheduled_beeps(delayed_audio, delayed_ts)
                self._bounded_put(self.audio_queue, {'audio': delayed_audio, 'ts': delayed_ts}, 'drops_audio_queue')
        finally:
            stream_in.stop_stream()
            stream_in.close()

    def _sync_thread(self):
        while not self.stop_event.is_set():
            try:
                video_pkt = self.video_queue.get(timeout=0.1)
            except Empty:
                continue

            v_ts = video_pkt['ts']
            matched_audio = None
            best_diff_ms = float('inf')
            for _ in range(8):
                try:
                    self.audio_sync_cache.append(self.audio_queue.get_nowait())
                except Empty:
                    break

            # Audio timestamps are intentionally delayed; match against expected delayed timestamp.
            expected_audio_ts = v_ts - (self.audio_delay.target_delay_ms / 1000.0)

            # Trim stale cached audio to bound memory and avoid bad matches.
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
                # keep video smooth under temporary audio mismatch
                silence = np.zeros(self.audio_chunk, dtype=np.float32)
                pkt = AVPacket(video_pkt['frame'], silence, v_ts, video_pkt.get('metadata', {}))
                self._bounded_put(self.output_queue, pkt, 'drops_stale')
                continue

            # Remove used audio packet from cache
            if best_idx is not None:
                del self.audio_sync_cache[best_idx]

            e2e_ms = (time.monotonic() - v_ts) * 1000.0
            self._safe_stat_set('sync_offset_ms', best_diff_ms)
            self._safe_stat_set('end_to_end_ms', e2e_ms)

            if e2e_ms > self.config.max_latency_ms:
                self._safe_stat_inc('drops_stale')
                continue

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
            format=self.audio_format,
            channels=self.audio_channels,
            rate=self.audio_rate,
            output=True,
            frames_per_buffer=self.audio_chunk,
        )
        if self.config.audio_output_device_index is not None:
            kwargs['output_device_index'] = self.config.audio_output_device_index

        audio_out = self.pyaudio.open(**kwargs)
        cv2.namedWindow('Production Streaming Filter', cv2.WINDOW_NORMAL)
        virtual_cam = None

        if self.use_virtual_cam:
            if not VIRTUAL_CAM_AVAILABLE:
                print('⚠ pyvirtualcam not installed; virtual cam disabled.')
            else:
                virtual_cam = pyvirtualcam.Camera(
                    width=1280,
                    height=720,
                    fps=30,
                    device=self.config.virtual_cam_device,
                )
                print(f'✓ Virtual camera active: {virtual_cam.device}')

        fps_count = 0
        start = time.monotonic()
        try:
            while not self.stop_event.is_set():
                try:
                    pkt = self.output_queue.get(timeout=0.1)
                except Empty:
                    continue

                frame = pkt.frame.copy()
                self._draw_stats_overlay(frame)
                cv2.imshow('Production Streaming Filter', frame)

                if virtual_cam is not None:
                    virtual_cam.send(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
                    virtual_cam.sleep_until_next_frame()

                audio_out.write(pkt.audio.astype(np.float32).tobytes())

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
    parser.add_argument('--blur-mode', default='heavy', choices=['pixelation', 'heavy', 'extreme', 'black'])
    parser.add_argument('--recall-mode', default='high', choices=['high', 'balanced', 'precision'])
    parser.add_argument('--target-latency-ms', type=float, default=150.0)
    parser.add_argument('--max-latency-ms', type=float, default=230.0)
    parser.add_argument('--sync-tolerance-ms', type=float, default=40.0)
    parser.add_argument('--use-virtual-cam', action='store_true')
    parser.add_argument('--virtual-cam-device', default='NeonGuard Virtual Camera')
    parser.add_argument('--audio-output-device-index', type=int, default=None)
    parser.add_argument('--health-log-path', default='runtime_health.jsonl')
    parser.add_argument('--health-log-interval-s', type=float, default=5.0)
    parser.add_argument('--disable-live-stt-censor', action='store_true')
    parser.add_argument('--stt-model-size', default='base')
    parser.add_argument('--stt-window-s', type=float, default=2.0)
    parser.add_argument('--stt-poll-interval-s', type=float, default=0.8)
    parser.add_argument('--stt-temp-wav', default='live_stt_window.wav')
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
    )

    print('=' * 70)
    print('PRODUCTION STREAMING FILTER')
    print('=' * 70)
    print(f'config: {cfg}')

    app = StreamingFilterProduction(cfg)
    app.run()
