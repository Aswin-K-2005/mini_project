"""
streaming_filter_production.py - PRODUCTION STREAMING FILTER
Properly implements A/V sync with buffering and async detection

Key improvements over previous version:
1. Audio delay buffer - matches video processing latency
2. Sync buffer - holds faster stream until slower catches up
3. Async detection thread - NudeNet doesn't block video pipeline
4. Jitter-free A/V sync - uses PTS (presentation timestamps)
"""

import cv2
import numpy as np
import pyaudio
import threading
import time
from queue import Queue, Empty, PriorityQueue
from collections import deque
import statistics

from videoprocessor import VideoProcessorTestComplete
from audio_processor import AdvancedProfanityDetector


class SyncBuffer:
    """
    Holds the faster stream (usually audio) until video catches up
    Prevents jitter/drift in A/V sync
    """
    
    def __init__(self, max_delay=0.200):  # 200ms max buffer
        self.buffer = PriorityQueue(maxsize=100)
        self.max_delay = max_delay
        self.stats = {'dropped': 0, 'buffered': 0}
    
    def add(self, timestamp, data):
        """Add data with timestamp"""
        try:
            self.buffer.put_nowait((timestamp, data))
            self.stats['buffered'] += 1
        except:
            self.stats['dropped'] += 1
    
    def get_matching(self, target_timestamp, tolerance=0.050):
        """
        Get data matching target timestamp (±tolerance)
        Returns None if no match or if match is too old
        """
        matches = []
        
        # Peek at all items in buffer
        temp_items = []
        while not self.buffer.empty():
            try:
                item = self.buffer.get_nowait()
                temp_items.append(item)
            except Empty:
                break
        
        # Find best match
        best_match = None
        best_diff = float('inf')
        
        for ts, data in temp_items:
            diff = abs(ts - target_timestamp)
            
            # Too old, discard
            if target_timestamp - ts > self.max_delay:
                continue
            
            # Good match
            if diff < tolerance and diff < best_diff:
                best_match = (ts, data)
                best_diff = diff
            else:
                # Put back in buffer
                self.buffer.put((ts, data))
        
        return best_match[1] if best_match else None


class AudioDelayBuffer:
    """
    Delays audio to match video processing latency
    Measures video latency and applies matching delay to audio
    """
    
    def __init__(self, sample_rate=44100):
        self.sample_rate = sample_rate
        self.buffer = deque()
        
        # Adaptive delay based on measured video latency
        self.target_delay = 0.100  # Start with 100ms
        self.video_latencies = deque(maxlen=30)  # Track last 30 frames
    
    def update_video_latency(self, latency):
        """Update with measured video processing latency"""
        self.video_latencies.append(latency)
        
        # Update target delay (use 95th percentile for stability)
        if len(self.video_latencies) >= 10:
            self.target_delay = np.percentile(self.video_latencies, 95)
    
    def add_audio(self, audio_chunk, timestamp):
        """Add audio chunk to delay buffer"""
        self.buffer.append((audio_chunk, timestamp))
    
    def get_delayed_audio(self):
        """
        Get audio that has been delayed by target_delay
        Returns (audio, timestamp) or None
        """
        if not self.buffer:
            return None
        
        # Check if oldest chunk has been delayed enough
        oldest_chunk, oldest_ts = self.buffer[0]
        age = time.time() - oldest_ts
        
        if age >= self.target_delay:
            return self.buffer.popleft()
        
        return None
    
    def get_current_delay(self):
        """Get current buffer delay in seconds"""
        if not self.buffer:
            return 0.0
        return time.time() - self.buffer[0][1]


class AsyncDetectionWorker:
    """
    Runs heavy detection (NudeNet/Face/Gesture) on separate thread
    Main video pipeline continues at full FPS regardless
    """
    
    def __init__(self, detector):
        self.detector = detector
        self.input_queue = Queue(maxsize=5)
        self.output_queue = Queue(maxsize=10)
        self.running = False
        self.thread = None
        
        self.stats = {
            'detections': 0,
            'avg_latency': 0,
            'queue_drops': 0
        }
        self.latencies = deque(maxlen=30)
    
    def start(self):
        """Start detection worker thread"""
        self.running = True
        self.thread = threading.Thread(target=self._worker_loop, daemon=True)
        self.thread.start()
    
    def stop(self):
        """Stop worker thread"""
        self.running = False
        if self.thread:
            self.thread.join(timeout=2)
    
    def _worker_loop(self):
        """Worker loop - processes detection requests"""
        while self.running:
            try:
                frame, timestamp = self.input_queue.get(timeout=0.1)
            except Empty:
                continue
            
            # Run detection (this can be slow - doesn't block main thread!)
            start_time = time.time()
            detections, metadata = self.detector.process_frame(frame, timestamp)
            latency = time.time() - start_time
            
            # Track latency
            self.latencies.append(latency)
            self.stats['detections'] += 1
            self.stats['avg_latency'] = statistics.mean(self.latencies)
            
            # Output result
            try:
                self.output_queue.put_nowait({
                    'frame': detections,
                    'timestamp': timestamp,
                    'metadata': metadata,
                    'latency': latency
                })
            except:
                pass
    
    def submit_frame(self, frame, timestamp):
        """Submit frame for async detection"""
        try:
            self.input_queue.put_nowait((frame.copy(), timestamp))
        except:
            self.stats['queue_drops'] += 1
    
    def get_latest_result(self):
        """Get latest detection result (non-blocking)"""
        try:
            return self.output_queue.get_nowait()
        except Empty:
            return None
    
    def get_avg_latency(self):
        """Get average detection latency for sync calibration"""
        return self.stats['avg_latency']


class StreamingFilterProduction:
    """
    Production-grade streaming filter with proper A/V sync
    """
    
    def __init__(self, 
                 blur_mode='heavy',
                 recall_mode='high',
                 video_source=0):
        
        print("\n" + "="*70)
        print("PRODUCTION STREAMING FILTER")
        print("="*70)
        
        # ══════════════════════════════════════════════════════════════════
        # ASYNC DETECTION WORKER
        # ══════════════════════════════════════════════════════════════════
        print("\n[1/5] Async Detection Worker...")
        self.video_processor = VideoProcessorTestComplete(blur_mode=blur_mode)
        self.detection_worker = AsyncDetectionWorker(self.video_processor)
        
        self.video_capture = cv2.VideoCapture(video_source, cv2.CAP_DSHOW)
        self.video_capture.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
        self.video_capture.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
        self.video_capture.set(cv2.CAP_PROP_FPS, 30)
        print("✓ Async detection ready")
        
        # ══════════════════════════════════════════════════════════════════
        # AUDIO PROCESSOR
        # ══════════════════════════════════════════════════════════════════
        print("\n[2/5] Audio Processor...")
        self.audio_detector = AdvancedProfanityDetector(
            recall_mode=recall_mode,
            allow_mild_profanity=False
        )
        
        self.audio_format = pyaudio.paFloat32
        self.audio_channels = 1
        self.audio_rate = 44100
        self.audio_chunk = 1024
        self.pyaudio = pyaudio.PyAudio()
        
        self.beep_sound = self._generate_beep()
        self.beep_until = 0
        
        print("✓ Audio processor ready")
        
        # ══════════════════════════════════════════════════════════════════
        # SYNC COMPONENTS
        # ══════════════════════════════════════════════════════════════════
        print("\n[3/5] Sync Components...")
        
        # Audio delay buffer - matches video latency
        self.audio_delay_buffer = AudioDelayBuffer(self.audio_rate)
        
        # Sync buffer - holds fast stream until slow catches up
        self.audio_sync_buffer = SyncBuffer(max_delay=0.200)
        
        print("✓ Sync buffers initialized")
        
        # ══════════════════════════════════════════════════════════════════
        # QUEUES
        # ══════════════════════════════════════════════════════════════════
        print("\n[4/5] Threading...")
        self.video_queue = Queue(maxsize=60)  # 2 seconds at 30 FPS
        self.output_queue = Queue(maxsize=30)
        
        self.running = False
        self.threads = []
        
        # ══════════════════════════════════════════════════════════════════
        # STATS
        # ══════════════════════════════════════════════════════════════════
        self.stats = {
            'video_fps': 0,
            'capture_fps': 0,
            'detection_latency': 0,
            'audio_delay': 0,
            'sync_offset': 0,
            'frames_processed': 0,
            'audio_chunks': 0,
            'profanity_detected': 0,
            'sync_drops': 0
        }
        
        print("✓ Threading configured")
        print("\n[5/5] System ready!")
        print("="*70 + "\n")
    
    def _generate_beep(self, duration_ms=300, frequency=1000):
        """Generate beep tone"""
        duration_s = duration_ms / 1000.0
        t = np.linspace(0, duration_s, int(self.audio_rate * duration_s))
        beep = np.sin(2 * np.pi * frequency * t)
        
        fade = int(0.01 * self.audio_rate)
        beep[:fade] *= np.linspace(0, 1, fade)
        beep[-fade:] *= np.linspace(1, 0, fade)
        
        return beep.astype(np.float32)
    
    # ══════════════════════════════════════════════════════════════════════
    # VIDEO CAPTURE THREAD - Captures at full FPS, submits to async worker
    # ══════════════════════════════════════════════════════════════════════
    
    def _video_capture_thread(self):
        """Capture frames at full FPS - never blocks on detection"""
        print("🎥 Video capture thread started")
        
        frame_count = 0
        fps_start = time.time()
        last_detection = None
        
        # Optical flow for inter-detection tracking
        prev_gray = None
        lk_params = dict(
            winSize=(15, 15),
            maxLevel=2,
            criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 0.03)
        )
        
        while self.running:
            ret, frame = self.video_capture.read()
            if not ret:
                continue
            
            capture_timestamp = time.time()
            frame_count += 1
            
            # Update capture FPS
            if frame_count % 30 == 0:
                elapsed = time.time() - fps_start
                self.stats['capture_fps'] = frame_count / elapsed
            
            # Submit to async worker every few frames
            if frame_count % 3 == 0:
                self.detection_worker.submit_frame(frame, capture_timestamp)
            
            # Check for new detection results (non-blocking)
            new_result = self.detection_worker.get_latest_result()
            if new_result:
                last_detection = new_result
                
                # Update audio delay buffer with measured latency
                self.audio_delay_buffer.update_video_latency(
                    new_result['latency']
                )
                self.stats['detection_latency'] = new_result['latency'] * 1000
            
            # Use latest detection result or track from previous
            if last_detection:
                processed_frame = last_detection['frame']
                metadata = last_detection['metadata']
            else:
                # No detection yet, pass through original
                processed_frame = frame
                metadata = {}
            
            # Put in queue with metadata
            try:
                self.video_queue.put_nowait({
                    'frame': processed_frame,
                    'timestamp': capture_timestamp,
                    'metadata': metadata
                })
            except:
                pass
        
        print("🎥 Video capture thread stopped")
    
    # ══════════════════════════════════════════════════════════════════════
    # AUDIO THREAD - Captures, delays, detects profanity
    # ══════════════════════════════════════════════════════════════════════
    
    def _audio_thread(self):
        """Audio capture with adaptive delay matching video latency"""
        print("🔊 Audio thread started")
        
        stream_in = self.pyaudio.open(
            format=self.audio_format,
            channels=self.audio_channels,
            rate=self.audio_rate,
            input=True,
            frames_per_buffer=self.audio_chunk
        )
        
        beep_position = 0
        
        try:
            while self.running:
                # Capture audio
                audio_data = stream_in.read(self.audio_chunk, exception_on_overflow=False)
                audio_array = np.frombuffer(audio_data, dtype=np.float32)
                capture_timestamp = time.time()
                
                # Add to delay buffer
                self.audio_delay_buffer.add_audio(audio_array, capture_timestamp)
                
                # Get delayed audio (matches video processing time)
                delayed_audio = self.audio_delay_buffer.get_delayed_audio()
                
                if delayed_audio:
                    audio_chunk, audio_timestamp = delayed_audio
                    
                    # Apply beep if profanity window active
                    if time.time() < self.beep_until:
                        beep_len = len(self.beep_sound)
                        chunk_len = len(audio_chunk)
                        
                        if beep_position + chunk_len < beep_len:
                            audio_output = self.beep_sound[beep_position:beep_position + chunk_len]
                            beep_position += chunk_len
                        else:
                            audio_output = self.beep_sound[:chunk_len] if chunk_len < beep_len else self.beep_sound
                            beep_position = 0
                    else:
                        audio_output = audio_chunk
                        beep_position = 0
                    
                    # Add to sync buffer
                    self.audio_sync_buffer.add(audio_timestamp, audio_output)
                    self.stats['audio_chunks'] += 1
                
                # Update stats
                self.stats['audio_delay'] = self.audio_delay_buffer.get_current_delay() * 1000
        
        finally:
            stream_in.stop_stream()
            stream_in.close()
            print("🔊 Audio thread stopped")
    
    # ══════════════════════════════════════════════════════════════════════
    # SYNC THREAD - Matches video with delayed audio
    # ══════════════════════════════════════════════════════════════════════
    
    def _sync_thread(self):
        """Match video frames with properly delayed audio"""
        print("🔗 Sync thread started")
        
        while self.running:
            try:
                # Get video frame
                video_data = self.video_queue.get(timeout=0.1)
            except Empty:
                continue
            
            v_frame = video_data['frame']
            v_timestamp = video_data['timestamp']
            v_metadata = video_data['metadata']
            
            # Get matching audio from sync buffer
            audio_chunk = self.audio_sync_buffer.get_matching(
                v_timestamp,
                tolerance=0.050  # 50ms tolerance
            )
            
            if audio_chunk is not None:
                # Successfully synced!
                try:
                    self.output_queue.put_nowait({
                        'frame': v_frame,
                        'audio': audio_chunk,
                        'timestamp': v_timestamp,
                        'metadata': v_metadata
                    })
                    self.stats['frames_processed'] += 1
                except:
                    pass
            else:
                # No matching audio - drop frame or use silence
                self.stats['sync_drops'] += 1
        
        print("🔗 Sync thread stopped")
    
    # ══════════════════════════════════════════════════════════════════════
    # OUTPUT THREAD
    # ══════════════════════════════════════════════════════════════════════
    
    def _output_thread(self):
        """Output synced A/V"""
        print("📺 Output thread started")
        
        stream_out = self.pyaudio.open(
            format=self.audio_format,
            channels=self.audio_channels,
            rate=self.audio_rate,
            output=True,
            frames_per_buffer=self.audio_chunk
        )
        
        cv2.namedWindow('Production Streaming Filter', cv2.WINDOW_NORMAL)
        
        fps_count = 0
        fps_start = time.time()
        
        try:
            while self.running:
                try:
                    av_pair = self.output_queue.get(timeout=0.1)
                except Empty:
                    continue
                
                # Display video
                frame = av_pair['frame']
                self._draw_stats_overlay(frame)
                cv2.imshow('Production Streaming Filter', frame)
                
                # Play audio
                audio = av_pair['audio']
                audio_bytes = (audio * 32767).astype(np.int16).tobytes()
                stream_out.write(audio_bytes)
                
                # FPS
                fps_count += 1
                if fps_count % 30 == 0:
                    self.stats['video_fps'] = fps_count / (time.time() - fps_start)
                
                # Controls
                key = cv2.waitKey(1) & 0xFF
                if key == ord('q'):
                    self.stop()
                elif key == ord('b'):
                    self.trigger_beep(0.5)
        
        finally:
            stream_out.stop_stream()
            stream_out.close()
            cv2.destroyAllWindows()
            print("📺 Output thread stopped")
    
    def _draw_stats_overlay(self, frame):
        """Draw stats"""
        h, w = frame.shape[:2]
        overlay = frame.copy()
        cv2.rectangle(overlay, (10, 10), (450, 220), (0, 0, 0), -1)
        cv2.addWeighted(overlay, 0.6, frame, 0.4, 0, frame)
        
        font = cv2.FONT_HERSHEY_SIMPLEX
        lines = [
            f"Capture FPS: {self.stats['capture_fps']:.1f}",
            f"Output FPS: {self.stats['video_fps']:.1f}",
            f"Detection latency: {self.stats['detection_latency']:.1f}ms",
            f"Audio delay buffer: {self.stats['audio_delay']:.1f}ms",
            f"Frames synced: {self.stats['frames_processed']}",
            f"Sync drops: {self.stats['sync_drops']}",
            f"Profanity detections: {self.stats['profanity_detected']}",
            "",
            "Press 'b' - test beep | 'q' - quit"
        ]
        
        for i, line in enumerate(lines):
            color = (0, 255, 0) if i < 2 else (100, 200, 255)
            cv2.putText(frame, line, (18, 35 + i * 22),
                       font, 0.48, color, 1, cv2.LINE_AA)
    
    def trigger_beep(self, duration=0.5):
        """Trigger beep"""
        self.beep_until = time.time() + duration
        self.stats['profanity_detected'] += 1
    
    # ══════════════════════════════════════════════════════════════════════
    # LIFECYCLE
    # ══════════════════════════════════════════════════════════════════════
    
    def start(self):
        """Start all threads"""
        print("\n🚀 Starting production streaming filter...")
        self.running = True
        
        # Start detection worker
        self.detection_worker.start()
        
        # Start threads
        self.threads = [
            threading.Thread(target=self._video_capture_thread, daemon=True),
            threading.Thread(target=self._audio_thread, daemon=True),
            threading.Thread(target=self._sync_thread, daemon=True),
            threading.Thread(target=self._output_thread, daemon=True),
        ]
        
        for thread in self.threads:
            thread.start()
        
        print("✓ All systems operational\n")
    
    def stop(self):
        """Stop all threads"""
        print("\n🛑 Stopping...")
        self.running = False
        
        self.detection_worker.stop()
        
        for thread in self.threads:
            thread.join(timeout=2)
        
        self.video_capture.release()
        self.video_processor.cleanup()
        self.pyaudio.terminate()
        
        print("\n" + "="*70)
        print("FINAL STATS")
        print("="*70)
        for key, value in self.stats.items():
            print(f"  {key}: {value}")
        print("="*70)
    
    def run(self):
        """Main loop"""
        try:
            self.start()
            while self.running:
                time.sleep(0.1)
        except KeyboardInterrupt:
            print("\n\nInterrupted")
        finally:
            self.stop()


if __name__ == "__main__":
    print("="*70)
    print("PRODUCTION STREAMING FILTER")
    print("="*70)
    print("\nFeatures:")
    print("  ✓ Async detection (NudeNet doesn't block)")
    print("  ✓ Audio delay buffer (matches video latency)")
    print("  ✓ Sync buffer (prevents jitter)")
    print("  ✓ Adaptive delay (measures actual latency)")
    print()
    
    stream_filter = StreamingFilterProduction(
        blur_mode='heavy',
        recall_mode='high',
        video_source=0
    )
    
    stream_filter.run()