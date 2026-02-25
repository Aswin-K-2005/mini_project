"""
video_processor.py - Full test with hand gesture detection
Includes multiple blur methods for maximum privacy
"""

import cv2
import numpy as np
from collections import deque
import time

# MediaPipe import with version compatibility
try:
    import mediapipe as mp
    MEDIAPIPE_AVAILABLE = True
    print("✓ MediaPipe imported successfully")
except ImportError:
    MEDIAPIPE_AVAILABLE = False
    print("⚠ MediaPipe not available - skipping gesture detection")


class VideoProcessorTestComplete:
    """
    Complete test version with:
    - Face detection (replaces NudeNet)
    - Hand gesture detection (MediaPipe)
    - Multiple blur modes (pixelation, heavy blur, black boxes)
    - Motion tracking
    """
    
    def __init__(self, blur_mode='heavy'):
        """
        blur_mode options:
        - 'pixelation': Fast pixelation (default for streaming)
        - 'heavy': Heavy Gaussian blur (stronger privacy)
        - 'extreme': Triple-pass blur (maximum privacy)
        - 'black': Black boxes (complete censoring)
        """
        print("\n🔧 Initializing Complete Video Processor...")
        
        self.blur_mode = blur_mode
        print(f"✓ Blur mode: {blur_mode.upper()}")
        
        # ══════════════════════════════════════════════════════════════════
        # FACE DETECTION
        # ══════════════════════════════════════════════════════════════════
        self.face_cascade = cv2.CascadeClassifier(
            cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
        )
        
        if self.face_cascade.empty():
            raise Exception("ERROR: Could not load face cascade!")
        print("✓ Face detector loaded")
        
        # ══════════════════════════════════════════════════════════════════
        # MEDIAPIPE HANDS - with version compatibility
        # ══════════════════════════════════════════════════════════════════
        self.hands = None
        self.hands_available = False
        
        if MEDIAPIPE_AVAILABLE:
            try:
                # Try to access mp.solutions (old API)
                if hasattr(mp, 'solutions'):
                    print("✓ Using MediaPipe solutions API")
                    mp_hands = mp.solutions.hands
                    self.hands = mp_hands.Hands(
                        static_image_mode=False,
                        max_num_hands=2,
                        min_detection_confidence=0.5,
                        min_tracking_confidence=0.5
                    )
                    self.mp_drawing = mp.solutions.drawing_utils
                    self.hands_available = True
                    print("✓ Hand tracking initialized (OLD API)")
                else:
                    # New API structure
                    print("⚠ MediaPipe new API detected - hand tracking disabled")
                    print("  (Install mediapipe<0.10.8 for hand tracking support)")
            except Exception as e:
                print(f"⚠ Hand tracking initialization failed: {e}")
        
        # Gesture patterns
        self.offensive_gestures = {
            'middle_finger': self._detect_middle_finger,
            'fist': self._detect_fist,
        }
        
        # ══════════════════════════════════════════════════════════════════
        # PERFORMANCE SETTINGS
        # ══════════════════════════════════════════════════════════════════
        self.frame_count = 0
        self.skip_frames = 2  # Detect every 3rd frame
        
        # Detection buffers
        self.detection_buffer = deque(maxlen=5)
        
        # Optical flow tracking
        self.lk_params = dict(
            winSize=(15, 15),
            maxLevel=2,
            criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 0.03)
        )
        self.prev_gray = None
        self.tracked_faces = []
        self.track_points = []
        
        # Stats
        self.stats = {
            'faces_detected': 0,
            'gestures_detected': 0,
            'frames_skipped': 0,
            'detection_mode': 'IDLE'
        }
        
        print("✓ Initialization complete\n")
    
    # ══════════════════════════════════════════════════════════════════════
    # GESTURE DETECTION
    # ══════════════════════════════════════════════════════════════════════
    
    def _detect_middle_finger(self, hand_landmarks):
        landmarks = hand_landmarks.landmark

        # Helper
        def is_finger_extended(tip, pip):
            return landmarks[tip].y < landmarks[pip].y

        def is_finger_folded(tip, pip):
            return landmarks[tip].y > landmarks[pip].y

        middle_extended = is_finger_extended(12, 10)
        index_folded = is_finger_folded(8, 6)
        ring_folded = is_finger_folded(16, 14)
        pinky_folded = is_finger_folded(20, 18)

        return middle_extended and index_folded and ring_folded and pinky_folded
    
    def _detect_fist(self, hand_landmarks):
        """Detect closed fist"""
        landmarks = hand_landmarks.landmark
        
        # All fingers closed
        fingers_closed = (
            landmarks[8].y > landmarks[5].y and
            landmarks[12].y > landmarks[9].y and
            landmarks[16].y > landmarks[13].y and
            landmarks[20].y > landmarks[17].y
        )
        
        return fingers_closed
    
    # ══════════════════════════════════════════════════════════════════════
    # DETECTION METHODS
    # ══════════════════════════════════════════════════════════════════════
    
    def detect_faces(self, frame):
        """Detect faces - replaces NudeNet"""
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        
        faces = self.face_cascade.detectMultiScale(
            gray,
            scaleFactor=1.2,
            minNeighbors=4,
            minSize=(50, 50)
        )
        
        blur_regions = []
        for (x, y, w, h) in faces:
            # Add padding for better coverage
            padding = 20
            blur_regions.append({
                'x': max(0, int(x - padding)),
                'y': max(0, int(y - padding)),
                'width': int(w + 2*padding),
                'height': int(h + 2*padding),
                'confidence': 0.95,
                'class': 'FACE'
            })
            self.stats['faces_detected'] += 1
        
        return blur_regions
    
    def detect_gestures(self, frame):
        """Detect hand gestures using MediaPipe"""
        if not self.hands_available:
            return []
        
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = self.hands.process(rgb_frame)
        
        blur_regions = []
        
        if results.multi_hand_landmarks:
            h, w, _ = frame.shape
            
            for hand_landmarks in results.multi_hand_landmarks:
                # Check each gesture
                for gesture_name, detect_func in self.offensive_gestures.items():
                    if detect_func(hand_landmarks):
                        # Get bounding box
                        x_coords = [lm.x for lm in hand_landmarks.landmark]
                        y_coords = [lm.y for lm in hand_landmarks.landmark]
                        
                        x_min = int(min(x_coords) * w)
                        x_max = int(max(x_coords) * w)
                        y_min = int(min(y_coords) * h)
                        y_max = int(max(y_coords) * h)
                        
                        # Add generous padding for gestures
                        padding = 40
                        blur_regions.append({
                            'x': max(0, x_min - padding),
                            'y': max(0, y_min - padding),
                            'width': min(w, x_max + padding) - max(0, x_min - padding),
                            'height': min(h, y_max + padding) - max(0, y_min - padding),
                            'type': gesture_name,
                            'confidence': 1.0
                        })
                        self.stats['gestures_detected'] += 1
        
        return blur_regions
    
    # ══════════════════════════════════════════════════════════════════════
    # MOTION TRACKING
    # ══════════════════════════════════════════════════════════════════════
    
    def update_tracking(self, gray_frame):
        """Update optical flow tracking"""
        if self.prev_gray is None or len(self.track_points) == 0:
            return
        
        new_points, status, error = cv2.calcOpticalFlowPyrLK(
            self.prev_gray, gray_frame, self.track_points, None, **self.lk_params
        )
        
        if new_points is not None and len(new_points) > 0:
            for i, (new_pt, old_pt) in enumerate(zip(new_points, self.track_points)):
                if i < len(self.tracked_faces) and status[i]:
                    dx = int(new_pt[0][0] - old_pt[0][0])
                    dy = int(new_pt[0][1] - old_pt[0][1])
                    
                    self.tracked_faces[i]['x'] += dx
                    self.tracked_faces[i]['y'] += dy
            
            self.track_points = new_points
    
    # ══════════════════════════════════════════════════════════════════════
    # BLUR METHODS - Multiple options for different privacy levels
    # ══════════════════════════════════════════════════════════════════════
    
    def _apply_pixelation(self, roi):
        """Fast pixelation - good for streaming"""
        h, w = roi.shape[:2]
        pixel_size = 12
        tiny = cv2.resize(roi, (pixel_size, pixel_size), 
                         interpolation=cv2.INTER_LINEAR)
        return cv2.resize(tiny, (w, h), interpolation=cv2.INTER_NEAREST)
    
    def _apply_heavy_blur(self, roi):
        """Heavy Gaussian blur - strong privacy"""
        h, w = roi.shape[:2]
        
        # Use largest safe kernel size
        k_size = min(99, (min(h, w) // 2) * 2 + 1)
        if k_size < 3:
            return roi
        
        return cv2.GaussianBlur(roi, (k_size, k_size), 50)
    
    def _apply_extreme_blur(self, roi):
        """Triple-pass blur - maximum privacy"""
        h, w = roi.shape[:2]
        
        # First pass: Heavy blur
        k_size = min(99, (min(h, w) // 2) * 2 + 1)
        if k_size < 3:
            return roi
        
        blurred = cv2.GaussianBlur(roi, (k_size, k_size), 50)
        
        # Second pass: Pixelate
        pixel_size = 8
        tiny = cv2.resize(blurred, (pixel_size, pixel_size),
                         interpolation=cv2.INTER_LINEAR)
        pixelated = cv2.resize(tiny, (w, h), interpolation=cv2.INTER_NEAREST)
        
        # Third pass: Light blur to smooth pixels
        final = cv2.GaussianBlur(pixelated, (15, 15), 10)
        
        return final
    
    def _apply_black_box(self, roi):
        """Complete censoring with black box"""
        return np.zeros_like(roi)
    
    def apply_blur(self, frame, regions):
        """Apply selected blur method to regions"""
        output = frame.copy()
        
        for region in regions:
            x = region['x']
            y = region['y']
            w = region['width']
            h = region['height']
            
            # Bounds checking
            frame_h, frame_w = frame.shape[:2]
            x = max(0, x)
            y = max(0, y)
            w = min(frame_w - x, w)
            h = min(frame_h - y, h)
            
            if w <= 0 or h <= 0:
                continue
            
            roi = frame[y:y+h, x:x+w]
            if roi.size == 0:
                continue
            
            # Apply selected blur method
            if self.blur_mode == 'pixelation':
                censored = self._apply_pixelation(roi)
            elif self.blur_mode == 'heavy':
                censored = self._apply_heavy_blur(roi)
            elif self.blur_mode == 'extreme':
                censored = self._apply_extreme_blur(roi)
            elif self.blur_mode == 'black':
                censored = self._apply_black_box(roi)
            else:
                censored = self._apply_heavy_blur(roi)  # Default
            
            output[y:y+h, x:x+w] = censored
            
            # Draw label
            label = region.get('class', region.get('type', 'FILTERED'))
            color = (0, 0, 255) if 'class' in region else (255, 0, 255)
            cv2.putText(output, f"[{label}]", (x + 5, y + 20),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
        
        return output
    
    # ══════════════════════════════════════════════════════════════════════
    # MAIN PROCESSING PIPELINE
    # ══════════════════════════════════════════════════════════════════════
    
    def process_frame(self, frame, timestamp):
        """Main processing function"""
        self.frame_count += 1
        blur_regions = []
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        
        # DETECTION MODE - every Nth frame
        if self.frame_count % (self.skip_frames + 1) == 0:
            self.stats['detection_mode'] = 'DETECTING'
            
            # Detect faces
            face_regions = self.detect_faces(frame)
            blur_regions.extend(face_regions)
            
            # Update tracking
            if face_regions:
                self.tracked_faces = face_regions.copy()
                self.track_points = []
                
                for region in face_regions:
                    cx = region['x'] + region['width'] // 2
                    cy = region['y'] + region['height'] // 2
                    self.track_points.append([[cx, cy]])
                
                if self.track_points:
                    self.track_points = np.float32(self.track_points).reshape(-1, 1, 2)
            
            self.detection_buffer.append(face_regions)
            self.prev_gray = gray.copy()
        
        # TRACKING MODE
        else:
            self.stats['detection_mode'] = 'TRACKING'
            self.stats['frames_skipped'] += 1
            
            self.update_tracking(gray)
            
            if self.tracked_faces:
                blur_regions.extend(self.tracked_faces)
            
            self.prev_gray = gray.copy()
        
        # GESTURE DETECTION - runs every frame (lightweight)
        gesture_regions = self.detect_gestures(frame)
        blur_regions.extend(gesture_regions)
        
        # Apply blur
        processed_frame = self.apply_blur(frame, blur_regions)
        
        metadata = {
            'timestamp': timestamp,
            'blur_count': len(blur_regions),
            'faces_detected': len([r for r in blur_regions if 'class' in r]),
            'gestures_detected': len([r for r in blur_regions if 'type' in r]),
            'detection_mode': self.stats['detection_mode'],
            'frame_count': self.frame_count
        }
        
        return processed_frame, metadata
    
    def cleanup(self):
        """Cleanup"""
        if self.hands_available and self.hands:
            self.hands.close()


# ══════════════════════════════════════════════════════════════════════════
# MAIN TEST
# ══════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    print("=" * 70)
    print("COMPLETE VIDEO PROCESSOR TEST")
    print("=" * 70)
    
    # Choose blur mode
    print("\nAvailable blur modes:")
    print("  1. pixelation  - Fast, good for streaming (recommended)")
    print("  2. heavy       - Strong Gaussian blur")
    print("  3. extreme     - Triple-pass maximum privacy")
    print("  4. black       - Complete black boxes")
    
    blur_choice = input("\nEnter blur mode (1-4, default=2): ").strip()
    blur_modes = {'1': 'pixelation', '2': 'heavy', '3': 'extreme', '4': 'black'}
    blur_mode = blur_modes.get(blur_choice, 'heavy')
    
    print("\n" + "=" * 70)
    print(f"Using blur mode: {blur_mode.upper()}")
    print("=" * 70)
    print("\nTests:")
    print("  ✓ Face detection (stands in for NudeNet)")
    print("  ✓ Hand gesture detection (MediaPipe)")
    print("  ✓ Motion tracking")
    print("  ✓ Frame skipping optimization")
    print("\nControls:")
    print("  - Move your face fast to test tracking")
    print("  - Make a FIST or MIDDLE FINGER to test gestures")
    print("  - Press 'q' to quit")
    print("=" * 70)
    
    processor = VideoProcessorTestComplete(blur_mode=blur_mode)
    cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)
    
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
    cap.set(cv2.CAP_PROP_FPS, 30)
    
    cv2.namedWindow('Complete Video Processor Test', cv2.WINDOW_NORMAL)
    
    fps_start = time.time()
    fps_count = 0
    fps_display = 0
    
    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                print("ERROR: Could not read frame")
                break
            
            timestamp = time.time()
            processed_frame, metadata = processor.process_frame(frame, timestamp)
            
            # FPS counter
            fps_count += 1
            if time.time() - fps_start >= 1.0:
                fps_display = fps_count
                fps_count = 0
                fps_start = time.time()
            
            # Draw stats
            h, w = processed_frame.shape[:2]
            overlay = processed_frame.copy()
            cv2.rectangle(overlay, (10, 10), (380, 150), (0, 0, 0), -1)
            cv2.addWeighted(overlay, 0.6, processed_frame, 0.4, 0, processed_frame)
            
            stats = [
                f"FPS: {fps_display}",
                f"Mode: {metadata['detection_mode']}",
                f"Blur: {blur_mode.upper()}",
                f"Faces: {metadata['faces_detected']}",
                f"Gestures: {metadata['gestures_detected']}"
            ]
            
            for i, text in enumerate(stats):
                color = (0, 255, 0) if i == 0 else (100, 200, 255)
                cv2.putText(processed_frame, text, (18, 32 + i * 25),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.55, color, 1, cv2.LINE_AA)
            
            cv2.imshow('Complete Video Processor Test', processed_frame)
            
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
                
    except KeyboardInterrupt:
        print("\n\nInterrupted by user")
    except Exception as e:
        print(f"\n\nERROR: {e}")
        import traceback
        traceback.print_exc()
    finally:
        print("\n" + "=" * 70)
        print("FINAL STATISTICS:")
        print(f"  Total faces detected: {processor.stats['faces_detected']}")
        print(f"  Total gestures detected: {processor.stats['gestures_detected']}")
        print(f"  Frames skipped: {processor.stats['frames_skipped']}")
        print(f"  Blur mode used: {blur_mode.upper()}")
        print("=" * 70)
        
        cap.release()
        cv2.destroyAllWindows()
        processor.cleanup()
        print("\nTest complete! Ready to swap in NudeNet.")