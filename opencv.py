import cv2
import time
from collections import deque
import numpy as np

cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)
cv2.namedWindow('Motion Tracked Filter', cv2.WINDOW_NORMAL)

face_cascade = cv2.CascadeClassifier(
    cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
)

skip_frames = 2
frame_count = 0
tracked_faces = []

# ══════════════════════════════════════════════════════════
# OPTICAL FLOW TRACKER — tracks object movement
# ══════════════════════════════════════════════════════════
lk_params = dict(
    winSize=(15, 15),
    maxLevel=2,
    criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 0.03)
)

prev_gray = None
track_points = []  # Points to track

fps_start = time.time()
fps_count = 0
fps_display = 0

while True:
    ret, frame = cap.read()
    if not ret:
        break

    h, w = frame.shape[:2]
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    frame_count += 1

    # ══════════════════════════════════════════════════════
    # EVERY Nth FRAME — run full detection
    # ══════════════════════════════════════════════════════
    if frame_count % (skip_frames + 1) == 0:
        faces = face_cascade.detectMultiScale(gray, 1.2, 4)
        
        if len(faces) > 0:
            tracked_faces = []
            track_points = []
            
            for (x, y, fw, fh) in faces:
                # Store the face rectangle
                tracked_faces.append([x, y, fw, fh])
                
                # Create tracking points (center of face)
                cx, cy = x + fw//2, y + fh//2
                track_points.append([[cx, cy]])
            
            # Convert to proper format for optical flow
            if len(track_points) > 0:
                track_points = np.float32(track_points).reshape(-1, 1, 2)
        
        prev_gray = gray.copy()
    
    # ══════════════════════════════════════════════════════
    # OTHER FRAMES — track face movement with optical flow
    # ══════════════════════════════════════════════════════
    elif prev_gray is not None and len(track_points) > 0:
        # Calculate optical flow (where did points move?)
        new_points, status, error = cv2.calcOpticalFlowPyrLK(
            prev_gray, gray, track_points, None, **lk_params
        )
        
        if new_points is not None:
            # Update face positions based on tracked movement
            for i, (new_pt, old_pt) in enumerate(zip(new_points, track_points)):
                if i < len(tracked_faces):
                    # Calculate movement delta
                    dx = int(new_pt[0][0] - old_pt[0][0])
                    dy = int(new_pt[0][1] - old_pt[0][1])
                    
                    # Update face rectangle position
                    tracked_faces[i][0] += dx  # x
                    tracked_faces[i][1] += dy  # y
            
            # Update tracking points
            track_points = new_points
        
        prev_gray = gray.copy()

    # ══════════════════════════════════════════════════════
    # BLUR using tracked positions
    # ══════════════════════════════════════════════════════
    for face_rect in tracked_faces:
        x, y, fw, fh = face_rect
        
        # Bounds checking
        x = max(0, x)
        y = max(0, y)
        fw = min(w - x, fw)
        fh = min(h - y, fh)
        
        if fw > 0 and fh > 0:
            roi = frame[y:y+fh, x:x+fw]
            if roi.size > 0:
                frame[y:y+fh, x:x+fw] = cv2.GaussianBlur(roi, (51, 51), 20)

    # FPS counter
    fps_count += 1
    if time.time() - fps_start >= 1.0:
        fps_display = fps_count
        fps_count = 0
        fps_start = time.time()

    cv2.putText(frame, f"FPS: {fps_display}", (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
    
    status = "DETECTING" if frame_count % (skip_frames + 1) == 0 else "TRACKING"
    cv2.putText(frame, status, (10, 60),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 1)

    cv2.imshow('Motion Tracked Filter', frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()