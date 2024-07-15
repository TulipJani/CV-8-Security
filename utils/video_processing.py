import cv2
import numpy as np

def process_frame(frame):
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    normalized = cv2.normalize(blurred, None, 0, 255, cv2.NORM_MINMAX)
    resized = cv2.resize(normalized, (640, 480))
    return resized

def extract_background(cap, num_frames=120):
    frames = []
    
    for _ in range(num_frames):
        ret, frame = cap.read()
        if not ret:
            break
        frames.append(process_frame(frame))
    
    if len(frames) > 0:
        background = np.median(frames, axis=0).astype(dtype=np.uint8)
        return background
    else:
        return None