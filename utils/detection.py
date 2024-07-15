import cv2
import numpy as np

def detect_unusual_activity(frame, background, threshold=30):
    detections = []
    
    if background is not None:
        frame_delta = cv2.absdiff(background, frame)
        thresh = cv2.threshold(frame_delta, threshold, 255, cv2.THRESH_BINARY)[1]
        thresh = cv2.dilate(thresh, None, iterations=2)
        
        contours, _ = cv2.findContours(thresh.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        for contour in contours:
            if cv2.contourArea(contour) > 500:
                (x, y, w, h) = cv2.boundingRect(contour)
                detections.append({
                    'x': x,
                    'y': y,
                    'w': w,
                    'h': h,
                    'label': 'Unusual Activity'
                })
    
    return detections

def analyze_motion(prev_frame, current_frame, min_area=500):
    frame_delta = cv2.absdiff(prev_frame, current_frame)
    thresh = cv2.threshold(frame_delta, 25, 255, cv2.THRESH_BINARY)[1]
    thresh = cv2.dilate(thresh, None, iterations=2)
    
    contours, _ = cv2.findContours(thresh.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    motion_info = []
    for contour in contours:
        if cv2.contourArea(contour) > min_area:
            (x, y, w, h) = cv2.boundingRect(contour)
            motion_info.append({
                'x': x,
                'y': y,
                'w': w,
                'h': h,
                'label': 'Motion Detected'
            })
    
    return motion_info