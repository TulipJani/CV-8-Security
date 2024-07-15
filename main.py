import streamlit as st
import cv2
import numpy as np
from utils.video_processing import process_frame, extract_background
from utils.detection import detect_unusual_activity, analyze_motion
from datetime import datetime
import tempfile

def main():
    st.set_page_config(layout="wide")

    # Create three columns
    left_column, middle_column, right_column = st.columns([1, 2, 1])

    with left_column:
        with st.container():
            st.subheader("System Details")
            st.write("Location: Main Entrance")
            st.write("Camera ID: CAM001")
            st.write("Date: " + datetime.now().strftime("%Y-%m-%d"))
            time_placeholder = st.empty()

        with st.container():
            st.subheader("System Status")
            status = st.empty()

        with st.container():
            st.subheader("Alerts")
            alerts = st.empty()

    with middle_column:
        st.subheader("Live Feed / Uploaded Video")
        video_source = st.radio("Select video source:", ("Live Feed", "Upload Video"))
        
        if video_source == "Upload Video":
            uploaded_file = st.file_uploader("Choose a video file", type=["mp4", "avi", "mov"])
            if uploaded_file is not None:
                tfile = tempfile.NamedTemporaryFile(delete=False) 
                tfile.write(uploaded_file.read())
                vf = cv2.VideoCapture(tfile.name)
            else:
                vf = None
        else:
            vf = cv2.VideoCapture(0)

        video_placeholder = st.empty()
        details_placeholder = st.empty()

    with right_column:
        with st.container():
            st.subheader("Detection Summary")
            col1, col2 = st.columns(2)
            with col1:
                person_count = st.empty()
                car_count = st.empty()
            with col2:
                unusual_activity_count = st.empty()

        with st.container():
            st.subheader("Recent Detections")
            recent_detections = st.empty()

    person_counter = 0
    car_counter = 0
    unusual_activity_counter = 0
    recent_detection_list = []

    if vf is not None and vf.isOpened():
        background = extract_background(vf, num_frames=60)
        prev_frame = None

        while True:
            ret, frame = vf.read()
            if not ret:
                break

            processed_frame = process_frame(frame)
            
            all_detections = []
            
            if prev_frame is not None:
                motion_info = analyze_motion(prev_frame, processed_frame)
                unusual_activities = detect_unusual_activity(processed_frame, background)
                
                all_detections = motion_info + unusual_activities
                
                for det in all_detections:
                    cv2.rectangle(frame, (det['x'], det['y']), (det['x'] + det['w'], det['y'] + det['h']), (0, 255, 0), 2)
                    cv2.putText(frame, det['label'], (det['x'], det['y'] - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)

                    if det['label'] == 'Person':
                        person_counter += 1
                    elif det['label'] == 'Car':
                        car_counter += 1
                    elif det['label'] == 'Unusual Activity':
                        unusual_activity_counter += 1

                    recent_detection_list.append(f"{det['label']} at ({det['x']}, {det['y']})")
                    if len(recent_detection_list) > 5:
                        recent_detection_list.pop(0)

            video_placeholder.image(frame, channels="BGR", use_column_width=True)
            details_placeholder.write(f"Detected objects: {len(all_detections)}")

            person_count.metric("Persons", person_counter)
            car_count.metric("Cars", car_counter)
            unusual_activity_count.metric("Unusual", unusual_activity_counter)
            recent_detections.write("\n".join(recent_detection_list))

            status.write("Active" if len(all_detections) > 0 else "All Clear")
            alerts.write("ALERT: Unusual Activity Detected!" if unusual_activity_counter > 0 else "No alerts")

            current_time = datetime.now().strftime("%H:%M:%S")
            time_placeholder.write(f"Time: {current_time}")

            prev_frame = processed_frame

            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

        vf.release()
    else:
        st.error("Error: Could not open video source.")

if __name__ == "__main__":
    main()