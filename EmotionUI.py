import streamlit as st
import cv2
from deepface import DeepFace
from ultralytics import YOLO
import time

# Load YOLOv8 model for face detection
yolo_model = YOLO(r'D:\abhishekEmotionDetection\train6\weights\best.pt')  # Replace with your YOLO model path

# Load DeepFace emotion model
try:
    emotion_model = DeepFace.build_model('Emotion')  # Preload the model
    st.sidebar.success("Emotion model loaded successfully!")
except Exception as e:
    st.sidebar.error(f"Error loading DeepFace model: {e}")

# Streamlit UI setup
st.title("Real-Time Emotion Detection")
st.sidebar.title("Settings")
st.sidebar.markdown("Adjust the configuration settings below:")

# Confidence threshold slider
confidence_threshold = st.sidebar.slider("Detection Confidence Threshold", 0.0, 1.0, 0.5, 0.01)

# Persistent state for video stream
if "start" not in st.session_state:
    st.session_state.start = False

# Start/Stop buttons
if st.sidebar.button("Start Detection"):
    st.session_state.start = True

if st.sidebar.button("Stop Detection"):
    st.session_state.start = False

# Placeholder for video stream
stframe = st.empty()

# Video processing loop
if st.session_state.start:
    cap = cv2.VideoCapture(0)  # Replace 0 with RTSP URL if needed
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

    while st.session_state.start:
        ret, frame = cap.read()
        if not ret:
            st.error("Failed to capture frame. Please check the camera connection.")
            break

        # YOLOv8 face detection
        results = yolo_model.predict(source=frame)

        for result in results:
            if hasattr(result, 'boxes'):
                for box in result.boxes:
                    confidence = box.conf[0].item()  # Confidence score
                    if confidence < confidence_threshold:
                        continue

                    # Get bounding box coordinates
                    x1, y1, x2, y2 = map(int, box.xyxy[0])
                    face = frame[y1:y2, x1:x2]  # Crop face

                    # Perform emotion detection on the cropped face
                    try:
                        emotion_result = DeepFace.analyze(face, actions=['emotion'], enforce_detection=False)
                        emotion = emotion_result[0]['dominant_emotion']
                    except Exception as e:
                        emotion = "Unknown"

                    # Draw bounding box and label on the frame
                    cv2.rectangle(frame, (x1, y1), (x2, y2), (255, 0, 0), 2)
                    cv2.putText(frame, emotion, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 255, 255), 2)

        # Convert frame to RGB
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        # Display the frame
        stframe.image(rgb_frame, channels="RGB", use_column_width=True)

        # Allow UI to update
        time.sleep(0.03)

    cap.release()
