
import cv2
from deepface import DeepFace
from ultralytics import YOLO

# Load YOLOv8 model for face detection
yolo_model = YOLO(r'D:\abhishekEmotionDetection\train6\weights\best.pt')  # Replace with the correct path to your YOLOv8 model

# Start capturing video from the camera
#rtsp_url = 'rtsp://ai:ai*12345@172.16.12.31:554/cam/realmonitor?channel=10&subtype=0'
cap = cv2.VideoCapture(0)  # Use RTSP URL for the camera

# Set camera resolution (optional, adjust as needed)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)  # Set width
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)  # Set height

# Load DeepFace model once
try:
    DeepFace.build_model('Emotion')  # Preload the model
except Exception as e:
    print(f"Error loading DeepFace model: {e}")

# Create a resizable window
cv2.namedWindow('Real-time Emotion Detection', cv2.WINDOW_NORMAL)

while True:
    ret, frame = cap.read()
    if not ret:
        print("Failed to capture frame. Exiting...")
        break

    # Get original frame dimensions
    original_height, original_width = frame.shape[:2]

    # YOLOv8 face detection
    results = yolo_model.predict(source=frame)  # Detect faces in the original frame

    # Iterate through the results
    for result in results:
        # Check if the result is a valid detection
        if hasattr(result, 'boxes'):
            for box in result.boxes:
                confidence = box.conf[0].item()  # Get confidence score
                if confidence < 0.5:  # Confidence threshold
                    continue
                
                # Extract face coordinates from the original frame
                x1, y1, x2, y2 = map(int, box.xyxy[0])  # Get bounding box coordinates

                # Extract the face region from the original frame
                face = frame[y1:y2, x1:x2]

                # Perform emotion analysis on the detected face
                try:
                    emotion_result = DeepFace.analyze(face, actions=['emotion'], enforce_detection=False)
                    emotion = emotion_result[0]['dominant_emotion']
                except Exception as e:
                    print(f"Error in emotion analysis: {e}")
                    continue

                # Draw rectangle around the face and display the emotion
                cv2.rectangle(frame, (x1, y1), (x2, y2), (255, 0, 0), 2)
                cv2.putText(frame, emotion, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 255, 255), 2)

    # Display the frame with detected faces and emotions
    cv2.imshow('Real-time Emotion Detection', frame)

    # Exit on pressing 'q'
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release the capture and close all windows
cap.release()
cv2.destroyAllWindows()