import cv2
import numpy as np
import face_recognition
import base64
import os
import time
import mediapipe as mp
import torch
import socketio
from ultralytics import YOLO
from io import BytesIO
from PIL import Image
import json
import glob
import uuid
import requests
import threading

# ----------------- Socket.IO Setup -----------------
sio = socketio.Client()

known_face_encodings = []
known_face_names = []
sessions = {}

def load_known_faces_from_images():
    print("[INFO] Loading known faces from saved images...")
    for path in glob.glob('received_*.jpg'):
        name = os.path.splitext(os.path.basename(path))[0].replace("received_", "")
        try:
            image = face_recognition.load_image_file(path)
            encodings = face_recognition.face_encodings(image)
            if encodings:
                known_face_encodings.append(encodings[0])
                known_face_names.append(name)
                print(f"[INFO] Loaded known face for '{name}' from {path}")
            else:
                print(f"[WARNING] No face found in saved image: {path}")
        except Exception as e:
            print(f"[ERROR] Failed to load {path}: {e}")

load_known_faces_from_images()

@sio.event
def connect():
    print("[INFO] Connected to Socket.IO Server")

@sio.event
def disconnect():
    print("[INFO] Disconnected from Socket.IO Server")

@sio.on('userPhoto')
def handle_face_data(data):
    try:
        token = data.get('token')
        image_dict = data.get('image_base64')

        if token and image_dict and 'buffer' in image_dict:
            image_buffer = image_dict['buffer']

            if isinstance(image_buffer, bytes):
                try:
                    image_str = image_buffer.decode()
                    if image_str.startswith("http"):
                        response = requests.get(image_str)
                        image_bytes = response.content
                    else:
                        image_bytes = base64.b64decode(image_str)
                except Exception as e:
                    print(f"[ERROR] Failed to decode bytes: {e}")
                    return
            elif isinstance(image_buffer, str):
                if image_buffer.startswith("http"):
                    response = requests.get(image_buffer)
                    if response.status_code != 200:
                        print(f"[ERROR] Failed to fetch image from URL: {response.status_code}")
                        return
                    image_bytes = response.content
                else:
                    image_bytes = base64.b64decode(image_buffer)
            else:
                raise ValueError("Unsupported format in image_base64['buffer']")

            try:
                image = Image.open(BytesIO(image_bytes)).convert('RGB')
            except Exception as e:
                print(f"[ERROR] PIL cannot open image: {e}")
                return

            np_image = np.array(image)
            face_locations = face_recognition.face_locations(np_image)
            face_encodings = face_recognition.face_encodings(np_image, face_locations)

            if face_encodings:
                known_face_encodings.append(face_encodings[0])
                known_face_names.append(token)
                image.save(f'received_{token}.jpg')
                sessions[token] = {
                    "name": token,
                    "face_encoding": face_encodings[0],
                    "last_seen": time.time(),
                    "events": []
                }
                print(f"[INFO] Photo received and processed for token {token}")
            else:
                print(f"[WARNING] No face found in the image for token {token}")
        else:
            print("[ERROR] Missing 'token' or image buffer in data.")
    except Exception as e:
        print(f"[ERROR] Failed to process face data: {e}")

# ----------------- GPU Check -----------------
device = 'cuda' if torch.cuda.is_available() else 'cpu'
print(f"[INFO] Running on {device.upper()}")

# ----------------- Load YOLO Models -----------------
model_path = r"C:\\Users\\Peter\\PycharmProjects\\pythonProject16\\yolov8n.pt"
if not os.path.exists(model_path):
    print(f"[ERROR] Model file not found: {model_path}")
    exit(1)

face_model = YOLO(model_path).to(device)
object_model = YOLO("best_yolov8.pt").to(device)

# ----------------- MediaPipe -----------------
mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils
hands = mp_hands.Hands(min_detection_confidence=0.7, min_tracking_confidence=0.7)

mp_pose = mp.solutions.pose
pose = mp_pose.Pose(min_detection_confidence=0.5, min_tracking_confidence=0.5)

def is_hand_closed(landmarks):
    fingertip_ids = [8, 12, 16, 20]
    knuckle_ids = [6, 10, 14, 18]
    return sum(1 for tip, knuckle in zip(fingertip_ids, knuckle_ids)
               if landmarks.landmark[tip].y > landmarks.landmark[knuckle].y) >= 3

def newItem(data):
    session_token = data.get("session_token")
    if session_token in sessions:
        sessions[session_token].setdefault("events", []).append({
            "timestamp": data.get("timestamp"),
            "status": data.get("status"),
            "object": data.get("object")
        })
    sio.emit('newItem', data)

# ----------------- Video Capture -----------------
cap = cv2.VideoCapture(1)
cap.set(3, 640)
cap.set(4, 360)

print("[INFO] Connecting to Socket.IO server...")
sio.connect("http://192.168.67.196:5001")

initial_objects = {}
sent_objects = {}
yolo_result = None

# ----------------- Threaded YOLO -----------------
def run_yolo_detection(frame):
    global yolo_result
    yolo_result = object_model.predict(frame, conf=0.3, device=0 if device == 'cuda' else 'cpu')[0]

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    frame = cv2.flip(frame, 1)
    resized_frame = cv2.resize(frame, (640, 360))
    rgb_frame = cv2.cvtColor(resized_frame, cv2.COLOR_BGR2RGB)

    yolo_thread = threading.Thread(target=run_yolo_detection, args=(resized_frame,))
    yolo_thread.start()

    # Face recognition
    face_locations = []
    face_results = face_model.predict(resized_frame, conf=0.3, device=0 if device == 'cuda' else 'cpu')
    for result in face_results:
        for box in result.boxes:
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            face_locations.append((y1, x2, y2, x1))

    session_token = None
    if face_locations:
        face_encodings = face_recognition.face_encodings(rgb_frame, face_locations)
        for (top, right, bottom, left), face_encoding in zip(face_locations, face_encodings):
            matches = face_recognition.compare_faces(known_face_encodings, face_encoding, tolerance=0.65)
            face_distances = face_recognition.face_distance(known_face_encodings, face_encoding)
            name = "Unknown"
            if len(face_distances) > 0:
                best_match_index = np.argmin(face_distances)
                if matches[best_match_index]:
                    name = known_face_names[best_match_index]
                    session_token = name
                    if session_token not in sessions:
                        sessions[session_token] = {
                            "name": name,
                            "face_encoding": face_encoding,
                            "last_seen": time.time(),
                            "events": []
                        }
            cv2.rectangle(resized_frame, (left, top), (right, bottom), (0, 255, 0), 2)
            cv2.putText(resized_frame, name, (left, top - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2)

    pose_results = pose.process(rgb_frame)
    if pose_results.pose_landmarks:
        mp_drawing.draw_landmarks(resized_frame, pose_results.pose_landmarks, mp_pose.POSE_CONNECTIONS)

    yolo_thread.join()

    current_objects = {}
    if yolo_result:
        yolo_output = yolo_result.plot()
        for box in yolo_result.boxes:
            cls_id = int(box.cls[0])
            obj_name = object_model.names[cls_id]
            current_objects[obj_name] = current_objects.get(obj_name, 0) + 1
    else:
        yolo_output = resized_frame

    catching_detected = False
    hand_results = hands.process(rgb_frame)
    if hand_results.multi_hand_landmarks:
        for hand_landmarks in hand_results.multi_hand_landmarks:
            mp_drawing.draw_landmarks(yolo_output, hand_landmarks, mp_hands.HAND_CONNECTIONS)
            catching_detected = is_hand_closed(hand_landmarks)
            status_text = "CATCHING" if catching_detected else "RELEASING"
            color = (0, 0, 255) if catching_detected else (0, 255, 0)
            cv2.putText(yolo_output, status_text, (20, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, color, 2)

    if session_token:
        # CATCH LOGIC
        if catching_detected:
            for obj, count in current_objects.items():
                already_sent = sent_objects.get((session_token, obj), 0)
                to_send = count - already_sent
                if to_send > 0:
                    sent_objects[(session_token, obj)] = already_sent + to_send
                    newItem({
                        "itemName": obj,
                        "state": "caught",
                        "token": session_token
                    })
                    print(f"[INFO] Caught: {obj} x{to_send} by {sessions[session_token]['name']}")

        # RELEASE LOGIC
        if not catching_detected:
            for (token, obj), count in list(sent_objects.items()):
                if token == session_token:
                    # If the object is back in frame with open hand
                    if obj in current_objects:
                        newItem({
                            "itemName": obj,
                            "state": "released",
                            "token": session_token
                        })
                        print(f"[INFO] Released: {obj} by {sessions[session_token]['name']}")
                        del sent_objects[(token, obj)]

    cv2.imshow("YOLO + Face + Hands + Pose", yolo_output)
    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

cap.release()
cv2.destroyAllWindows()
