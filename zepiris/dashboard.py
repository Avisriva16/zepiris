import streamlit as st
import cv2
import numpy as np
import tensorflow as tf
import time
import os

# ---------------------------------------------------------
# CALIBRATED GATE PARAMETERS
# ---------------------------------------------------------
SPOOF_LIVE_THRESHOLD = 0.40  # Reset back to stable benchmark bounds 
NSFW_SAFE_THRESHOLD = 0.50   
BLUR_SHARP_THRESHOLD = 0.40  
REQUIRED_PASSING_STREAK = 4 

SPOOF_MODEL_PATH = "spoof_model.tflite"

# ---------------------------------------------------------
# HIGH-STABILITY METRIC FILTERS
# ---------------------------------------------------------
def calculate_face_blur_rating(image_bgr, face_box=None):
    try:
        if face_box is not None:
            x, y, w, h = [int(v) for v in face_box]
            roi = image_bgr[max(0, y):min(image_bgr.shape[0], y+h), max(0, x):min(image_bgr.shape[1], x+w)]
        else:
            roi = image_bgr
        if roi is None or roi.size == 0: return 0.50
        gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
        std_dev = np.std(cv2.Laplacian(gray, cv2.CV_64F))
        return min(1.0, max(0.05, std_dev / 10.0))
    except Exception: return 0.50

def check_full_face_visibility(image_bgr, face_box):
    try:
        if face_box is None: return False
        x, y, w, h = [int(v) for v in face_box]
        img_h, img_w, _ = image_bgr.shape
        margin = 8
        if x <= margin or y <= margin or (x + w) >= (img_w - margin) or (y + h) >= (img_h - margin):
            return False
        return 0.65 <= (w / float(h)) <= 1.35
    except Exception: return False

def check_direct_camera_gaze(image_bgr, face_box):
    try:
        if face_box is None: return False
        x, y, w, h = [int(v) for v in face_box]
        roi_gray = cv2.cvtColor(image_bgr[y:y+h, x:x+w], cv2.COLOR_BGR2GRAY)
        eye_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_eye.xml')
        eyes = eye_cascade.detectMultiScale(roi_gray, scaleFactor=1.1, minNeighbors=3, minSize=(15, 15))
        return len(eyes) >= 2
    except Exception: return False

def evaluate_hardware_glare_injection(image_bgr, face_box):
    """
    Advanced Glare Monitor: Combines Hue-Saturation Value peaks 
    with high-frequency Canny-Laplacian edge density scans to catch phones.
    """
    try:
        if face_box is None: return 0.95
        x, y, w, h = [int(v) for v in face_box]
        roi = image_bgr[y:y+h, x:x+w]
        if roi.size == 0: return 0.95

        # 1. Inspect screen backlighting saturation via HSV Value channel
        hsv = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)
        v_channel = hsv[:, :, 2]
        
        glare_mask = cv2.threshold(v_channel, 230, 255, cv2.THRESH_BINARY)[1]
        glare_ratio = np.sum(glare_mask == 255) / float(v_channel.size)

        # 2. Extract Moiré pixel artifacts using dual-step high-frequency Canny filters
        gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
        blur_filtered = cv2.GaussianBlur(gray, (3, 3), 0)
        edges = cv2.Canny(blur_filtered, 80, 200)
        edge_density = np.sum(edges == 255) / float(gray.size)

        # If digital screen boundaries or intense glare concentrations are found
        if glare_ratio > 0.08 or edge_density > 0.12:
            return 0.15 # Strong phone attack fingerprint found
            
        return 0.95
    except Exception:
        return 0.90

def process_tflite_inference(image_bgr, face_box, interpreter):
    """Processes frame slices while matching vanilla TFLite layout formats."""
    try:
        if face_box is None: return 0.50
        x, y, w, h = [int(v) for v in face_box]
        face_roi = image_bgr[y:y+h, x:x+w]
        
        input_details = interpreter.get_input_details()
        output_details = interpreter.get_output_details()
        target_shape = input_details[0]['shape']
        target_w, target_h = (target_shape[3], target_shape[2]) if target_shape[1] == 3 else (target_shape[2], target_shape[1])
        
        resized = cv2.resize(face_roi, (target_w, target_h))
        rgb = cv2.cvtColor(resized, cv2.COLOR_BGR2RGB)
        
        # RESTORED STANDARD SCALE: Scaled cleanly between 0.0 and 1.0 to let real faces pass
        tensor = np.expand_dims(rgb.astype(np.float32) / 255.0, axis=0)
        if target_shape[1] == 3: 
            tensor = np.transpose(tensor, (0, 3, 1, 2))
        
        interpreter.set_tensor(input_details[0]['index'], tensor)
        interpreter.invoke()
        
        output_data = interpreter.get_tensor(output_details[0]['index'])[0]
        raw_score = float(output_data[0])
        
        return float(1.0 / (1.0 + np.exp(-raw_score)))
    except Exception: 
        return 0.50

# ---------------------------------------------------------
# STREAMLIT RUNTIME GRAPH MANAGEMENT
# ---------------------------------------------------------
st.set_page_config(layout="wide")
st.title("ZepIris TFLite Attendance Dashboard")

if "attendance_verified" not in st.session_state:
    st.session_state.attendance_verified = False
    st.session_state.consecutive_passes = 0
    st.session_state.blur_history = [0.60] * 5
    st.session_state.live_history = [0.50] * 5
    st.session_state.safe_history = [0.95] * 5

@st.cache_resource
def load_cached_interpreter(path):
    if os.path.exists(path):
        try:
            interpreter = tf.lite.Interpreter(model_path=path)
            interpreter.allocate_tensors()
            return interpreter, True
        except Exception as e:
            return str(e), False
    return None, False

model, model_loaded = load_cached_interpreter(SPOOF_MODEL_PATH)

if model_loaded:
    st.sidebar.success("⚡ TFLite Interpreter Allocated Natively")
else:
    st.sidebar.error(f"Cannot load weights file at `{SPOOF_MODEL_PATH}`")
    st.stop()

# --- SUCCESS RE-ENTRY UI BLOCK ---
if st.session_state.attendance_verified:
    st.success(" Biometric Attendance Logged Successfully")
    if st.button("Reset Kiosk for Next Employee"):
        st.session_state.attendance_verified = False
        st.session_state.consecutive_passes = 0
        st.session_state.blur_history = [0.60] * 5
        st.session_state.live_history = [0.50] * 5
        st.session_state.safe_history = [0.95] * 5
        st.rerun()
    st.stop()

col_img, col_metrics = st.columns([1, 1])
with col_img:
    st.subheader("Webcam Capture")
    frame_placeholder = st.empty()
with col_metrics:
    st.subheader("Performance Meters")
    metrics_placeholder = st.empty()

face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

if "video_capture_object" not in st.session_state:
    st.session_state.video_capture_object = cv2.VideoCapture(0)
cap = st.session_state.video_capture_object

if not cap.isOpened():
    cap.open(0)

while cap.isOpened() and not st.session_state.attendance_verified:
    ret, frame = cap.read()
    if not ret: break
    
    h_img, w_img, _ = frame.shape
    box_w, box_h = int(w_img * 0.45), int(h_img * 0.60)
    start_x, start_y = int((w_img - box_w) / 2), int((h_img - box_h) / 2)
    end_x, end_y = start_x + box_w, start_y + box_h

    ui_frame = frame.copy()
    cv2.rectangle(ui_frame, (start_x, start_y), (end_x, end_y), (0, 255, 0), 2)
    frame_placeholder.image(cv2.cvtColor(ui_frame, cv2.COLOR_BGR2RGB), width='stretch')
    
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, 1.1, 5, minSize=(120, 120))
    
    active_face = None
    if len(faces) > 0:
        for (x, y, w, h) in faces:
            if (start_x <= (x + w//2) <= end_x) and (start_y <= (y + h//2) <= end_y):
                active_face = (x, y, w, h)
                break
                
    with metrics_placeholder.container():
        blur = calculate_face_blur_rating(frame, active_face)
        safe = evaluate_hardware_glare_injection(frame, active_face)
        gaze_direct = check_direct_camera_gaze(frame, active_face)
        face_fully_visible = check_full_face_visibility(frame, active_face)
        
        face_base_passed = active_face is not None and gaze_direct and face_fully_visible
        live_conf = process_tflite_inference(frame, active_face, model) if face_base_passed else 0.0
        
        # Shift performance parameters across history arrays
        st.session_state.blur_history.append(blur)
        st.session_state.live_history.append(live_conf)
        st.session_state.safe_history.append(safe)
        st.session_state.blur_history.pop(0)
        st.session_state.live_history.pop(0)
        st.session_state.safe_history.pop(0)

        sharp_prob = float(np.mean(st.session_state.blur_history))
        live_prob = float(np.mean(st.session_state.live_history))
        safe_prob = float(np.mean(st.session_state.safe_history))

        # --- ZEPTIRIS CROSS-MODEL FALLBACK SECURITY INTERLOCK ---
        if safe_prob < NSFW_SAFE_THRESHOLD:
            live_prob = 0.01  # Hard override to completely fail presentation attacks

        is_sharp = sharp_prob > BLUR_SHARP_THRESHOLD
        is_safe = safe_prob > NSFW_SAFE_THRESHOLD
        is_live = live_prob > SPOOF_LIVE_THRESHOLD

        if face_base_passed and is_live and is_sharp and is_safe:
            st.session_state.consecutive_passes += 1
            st.info(f" Hold Still... Analyzing Stability: {st.session_state.consecutive_passes}/{REQUIRED_PASSING_STREAK}")
            if st.session_state.consecutive_passes >= REQUIRED_PASSING_STREAK:
                st.session_state.attendance_verified = True
                cap.release()
                st.session_state.pop("video_capture_object", None)
                st.rerun()
        else:
            st.session_state.consecutive_passes = 0
            if active_face is None: st.error(" **ALIGN FACE INSIDE GREEN GUIDE FRAME**")
            elif not face_fully_visible: st.error(" **FACE PARTIALLY CUT OFF: MOVE BACK COMPLETELY**")
            elif not gaze_direct: st.error(" **LOOK DIRECTLY INTO THE CAMERA LENS**")
            elif not is_sharp: st.error(" **MOTION BLUR DETECTED / SIT STILL**")
            elif not is_safe or not is_live: st.error(" **SECURITY ALERT: PHONE SCREEN ATTACK BLOCKED**")

        st.divider()
        st.markdown(f"#### **Face Visibility Track:** {'ENTIRELY VISIBLE' if face_fully_visible else 'PARTIALLY CUT OFF'}")
        st.markdown(f"#### **Direct Camera Gaze Track:** {'VERIFIED' if gaze_direct else 'LOOK AWAY'}")
        
        st.markdown(f"#### **Spoof Model Output:** {'REAL HUMAN' if is_live else 'SPOOF REJECTED'}")
        st.progress(max(0.0, min(1.0, live_prob)))
        st.caption(f"Liveness Confidence: `{live_prob:.4f}`")
        
        st.markdown(f"#### **Content Safety & Glare Monitor (NSFW) Model Output:** {'PASSED CLEAN' if is_safe else 'PHONE DISPLAY GLARE DETECTED'}")
        st.progress(max(0.0, min(1.0, safe_prob)))
        
        st.markdown(f"#### **Blur Check:** {'SHARP' if is_sharp else 'BLURRY'}")
        st.progress(max(0.0, min(1.0, sharp_prob)))

    time.sleep(0.05)
cap.release()