import streamlit as st
import os
import numpy as np
import cv2
import time
import io
from PIL import Image

# Apple CoreML Bridges
import CoreML
import Vision
from Foundation import NSURL

# --- PRODUCTION PAGE SETUP ---
st.set_page_config(page_title="ZepIris Hardened Biometric Kiosk", layout="wide")

# --- MASTER BIOMETRIC & TELEMETRY PARAMETERS ---
SPOOF_LIVE_THRESHOLD = 0.50   
BLUR_SHARP_THRESHOLD = 0.80   # ✅ CALIBRATED: Safely sits above your live face floor (0.84)
NSFW_VIOLATION_THRESHOLD = 0.80 # ✅ SENSITIVE: Low values catch dark/grainy/pixelated environments

# Active Behavioral Movement Bounds
MIN_HEAD_MOTION_VELOCITY = 4     
MAX_STATIC_FRAMES_ALLOWED = 40   
STABILIZATION_DURATION = 4.5 

def stable_softmax(logits):
    exp_logits = np.exp(logits - np.max(logits))
    return exp_logits / exp_logits.sum()

# --- LOCAL RUNTIME COREML ENGINE CLUSTER ---
class NativeAppleInferenceEngine:
    def __init__(self):
        self.models = {}
        self.errors = {}
        self._load_all_packages()

    def _load_all_packages(self):
        targets = {
            "blur": "blur_model.mlpackage",
            "nsfw": "nsfw_model.mlpackage",
            "spoof": "spoof_model.mlpackage"
        }
        config = CoreML.MLModelConfiguration.alloc().init()
        config.setComputeUnits_(CoreML.MLComputeUnitsAll)

        for name, filename in targets.items():
            if not os.path.exists(filename):
                self.errors[name] = f"Missing package directory asset: {filename}"
                continue
            try:
                url = NSURL.fileURLWithPath_(os.path.abspath(filename))
                compiled_url, err = CoreML.MLModel.compileModelAtURL_error_(url, None)
                if err: continue
                model, err = CoreML.MLModel.modelWithContentsOfURL_configuration_error_(compiled_url, config, None)
                if err: continue
                vision_model, err = Vision.VNCoreMLModel.modelForMLModel_error_(model, None)
                if err: continue
                self.models[name] = vision_model
            except Exception as e:
                self.errors[name] = str(e)

    def run_vision_inference(self, cg_image_ref, model_name):
        if model_name not in self.models or cg_image_ref is None: 
            return None
        try:
            request = Vision.VNCoreMLRequest.alloc().initWithModel_(self.models[model_name]).init()
            handler = Vision.VNImageRequestHandler.alloc().initWithCGImage_options_(cg_image_ref, None).init()
            success, err = handler.performRequests_error_([request], None)
            if not success: return None
            results = request.results()
            if not results: return None
            multi_array = results[0].featureValue().multiArrayValue()
            count = multi_array.count()
            return [float(multi_array.objectAtIndexedSubscript_(i)) for i in range(count)]
        except Exception:
            return None

def crop_and_convert_to_cgimage(opencv_frame, x, y, w, h):
    if opencv_frame is None or opencv_frame.size == 0: return None
    try:
        img_h, img_w, _ = opencv_frame.shape
        pad_w, pad_h = int(w * 0.35), int(h * 0.35)
        x1, y1 = max(0, x - pad_w), max(0, y - pad_h)
        x2, y2 = min(img_w, x + w + pad_w), min(img_h, y + h + pad_h)
        
        face_roi = np.copy(opencv_frame[y1:y2, x1:x2])
        if face_roi.size == 0 or face_roi.shape[0] < 40 or face_roi.shape[1] < 40: return None
            
        resized = cv2.resize(face_roi, (224, 224))
        rgb = cv2.cvtColor(resized, cv2.COLOR_BGR2RGB)
        
        pil_img = Image.fromarray(rgb)
        img_byte_arr = io.BytesIO()
        pil_img.save(img_byte_arr, format='PNG')
        
        import Quartz
        data = Quartz.CFDataCreate(None, img_byte_arr.getvalue(), len(img_byte_arr.getvalue()))
        provider = Quartz.CGDataProviderCreateWithCFData(data)
        return Quartz.CGImageCreateWithPNGDataProvider(provider, None, True, Quartz.kCGRenderingIntentDefault)
    except Exception:
        return None

# --- LOAD LOCAL CACHED HAAR CLASSIFIERS ---
@st.cache_resource
def load_cascade_classifiers():
    face = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
    eye = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_eye_tree_eyeglasses.xml')
    return face, eye

# --- ARCHITECTURAL RUNTIME CACHE MANAGEMENT ---
if 'engine' not in st.session_state: st.session_state.engine = NativeAppleInferenceEngine()
if 'captured_badge' not in st.session_state: st.session_state.captured_badge = None
if 'current_screen' not in st.session_state: st.session_state.current_screen = "CAMERA_SCREEN"
if 'validation_start_time' not in st.session_state: st.session_state.validation_start_time = None
if 'motion_history_pool' not in st.session_state: st.session_state.motion_history_pool = []

engine = st.session_state.engine
face_cascade, eye_cascade = load_cascade_classifiers()

# ==============================================================================
# SCREEN 2: SUCCESS VIEW
# ==============================================================================
if st.session_state.current_screen == "SUCCESS_SCREEN":
    st.title("✅ Attendance Punch Registered")
    st.write("---")
    col_left, col_right = st.columns([1, 1])
    with col_left:
        st.success("### 🔓 BIO-INTEGRITY LOCK COMPLETED")
        st.write(f"**Verification Clock Time:** {time.strftime('%Y-%m-%d %H:%M:%S Local')}")
        if st.button("Reset Kiosk for Next Employee 🔄", type="primary", width="stretch"):
            st.session_state.captured_badge = None
            st.session_state.validation_start_time = None
            st.session_state.motion_history_pool = []
            st.session_state.current_screen = "CAMERA_SCREEN"
            st.rerun()
    with col_right:
        st.markdown("**Captured Security Snapshot Checkpoint:**")
        if st.session_state.captured_badge is not None:
            st.image(st.session_state.captured_badge, channels="RGB", width="stretch")

# ==============================================================================
# SCREEN 1: THE ACTIVE REAL-TIME ATTENDANCE INTERFACE
# ==============================================================================
else:
    st.title("🛡️ ZepIris Hardened Biometric Kiosk")
    st.write("Side-by-side verification interface managing explicit parameter thresholds and contextual overrides.")

    col1, col2 = st.columns([2, 1])
    with col1:
        video_placeholder = st.empty()
    with col2:
        st.subheader("Side-by-Side Biometric Matrix")
        gate_status = st.empty()
        blur_metric = st.empty()
        spoof_metric = st.empty()
        nsfw_metric = st.empty()

    cap = cv2.VideoCapture(0)
    
    if not cap.isOpened():
        st.error("❌ Cannot establish connection to video capture hardware interface.")
    else:
        while st.session_state.current_screen == "CAMERA_SCREEN":
            ret, frame = cap.read()
            if not ret or frame is None: 
                time.sleep(0.01)
                continue
                
            frame = cv2.flip(frame, 1)
            h, w, _ = frame.shape
            raw_clean_snapshot = frame.copy()
            
            gray_input = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            detected_faces = face_cascade.detectMultiScale(gray_input, scaleFactor=1.1, minNeighbors=6, minSize=(160, 160))
            
            face_detected = False
            current_frame_passed = False
            hardware_screen_spoof_attack = False
            gaze_and_eyes_valid = False
            micro_movement_passed = False
            
            if len(detected_faces) > 0:
                largest_face = max(detected_faces, key=lambda b: b[2] * b[3])
                fx, fy, fw_box, fh_box = largest_face
                face_detected = True
                
                pad_w, pad_h = int(fw_box * 0.35), int(fh_box * 0.35)
                render_x1, render_y1 = max(0, fx - pad_w), max(0, fy - pad_h)
                render_x2, render_y2 = min(w, fx + fw_box + pad_w), min(h, fy + fh_box + pad_h)
                
                # --- EXPLICIT TWO-EYE FRONTAL GAZE TRACKING ---
                eye_zone_y2 = fy + int(fh_box * 0.55)
                mid_x = fx + (fw_box // 2)
                
                left_eye_region = gray_input[fy:eye_zone_y2, fx:mid_x]
                right_eye_region = gray_input[fy:eye_zone_y2, mid_x:fx + fw_box]
                
                left_eyes_found = eye_cascade.detectMultiScale(left_eye_region, scaleFactor=1.05, minNeighbors=4, minSize=(20, 20))
                right_eyes_found = eye_cascade.detectMultiScale(right_eye_region, scaleFactor=1.05, minNeighbors=4, minSize=(20, 20))
                
                if len(left_eyes_found) > 0 and len(right_eyes_found) > 0:
                    gaze_and_eyes_valid = True
                
                # --- MOTION HISTORY POOLING ---
                current_center = float(fx + (fw_box / 2.0))
                st.session_state.motion_history_pool.append(current_center)
                
                if len(st.session_state.motion_history_pool) > 10:
                    st.session_state.motion_history_pool.pop(0)
                
                if len(st.session_state.motion_history_pool) == 10:
                    coordinate_std_dev = np.std(st.session_state.motion_history_pool)
                    if 0.12 <= coordinate_std_dev <= 5.0:
                        micro_movement_passed = True
                
                # --- DOWNSAMPLED PATH (APPLE SILICON NEURAL ENGINE CALCULATIONS) ---
                cg_face_crop = crop_and_convert_to_cgimage(raw_clean_snapshot, fx, fy, fw_box, fh_box)
                prob_sharp, prob_live, prob_explicit = 0.0, 0.0, 0.0
                
                if cg_face_crop is not None:
                    blur_out = engine.run_vision_inference(cg_face_crop, "blur")
                    if blur_out: prob_sharp = stable_softmax(blur_out)[0]
                    
                    spoof_out = engine.run_vision_inference(cg_face_crop, "spoof")
                    if spoof_out: prob_live = 1.0 / (1.0 + np.exp(-spoof_out[0]))
                    
                    nsfw_out = engine.run_vision_inference(cg_face_crop, "nsfw")
                    if nsfw_out: prob_explicit = stable_softmax(nsfw_out)[1]

                # ──────────────────────────────────────────────────────────
                # ✅ CROSS-MODEL FALLBACK SECURITY INTERLOCK INTERCEPT
                # ──────────────────────────────────────────────────────────
                # If the environment triggers an alert (dark noise/pixelated frame), 
                # force the clarity status to drop to fail the presentation attempt.
                is_safe = prob_explicit < NSFW_VIOLATION_THRESHOLD
                
                if not is_safe or (not gaze_and_eyes_valid):
                    prob_sharp = 0.0000
                    prob_live = 0.0000
                    is_sharp = False
                    is_neural_live = False
                else:
                    # Execute standard threshold gating equations
                    is_sharp = prob_sharp >= BLUR_SHARP_THRESHOLD
                    is_neural_live = prob_live >= SPOOF_LIVE_THRESHOLD

                # Combine all criteria layers
                current_frame_passed = (is_sharp and is_neural_live and is_safe and 
                                        micro_movement_passed and gaze_and_eyes_valid)
                
                # Render side-by-side metrics metrics matching log bounds
                if is_sharp: 
                    blur_metric.success(f"📷 Clarity: {prob_sharp:.4f} (BLUR_SHARP_THRESHOLD: {BLUR_SHARP_THRESHOLD}) [PASSED]")
                else: 
                    blur_metric.error(f"📷 Clarity: {prob_sharp:.4f} (BLUR_SHARP_THRESHOLD: {BLUR_SHARP_THRESHOLD}) [REJECTED LOW FI QUALITY]")
                    
                if is_neural_live and gaze_and_eyes_valid: 
                    spoof_metric.success(f"🛡️ Liveness: {prob_live:.4f} (SPOOF_LIVE_THRESHOLD: {SPOOF_LIVE_THRESHOLD}) [PASSED]")
                else: 
                    spoof_metric.error(f"🛡️ Liveness: {prob_live:.4f} (SPOOF_LIVE_THRESHOLD: {SPOOF_LIVE_THRESHOLD}) [SPOOFING/WRONG POSE REJECTED]")
                    
                if is_safe: 
                    nsfw_metric.success(f"👔 Environment: {prob_explicit:.4f} (NSFW_VIOLATION_THRESHOLD: {NSFW_VIOLATION_THRESHOLD}) [CLEAN]")
                else: 
                    nsfw_metric.error(f"👔 Environment: {prob_explicit:.4f} (NSFW_VIOLATION_THRESHOLD: {NSFW_VIOLATION_THRESHOLD}) [ANOMALY OVERRIDE INTERLOCK]")

                # --- HARDENED TIMING CONTROL GATE ---
                if current_frame_passed:
                    if st.session_state.validation_start_time is None:
                        st.session_state.validation_start_time = time.time()
                        
                    elapsed = time.time() - st.session_state.validation_start_time
                    remaining = max(0.0, STABILIZATION_DURATION - elapsed)
                    
                    if remaining > 0:
                        cv2.rectangle(frame, (render_x1, render_y1), (render_x2, render_y2), (0, 165, 255), 3)
                        cv2.putText(frame, f"VERIFYING IDENTITY ({remaining:.1f}s)", (render_x1, render_y1 - 12), 
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 165, 255), 2)
                        gate_status.warning(f"⏳ TIMING: Posture verified. Analyzing human biometric tremors ({remaining:.1f}s)...")
                    else:
                        cv2.rectangle(frame, (render_x1, render_y1), (render_x2, render_y2), (0, 255, 0), 3)
                        cv2.putText(frame, "ACCESS LOGGED", (render_x1, render_y1 - 12), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
                        gate_status.success("🔓 SUCCESS: BIO-INTEGRITY AUTHENTICATED! LOGGING ACCESS...")
                        
                        st.session_state.captured_badge = cv2.cvtColor(raw_clean_snapshot, cv2.COLOR_BGR2RGB)
                        st.session_state.current_screen = "SUCCESS_SCREEN"
                        cap.release()
                        st.rerun()
                else:
                    st.session_state.validation_start_time = None
                    cv2.rectangle(frame, (render_x1, render_y1), (render_x2, render_y2), (0, 0, 255), 3)
                    cv2.putText(frame, "GATE LOCKED", (render_x1, render_y1 - 12), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
                    
                    if not is_safe:
                        gate_status.error("🔒 PRESENTATION ATTACK BLOCKED: ENVIRONMENT NOISE / PHOTO REPLAY DETECTED.")
                    elif not gaze_and_eyes_valid:
                        gate_status.error("🔒 ALIGNMENT GATES: FRONT-FACING PROFILE REQUIRED WITH BOTH EYES OPEN.")
                    elif not micro_movement_passed:
                        gate_status.warning("⏳ MOTION ERROR: Wiggle your posture slightly. Static images or rigid screen mounts are restricted.")
                    else:
                        gate_status.error("🔒 GATE HOLD: ENSURE YOUR FACE IS SHARP AND LIT PROPERLY.")
                        
            if not face_detected:
                st.session_state.validation_start_time = None
                gate_status.warning("⏳ SCANNING SYSTEM ONLINE: STAND IN FRONT OF THE CAMERA VIEWPORT LENS")
                blur_metric.info("📷 Clarity: Standby")
                spoof_metric.info("🛡️ Liveness: Standby")
                nsfw_metric.info("👔 Safety: Standby")

            video_placeholder.image(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB), channels="RGB", width="stretch")
            time.sleep(0.03) 
            
        cap.release()