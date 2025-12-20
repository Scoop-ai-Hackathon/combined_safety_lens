import cv2
import mediapipe as mp
import streamlit as st
import numpy as np
import time
import os
from datetime import datetime

# Import MediaPipe Tasks API
from mediapipe.tasks import python
from mediapipe.tasks.python import vision

# --- Configuration & Setup ---
st.set_page_config(
    page_title="Safety Lens: Active Sentinel",
    page_icon="🛡️",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# --- Spoon OS Integration Stub ---
def trigger_spoon_agent_incident():
    """
    Mock function to simulate calling the Spoon Agent.
    In a real scenario, this would trigger an on-chain transaction or agent workflow.
    """
    timestamp = datetime.now().strftime("%H:%M:%S")
    print(f"[{timestamp}] 🚨 Wait! Calling Spoon Agent via x402 to log incident on-chain...")
    return f"Incident logged at {timestamp} via Spoon Agent (x402)"

def save_screenshot(frame):
    """
    Save a screenshot of the current frame when pause is triggered.
    Saves to the screenshots/ folder with a timestamped filename.
    """
    screenshots_dir = "screenshots"
    os.makedirs(screenshots_dir, exist_ok=True)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"incident_{timestamp}.jpg"
    filepath = os.path.join(screenshots_dir, filename)

    cv2.imwrite(filepath, frame)
    print(f"[{datetime.now().strftime('%H:%M:%S')}] 📸 Screenshot saved: {filepath}")
    return filepath

# --- Custom CSS for Animation & Styling ---
def load_css():
    st.markdown("""
        <style>
        /* Modern Dark Theme Adjustments */
        .stApp {
            background-color: #0e1117;
        }
        
        /* Status Badge Styling */
        .status-badge {
            padding: 1rem;
            border-radius: 10px;
            text-align: center;
            font-weight: bold;
            font-size: 24px;
            margin-bottom: 20px;
            transition: all 0.3s ease;
        }
        .status-safe {
            background-color: rgba(0, 255, 0, 0.1);
            color: #00ff00;
            border: 2px solid #00ff00;
            box-shadow: 0 0 15px rgba(0, 255, 0, 0.2);
        }
        .status-danger {
            background-color: rgba(255, 0, 0, 0.2);
            color: #ff0000;
            border: 2px solid #ff0000;
            box-shadow: 0 0 20px rgba(255, 0, 0, 0.5);
            animation: pulse 1s infinite;
        }
        
        /* Machine Animation */
        .machine-container {
            display: flex;
            justify-content: center;
            align-items: center;
            height: 200px;
            margin: 20px 0;
            background: rgba(255, 255, 255, 0.05);
            border-radius: 15px;
        }
        
        .gear-icon {
            font-size: 80px;
            display: inline-block;
        }
        
        .spinning {
            animation: spin 3s linear infinite;
        }
        
        .stopped {
            animation: none;
            opacity: 0.5;
            transform: scale(0.9);
        }
        
        @keyframes spin {
            0% { transform: rotate(0deg); }
            100% { transform: rotate(360deg); }
        }
        
        @keyframes pulse {
            0% { box-shadow: 0 0 0 0 rgba(255, 0, 0, 0.7); }
            70% { box-shadow: 0 0 0 20px rgba(255, 0, 0, 0); }
            100% { box-shadow: 0 0 0 0 rgba(255, 0, 0, 0); }
        }
        
        /* Log Area */
        .log-area {
            font-family: 'Courier New', monospace;
            font-size: 12px;
            background-color: #111;
            padding: 10px;
            border-radius: 5px;
            max-height: 200px;
            overflow-y: auto;
            border: 1px solid #333;
        }
        </style>
    """, unsafe_allow_html=True)

# --- MediaPipe Tasks Setup ---
@st.cache_resource
def get_detector():
    model_path = 'hand_landmarker.task'
    base_options = python.BaseOptions(model_asset_path=model_path)
    options = vision.HandLandmarkerOptions(
        base_options=base_options,
        num_hands=2,
        min_hand_detection_confidence=0.3, # Lowered for better recall in bad lighting
        min_hand_presence_confidence=0.3, # Lowered
        min_tracking_confidence=0.3,
        running_mode=vision.RunningMode.IMAGE # Using IMAGE mode for simplicity in the streamlit loop
    )
    detector = vision.HandLandmarker.create_from_options(options)
    return detector

def draw_landmarks_and_box(frame, detection_result):
    hand_detected = False
    
    if detection_result.hand_landmarks:
        hand_detected = True
        h, w, c = frame.shape
        
        for hand_landmarks in detection_result.hand_landmarks:
            # Draw Landmarks
            # We implemented simple drawing since existing drawing_utils are missing
            points = []
            for lm in hand_landmarks:
                pixel_x, pixel_y = int(lm.x * w), int(lm.y * h)
                points.append((pixel_x, pixel_y))
                cv2.circle(frame, (pixel_x, pixel_y), 2, (0, 255, 0), -1)
            
            # Simple connections (finger bones) logic is complex to hardcode perfectly, 
            # so we focus on the bounding box which is the key requirement.
            
            # Bounding Box Logic
            x_coords = [p[0] for p in points]
            y_coords = [p[1] for p in points]
            
            x_min, x_max = min(x_coords), max(x_coords)
            y_min, y_max = min(y_coords), max(y_coords)
            
            margin = 30
            x_min = max(0, x_min - margin)
            y_min = max(0, y_min - margin)
            x_max = min(w, x_max + margin)
            y_max = min(h, y_max + margin)
            
            cv2.rectangle(frame, (x_min, y_min), (x_max, y_max), (0, 0, 255), 4)

    return hand_detected

# --- Main Application ---
def main():
    load_css()
    
    # Session State Initialization
    if 'monitor_active' not in st.session_state:
        st.session_state.monitor_active = False
    if 'current_status' not in st.session_state:
        st.session_state.current_status = "SAFE"
    if 'incident_log' not in st.session_state:
        st.session_state.incident_log = []
    if 'pinch_frame_count' not in st.session_state:
        st.session_state.pinch_frame_count = 0
    if 'impact_frame_count' not in st.session_state:
        st.session_state.impact_frame_count = 0
    if 'hazard_count_l' not in st.session_state:
        st.session_state.hazard_count_l = 0
    if 'hazard_count_r' not in st.session_state:
        st.session_state.hazard_count_r = 0
    if 'hazard_count_t' not in st.session_state:
        st.session_state.hazard_count_t = 0
    if 'hazard_count_b' not in st.session_state:
        st.session_state.hazard_count_b = 0
    if 'hazard_latched' not in st.session_state:
        st.session_state.hazard_latched = False
    
    # Header
    st.title("🛡️ Safety Lens: Active Sentinel")
    st.markdown("### Real-time Factory Safety & Incident Response")
    
    # Layout using Columns
    col_video, col_dashboard = st.columns([2, 1])
    
    with col_dashboard:
        # Controls
        if not st.session_state.monitor_active:
            # The logs showed warnings likely from image updates loop. We will try to be safe.
            if st.button("▶️ START SYSTEM", type="primary", help="Start the safety monitoring system"):
                st.session_state.monitor_active = True
                st.session_state.current_status = "SAFE"
                # Reset all counters on fresh start
                st.session_state.pinch_frame_count = 0
                st.session_state.impact_frame_count = 0
                st.session_state.hazard_count_l = 0
                st.session_state.hazard_count_r = 0
                st.session_state.hazard_count_t = 0
                st.session_state.hazard_count_b = 0
                st.session_state.hazard_latched = False
                st.rerun()
        else:
            if st.button("⏹️ EMERGENCY STOP / RESET", type="secondary", help="Stop the system and reset state"):
                st.session_state.monitor_active = False
                st.session_state.current_status = "SAFE"
                st.session_state.hazard_latched = False
                st.rerun()

        st.markdown("---")
        st.markdown("#### 🏭 Machine Status")
        
        # Placeholders for dynamic updates
        vehicle_status_ph = st.empty()
        machine_anim_ph = st.empty()
        
        st.markdown("#### 📋 Incident Log")
        log_ph = st.empty()

    # Video Placeholder in Left Column
    # Video Placeholder in Left Column
    with col_video:
        video_ph = st.empty()
        if not st.session_state.monitor_active:
            if st.session_state.get("current_status") == "DANGER":
                video_ph.error("🚨 EMERGENCY STOP TRIGGERED. SYSTEM HALTED. PRESS START TO RESET.")
                vehicle_status_ph.markdown(
                    '<div class="status-badge status-danger">🚫 STOPPED</div>', 
                    unsafe_allow_html=True
                )
                machine_anim_ph.markdown(
                    '<div class="machine-container"><div class="gear-icon stopped">⚙️</div></div>', 
                    unsafe_allow_html=True
                )
                
                # --- AUDIO ALARM ---
                import base64
                try:
                    with open("alarm.wav", "rb") as f:
                        audio_bytes = f.read()
                    audio_b64 = base64.b64encode(audio_bytes).decode()
                    audio_html = f"""
                        <audio autoplay>
                            <source src="data:audio/wav;base64,{audio_b64}" type="audio/wav">
                        </audio>
                    """
                    st.markdown(audio_html, unsafe_allow_html=True)
                except Exception as e:
                    print(f"Audio Error: {e}")
                    
            else:
                video_ph.info("System is OFFLINE. Press START to active Active Sentinel.")
                
                # Show static machine state offline
                vehicle_status_ph.markdown(
                    '<div class="status-badge" style="border-color: #666; color: #666;">OFFLINE</div>', 
                    unsafe_allow_html=True
                )
                machine_anim_ph.markdown(
                    '<div class="machine-container"><div class="gear-icon stopped">⚙️</div></div>', 
                    unsafe_allow_html=True
                )
            return

    # --- Sidebar Configuration ---
    st.sidebar.title("⚙️ System Config")
    camera_index = st.sidebar.number_input("Camera Index", min_value=0, max_value=10, value=0, step=1, help="Change this if webcam doesn't load (try 0, 1, or 2).")
    
    # Debug info
    st.sidebar.markdown("---")
    st.sidebar.markdown("### 🛠️ Debug Info")
    fps_ph = st.sidebar.empty()
    frame_count_ph = st.sidebar.empty()
    status_ph = st.sidebar.empty()
    
    # --- Active Monitoring Loop ---
    # Use selected camera index
    print(f"Attempting to open camera index {camera_index} with CAP_AVFOUNDATION...")
    cap = cv2.VideoCapture(camera_index, cv2.CAP_AVFOUNDATION)
    
    # Check if opened successfully
    if not cap.isOpened():
        print(f"Failed to open camera index {camera_index}")
        st.error(f"🚨 Error: Could not access webcam at Index {camera_index}. Please try a different index in the sidebar.")
        st.session_state.monitor_active = False
        # Stop background animation if camera fails initially
        video_ph.markdown("### Camera Failed. check settings.")
        return

    print("Camera opened successfully. Loading model...")
    try:
        detector = get_detector()
        print("Model loaded successfully.")
    except Exception as e:
        # Fallback to ensure users know what to do
        st.error(f"Failed to load MediaPipe model: {e}")
        st.info("Ensure 'hand_landmarker.task' is present in the directory.")
        return
    
    frame_count = 0
    start_time = time.time()
    prev_gray = None
    prev_hand_area = None # For compression detection
    prev_hand_centroid = None
    
    # Signal Buffers (For "Around the same time" logic)
    motion_buffer = 0
    squeeze_buffer = 0

    status_ph.info("Starting loop...")

    try:
        while st.session_state.monitor_active:
            ret, frame = cap.read()
            if not ret:
                st.error(f"Failed to retrieve frame from Camera {camera_index}.")
                status_ph.error("Frame retrieval failed.")
                break
            
            # Reduce resolution for Optical Flow speed
            h, w, c = frame.shape
            small_w, small_h = 320, 240
            frame_small = cv2.resize(frame, (small_w, small_h))
            gray_small = cv2.cvtColor(frame_small, cv2.COLOR_BGR2GRAY)
            
            frame_count += 1
            if frame_count % 30 == 0:
                elapsed = time.time() - start_time
                fps = frame_count / elapsed
                fps_ph.write(f"FPS: {fps:.1f}")
                frame_count_ph.write(f"Frames: {frame_count}")
            
            # --- LATCHED HAZARD STATE ---
            if st.session_state.get('hazard_latched'):
                  # Update Dashboard UI to Stopped State
                  vehicle_status_ph.markdown(
                      '<div class="status-badge status-danger">🚫 STOPPED</div>', 
                      unsafe_allow_html=True
                  )
                  machine_anim_ph.markdown(
                      '<div class="machine-container"><div class="gear-icon stopped">⚙️</div><div style="text-align:center; color:red; font-weight:bold;">SYSTEM HALTED</div></div>', 
                      unsafe_allow_html=True
                  )

                  cv2.putText(frame, "SYSTEM HALTED", (50, 100), cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0,0,255), 4)
                  cv2.putText(frame, "PRESS 'RESET' TO RESUME", (50, 160), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0,0,255), 3)
                  
                  # Add a red border
                  cv2.rectangle(frame, (0,0), (w,h), (0,0,255), 20)
                  
                  final_display_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                  video_ph.image(final_display_rgb, channels="RGB")
                 
                  # Optional: Small sleep to reduce CPU usage in halted state
                  time.sleep(0.03) 
                  continue

            # 1. Calculate Optical Flow (if we have a previous frame)
            pinch_detected = False
            avg_u_left = 0
            avg_u_right = 0
            
            if prev_gray is not None:
                # Farneback Optical Flow on DOWN-SAMPLED image for speed
                # Parameters optimized for speed: iterations=1, poly_n=5, etc.
                flow = cv2.calcOpticalFlowFarneback(prev_gray, gray_small, None, 0.5, 2, 10, 2, 5, 1.1, 0)
                # flow[..., 0] is x-velocity (u), flow[..., 1] is y-velocity (v)
            
            prev_gray = gray_small
                
            # 2. Hand Detection (Can run on full frame or small, keeping full for accuracy)
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=frame_rgb)
            detection_result = detector.detect(mp_image)
            
            # 3. Process Logic
            hand_box = None
            hand_detected = False
            
            # Print debug info ever 60 frames to avoid spam
            if frame_count % 60 == 0:
                print(f"Frame {frame_count}: Hand Detected? {bool(detection_result.hand_landmarks)}")
                print(f"Frame {frame_count}: Flow in locals? {'flow' in locals()}")
                print(f"Frame Stats: Mean={np.mean(frame):.2f} Max={np.max(frame)} Shape={frame.shape}")
            
            # Scaling factors for mapping full-res box to small-res flow
            sx = small_w / w
            sy = small_h / h

            if detection_result.hand_landmarks:
                hand_detected = True
                for hand_landmarks in detection_result.hand_landmarks:
                    # Basic drawing of landmarks on FULL frame
                    x_coords = [int(lm.x * w) for lm in hand_landmarks]
                    y_coords = [int(lm.y * h) for lm in hand_landmarks]
                    
                    x_min, x_max = min(x_coords), max(x_coords)
                    y_min, y_max = min(y_coords), max(y_coords)
                    
                    # Margin for the Bounding Box
                    margin = 30
                    x_min = max(0, x_min - margin)
                    y_min = max(0, y_min - margin)
                    x_max = min(w, x_max + margin)
                    y_max = min(h, y_max + margin)
                    
                    cv2.rectangle(frame, (x_min, y_min), (x_max, y_max), (0, 255, 0), 2)
                    
                    # Store box for Optical Flow analysis
                    hand_box = (x_min, y_min, x_max, y_max)
                    
                    # --- Compression Detection (Rapid Area Shrink) ---
                    current_raw_area = (x_max - x_min) * (y_max - y_min)
                    compression_detected = False
                    
                    # Exponential Moving Average for Smoothing (Reduce jitter)
                    # Use 50% from history, 50% new
                    if prev_hand_area is not None:
                         current_area = 0.5 * current_raw_area + 0.5 * prev_hand_area
                    else:
                         current_area = current_raw_area
                    
                    if prev_hand_area is not None:
                        # Calculate % change
                        delta = current_area - prev_hand_area
                        percent_change = delta / prev_hand_area
                        
                        # Debug Area
                        cv2.putText(frame, f"Area: {int(current_area)} ({percent_change*100:.1f}%)", (x_min, y_max+20), 
                                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
                                   
                        # Threshold: If shrinks by > 5% in one frame (tuned from 7%)
                        # Note: -0.05 means 5% reduction - filters out jitter
                        if percent_change < -0.05: 
                             compression_detected = True
                             cv2.putText(frame, "SQUEEZE!", (x_min, y_min-30), 
                                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
                        
                        # Log significant changes for debug
                        if abs(percent_change) > 0.05:
                             print(f"Frame {frame_count}: Area Delta {percent_change:.3f}")
                    
                    prev_hand_area = current_area
                    
                    # --- Pinch Detection Logic (Optical Flow) ---
                    if 'flow' in locals():
                        # Map Hand Box to Small Flow Coordinates
                        s_x_min = int(x_min * sx)
                        s_x_max = int(x_max * sx)
                        s_y_min = int(y_min * sy)
                        s_y_max = int(y_max * sy)

                        # Define ROIs for Flow Analysis (in Small Coords)
                        # Left ROI: Region strictly to the left of the hand box
                        roi_w = int(100 * sx) # Tighter ROI for focus
                        l_x1 = max(0, s_x_min - roi_w)
                        l_x2 = s_x_min
                        
                        # Right ROI
                        r_x1 = s_x_max
                        r_x2 = min(small_w, s_x_max + roi_w)
                        
                        # Thresholds (TUNED FOR ROBUSTNESS)
                        velocity_thresh = 4.0 # Increased from 3.0 to ignore hand drift
                        pixel_thresh = 400    # Increased from 200 to ignore small artifacts
                        
                        # --- HAND MOVEMENT SUPPRESSION ---
                        # Calculate Hand Centroid Speed to ignore self-induced flow
                        current_centroid = ((x_min + x_max)//2, (y_min + y_max)//2)
                        hand_speed = 0.0
                        suppress_hazards = False
                        
                        if 'prev_hand_centroid' in locals() and prev_hand_centroid is not None:
                             dist = np.sqrt((current_centroid[0]-prev_hand_centroid[0])**2 + 
                                          (current_centroid[1]-prev_hand_centroid[1])**2)
                             hand_speed = dist
                             
                             # If hand moves > 15 pixels/frame, assume flow is self-induced
                             if hand_speed > 15.0:
                                  suppress_hazards = True
                                  cv2.putText(frame, "HAND MOVING - SUPPRESSED", (50, 230), 
                                             cv2.FONT_HERSHEY_SIMPLEX, 0.6, (100,255,100), 2)
                        
                        prev_hand_centroid = current_centroid
                        
                        # Logic: Count pixels moving TOWARDS hand
                        moving_pixels_left = 0
                        moving_pixels_right = 0
                        moving_pixels_top = 0
                        moving_pixels_bottom = 0
                        
                        # --- HORIZONTAL CHECK ---
                        # Pre-calculate counts for cancellation logic
                        moving_pixels_left = 0      # Left ROI moving Right (+)
                        moving_pixels_left_away = 0 # Left ROI moving Left (-)
                        moving_pixels_right = 0     # Right ROI moving Left (-)
                        moving_pixels_right_away = 0# Right ROI moving Right (+)
                        
                        # Left ROI Analysis
                        if l_x2 > l_x1:
                            u_left = flow[s_y_min:s_y_max, l_x1:l_x2, 0]
                            moving_pixels_left = np.count_nonzero(u_left > velocity_thresh)
                            moving_pixels_left_away = np.count_nonzero(u_left < -velocity_thresh)

                        # Right ROI Analysis
                        if r_x2 > r_x1:
                            u_right = flow[s_y_min:s_y_max, r_x1:r_x2, 0]
                            moving_pixels_right = np.count_nonzero(u_right < -velocity_thresh)
                            moving_pixels_right_away = np.count_nonzero(u_right > velocity_thresh)
                            
                        # GLOBAL TRANSLATION CANCELLATION
                        # If Pixels Left move Right AND Pixels Right move Right -> Hand is moving Right -> IGNORE LEFT HAZARD
                        translation_right = (moving_pixels_left > pixel_thresh) and (moving_pixels_right_away > pixel_thresh)
                        
                        # If Pixels Right move Left AND Pixels Left move Left -> Hand is moving Left -> IGNORE RIGHT HAZARD
                        translation_left = (moving_pixels_right > pixel_thresh) and (moving_pixels_left_away > pixel_thresh)

                        # Determine ACTIVE states with cancellation AND suppression
                        left_active = (moving_pixels_left > pixel_thresh) and not translation_right and not suppress_hazards
                        right_active = (moving_pixels_right > pixel_thresh) and not translation_left and not suppress_hazards

                        # Visualize Left
                        d_x1, d_x2 = int(l_x1 / sx), int(l_x2 / sx)
                        color_l = (0, 0, 255) if left_active else (200, 200, 200)
                        if translation_right: color_l = (0, 255, 0) # Green if ignored due to translation
                        thickness_l = 3 if left_active else 1
                        cv2.rectangle(frame, (d_x1, y_min), (d_x2, y_max), color_l, thickness_l)
                        if left_active:
                             cv2.putText(frame, f"SIDE! ({moving_pixels_left})", (d_x1, y_min-10), 
                                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,0,255), 2)
                        
                        # Visualize Right
                        d_x1r, d_x2r = int(r_x1 / sx), int(r_x2 / sx)
                        color_r = (0, 0, 255) if right_active else (200, 200, 200)
                        if translation_left: color_r = (0, 255, 0) # Green if ignored
                        thickness_r = 3 if right_active else 1
                        cv2.rectangle(frame, (d_x1r, y_min), (d_x2r, y_max), color_r, thickness_r)
                        if right_active:
                             cv2.putText(frame, f"SIDE! ({moving_pixels_right})", (d_x1r, y_min-10), 
                                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,0,255), 2)
                                           
                        # --- VERTICAL CHECK ---
                        # Top ROI: Region strictly ABOVE the hand box
                        roi_h = int(100 * sy)
                        t_y1 = max(0, s_y_min - roi_h)
                        t_y2 = s_y_min
                        
                        # Bottom ROI: Region strictly BELOW
                        b_y1 = s_y_max
                        b_y2 = min(small_h, s_y_max + roi_h)
                        
                        moving_pixels_top = 0       # Top ROI moving Down (+)
                        moving_pixels_top_away = 0  # Top ROI moving Up (-)
                        moving_pixels_bottom = 0    # Bottom ROI moving Up (-)
                        moving_pixels_bottom_away = 0 # Bottom ROI moving Down (+)

                        # Top Analysis
                        if t_y2 > t_y1:
                            v_top = flow[t_y1:t_y2, s_x_min:s_x_max, 1]
                            moving_pixels_top = np.count_nonzero(v_top > velocity_thresh)
                            moving_pixels_top_away = np.count_nonzero(v_top < -velocity_thresh)

                        # Bottom Analysis
                        if b_y2 > b_y1:
                            v_bottom = flow[b_y1:b_y2, s_x_min:s_x_max, 1]
                            moving_pixels_bottom = np.count_nonzero(v_bottom < -velocity_thresh)
                            moving_pixels_bottom_away = np.count_nonzero(v_bottom > velocity_thresh)

                        # Check Translation
                        # Moving Down: Top moves Down AND Bottom moves Down
                        translation_down = (moving_pixels_top > pixel_thresh) and (moving_pixels_bottom_away > pixel_thresh)
                        # Moving Up: Bottom moves Up AND Top moves Up
                        translation_up = (moving_pixels_bottom > pixel_thresh) and (moving_pixels_top_away > pixel_thresh)
                        
                        # Determine Active
                        top_active = (moving_pixels_top > pixel_thresh) and not translation_down and not suppress_hazards
                        bottom_active = (moving_pixels_bottom > pixel_thresh) and not translation_up and not suppress_hazards

                        # Visualize Top
                        d_y1, d_y2 = int(t_y1 / sy), int(t_y2 / sy)
                        color_t = (0, 0, 255) if top_active else (200, 200, 200)
                        if translation_down: color_t = (0, 255, 0)
                        thickness_t = 3 if top_active else 1
                        cv2.rectangle(frame, (x_min, d_y1), (x_max, d_y2), color_t, thickness_t)
                        if top_active:
                            cv2.putText(frame, f"TOP! ({moving_pixels_top})", (x_min, d_y1-10),
                                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,0,255), 2)

                        # Visualize Bottom
                        d_y1b, d_y2b = int(b_y1 / sy), int(b_y2 / sy)
                        color_b = (0, 0, 255) if bottom_active else (200, 200, 200)
                        if translation_up: color_b = (0, 255, 0)
                        thickness_b = 3 if bottom_active else 1
                        cv2.rectangle(frame, (x_min, d_y1b), (x_max, d_y2b), color_b, thickness_b)
                        if bottom_active:
                            cv2.putText(frame, f"BOT! ({moving_pixels_bottom})", (x_min, d_y2b+20),
                                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,0,255), 2)
                        
                        # PINCH CONDITION (SMART TRIGGER - BUFFERED):
                        
                        # 1. Update Buffers (Latch the signals)
                        any_motion_active = left_active or right_active or top_active or bottom_active
                        if any_motion_active:
                            motion_buffer = 5 # Reduced from 8: Tighter "simultaneity" window
                        
                        if compression_detected:
                            squeeze_buffer = 5 # Reduced from 8
                            
                        # Decay buffers
                        motion_buffer = max(0, motion_buffer - 1)
                        squeeze_buffer = max(0, squeeze_buffer - 1)
                        
                        # 2. Combined Trigger (Buffered)
                        # If start motion AND squeeze happen roughly together
                        buffered_trigger = (motion_buffer > 0) and (squeeze_buffer > 0)
                        
                        # Debug Text on Frame
                        cv2.putText(frame, f"Buf Mot:{motion_buffer} Sq:{squeeze_buffer} Trig:{buffered_trigger}", (50, 80), 
                                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 2)
                        
                        # Logic A: Sustained Slow Pinch (Strict Pincer)
                        strict_pincer = (left_active and right_active) or (top_active and bottom_active)
                        if strict_pincer:
                            if 'pinch_frame_count' not in st.session_state:
                                st.session_state.pinch_frame_count = 0
                            st.session_state.pinch_frame_count += 1
                        else:
                            st.session_state.pinch_frame_count = 0 # Strict reset for slow pinch
                        
                        # Logic B: Buffered Impact (Leaky Bucket)
                        if 'impact_frame_count' not in st.session_state:
                                st.session_state.impact_frame_count = 0
                                
                        if buffered_trigger:
                             st.session_state.impact_frame_count += 1
                             # Warning Text
                             cv2.putText(frame, f"IMPACT ACCUMULATING... {st.session_state.impact_frame_count}", (50, 110), 
                                       cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0,165,255), 2)
                        else:
                             # Leaky Bucket: Decay slowly instead of hard reset
                             # This allows for "2 frames" gap that user mentioned
                             st.session_state.impact_frame_count = max(0, st.session_state.impact_frame_count - 1)
                        
                        # Logic C: "Any Hazard" Aggressive Trigger - PER SIDE
                        # User Request: "More or less 5 frames close to each other", summed separately.
                        # Implementation: Leaky Bucket with +2 Increment / -1 Decay.
                        # Threshold 10 => Equivalent to 5 consecutive frames, but allows gaps.
                        
                        if 'hazard_count_l' not in st.session_state: st.session_state.hazard_count_l = 0
                        if 'hazard_count_r' not in st.session_state: st.session_state.hazard_count_r = 0
                        if 'hazard_count_t' not in st.session_state: st.session_state.hazard_count_t = 0
                        if 'hazard_count_b' not in st.session_state: st.session_state.hazard_count_b = 0
                        
                        # Left
                        if left_active: st.session_state.hazard_count_l += 2
                        else: st.session_state.hazard_count_l = max(0, st.session_state.hazard_count_l - 1)
                        
                        # Right
                        if right_active: st.session_state.hazard_count_r += 2
                        else: st.session_state.hazard_count_r = max(0, st.session_state.hazard_count_r - 1)
                        
                        # Top
                        if top_active: st.session_state.hazard_count_t += 2
                        else: st.session_state.hazard_count_t = max(0, st.session_state.hazard_count_t - 1)

                        # Bottom
                        if bottom_active: st.session_state.hazard_count_b += 2
                        else: st.session_state.hazard_count_b = max(0, st.session_state.hazard_count_b - 1)
                        
                        # Max threat level
                        max_hazard = max(st.session_state.hazard_count_l, 
                                         st.session_state.hazard_count_r,
                                         st.session_state.hazard_count_t,
                                         st.session_state.hazard_count_b)
                        
                        # Visualize Hazard Building
                        if max_hazard > 0:
                             # Display the count to the user so they can see it building up
                             cv2.putText(frame, f"HAZARD LVL: {max_hazard}/10", (50, 170), 
                                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,165,255), 2)
                             
                        if max_hazard >= 10: # Threshold 10 (approx 5 detect frames @ +2 weight)
                            pinch_detected = True
                            cv2.putText(frame, "FAST SIDE HAZARD", (50, 200), 
                                       cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0,0,255), 2)
                        
                        # Logic A Result (Pincer Only - Fast Trigger)
                        if st.session_state.pinch_frame_count > 3: # Was 5
                            pinch_detected = True
                            cv2.putText(frame, "PINCH DETECTED", (50, 140), 
                                       cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0,0,255), 2)
                        
                        # Logic B Result (Trigger threshold)
                        # Increased to 12 (approx 0.5-0.6s) to be less "trigger happy"
                        if st.session_state.impact_frame_count > 12: 
                            pinch_detected = True
                            cv2.putText(frame, "IMPACT CONFIRMED", (50, 140), 
                                       cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0,0,255), 2)

            # --- Update State Logic ---

            # --- Update State Logic ---
            previous_status = st.session_state.current_status
            
            # Logic: If Pinch Detected -> EMERGENCY STOP
            #        If Just Hand -> WARNING (Yellow?) or Safe for now (per user request: "detect when metal is close")
            
            if pinch_detected:
                st.session_state.current_status = "DANGER"
                cv2.putText(frame, "CRUSH HAZARD DETECTED!", (50, 150), 
                           cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 4)
                cv2.putText(frame, "EMERGENCY STOP", (50, 100), 
                           cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 4)
                
                # Draw big visual indicators of the squeeze
                if hand_box:
                     cv2.line(frame, (hand_box[0]-50, hand_box[1]+50), (hand_box[0], hand_box[1]+50), (0,0,255), 5) # Arrow logic simplified
                     cv2.line(frame, (hand_box[2]+50, hand_box[1]+50), (hand_box[2], hand_box[1]+50), (0,0,255), 5)
                
                if previous_status == "SAFE":
                    log_msg = trigger_spoon_agent_incident()
                    st.session_state.incident_log.insert(0, log_msg)
                
                status_ph.error("🚨 CRUSH HAZARD! SYSTEM HALTED.")
                
                # LATCHING STOP: Set Latched Flag
                st.session_state.hazard_latched = True
                
                # Update visuals immediately
                final_frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                video_ph.image(final_frame_rgb, channels="RGB")
                continue
                
            elif hand_detected:
                # If Hand is present but no pinch, we can call it "WARNING" or keep SAFE
                # For this demo, user said "detect when metal is close". 
                # Let's keep it SAFE until the squeeze happens, or maybe YELLOW warning.
                st.session_state.current_status = "SAFE" 
                cv2.putText(frame, "Hand Detected - Monitoring...", (50, 50), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2)
                status_ph.warning("Hand Monitored.")
            else:
                st.session_state.current_status = "SAFE"
                cv2.putText(frame, "ZONE CLEAR - NO HAND DETECTED", (50, 50), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.8, (100, 255, 100), 2)
                status_ph.success("Zone Clear.")

            # --- Render Dashboard UI (Animation & Badges) ---
            if st.session_state.current_status == "SAFE":
                vehicle_status_ph.markdown(
                    '<div class="status-badge status-safe">OPERATIONAL</div>', 
                    unsafe_allow_html=True
                )
                machine_anim_ph.markdown(
                    '<div class="machine-container"><div class="gear-icon spinning">⚙️</div></div>', 
                    unsafe_allow_html=True
                )
            else:
                vehicle_status_ph.markdown(
                    '<div class="status-badge status-danger">🚫 STOPPED</div>', 
                    unsafe_allow_html=True
                )
                machine_anim_ph.markdown(
                    '<div class="machine-container"><div class="gear-icon stopped">⚙️</div></div>', 
                    unsafe_allow_html=True
                )

            # Update Log
            log_html = "<br>".join([f"• {msg}" for msg in st.session_state.incident_log[:5]])
            log_ph.markdown(f'<div class="log-area">{log_html}</div>', unsafe_allow_html=True)

            # Display Video
            # Using simple display first to ensure it works
            # Convert the MODIFIED frame (with drawings) to RGB for display
            final_display_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            video_ph.image(final_display_rgb, channels="RGB")
            
            # Remove explicit sleep if not needed, loop drives via camera speed
            # time.sleep(0.01)

    finally:
        if 'cap' in locals() and cap.isOpened():
            cap.release()
        
if __name__ == "__main__":
    main()
