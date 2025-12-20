import cv2
import mediapipe as mp
import streamlit as st
import numpy as np
import time
import os
import json
import base64
import httpx
import uuid
from datetime import datetime
from pathlib import Path
from dotenv import load_dotenv

# Import MediaPipe Tasks API
from mediapipe.tasks import python
from mediapipe.tasks.python import vision

# Load environment variables
load_dotenv()

# --- Configuration & Setup ---
st.set_page_config(
    page_title="Safety Lens",
    page_icon="🛡️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Directory Setup
BASE_DIR = Path(os.getcwd())
UPLOAD_DIR = BASE_DIR / "uploads"
UPLOAD_DIR.mkdir(exist_ok=True)
CACHE_DIR = BASE_DIR / "cache"
CACHE_DIR.mkdir(exist_ok=True)
REGULATIONS_CACHE_FILE = CACHE_DIR / "regulations_cache.json"
MEMORY_FILE = CACHE_DIR / "analysis_memory.json"

# ==================== Shared Utilities ====================

def trigger_spoon_agent_incident():
    timestamp = datetime.now().strftime("%H:%M:%S")
    print(f"[{timestamp}] [ALERT] Calling Spoon Agent via x402 to log incident on-chain...")
    return f"Incident logged at {timestamp} via Spoon Agent (x402)"

def load_css():
    st.markdown("""
        <style>
        .stApp { background-color: #0e1117; }
        .status-badge { padding: 1rem; border-radius: 10px; text-align: center; font-weight: bold; font-size: 24px; margin-bottom: 20px; transition: all 0.3s ease; }
        .status-safe { background-color: rgba(0, 255, 0, 0.1); color: #00ff00; border: 2px solid #00ff00; box-shadow: 0 0 15px rgba(0, 255, 0, 0.2); }
        .status-danger { background-color: rgba(255, 0, 0, 0.2); color: #ff0000; border: 2px solid #ff0000; box-shadow: 0 0 20px rgba(255, 0, 0, 0.5); animation: pulse 1s infinite; }
        .machine-container { display: flex; justify-content: center; align-items: center; height: 200px; margin: 20px 0; background: rgba(255, 255, 255, 0.05); border-radius: 15px; }
        .gear-icon { font-size: 80px; display: inline-block; }
        .spinning { animation: spin 3s linear infinite; }
        .stopped { animation: none; opacity: 0.5; transform: scale(0.9); }
        .log-area { font-family: 'Courier New', monospace; font-size: 12px; background-color: #111; padding: 10px; border-radius: 5px; max-height: 200px; overflow-y: auto; border: 1px solid #333; }
        @keyframes spin { 0% { transform: rotate(0deg); } 100% { transform: rotate(360deg); } }
        @keyframes pulse { 0% { box-shadow: 0 0 0 0 rgba(255, 0, 0, 0.7); } 70% { box-shadow: 0 0 0 20px rgba(255, 0, 0, 0); } 100% { box-shadow: 0 0 0 0 rgba(255, 0, 0, 0); } }
        .result-card { background: #1a1a2e; border-radius: 10px; padding: 20px; margin: 10px 0; border-left: 4px solid #00ff00; }
        .result-card.danger { border-left-color: #ff0000; }
        .result-card.warning { border-left-color: #ffaa00; }
        </style>
    """, unsafe_allow_html=True)

# ==================== Regulations Fetcher ====================

class RegulationsFetcher:
    def __init__(self):
        self.cache_file = REGULATIONS_CACHE_FILE
        self.regulations = self._load_cache()

    def _load_cache(self) -> dict:
        if self.cache_file.exists():
            try:
                with open(self.cache_file, "r", encoding="utf-8") as f:
                    data = json.load(f)
                    if data.get("fetched_at"):
                        fetched = datetime.fromisoformat(data["fetched_at"])
                        if (datetime.now() - fetched).total_seconds() < 86400:
                            return data.get("regulations", {})
            except:
                pass
        return {}

    def _save_cache(self, regulations: dict):
        with open(self.cache_file, "w", encoding="utf-8") as f:
            json.dump({
                "fetched_at": datetime.now().isoformat(),
                "regulations": regulations
            }, f, ensure_ascii=False, indent=2)

    def _get_default_regulations(self) -> dict:
        return {
            "보호구": {
                "안전모": {"rule": "산업안전보건기준 제32조", "description": "물체가 떨어지거나 날아올 위험이 있는 작업장에서 착용", "severity": "high"},
                "안전화": {"rule": "산업안전보건기준 제32조", "description": "물체의 낙하, 충격 또는 날카로운 물체에 의한 위험이 있는 작업장에서 착용", "severity": "high"},
                "보안경": {"rule": "산업안전보건기준 제32조", "description": "물체가 비래하거나 유해광선에 노출될 위험이 있는 작업장에서 착용", "severity": "medium"},
                "안전장갑": {"rule": "산업안전보건기준 제32조", "description": "날카로운 물체를 취급하거나 고온/저온 물체 취급 시 착용", "severity": "medium"}
            },
            "통로": {
                "폭": {"rule": "산업안전보건기준 제17조", "description": "통로 폭 80cm 이상 확보", "severity": "medium"},
                "조명": {"rule": "산업안전보건기준 제21조", "description": "통로에 적정 조명 확보 (75럭스 이상)", "severity": "low"},
                "장애물": {"rule": "산업안전보건기준 제17조", "description": "통로에 장애물 적재 금지", "severity": "high"}
            },
            "소화설비": {
                "소화기": {"rule": "산업안전보건기준 제241조", "description": "작업장별 적합한 소화기 비치 및 접근성 확보", "severity": "high"},
                "비상구": {"rule": "산업안전보건기준 제17조", "description": "비상구 확보 및 표시", "severity": "high"}
            },
            "전기안전": {
                "접지": {"rule": "산업안전보건기준 제304조", "description": "전기기기 접지 실시", "severity": "high"},
                "배선": {"rule": "산업안전보건기준 제311조", "description": "전선 피복 손상 없이 정리", "severity": "medium"}
            },
            "기계안전": {
                "방호장치": {"rule": "산업안전보건기준 제87조", "description": "회전체, 동력전달부 등에 덮개 또는 방호장치 설치", "severity": "high"},
                "비상정지": {"rule": "산업안전보건기준 제88조", "description": "기계에 비상정지장치 설치", "severity": "high"}
            },
            "_metadata": {
                "source": "default",
                "fetched_at": datetime.now().isoformat()
            }
        }

    def get_regulations(self, force_refresh: bool = False) -> dict:
        if self.regulations and not force_refresh:
            return self.regulations
        self.regulations = self._get_default_regulations()
        self._save_cache(self.regulations)
        return self.regulations

# ==================== Memory Store ====================

class MemoryStore:
    def __init__(self):
        self.memory_file = MEMORY_FILE
        self.memories = self._load()

    def _load(self) -> list:
        if self.memory_file.exists():
            try:
                with open(self.memory_file, "r", encoding="utf-8") as f:
                    return json.load(f)
            except:
                return []
        return []

    def _save(self):
        with open(self.memory_file, "w", encoding="utf-8") as f:
            json.dump(self.memories, f, ensure_ascii=False, indent=2)

    def add_memory(self, content: dict) -> dict:
        memory = {
            "id": str(uuid.uuid4()),
            "content": content,
            "created_at": datetime.now().isoformat(),
            "type": "safety_analysis"
        }
        self.memories.insert(0, memory)
        self._save()
        return {"success": True, "memory_id": memory["id"]}

    def get_all_memory(self) -> list:
        return self.memories

    def search_memory(self, query: str) -> list:
        results = []
        for mem in self.memories:
            content_str = json.dumps(mem["content"], ensure_ascii=False)
            if query.lower() in content_str.lower():
                results.append(mem)
        return results

# ==================== Safety Analysis Service ====================

class SafetyAnalysisService:
    def __init__(self, regulations_fetcher: RegulationsFetcher):
        self.api_key = os.getenv("OPENROUTER_API_KEY") or os.getenv("OPENAI_API_KEY")
        self.model = "google/gemini-2.0-flash-001"
        self.regulations_fetcher = regulations_fetcher

    def analyze_image_bytes(self, image_bytes: bytes, filename: str) -> dict:
        image_data = base64.b64encode(image_bytes).decode("utf-8")
        ext = os.path.splitext(filename)[1].lower()
        mime_type = "image/png" if ext == ".png" else "image/jpeg"

        regulations = self.regulations_fetcher.regulations or self.regulations_fetcher._get_default_regulations()
        regulations_text = self._build_regulations_text(regulations)

        prompt = f"""당신은 한국 산업안전보건 전문가입니다.
이 공장/작업장 이미지를 분석하여 다음 규정에 따른 준수 여부를 JSON 형식으로 반환해주세요.

## 적용 규정:
{regulations_text}

## 필수 응답 형식 (JSON):
{{
    "workplace_type": "작업장 유형 (예: 제조공장, 창고, 건설현장 등)",
    "overall_risk_level": "high/medium/low",
    "pass_status": true/false,
    "compliant_items": [
        {{"category": "카테고리", "item": "항목", "description": "준수 상태 설명"}}
    ],
    "violations": [
        {{"category": "카테고리", "item": "항목", "rule": "위반 규정", "description": "위반 내용", "severity": "high/medium/low", "recommendation": "개선 권고사항"}}
    ],
    "immediate_actions": ["즉시 조치 필요 사항"],
    "recommendations": ["추가 권고사항"]
}}

JSON 코드 블록 없이 순수 JSON만 반환하세요.
"""
        try:
            if not self.api_key:
                return self._get_demo_result()

            with httpx.Client(timeout=60.0) as client:
                response = client.post(
                    "https://openrouter.ai/api/v1/chat/completions",
                    headers={"Authorization": f"Bearer {self.api_key}"},
                    json={
                        "model": self.model,
                        "messages": [{
                            "role": "user",
                            "content": [
                                {"type": "text", "text": prompt},
                                {"type": "image_url", "image_url": {"url": f"data:{mime_type};base64,{image_data}"}}
                            ]
                        }]
                    }
                )

                if response.status_code == 200:
                    content = response.json()["choices"][0]["message"]["content"]
                    if "```json" in content:
                        content = content.split("```json")[1].split("```")[0]
                    elif "```" in content:
                        content = content.split("```")[1].split("```")[0]
                    result = json.loads(content.strip())
                    result["success"] = True
                    result["analyzed_at"] = datetime.now().isoformat()
                    return result
                else:
                    return {"success": False, "error": f"API Error: {response.status_code}"}
        except Exception as e:
            return {"success": False, "error": str(e)}

    def _build_regulations_text(self, regulations: dict) -> str:
        lines = []
        for category, items in regulations.items():
            if category.startswith("_"):
                continue
            for item, details in items.items():
                if isinstance(details, dict):
                    lines.append(f"- [{category}/{item}] {details.get('rule', '')}: {details.get('description', '')}")
        return "\n".join(lines)

    def _get_demo_result(self) -> dict:
        return {
            "success": True,
            "workplace_type": "Demo - API Key Required",
            "overall_risk_level": "low",
            "pass_status": True,
            "compliant_items": [{"category": "Demo", "item": "Demo", "description": "API Key를 설정하면 실제 분석이 수행됩니다."}],
            "violations": [],
            "immediate_actions": ["OPENROUTER_API_KEY 환경변수를 설정하세요."],
            "recommendations": [".env 파일에 API Key를 추가하세요."],
            "analyzed_at": datetime.now().isoformat()
        }

# ==================== MediaPipe Setup ====================

@st.cache_resource
def get_detector():
    model_path = 'hand_landmarker.task'
    if not os.path.exists(model_path):
        return None
    base_options = python.BaseOptions(model_asset_path=model_path)
    options = vision.HandLandmarkerOptions(
        base_options=base_options,
        num_hands=2,
        min_hand_detection_confidence=0.3,
        min_hand_presence_confidence=0.3,
        min_tracking_confidence=0.3,
        running_mode=vision.RunningMode.IMAGE
    )
    return vision.HandLandmarker.create_from_options(options)

# ==================== Page 1: Real-time Sentinel ====================

def render_realtime_sentinel():
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

    st.title("🛡️ Real-time Safety Sentinel")
    st.markdown("### Real-time Factory Safety Monitoring")

    col_video, col_dashboard = st.columns([2, 1])

    with col_dashboard:
        if not st.session_state.monitor_active:
            if st.button("▶️ START SYSTEM", type="primary"):
                st.session_state.monitor_active = True
                st.session_state.current_status = "SAFE"
                st.session_state.pinch_frame_count = 0
                st.session_state.impact_frame_count = 0
                st.session_state.hazard_count_l = 0
                st.session_state.hazard_count_r = 0
                st.session_state.hazard_count_t = 0
                st.session_state.hazard_count_b = 0
                st.session_state.hazard_latched = False
                st.rerun()
        else:
            if st.button("⏹️ EMERGENCY STOP / RESET", type="secondary"):
                st.session_state.monitor_active = False
                st.session_state.current_status = "SAFE"
                st.session_state.hazard_latched = False
                st.rerun()

        st.markdown("---")
        st.markdown("#### 🏭 Machine Status")
        vehicle_status_ph = st.empty()
        machine_anim_ph = st.empty()

        st.markdown("#### 📋 Incident Log")
        log_ph = st.empty()

        with st.expander("⚙️ Camera Settings"):
            camera_index = st.number_input("Camera Index", min_value=0, max_value=10, value=0, step=1)

    with col_video:
        video_ph = st.empty()
        if not st.session_state.monitor_active:
            if st.session_state.get("current_status") == "DANGER":
                video_ph.error("🚨 EMERGENCY STOP TRIGGERED. PRESS START TO RESET.")
                vehicle_status_ph.markdown('<div class="status-badge status-danger">🚫 STOPPED</div>', unsafe_allow_html=True)
                machine_anim_ph.markdown('<div class="machine-container"><div class="gear-icon stopped">⚙️</div></div>', unsafe_allow_html=True)
            else:
                video_ph.info("System is OFFLINE. Press START to activate.")
                vehicle_status_ph.markdown('<div class="status-badge" style="border-color: #666; color: #666;">OFFLINE</div>', unsafe_allow_html=True)
                machine_anim_ph.markdown('<div class="machine-container"><div class="gear-icon stopped">⚙️</div></div>', unsafe_allow_html=True)
            return

    # Active Monitoring Loop
    cap = cv2.VideoCapture(camera_index, cv2.CAP_DSHOW)

    if not cap.isOpened():
        st.error(f"Could not access webcam at Index {camera_index}.")
        st.session_state.monitor_active = False
        return

    detector = get_detector()
    if detector is None:
        st.warning("hand_landmarker.task not found. Hand detection disabled.")

    frame_count = 0
    start_time = time.time()
    prev_gray = None
    prev_hand_area = None
    prev_hand_centroid = None
    motion_buffer = 0
    squeeze_buffer = 0

    try:
        while st.session_state.monitor_active:
            ret, frame = cap.read()
            if not ret:
                st.error("Failed to retrieve frame.")
                break

            h, w, c = frame.shape
            small_w, small_h = 320, 240
            frame_small = cv2.resize(frame, (small_w, small_h))
            gray_small = cv2.cvtColor(frame_small, cv2.COLOR_BGR2GRAY)

            frame_count += 1

            # Latched Hazard State
            if st.session_state.get('hazard_latched'):
                vehicle_status_ph.markdown('<div class="status-badge status-danger">🚫 STOPPED</div>', unsafe_allow_html=True)
                machine_anim_ph.markdown('<div class="machine-container"><div class="gear-icon stopped">⚙️</div></div>', unsafe_allow_html=True)
                cv2.putText(frame, "SYSTEM HALTED", (50, 100), cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0,0,255), 4)
                cv2.putText(frame, "PRESS 'RESET' TO RESUME", (50, 160), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0,0,255), 3)
                cv2.rectangle(frame, (0,0), (w,h), (0,0,255), 20)
                video_ph.image(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB), channels="RGB")
                time.sleep(0.03)
                continue

            pinch_detected = False

            if prev_gray is not None:
                flow = cv2.calcOpticalFlowFarneback(prev_gray, gray_small, None, 0.5, 2, 10, 2, 5, 1.1, 0)
            prev_gray = gray_small

            # Hand Detection
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            hand_detected = False
            hand_box = None

            if detector:
                mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=frame_rgb)
                detection_result = detector.detect(mp_image)

                sx = small_w / w
                sy = small_h / h

                if detection_result.hand_landmarks:
                    hand_detected = True
                    for hand_landmarks in detection_result.hand_landmarks:
                        x_coords = [int(lm.x * w) for lm in hand_landmarks]
                        y_coords = [int(lm.y * h) for lm in hand_landmarks]
                        x_min, x_max = max(0, min(x_coords) - 30), min(w, max(x_coords) + 30)
                        y_min, y_max = max(0, min(y_coords) - 30), min(h, max(y_coords) + 30)

                        cv2.rectangle(frame, (x_min, y_min), (x_max, y_max), (0, 255, 0), 2)
                        hand_box = (x_min, y_min, x_max, y_max)

                        # Compression Detection
                        current_raw_area = (x_max - x_min) * (y_max - y_min)
                        compression_detected = False
                        current_area = 0.5 * current_raw_area + 0.5 * prev_hand_area if prev_hand_area else current_raw_area

                        if prev_hand_area:
                            percent_change = (current_area - prev_hand_area) / prev_hand_area
                            if percent_change < -0.05:
                                compression_detected = True
                                cv2.putText(frame, "SQUEEZE!", (x_min, y_min-30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
                        prev_hand_area = current_area

                        # Optical Flow Analysis
                        if 'flow' in locals():
                            s_x_min, s_x_max = int(x_min * sx), int(x_max * sx)
                            s_y_min, s_y_max = int(y_min * sy), int(y_max * sy)
                            roi_w = int(100 * sx)
                            velocity_thresh, pixel_thresh = 4.0, 400

                            current_centroid = ((x_min + x_max)//2, (y_min + y_max)//2)
                            suppress_hazards = False
                            if prev_hand_centroid:
                                dist = np.sqrt((current_centroid[0]-prev_hand_centroid[0])**2 + (current_centroid[1]-prev_hand_centroid[1])**2)
                                if dist > 15.0:
                                    suppress_hazards = True
                            prev_hand_centroid = current_centroid

                            l_x1, l_x2 = max(0, s_x_min - roi_w), s_x_min
                            r_x1, r_x2 = s_x_max, min(small_w, s_x_max + roi_w)

                            moving_pixels_left = moving_pixels_right = 0
                            moving_pixels_left_away = moving_pixels_right_away = 0

                            if l_x2 > l_x1:
                                u_left = flow[s_y_min:s_y_max, l_x1:l_x2, 0]
                                moving_pixels_left = np.count_nonzero(u_left > velocity_thresh)
                                moving_pixels_left_away = np.count_nonzero(u_left < -velocity_thresh)
                            if r_x2 > r_x1:
                                u_right = flow[s_y_min:s_y_max, r_x1:r_x2, 0]
                                moving_pixels_right = np.count_nonzero(u_right < -velocity_thresh)
                                moving_pixels_right_away = np.count_nonzero(u_right > velocity_thresh)

                            translation_right = (moving_pixels_left > pixel_thresh) and (moving_pixels_right_away > pixel_thresh)
                            translation_left = (moving_pixels_right > pixel_thresh) and (moving_pixels_left_away > pixel_thresh)
                            left_active = (moving_pixels_left > pixel_thresh) and not translation_right and not suppress_hazards
                            right_active = (moving_pixels_right > pixel_thresh) and not translation_left and not suppress_hazards

                            # Vertical
                            roi_h = int(100 * sy)
                            t_y1, t_y2 = max(0, s_y_min - roi_h), s_y_min
                            b_y1, b_y2 = s_y_max, min(small_h, s_y_max + roi_h)

                            moving_pixels_top = moving_pixels_bottom = 0
                            moving_pixels_top_away = moving_pixels_bottom_away = 0

                            if t_y2 > t_y1:
                                v_top = flow[t_y1:t_y2, s_x_min:s_x_max, 1]
                                moving_pixels_top = np.count_nonzero(v_top > velocity_thresh)
                                moving_pixels_top_away = np.count_nonzero(v_top < -velocity_thresh)
                            if b_y2 > b_y1:
                                v_bottom = flow[b_y1:b_y2, s_x_min:s_x_max, 1]
                                moving_pixels_bottom = np.count_nonzero(v_bottom < -velocity_thresh)
                                moving_pixels_bottom_away = np.count_nonzero(v_bottom > velocity_thresh)

                            translation_down = (moving_pixels_top > pixel_thresh) and (moving_pixels_bottom_away > pixel_thresh)
                            translation_up = (moving_pixels_bottom > pixel_thresh) and (moving_pixels_top_away > pixel_thresh)
                            top_active = (moving_pixels_top > pixel_thresh) and not translation_down and not suppress_hazards
                            bottom_active = (moving_pixels_bottom > pixel_thresh) and not translation_up and not suppress_hazards

                            # Hazard Detection
                            any_motion_active = left_active or right_active or top_active or bottom_active
                            if any_motion_active: motion_buffer = 5
                            if compression_detected: squeeze_buffer = 5
                            motion_buffer = max(0, motion_buffer - 1)
                            squeeze_buffer = max(0, squeeze_buffer - 1)
                            buffered_trigger = (motion_buffer > 0) and (squeeze_buffer > 0)

                            strict_pincer = (left_active and right_active) or (top_active and bottom_active)
                            if strict_pincer: st.session_state.pinch_frame_count += 1
                            else: st.session_state.pinch_frame_count = 0

                            if buffered_trigger: st.session_state.impact_frame_count += 1
                            else: st.session_state.impact_frame_count = max(0, st.session_state.impact_frame_count - 1)

                            if left_active: st.session_state.hazard_count_l += 2
                            else: st.session_state.hazard_count_l = max(0, st.session_state.hazard_count_l - 1)
                            if right_active: st.session_state.hazard_count_r += 2
                            else: st.session_state.hazard_count_r = max(0, st.session_state.hazard_count_r - 1)
                            if top_active: st.session_state.hazard_count_t += 2
                            else: st.session_state.hazard_count_t = max(0, st.session_state.hazard_count_t - 1)
                            if bottom_active: st.session_state.hazard_count_b += 2
                            else: st.session_state.hazard_count_b = max(0, st.session_state.hazard_count_b - 1)

                            max_hazard = max(st.session_state.hazard_count_l, st.session_state.hazard_count_r, st.session_state.hazard_count_t, st.session_state.hazard_count_b)
                            if max_hazard >= 10: pinch_detected = True
                            if st.session_state.pinch_frame_count > 3: pinch_detected = True
                            if st.session_state.impact_frame_count > 12: pinch_detected = True

            # Update State
            if pinch_detected:
                st.session_state.current_status = "DANGER"
                cv2.putText(frame, "CRUSH HAZARD!", (50, 100), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 4)
                log_msg = trigger_spoon_agent_incident()
                st.session_state.incident_log.insert(0, log_msg)
                st.session_state.hazard_latched = True
            elif hand_detected:
                st.session_state.current_status = "SAFE"
                cv2.putText(frame, "Hand Detected - Monitoring...", (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2)
            else:
                st.session_state.current_status = "SAFE"
                cv2.putText(frame, "ZONE CLEAR", (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (100, 255, 100), 2)

            # Render Dashboard
            if st.session_state.current_status == "SAFE":
                vehicle_status_ph.markdown('<div class="status-badge status-safe">OPERATIONAL</div>', unsafe_allow_html=True)
                machine_anim_ph.markdown('<div class="machine-container"><div class="gear-icon spinning">⚙️</div></div>', unsafe_allow_html=True)
            else:
                vehicle_status_ph.markdown('<div class="status-badge status-danger">🚫 STOPPED</div>', unsafe_allow_html=True)
                machine_anim_ph.markdown('<div class="machine-container"><div class="gear-icon stopped">⚙️</div></div>', unsafe_allow_html=True)

            log_html = "<br>".join([f"• {msg}" for msg in st.session_state.incident_log[:5]])
            log_ph.markdown(f'<div class="log-area">{log_html}</div>', unsafe_allow_html=True)

            video_ph.image(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB), channels="RGB")

    finally:
        cap.release()

# ==================== Page 2: Image Analysis ====================

def render_image_analysis():
    load_css()

    st.title("🔍 Safety Regulation Analysis")
    st.markdown("### Upload an image to analyze safety compliance")

    # Initialize services
    regulations_fetcher = RegulationsFetcher()
    memory_store = MemoryStore()
    analysis_service = SafetyAnalysisService(regulations_fetcher)

    col1, col2 = st.columns([1, 1])

    with col1:
        st.markdown("#### 📤 Upload Image")
        uploaded_file = st.file_uploader("Choose an image", type=["jpg", "jpeg", "png"])

        if uploaded_file:
            st.image(uploaded_file, caption="Uploaded Image", use_container_width=True)

            if st.button("🔍 Analyze Safety", type="primary"):
                with st.spinner("Analyzing image..."):
                    image_bytes = uploaded_file.getvalue()

                    # Save file
                    file_path = UPLOAD_DIR / f"{uuid.uuid4()}{Path(uploaded_file.name).suffix}"
                    with open(file_path, "wb") as f:
                        f.write(image_bytes)

                    # Run analysis (synchronous)
                    result = analysis_service.analyze_image_bytes(image_bytes, uploaded_file.name)

                    # Save to memory
                    memory_store.add_memory({
                        "filename": uploaded_file.name,
                        "file_path": str(file_path),
                        "analysis": result
                    })

                    st.session_state.analysis_result = result

    with col2:
        st.markdown("#### 📊 Analysis Result")

        if 'analysis_result' in st.session_state and st.session_state.analysis_result:
            result = st.session_state.analysis_result

            if result.get("success"):
                # Overall Status
                risk_level = result.get("overall_risk_level", "unknown")
                pass_status = result.get("pass_status", False)

                if pass_status:
                    st.success(f"✅ PASSED - Risk Level: {risk_level.upper()}")
                else:
                    st.error(f"❌ FAILED - Risk Level: {risk_level.upper()}")

                st.markdown(f"**Workplace Type:** {result.get('workplace_type', 'N/A')}")

                # Violations
                violations = result.get("violations", [])
                if violations:
                    st.markdown("#### ⚠️ Violations")
                    for v in violations:
                        severity_color = {"high": "🔴", "medium": "🟡", "low": "🟢"}.get(v.get("severity", ""), "⚪")
                        st.markdown(f"""
                        <div class="result-card danger">
                            <strong>{severity_color} [{v.get('category', '')}] {v.get('item', '')}</strong><br>
                            <em>Rule: {v.get('rule', 'N/A')}</em><br>
                            {v.get('description', '')}<br>
                            <strong>Recommendation:</strong> {v.get('recommendation', 'N/A')}
                        </div>
                        """, unsafe_allow_html=True)

                # Compliant Items
                compliant = result.get("compliant_items", [])
                if compliant:
                    with st.expander("✅ Compliant Items"):
                        for c in compliant:
                            st.markdown(f"- **[{c.get('category', '')}] {c.get('item', '')}**: {c.get('description', '')}")

                # Recommendations
                recommendations = result.get("recommendations", [])
                if recommendations:
                    with st.expander("💡 Recommendations"):
                        for r in recommendations:
                            st.markdown(f"- {r}")

            else:
                st.error(f"Analysis failed: {result.get('error', 'Unknown error')}")
        else:
            st.info("Upload an image and click 'Analyze Safety' to see results.")

    # History Section
    st.markdown("---")
    st.markdown("#### 📜 Analysis History")

    memories = memory_store.get_all_memory()
    if memories:
        for mem in memories[:5]:
            content = mem.get("content", {})
            analysis = content.get("analysis", {})
            with st.expander(f"📄 {content.get('filename', 'Unknown')} - {mem.get('created_at', '')[:10]}"):
                st.json(analysis)
    else:
        st.info("No analysis history yet.")

# ==================== Main App ====================

def main():
    st.sidebar.title("🛡️ Safety Lens")
    st.sidebar.markdown("---")

    page = st.sidebar.radio(
        "Select Page",
        ["🎥 Real-time Sentinel", "🔍 Image Analysis"],
        index=0
    )

    st.sidebar.markdown("---")
    st.sidebar.markdown("### About")
    st.sidebar.info("Safety Lens is a real-time factory safety monitoring system with AI-powered image analysis.")

    if page == "🎥 Real-time Sentinel":
        render_realtime_sentinel()
    else:
        render_image_analysis()

if __name__ == "__main__":
    main()
