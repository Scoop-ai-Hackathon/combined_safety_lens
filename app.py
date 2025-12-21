import os
import time
import json
import base64
import hashlib
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Optional, Dict, Any

import cv2
import numpy as np
import streamlit as st
from dotenv import load_dotenv

from streamlit_webrtc import (
    webrtc_streamer,
    VideoProcessorBase,
    RTCConfiguration,
    WebRtcMode,
)

import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision

import asyncio
from spoon_ai.agents.toolcall import ToolCallAgent
from spoon_ai.chat import ChatBot
from spoon_ai.tools import ToolManager


# ----------------------------
# ENV / CONFIG
# ----------------------------
load_dotenv()

# spoon-core 문서 가이드: OpenRouter를 OpenAI provider로 붙이기
# (OPENAI_API_KEY에 OpenRouter 키를 넣고 base_url을 openrouter로 둠) :contentReference[oaicite:3]{index=3}
if not os.getenv("OPENAI_API_KEY") and os.getenv("OPENROUTER_API_KEY"):
    os.environ["OPENAI_API_KEY"] = os.getenv("OPENROUTER_API_KEY")

OPENROUTER_BASE_URL = "https://openrouter.ai/api/v1"
DEFAULT_MODEL = os.getenv("DEFAULT_MODEL", "google/gemini-2.0-flash-001")

# Enhanced RTC configuration for EC2/Cloud deployment
# Multiple STUN servers for better NAT traversal and reliability
RTC_CONFIGURATION = RTCConfiguration({
    "iceServers": [
        {"urls": ["stun:stun.l.google.com:19302"]},
        {"urls": ["stun:stun1.l.google.com:19302"]},
        {"urls": ["stun:stun2.l.google.com:19302"]},
        {"urls": ["stun:stun3.l.google.com:19302"]},
        {"urls": ["stun:stun4.l.google.com:19302"]},
    ]
})

st.set_page_config(page_title="Safety Lens (WebRTC + SpoonOS)", layout="wide")


# ----------------------------
# Rule Set (MVP용 예시)
# 실제로는: 산업안전관리법/사내 SOP를 파일로 로딩해서 넣는 식으로 확장
# ----------------------------
FACTORY_RULESET = """
[Factory Safety Rule Set - MVP]
R1. 위험 구역(프레스/절단/롤러 인근)에는 작업자 손이 들어오지 않도록 가드/센서가 활성화되어야 한다.
R2. 위험 구역에 손이 감지되면 즉시 비상정지(EMERGENCY STOP)를 트리거하고 재가동은 관리자 승인 후에만 가능하다.
R3. 협착(손 주변의 급격한 면적 감소/압박) 징후가 관측되면 '중대 위험'으로 분류한다.
R4. 중대 위험 발생 시, 사건 증적(프레임 캡처, 시간, 규정 버전, 판정 결과)을 기록하고 보고서를 생성한다.
R5. 자동 서명(승인)은 "위반 없음(Compliant)"일 때만 가능하며,
    위반 또는 불확실(Review Required)일 경우 인간 안전관리자가 최종 확인/서명한다.
"""


# ----------------------------
# Utilities
# ----------------------------
def sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def save_screenshot_jpg(jpg_bytes: bytes) -> str:
    out_dir = Path("screenshots")
    out_dir.mkdir(exist_ok=True)
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    path = out_dir / f"incident_{ts}.jpg"
    path.write_bytes(jpg_bytes)
    return str(path)


@dataclass
class IncidentPacket:
    ts_iso: str
    jpg_bytes: bytes
    jpg_sha256: str
    screenshot_path: str
    detector_signals: Dict[str, Any]


# ----------------------------
# SpoonOS (spoon_ai) LLM Agent
# ----------------------------
class IncidentReportAgent(ToolCallAgent):
    name: str = "incident_report_agent"
    description: str = "Generates compliance report from an incident snapshot + rules"
    system_prompt: str = """
You are an industrial safety compliance assistant.
You will receive:
- a factory rule set (text)
- incident metadata (signals + hash + timestamp)
Your job:
1) decide whether the incident is COMPLIANT or VIOLATION or REVIEW_REQUIRED
2) list violated rules if any (R1..Rn)
3) write a concise incident report in Korean for a factory safety manager
4) output MUST be valid JSON with keys:
   decision: one of ["COMPLIANT","VIOLATION","REVIEW_REQUIRED"]
   violated_rules: string array
   risk_level: one of ["LOW","MEDIUM","HIGH","CRITICAL"]
   summary: string
   recommendations: string array
   signable: boolean  (true only when decision == COMPLIANT)
   report_text: string (human-readable report)
Keep it realistic: If uncertain, choose REVIEW_REQUIRED.
"""

    # tools 없이도 ToolCallAgent를 쓸 수 있게 빈 ToolManager
    available_tools: ToolManager = ToolManager([])


async def generate_report_with_spoonos(incident: IncidentPacket) -> Dict[str, Any]:
    # OpenRouter를 OpenAI provider로 호출 (spoon-core 문서 방식) :contentReference[oaicite:4]{index=4}
    bot = ChatBot(
        llm_provider="openai",
        model_name=DEFAULT_MODEL,
        base_url=OPENROUTER_BASE_URL,
    )

    agent = IncidentReportAgent(llm=bot)

    # MVP: 이미지 자체를 비전으로 “해석”하기보다는,
    #       우리가 감지한 시그널(손 감지, hazard_count 등)을 근거로 규정 판단.
    # (추후: 멀티모달 모델/메시지 포맷으로 이미지도 같이 넣을 수 있음)
    payload = {
        "ruleset": FACTORY_RULESET.strip(),
        "incident": {
            "ts_iso": incident.ts_iso,
            "jpg_sha256": incident.jpg_sha256,
            "screenshot_path": incident.screenshot_path,
            "signals": incident.detector_signals,
        },
        "instruction": "Return the JSON only. Do not include markdown fences."
    }

    user_message = json.dumps(payload, ensure_ascii=False)

    raw = await agent.run(user_message)

    # raw가 문자열일 가능성이 높음 → JSON 파싱 시도
    try:
        return json.loads(raw)
    except Exception:
        # 모델이 JSON을 약간 깨먹어도 최소 복구
        return {
            "decision": "REVIEW_REQUIRED",
            "violated_rules": [],
            "risk_level": "MEDIUM",
            "summary": "모델 출력이 JSON으로 파싱되지 않았습니다.",
            "recommendations": ["안전관리자가 수동으로 내용을 확인하세요."],
            "signable": False,
            "report_text": raw,
        }


# ----------------------------
# MediaPipe Hand Detector
# ----------------------------
@st.cache_resource
def get_hand_detector():
    model_path = Path("models") / "hand_landmarker.task"
    if not model_path.exists():
        raise FileNotFoundError(
            f"MediaPipe model not found: {model_path}\n"
            f"Put 'hand_landmarker.task' into ./models/"
        )

    base_options = python.BaseOptions(model_asset_path=str(model_path))
    options = vision.HandLandmarkerOptions(
        base_options=base_options,
        num_hands=2,
        min_hand_detection_confidence=0.3,
        min_hand_presence_confidence=0.3,
        min_tracking_confidence=0.3,
        running_mode=vision.RunningMode.IMAGE,
    )
    return vision.HandLandmarker.create_from_options(options)


# ----------------------------
# Video Processor (WebRTC frames)
# ----------------------------
class SafetyVideoProcessor(VideoProcessorBase):
    """
    브라우저에서 들어오는 프레임을 처리해:
    - 손 bounding box
    - 면적 급감(압박) + 주변 모션(간단)으로 hazard_count 누적
    - DANGER 되면 latch + incident packet 준비
    """
    def __init__(self):
        self.detector = get_hand_detector()

        self.hazard_latched = False
        self.hazard_count = 0

        self.prev_gray_small = None
        self.prev_hand_area = None

        self.incident_pending: Optional[IncidentPacket] = None
        self._incident_emitted = False

        self.last_ts = time.time()
        self.fps = 0.0

    def _update_fps(self):
        now = time.time()
        dt = now - self.last_ts
        if dt > 0:
            self.fps = 0.9 * self.fps + 0.1 * (1.0 / dt)
        self.last_ts = now

    def recv(self, frame):
        self._update_fps()

        img_bgr = frame.to_ndarray(format="bgr24")
        h, w, _ = img_bgr.shape

        # latch 상태면 화면에 표시만
        if self.hazard_latched:
            cv2.rectangle(img_bgr, (0, 0), (w, h), (0, 0, 255), 18)
            cv2.putText(img_bgr, "SYSTEM HALTED", (40, 90),
                        cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0, 0, 255), 4)
            cv2.putText(img_bgr, "Generating report...", (40, 140),
                        cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 0, 255), 3)
            return frame.from_ndarray(img_bgr, format="bgr24")

        # optical flow용 downsample
        small_w, small_h = 320, 240
        img_small = cv2.resize(img_bgr, (small_w, small_h))
        gray_small = cv2.cvtColor(img_small, cv2.COLOR_BGR2GRAY)

        flow = None
        if self.prev_gray_small is not None:
            flow = cv2.calcOpticalFlowFarneback(
                self.prev_gray_small, gray_small, None,
                0.5, 2, 10, 2, 5, 1.1, 0
            )
        self.prev_gray_small = gray_small

        # hand detect
        img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=img_rgb)
        det = self.detector.detect(mp_image)

        hand_detected = False
        hazard_detected = False
        compression = False
        motion_toward = False

        if det.hand_landmarks:
            hand_detected = True
            # 한 손만 MVP로 처리
            hand = det.hand_landmarks[0]
            xs = [int(lm.x * w) for lm in hand]
            ys = [int(lm.y * h) for lm in hand]
            x_min, x_max = max(0, min(xs) - 30), min(w, max(xs) + 30)
            y_min, y_max = max(0, min(ys) - 30), min(h, max(ys) + 30)

            cv2.rectangle(img_bgr, (x_min, y_min), (x_max, y_max), (0, 255, 0), 2)

            # area shrink (압박 단서)
            area = (x_max - x_min) * (y_max - y_min)
            if self.prev_hand_area is not None:
                area_smooth = 0.5 * area + 0.5 * self.prev_hand_area
                pct = (area_smooth - self.prev_hand_area) / max(1.0, self.prev_hand_area)
                if pct < -0.05:
                    compression = True
                    cv2.putText(img_bgr, "SQUEEZE!", (x_min, y_min - 10),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
                self.prev_hand_area = area_smooth
            else:
                self.prev_hand_area = float(area)

            # motion toward hand (아주 단순)
            if flow is not None:
                sx = small_w / w
                sy = small_h / h
                s_x1, s_x2 = int(x_min * sx), int(x_max * sx)
                s_y1, s_y2 = int(y_min * sy), int(y_max * sy)

                roi_w = int(100 * sx)
                vel_th = 4.0
                pix_th = 350

                # left ROI: u > +vel_th
                lx1, lx2 = max(0, s_x1 - roi_w), s_x1
                left_cnt = 0
                if lx2 > lx1:
                    u = flow[s_y1:s_y2, lx1:lx2, 0]
                    left_cnt = np.count_nonzero(u > vel_th)

                # right ROI: u < -vel_th
                rx1, rx2 = s_x2, min(small_w, s_x2 + roi_w)
                right_cnt = 0
                if rx2 > rx1:
                    u = flow[s_y1:s_y2, rx1:rx2, 0]
                    right_cnt = np.count_nonzero(u < -vel_th)

                motion_toward = (left_cnt > pix_th) or (right_cnt > pix_th)
                if motion_toward:
                    cv2.putText(img_bgr, "MOTION NEAR HAND", (40, 50),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2)

            # leaky bucket
            if motion_toward and compression:
                self.hazard_count += 2
            else:
                self.hazard_count = max(0, self.hazard_count - 1)

            cv2.putText(img_bgr, f"HAZARD: {self.hazard_count}/10", (40, 90),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 165, 255), 2)

            if self.hazard_count >= 10:
                hazard_detected = True

        # 상태 표시
        if hazard_detected and not self._incident_emitted:
            # latch + incident packet
            self.hazard_latched = True
            ts_iso = datetime.utcnow().replace(microsecond=0).isoformat() + "Z"

            ok, buf = cv2.imencode(".jpg", img_bgr, [int(cv2.IMWRITE_JPEG_QUALITY), 90])
            jpg_bytes = buf.tobytes() if ok else b""
            jpg_hash = sha256_bytes(jpg_bytes)
            path = save_screenshot_jpg(jpg_bytes)

            self.incident_pending = IncidentPacket(
                ts_iso=ts_iso,
                jpg_bytes=jpg_bytes,
                jpg_sha256=jpg_hash,
                screenshot_path=path,
                detector_signals={
                    "hand_detected": hand_detected,
                    "compression": compression,
                    "motion_toward": motion_toward,
                    "hazard_count": self.hazard_count,
                    "fps_estimate": round(self.fps, 1),
                }
            )
            self._incident_emitted = True

            cv2.putText(img_bgr, "CRUSH HAZARD DETECTED!", (40, 150),
                        cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 0, 255), 4)
            cv2.putText(img_bgr, "EMERGENCY STOP", (40, 120),
                        cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 0, 255), 4)

        else:
            if hand_detected:
                cv2.putText(img_bgr, "Hand Detected - Monitoring...", (40, 50),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2)
            else:
                cv2.putText(img_bgr, "ZONE CLEAR - NO HAND", (40, 50),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.8, (100, 255, 100), 2)

        cv2.putText(img_bgr, f"FPS~ {self.fps:.1f}", (40, h - 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

        return frame.from_ndarray(img_bgr, format="bgr24")


# ----------------------------
# Streamlit UI
# ----------------------------
def ensure_state():
    if "report" not in st.session_state:
        st.session_state.report = None
    if "report_status" not in st.session_state:
        st.session_state.report_status = "IDLE"  # IDLE / RUNNING / DONE / ERROR
    if "incident_log" not in st.session_state:
        st.session_state.incident_log = []
    if "last_incident_hash" not in st.session_state:
        st.session_state.last_incident_hash = None


def main():
    ensure_state()

    st.title("🛡️ Safety Lens: WebRTC + SpoonOS (OpenRouter Gemini)")
    st.caption("브라우저 카메라 → 위험 감지(협착) → 스크린샷 캡처 → SpoonOS로 규정 비교 보고서 자동 생성")

    left, right = st.columns([2, 1])

    with right:
        st.subheader("🏭 상태")
        st.write(f"- 모델: `{DEFAULT_MODEL}`")
        st.write(f"- LLM 라우팅: OpenRouter (OpenAI provider)")

        st.subheader("📋 Incident Log")
        if st.session_state.incident_log:
            for item in st.session_state.incident_log[:8]:
                st.write(f"• {item}")
        else:
            st.info("No incidents yet.")

        st.subheader("🧾 보고서")
        if st.session_state.report_status == "RUNNING":
            st.warning("보고서 생성 중...")
        elif st.session_state.report_status == "ERROR":
            st.error("보고서 생성 실패. 콘솔/로그 확인.")
        elif st.session_state.report_status == "DONE" and st.session_state.report:
            rep = st.session_state.report
            st.write(f"**Decision:** {rep.get('decision')}")
            st.write(f"**Risk:** {rep.get('risk_level')}")
            st.write(f"**Signable:** {rep.get('signable')}")
            st.write("**Violated:**", rep.get("violated_rules"))
            st.text_area("Report Text", rep.get("report_text", ""), height=260)
        else:
            st.caption("위험 감지 시 자동 생성됩니다.")

    with left:
        ctx = webrtc_streamer(
            key="safety-lens",
            mode=WebRtcMode.SENDRECV,
            rtc_configuration=RTC_CONFIGURATION,
            video_processor_factory=SafetyVideoProcessor,
            media_stream_constraints={"video": True, "audio": False},
            async_processing=True,
        )

        # incident pending을 메인 스레드에서 감지하고, LLM 호출
        if ctx.video_processor:
            vp = ctx.video_processor
            incident = getattr(vp, "incident_pending", None)

            if incident and incident.jpg_sha256 != st.session_state.last_incident_hash:
                st.session_state.last_incident_hash = incident.jpg_sha256
                ts = datetime.now().strftime("%H:%M:%S")
                st.session_state.incident_log.insert(
                    0,
                    f"[{ts}] 🚨 DANGER latched. Captured={incident.screenshot_path}, sha256={incident.jpg_sha256[:12]}..."
                )

                # LLM 보고서 생성
                st.session_state.report_status = "RUNNING"
                st.session_state.report = None

                try:
                    with st.spinner("SpoonOS로 규정 비교 보고서 생성 중..."):
                        report = asyncio.run(generate_report_with_spoonos(incident))
                    st.session_state.report = report
                    st.session_state.report_status = "DONE"

                    st.session_state.incident_log.insert(
                        0,
                        f"[{ts}] ✅ Report generated. decision={report.get('decision')} signable={report.get('signable')}"
                    )
                except Exception as e:
                    st.session_state.report_status = "ERROR"
                    st.session_state.incident_log.insert(0, f"[{ts}] ❌ Report error: {e}")

                # rerun to refresh right panel
                st.rerun()


if __name__ == "__main__":
    main()
