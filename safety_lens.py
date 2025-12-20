import cv2
import numpy as np
import time
from datetime import datetime

import streamlit as st
from streamlit_webrtc import (
    webrtc_streamer,
    VideoProcessorBase,
    RTCConfiguration,
    WebRtcMode,  # ✅ 추가 (mode에 문자열이 아니라 Enum을 넣어야 함)
)

import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision


st.set_page_config(page_title="Safety Lens: Active Sentinel (WebRTC)", layout="wide")

RTC_CONFIGURATION = RTCConfiguration(
    {"iceServers": [{"urls": ["stun:stun.l.google.com:19302"]}]}
)


@st.cache_resource
def get_detector():
    model_path = "hand_landmarker.task"
    base_options = python.BaseOptions(model_asset_path=model_path)
    options = vision.HandLandmarkerOptions(
        base_options=base_options,
        num_hands=2,
        min_hand_detection_confidence=0.3,
        min_hand_presence_confidence=0.3,
        min_tracking_confidence=0.3,
        running_mode=vision.RunningMode.IMAGE,
    )
    return vision.HandLandmarker.create_from_options(options)


def trigger_spoon_agent_incident():
    ts = datetime.now().strftime("%H:%M:%S")
    return f"Incident logged at {ts} via Spoon Agent (x402)"


class SafetyVideoProcessor(VideoProcessorBase):
    """
    브라우저 카메라 프레임을 받아서:
    - 손 탐지 + 박스 표시
    - optical flow + 면적 감소 기반 hazard 감지(간단 버전)
    - hazard_latched 되면 화면에 빨간 테두리 + STOP 표시
    """

    def __init__(self):
        self.detector = get_detector()

        self.hazard_latched = False
        self.current_status = "SAFE"
        self.incident_log = []

        self.prev_gray = None
        self.prev_hand_area = None

        # leaky bucket counters
        self.hazard_count = 0

        self.last_frame_time = time.time()
        self.fps = 0.0

    def _update_fps(self):
        now = time.time()
        dt = now - self.last_frame_time
        if dt > 0:
            self.fps = 0.9 * self.fps + 0.1 * (1.0 / dt)
        self.last_frame_time = now

    def recv(self, frame):
        self._update_fps()

        img_bgr = frame.to_ndarray(format="bgr24")
        h, w, _ = img_bgr.shape

        # 이미 hazard latch면: 계속 STOP 화면
        if self.hazard_latched:
            cv2.rectangle(img_bgr, (0, 0), (w, h), (0, 0, 255), 20)
            cv2.putText(
                img_bgr,
                "SYSTEM HALTED",
                (50, 100),
                cv2.FONT_HERSHEY_SIMPLEX,
                1.5,
                (0, 0, 255),
                4,
            )
            cv2.putText(
                img_bgr,
                "PRESS RESET BUTTON",
                (50, 160),
                cv2.FONT_HERSHEY_SIMPLEX,
                1.0,
                (0, 0, 255),
                3,
            )
            return frame.from_ndarray(img_bgr, format="bgr24")

        # Optical flow 용 다운샘플
        small_w, small_h = 320, 240
        img_small = cv2.resize(img_bgr, (small_w, small_h))
        gray_small = cv2.cvtColor(img_small, cv2.COLOR_BGR2GRAY)

        flow = None
        if self.prev_gray is not None:
            flow = cv2.calcOpticalFlowFarneback(
                self.prev_gray, gray_small, None, 0.5, 2, 10, 2, 5, 1.1, 0
            )
        self.prev_gray = gray_small

        # HandLandmarker
        img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=img_rgb)
        det = self.detector.detect(mp_image)

        hand_detected = False
        hazard_detected = False

        sx = small_w / w
        sy = small_h / h

        if det.hand_landmarks:
            hand_detected = True

            for hand_landmarks in det.hand_landmarks:
                xs = [int(lm.x * w) for lm in hand_landmarks]
                ys = [int(lm.y * h) for lm in hand_landmarks]
                x_min, x_max = max(0, min(xs) - 30), min(w, max(xs) + 30)
                y_min, y_max = max(0, min(ys) - 30), min(h, max(ys) + 30)

                # 손 박스
                cv2.rectangle(img_bgr, (x_min, y_min), (x_max, y_max), (0, 255, 0), 2)

                # 면적 감소(압축) 감지
                area = (x_max - x_min) * (y_max - y_min)
                compression = False
                if self.prev_hand_area is not None:
                    # 간단 EMA
                    area_smooth = 0.5 * area + 0.5 * self.prev_hand_area
                    percent_change = (area_smooth - self.prev_hand_area) / max(
                        1, self.prev_hand_area
                    )
                    if percent_change < -0.05:
                        compression = True
                        cv2.putText(
                            img_bgr,
                            "SQUEEZE!",
                            (x_min, y_min - 10),
                            cv2.FONT_HERSHEY_SIMPLEX,
                            0.7,
                            (0, 0, 255),
                            2,
                        )
                    self.prev_hand_area = area_smooth
                else:
                    self.prev_hand_area = area

                # Optical flow 기반 "금속 접근" 같은 방향성 감지 (아주 단순화)
                motion_toward = False
                if flow is not None:
                    s_x1, s_x2 = int(x_min * sx), int(x_max * sx)
                    s_y1, s_y2 = int(y_min * sy), int(y_max * sy)

                    roi_w = int(100 * sx)
                    roi_h = int(100 * sy)

                    # 좌/우/상/하 ROI에서 hand 쪽으로 향하는 픽셀 수 세기
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

                    # top ROI: v > +vel_th (downward)
                    ty1, ty2 = max(0, s_y1 - roi_h), s_y1
                    top_cnt = 0
                    if ty2 > ty1:
                        v = flow[ty1:ty2, s_x1:s_x2, 1]
                        top_cnt = np.count_nonzero(v > vel_th)

                    # bottom ROI: v < -vel_th (upward)
                    by1, by2 = s_y2, min(small_h, s_y2 + roi_h)
                    bot_cnt = 0
                    if by2 > by1:
                        v = flow[by1:by2, s_x1:s_x2, 1]
                        bot_cnt = np.count_nonzero(v < -vel_th)

                    motion_toward = (
                        left_cnt > pix_th
                        or right_cnt > pix_th
                        or top_cnt > pix_th
                        or bot_cnt > pix_th
                    )

                    if motion_toward:
                        cv2.putText(
                            img_bgr,
                            "MOTION NEAR HAND",
                            (50, 50),
                            cv2.FONT_HERSHEY_SIMPLEX,
                            0.8,
                            (0, 255, 255),
                            2,
                        )

                # leaky bucket: motion_toward + compression이 같이 오면 위험 누적
                if motion_toward and compression:
                    self.hazard_count += 2
                else:
                    self.hazard_count = max(0, self.hazard_count - 1)

                cv2.putText(
                    img_bgr,
                    f"HAZARD: {self.hazard_count}/10",
                    (50, 90),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.7,
                    (0, 165, 255),
                    2,
                )

                if self.hazard_count >= 10:
                    hazard_detected = True

                # 한 손만 처리
                break

        # 상태 업데이트
        if hazard_detected:
            self.current_status = "DANGER"
            cv2.putText(
                img_bgr,
                "CRUSH HAZARD DETECTED!",
                (50, 150),
                cv2.FONT_HERSHEY_SIMPLEX,
                1.0,
                (0, 0, 255),
                4,
            )
            cv2.putText(
                img_bgr,
                "EMERGENCY STOP",
                (50, 110),
                cv2.FONT_HERSHEY_SIMPLEX,
                1.0,
                (0, 0, 255),
                4,
            )

            self.hazard_latched = True
            self.incident_log.insert(0, trigger_spoon_agent_incident())
        else:
            self.current_status = "SAFE"
            if hand_detected:
                cv2.putText(
                    img_bgr,
                    "Hand Detected - Monitoring...",
                    (50, 50),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.8,
                    (0, 255, 255),
                    2,
                )
            else:
                cv2.putText(
                    img_bgr,
                    "ZONE CLEAR - NO HAND",
                    (50, 50),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.8,
                    (100, 255, 100),
                    2,
                )

        cv2.putText(
            img_bgr,
            f"FPS ~ {self.fps:.1f}",
            (50, h - 30),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (255, 255, 255),
            2,
        )

        return frame.from_ndarray(img_bgr, format="bgr24")


def main():
    st.title("🛡️ Safety Lens: Active Sentinel (WebRTC)")
    st.caption("브라우저 카메라 → WebRTC로 프레임 수신 → 서버에서 MediaPipe 처리")

    col_video, col_dash = st.columns([2, 1])

    with col_dash:
        st.subheader("🏭 Status")
        status_box = st.empty()

        st.subheader("🧰 Controls")
        reset_clicked = st.button("🔁 RESET (Unlatch)", type="primary")

        st.subheader("📋 Incident Log")
        log_box = st.empty()

    with col_video:
        ctx = webrtc_streamer(
            key="safety-webrtc",
            mode=WebRtcMode.SENDRECV,  # ✅ 문자열 대신 Enum 사용
            rtc_configuration=RTC_CONFIGURATION,
            video_processor_factory=SafetyVideoProcessor,
            media_stream_constraints={"video": True, "audio": False},
            async_processing=True,
        )

    # UI 업데이트(메인 스레드에서 processor 상태를 읽기)
    if ctx.video_processor:
        vp = ctx.video_processor

        if reset_clicked:
            vp.hazard_latched = False
            vp.hazard_count = 0
            vp.current_status = "SAFE"

        if vp.current_status == "SAFE":
            status_box.success("OPERATIONAL")
        else:
            status_box.error("🚨 STOPPED (LATCHED)")

        # 로그 표시
        if vp.incident_log:
            log_box.write("\n".join([f"• {x}" for x in vp.incident_log[:8]]))
        else:
            log_box.info("No incidents yet.")


if __name__ == "__main__":
    main()
