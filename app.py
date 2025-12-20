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
import threading
from datetime import datetime
from pathlib import Path
from dotenv import load_dotenv
import av

# Import MediaPipe Tasks API
from mediapipe.tasks import python
from mediapipe.tasks.python import vision

# WebRTC for browser camera access
from streamlit_webrtc import webrtc_streamer, WebRtcMode, VideoProcessorBase, RTCConfiguration

# Load environment variables
load_dotenv()

# --- Configuration & Setup ---
st.set_page_config(
    page_title="Safety Lens",
    page_icon="🛡️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Report Directory Setup
REPORTS_DIR = Path(os.getcwd()) / "reports"
REPORTS_DIR.mkdir(exist_ok=True)
CAPTURES_DIR = Path(os.getcwd()) / "captures"
CAPTURES_DIR.mkdir(exist_ok=True)

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
        /* ===== Global Styles ===== */
        @import url('https://fonts.googleapis.com/css2?family=Noto+Sans+KR:wght@300;400;500;700&display=swap');

        .stApp {
            background: linear-gradient(135deg, #1a1a2e 0%, #16213e 100%);
            font-family: 'Noto Sans KR', 'Segoe UI', sans-serif;
        }

        /* Main content text color */
        .stApp, .stApp p, .stApp span, .stApp label, .stApp div {
            color: #e0e0e0 !important;
            font-size: 1.15rem !important;
        }

        h1, h2, h3, h4, h5, h6 {
            color: #00d4ff !important;
            font-weight: 600 !important;
        }

        /* Base font size increase */
        html, body {
            font-size: 18px !important;
        }

        /* ===== Sidebar Styles ===== */
        [data-testid="stSidebar"] {
            background: linear-gradient(180deg, #0f0f1a 0%, #1a1a2e 100%) !important;
            border-right: 1px solid rgba(0, 212, 255, 0.2);
        }

        [data-testid="stSidebar"] .stRadio label {
            font-size: 1.1rem !important;
            padding: 10px 15px !important;
            border-radius: 10px;
            transition: all 0.3s ease;
        }

        [data-testid="stSidebar"] .stRadio label:hover {
            background: rgba(0, 212, 255, 0.1);
        }

        /* ===== Card Styles ===== */
        .card {
            background: rgba(255, 255, 255, 0.05) !important;
            border-radius: 20px !important;
            padding: 25px !important;
            border: 1px solid rgba(255, 255, 255, 0.1) !important;
            backdrop-filter: blur(10px);
            margin-bottom: 20px;
        }

        /* ===== Status Badge ===== */
        .status-badge {
            padding: 1.5rem 2.5rem;
            border-radius: 15px;
            text-align: center;
            font-weight: bold;
            font-size: 36px;
            margin-bottom: 25px;
            transition: all 0.3s ease;
            backdrop-filter: blur(10px);
        }

        .status-safe {
            background: rgba(0, 255, 136, 0.15);
            color: #00ff88 !important;
            border: 2px solid #00ff88;
            box-shadow: 0 0 30px rgba(0, 255, 136, 0.2);
        }

        .status-danger {
            background: rgba(255, 68, 68, 0.2);
            color: #ff4444 !important;
            border: 2px solid #ff4444;
            box-shadow: 0 0 30px rgba(255, 68, 68, 0.4);
            animation: pulse 1.5s infinite;
        }

        /* ===== Machine Container ===== */
        .machine-container {
            display: flex;
            justify-content: center;
            align-items: center;
            height: 200px;
            margin: 20px 0;
            background: rgba(255, 255, 255, 0.03);
            border-radius: 20px;
            border: 1px solid rgba(255, 255, 255, 0.1);
        }

        .gear-icon {
            font-size: 100px;
            display: inline-block;
        }

        .spinning {
            animation: spin 3s linear infinite;
        }

        .stopped {
            animation: none;
            opacity: 0.4;
            transform: scale(0.85);
        }

        /* ===== Log Area ===== */
        .log-area {
            font-family: 'Courier New', monospace;
            font-size: 1.1rem;
            background: rgba(0, 0, 0, 0.3);
            padding: 20px;
            border-radius: 12px;
            max-height: 250px;
            overflow-y: auto;
            border: 1px solid rgba(0, 212, 255, 0.2);
            color: #00d4ff !important;
            line-height: 1.6;
        }

        /* ===== Animations ===== */
        @keyframes spin {
            0% { transform: rotate(0deg); }
            100% { transform: rotate(360deg); }
        }

        @keyframes pulse {
            0% { box-shadow: 0 0 0 0 rgba(255, 68, 68, 0.6); }
            70% { box-shadow: 0 0 0 25px rgba(255, 68, 68, 0); }
            100% { box-shadow: 0 0 0 0 rgba(255, 68, 68, 0); }
        }

        /* ===== Result Cards ===== */
        .result-card {
            background: rgba(255, 255, 255, 0.05);
            border-radius: 15px;
            padding: 20px;
            margin: 15px 0;
            border-left: 4px solid #00ff88;
            border: 1px solid rgba(255, 255, 255, 0.1);
        }

        .result-card.danger {
            border-left: 4px solid #ff4444;
            background: rgba(255, 68, 68, 0.1);
        }

        .result-card.warning {
            border-left: 4px solid #ffaa00;
            background: rgba(255, 170, 0, 0.1);
        }

        /* ===== Buttons ===== */
        .stButton > button {
            background: linear-gradient(90deg, #00d4ff, #7b2cbf) !important;
            color: white !important;
            border: none !important;
            padding: 15px 35px !important;
            border-radius: 25px !important;
            font-size: 1.25rem !important;
            font-weight: 600 !important;
            transition: all 0.3s ease !important;
            box-shadow: 0 4px 15px rgba(0, 212, 255, 0.3);
        }

        .stButton > button:hover {
            transform: translateY(-2px) !important;
            box-shadow: 0 8px 25px rgba(0, 212, 255, 0.4) !important;
        }

        .stButton > button[kind="secondary"] {
            background: rgba(255, 68, 68, 0.2) !important;
            border: 2px solid #ff4444 !important;
            color: #ff4444 !important;
        }

        /* ===== Download Button ===== */
        .stDownloadButton > button {
            background: linear-gradient(90deg, #00d4ff, #7b2cbf) !important;
            color: white !important;
            border: none !important;
            padding: 15px 35px !important;
            border-radius: 25px !important;
            font-size: 1.2rem !important;
            font-weight: 600 !important;
            box-shadow: 0 4px 15px rgba(0, 212, 255, 0.3);
        }

        /* ===== Input Fields ===== */
        .stTextInput > div > div > input,
        .stNumberInput > div > div > input,
        .stSelectbox > div > div {
            background: rgba(255, 255, 255, 0.05) !important;
            border: 1px solid rgba(0, 212, 255, 0.3) !important;
            border-radius: 10px !important;
            color: #e0e0e0 !important;
            font-size: 1rem !important;
        }

        /* ===== Expander ===== */
        .streamlit-expanderHeader {
            background: rgba(255, 255, 255, 0.05) !important;
            border-radius: 12px !important;
            border: 1px solid rgba(0, 212, 255, 0.2) !important;
            font-size: 1.3rem !important;
            color: #e0e0e0 !important;
            padding: 15px !important;
        }

        .streamlit-expanderContent {
            background: rgba(255, 255, 255, 0.03) !important;
            border: 1px solid rgba(255, 255, 255, 0.1) !important;
            border-radius: 0 0 12px 12px !important;
            padding: 20px !important;
        }

        /* Expander text */
        [data-testid="stExpander"] p {
            font-size: 1.15rem !important;
        }

        /* ===== Metrics ===== */
        [data-testid="stMetricValue"] {
            font-size: 3rem !important;
            font-weight: 700 !important;
            color: #00d4ff !important;
        }

        [data-testid="stMetricLabel"] {
            font-size: 1.3rem !important;
            color: #888 !important;
        }

        /* ===== File Uploader ===== */
        [data-testid="stFileUploader"] {
            background: rgba(0, 212, 255, 0.05) !important;
            border: 2px dashed rgba(0, 212, 255, 0.4) !important;
            border-radius: 15px !important;
            padding: 30px !important;
        }

        [data-testid="stFileUploader"]:hover {
            border-color: #00d4ff !important;
            background: rgba(0, 212, 255, 0.1) !important;
        }

        /* ===== Info/Warning/Error boxes ===== */
        .stAlert {
            border-radius: 12px !important;
            font-size: 1.2rem !important;
            padding: 20px !important;
        }

        .stAlert p {
            font-size: 1.2rem !important;
        }

        /* ===== Image captions ===== */
        .stImage > div > div > p {
            color: #888 !important;
            font-size: 0.95rem !important;
            text-align: center;
        }

        /* ===== Markdown text ===== */
        .stMarkdown p {
            font-size: 1.2rem !important;
            line-height: 1.8 !important;
        }

        .stMarkdown strong {
            color: #00d4ff !important;
        }

        .stMarkdown li {
            font-size: 1.15rem !important;
            line-height: 1.7 !important;
        }

        /* ===== Title styling ===== */
        .stApp h1 {
            font-size: 3rem !important;
            background: linear-gradient(90deg, #00d4ff, #7b2cbf);
            -webkit-background-clip: text;
            -webkit-text-fill-color: transparent;
            background-clip: text;
            margin-bottom: 15px !important;
        }

        .stApp h2 {
            font-size: 2rem !important;
            border-bottom: 2px solid rgba(0, 212, 255, 0.3);
            padding-bottom: 12px;
        }

        .stApp h3 {
            font-size: 1.6rem !important;
        }

        .stApp h4 {
            font-size: 1.4rem !important;
        }

        /* ===== Divider ===== */
        hr {
            border-color: rgba(0, 212, 255, 0.2) !important;
            margin: 25px 0 !important;
        }

        /* ===== Radio buttons ===== */
        .stRadio > label {
            font-size: 1.3rem !important;
        }

        .stRadio > div {
            font-size: 1.2rem !important;
        }

        .stRadio > div > label {
            font-size: 1.2rem !important;
            padding: 12px 15px !important;
        }

        /* ===== Scrollbar ===== */
        ::-webkit-scrollbar {
            width: 8px;
            height: 8px;
        }

        ::-webkit-scrollbar-track {
            background: rgba(255, 255, 255, 0.05);
            border-radius: 10px;
        }

        ::-webkit-scrollbar-thumb {
            background: rgba(0, 212, 255, 0.4);
            border-radius: 10px;
        }

        ::-webkit-scrollbar-thumb:hover {
            background: rgba(0, 212, 255, 0.6);
        }

        /* ===== Analysis result box ===== */
        .analysis-box {
            background: rgba(255, 68, 68, 0.1);
            padding: 25px;
            border-radius: 15px;
            border: 1px solid rgba(255, 68, 68, 0.3);
            margin: 20px 0;
        }

        .analysis-box p {
            font-size: 1.3rem !important;
            margin: 12px 0 !important;
            line-height: 1.6 !important;
        }

        /* ===== Tech badges ===== */
        .tech-badge {
            display: inline-block;
            background: rgba(0, 212, 255, 0.15);
            color: #00d4ff !important;
            padding: 5px 15px;
            border-radius: 20px;
            font-size: 0.9rem;
            margin: 5px;
            border: 1px solid rgba(0, 212, 255, 0.3);
        }
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

    def _get_default_regulations(self, analysis_type: str = "factory") -> dict:
        if analysis_type == "building_plan":
            return self._get_building_plan_regulations()
        elif analysis_type == "construction_site":
            return self._get_construction_site_regulations()
        else:
            return self._get_factory_regulations()

    def _get_factory_regulations(self) -> dict:
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
                "type": "factory",
                "fetched_at": datetime.now().isoformat()
            }
        }

    def _get_building_plan_regulations(self) -> dict:
        """건축 도면 관련 법규"""
        return {
            "건축구조": {
                "내진설계": {"rule": "건축법 제48조, 건축물의 구조기준 등에 관한 규칙", "description": "지진에 대비한 내진설계 적용 (3층 이상 또는 연면적 500㎡ 이상)", "severity": "high"},
                "구조안전": {"rule": "건축법 제48조", "description": "구조내력 기준 충족 여부", "severity": "high"},
                "기초설계": {"rule": "건축물의 구조기준 등에 관한 규칙 제18조", "description": "지반조사에 따른 적정 기초 설계", "severity": "high"}
            },
            "피난안전": {
                "피난계단": {"rule": "건축법 시행령 제35조", "description": "5층 이상 건물은 피난계단 또는 특별피난계단 설치", "severity": "high"},
                "피난통로폭": {"rule": "건축법 시행령 제15조의2", "description": "복도 폭 1.2m 이상 (양측 거실) 또는 1.5m 이상", "severity": "high"},
                "출구수": {"rule": "건축법 시행령 제39조", "description": "관람집회시설 등은 2개소 이상 출구 확보", "severity": "high"},
                "피난거리": {"rule": "건축법 시행령 제34조", "description": "거실에서 직통계단까지 보행거리 30m 이내", "severity": "high"}
            },
            "방화구획": {
                "면적구획": {"rule": "건축법 시행령 제46조", "description": "10층 이하 층 1,000㎡마다, 11층 이상 층 200㎡마다 방화구획", "severity": "high"},
                "수직구획": {"rule": "건축법 시행령 제46조", "description": "3층 이상 층과 지하층은 층별 방화구획", "severity": "high"},
                "방화문": {"rule": "건축법 시행령 제64조", "description": "방화구획에 갑종 또는 을종 방화문 설치", "severity": "high"}
            },
            "소방시설": {
                "스프링클러": {"rule": "소방시설 설치 및 관리에 관한 법률 시행령", "description": "특정소방대상물에 스프링클러 설치", "severity": "high"},
                "비상경보": {"rule": "소방시설 설치 및 관리에 관한 법률", "description": "비상경보설비 및 자동화재탐지설비 설치", "severity": "high"},
                "피난유도": {"rule": "소방시설 설치 및 관리에 관한 법률", "description": "피난구유도등, 통로유도등 설치", "severity": "medium"}
            },
            "장애인편의": {
                "경사로": {"rule": "장애인등편의법 시행령", "description": "주출입구 접근로에 경사로 설치 (기울기 1/12 이하)", "severity": "medium"},
                "승강기": {"rule": "장애인등편의법 시행령", "description": "6층 이상 또는 연면적 2,000㎡ 이상 시 장애인용 승강기 설치", "severity": "medium"},
                "화장실": {"rule": "장애인등편의법 시행령", "description": "장애인 화장실 설치 (유효폭 0.9m 이상)", "severity": "medium"},
                "출입문": {"rule": "장애인등편의법 시행령", "description": "출입문 유효폭 0.9m 이상", "severity": "medium"}
            },
            "채광환기": {
                "채광면적": {"rule": "건축법 시행령 제51조", "description": "거실 바닥면적의 1/10 이상 채광 창문", "severity": "medium"},
                "환기면적": {"rule": "건축법 시행령 제51조", "description": "거실 바닥면적의 1/20 이상 환기 창문", "severity": "medium"}
            },
            "주차시설": {
                "주차대수": {"rule": "주차장법 시행령", "description": "용도별 건축물 연면적에 따른 주차대수 확보", "severity": "medium"},
                "주차구획": {"rule": "주차장법 시행규칙", "description": "일반 2.5m×5.0m, 장애인 3.3m×5.0m 이상", "severity": "low"}
            },
            "_metadata": {
                "source": "default",
                "type": "building_plan",
                "fetched_at": datetime.now().isoformat()
            }
        }

    def _get_construction_site_regulations(self) -> dict:
        """건설 현장 관련 법규"""
        return {
            "개인보호구": {
                "안전모": {"rule": "산업안전보건기준에 관한 규칙 제32조", "description": "낙하물 위험장소에서 안전모 착용 필수", "severity": "high"},
                "안전대": {"rule": "산업안전보건기준에 관한 규칙 제44조", "description": "높이 2m 이상 추락위험장소에서 안전대 착용", "severity": "high"},
                "안전화": {"rule": "산업안전보건기준에 관한 규칙 제32조", "description": "중량물 취급, 못 등 위험물 산재 장소에서 안전화 착용", "severity": "high"},
                "반사조끼": {"rule": "산업안전보건기준에 관한 규칙", "description": "차량 통행 구역에서 반사조끼 착용", "severity": "medium"}
            },
            "추락방지": {
                "안전난간": {"rule": "산업안전보건기준에 관한 규칙 제13조", "description": "개구부, 작업발판 끝에 높이 90cm 이상 안전난간 설치", "severity": "high"},
                "개구부덮개": {"rule": "산업안전보건기준에 관한 규칙 제14조", "description": "바닥 개구부에 덮개 또는 안전난간 설치", "severity": "high"},
                "안전망": {"rule": "산업안전보건기준에 관한 규칙 제42조", "description": "추락높이 10m 이상 시 안전망 설치", "severity": "high"},
                "작업발판": {"rule": "산업안전보건기준에 관한 규칙 제56조", "description": "발판 폭 40cm 이상, 틈새 3cm 이하", "severity": "high"}
            },
            "비계안전": {
                "비계설치": {"rule": "산업안전보건기준에 관한 규칙 제57조", "description": "비계 기둥 간격 띠장 방향 1.85m 이하", "severity": "high"},
                "작업발판폭": {"rule": "산업안전보건기준에 관한 규칙 제58조", "description": "비계 작업발판 폭 40cm 이상", "severity": "high"},
                "벽이음": {"rule": "산업안전보건기준에 관한 규칙 제60조", "description": "강관비계는 수직 5m, 수평 5.5m 간격으로 벽이음", "severity": "high"}
            },
            "굴착안전": {
                "토사붕괴방지": {"rule": "산업안전보건기준에 관한 규칙 제338조", "description": "굴착깊이 2m 이상 시 흙막이 또는 경사면 확보", "severity": "high"},
                "굴착기울기": {"rule": "산업안전보건기준에 관한 규칙 제339조", "description": "지반별 굴착면 기울기 준수 (보통흙 1:1 이상)", "severity": "high"},
                "점검": {"rule": "산업안전보건기준에 관한 규칙 제340조", "description": "굴착작업 전 지반상태 점검", "severity": "medium"}
            },
            "양중작업": {
                "크레인안전": {"rule": "산업안전보건기준에 관한 규칙 제132조", "description": "정격하중 초과 금지, 과부하방지장치 설치", "severity": "high"},
                "신호수배치": {"rule": "산업안전보건기준에 관한 규칙 제38조", "description": "양중작업 시 신호수 배치", "severity": "high"},
                "작업반경": {"rule": "산업안전보건기준에 관한 규칙", "description": "양중기 작업반경 내 출입금지", "severity": "high"}
            },
            "전기안전": {
                "가설전기": {"rule": "산업안전보건기준에 관한 규칙 제304조", "description": "가설전기 접지 및 누전차단기 설치", "severity": "high"},
                "전선관리": {"rule": "산업안전보건기준에 관한 규칙 제311조", "description": "전선 피복 손상 방지, 통로 횡단 시 매립 또는 방호", "severity": "high"},
                "습윤장소": {"rule": "산업안전보건기준에 관한 규칙 제303조", "description": "습윤장소 전기기계기구 방수형 사용", "severity": "high"}
            },
            "화재예방": {
                "소화기비치": {"rule": "산업안전보건기준에 관한 규칙 제241조", "description": "용접, 절단 작업장 반경 5m 내 소화기 비치", "severity": "high"},
                "인화물질": {"rule": "산업안전보건기준에 관한 규칙 제232조", "description": "인화성 물질 저장소 환기 및 화기엄금", "severity": "high"},
                "용접작업": {"rule": "산업안전보건기준에 관한 규칙 제241조", "description": "용접 시 불꽃 비산방지 조치", "severity": "high"}
            },
            "통행안전": {
                "통로확보": {"rule": "산업안전보건기준에 관한 규칙 제17조", "description": "통로 폭 0.9m 이상 확보", "severity": "medium"},
                "조명": {"rule": "산업안전보건기준에 관한 규칙 제21조", "description": "작업장 조명 150럭스 이상", "severity": "medium"},
                "정리정돈": {"rule": "산업안전보건기준에 관한 규칙 제3조", "description": "자재 정리정돈 및 통로 장애물 제거", "severity": "medium"}
            },
            "_metadata": {
                "source": "default",
                "type": "construction_site",
                "fetched_at": datetime.now().isoformat()
            }
        }

    def get_regulations(self, force_refresh: bool = False) -> dict:
        if self.regulations and not force_refresh:
            return self.regulations
        self.regulations = self._get_default_regulations()
        self._save_cache(self.regulations)
        return self.regulations

# ==================== Report Manager ====================

REPORTS_META_FILE = CACHE_DIR / "reports_meta.json"

class ReportManager:
    """Manages saved incident reports"""

    def __init__(self):
        self.reports_dir = REPORTS_DIR
        self.meta_file = REPORTS_META_FILE
        self.reports = self._load_meta()

    def _load_meta(self) -> list:
        if self.meta_file.exists():
            try:
                with open(self.meta_file, "r", encoding="utf-8") as f:
                    return json.load(f)
            except:
                return []
        return []

    def _save_meta(self):
        with open(self.meta_file, "w", encoding="utf-8") as f:
            json.dump(self.reports, f, ensure_ascii=False, indent=2)

    def save_report(self, report_id: str, html_content: str, analysis: dict,
                    incident_time: str, capture_base64: str) -> dict:
        """Save report to file system and metadata"""
        # Save HTML file
        html_path = self.reports_dir / f"safety_report_{report_id}.html"
        with open(html_path, "w", encoding="utf-8") as f:
            f.write(html_content)

        # Save capture image
        capture_path = CAPTURES_DIR / f"capture_{report_id}.jpg"
        with open(capture_path, "wb") as f:
            f.write(base64.b64decode(capture_base64))

        # Create metadata
        meta = {
            "id": report_id,
            "incident_time": incident_time,
            "created_at": datetime.now().isoformat(),
            "html_path": str(html_path),
            "capture_path": str(capture_path),
            "risk_level": analysis.get("overall_risk_level", "unknown"),
            "pass_status": analysis.get("pass_status", False),
            "workplace_type": analysis.get("workplace_type", "N/A"),
            "violations_count": len(analysis.get("violations", [])),
            "compliant_count": len(analysis.get("compliant_items", [])),
            "analysis_summary": {
                "violations": analysis.get("violations", []),
                "immediate_actions": analysis.get("immediate_actions", [])
            }
        }

        self.reports.insert(0, meta)
        self._save_meta()

        return {"success": True, "report_id": report_id, "path": str(html_path)}

    def get_all_reports(self) -> list:
        return self.reports

    def get_report(self, report_id: str) -> dict:
        for report in self.reports:
            if report["id"] == report_id:
                return report
        return None

    def get_report_html(self, report_id: str) -> str:
        report = self.get_report(report_id)
        if report and Path(report["html_path"]).exists():
            with open(report["html_path"], "r", encoding="utf-8") as f:
                return f.read()
        return None

    def delete_report(self, report_id: str) -> bool:
        report = self.get_report(report_id)
        if report:
            # Delete files
            try:
                if Path(report["html_path"]).exists():
                    os.remove(report["html_path"])
                if Path(report["capture_path"]).exists():
                    os.remove(report["capture_path"])
            except:
                pass

            # Remove from meta
            self.reports = [r for r in self.reports if r["id"] != report_id]
            self._save_meta()
            return True
        return False

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

    def analyze_image_bytes(self, image_bytes: bytes, filename: str, analysis_type: str = "factory") -> dict:
        image_data = base64.b64encode(image_bytes).decode("utf-8")
        ext = os.path.splitext(filename)[1].lower()
        mime_type = "image/png" if ext == ".png" else "image/jpeg"

        regulations = self.regulations_fetcher._get_default_regulations(analysis_type)
        regulations_text = self._build_regulations_text(regulations)

        prompt = self._get_prompt_by_type(analysis_type, regulations_text)
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

    def _get_prompt_by_type(self, analysis_type: str, regulations_text: str) -> str:
        """분석 유형별 맞춤 프롬프트 반환"""

        base_response_format = """
        중요: 모든 텍스트 내용은 반드시 한국어로 작성하세요!

        다음 JSON 형식으로만 응답하세요 (마크다운이나 코드 블록 없이):
        {
            "workplace_type": "작업장/건물/현장 유형을 한국어로 설명",
            "overall_risk_level": "high" 또는 "medium" 또는 "low",
            "pass_status": true 또는 false,
            "violations": [
                {
                    "category": "카테고리명 (한국어)",
                    "item": "위반 항목명 (한국어)",
                    "severity": "high" 또는 "medium" 또는 "low",
                    "description": "위반 내용 상세 설명 (한국어)",
                    "rule": "관련 법규/규정 (한국어)",
                    "recommendation": "개선 방안 (한국어)"
                }
            ],
            "compliant_items": [
                {
                    "category": "카테고리명 (한국어)",
                    "item": "준수 항목명 (한국어)",
                    "description": "준수 내용 설명 (한국어)"
                }
            ],
            "immediate_actions": ["즉시 조치사항 1 (한국어)", "즉시 조치사항 2 (한국어)"],
            "recommendations": ["권고사항 1 (한국어)", "권고사항 2 (한국어)"]
        }
        """

        if analysis_type == "building_plan":
            return f"""당신은 전문 건축법규 분석가입니다. 제공된 건축 도면/설계도를 분석하여 대한민국 건축법, 건축물의 피난·방화구조 등의 기준에 관한 규칙, 주차장법, 장애인·노인·임산부 등의 편의증진 보장에 관한 법률 등 관련 법규 위반 여부를 상세히 검토하세요.

분석 대상:
- 건축 도면, 평면도, 단면도, 입면도
- 설계 도서, CAD 도면
- 건축 계획안, 배치도

검토 항목:
1. 피난 안전: 피난계단 설치 기준, 피난통로 폭, 비상구 위치
2. 방화 구획: 방화벽, 방화문, 스프링클러 설치 기준
3. 구조 안전: 내진설계 기준, 구조안전 기준
4. 주차시설: 주차대수 기준, 장애인주차구역
5. 장애인 편의시설: 경사로, 점자블록, 장애인화장실
6. 채광/환기: 거실 채광 기준, 환기시설 기준
7. 건폐율/용적률: 법정 기준 준수 여부

관련 법규:
{regulations_text}

{base_response_format}

위반사항이 있으면 violations에 상세히 기록하고, 법규를 준수한 항목은 compliant_items에 기록하세요.
도면에서 확인 가능한 모든 법규 위반 가능성을 분석하세요."""

        elif analysis_type == "construction_site":
            return f"""당신은 전문 건설현장 안전관리 전문가입니다. 제공된 건설현장 사진을 분석하여 산업안전보건법, 건설기술 진흥법, 산업안전보건기준에 관한 규칙 등 관련 법규 위반 여부를 상세히 검토하세요.

분석 대상:
- 건설현장 사진, 시공 현장 이미지
- 작업자 안전장비 착용 상태
- 안전시설물 설치 현황

검토 항목:
1. 개인보호구(PPE): 안전모, 안전화, 안전벨트, 안전조끼 착용
2. 추락 방지: 안전난간, 개구부 덮개, 추락방지망, 안전대 부착설비
3. 비계 안전: 비계 설치 기준, 안전발판, 작업발판 폭
4. 굴착 안전: 토사붕괴 방지, 흙막이, 경사면 안전
5. 양중 작업: 크레인 안전, 인양물 경고표지, 유도자 배치
6. 전기 안전: 전기설비 접지, 누전차단기, 가설전기 기준
7. 화재 예방: 소화기 비치, 용접작업 화재감시
8. 통행 안전: 통행로 확보, 표지판 설치, 조명 확보

관련 법규:
{regulations_text}

{base_response_format}

현장에서 발견된 모든 안전 위반사항을 상세히 분석하고, 즉시 시정이 필요한 사항과 개선 권고사항을 명확히 구분하세요."""

        else:  # factory (default)
            return f"""당신은 전문 산업안전 분석가입니다. 제공된 공장/제조시설 이미지를 분석하여 산업안전보건법, 산업안전보건기준에 관한 규칙 등 관련 법규 위반 여부를 상세히 검토하세요.

분석 대상:
- 공장, 제조시설, 작업장 이미지
- 기계설비, 생산라인
- 작업 환경, 안전시설

검토 항목:
1. 개인보호구(PPE): 안전모, 보안경, 안전장갑, 방진마스크 등
2. 기계 안전: 방호장치, 비상정지장치, 경고표지
3. 전기 안전: 전기설비 상태, 접지, 누전차단기
4. 화학물질 관리: 경고표지, MSDS 비치, 보관기준
5. 정리정돈: 통행로 확보, 작업장 청결
6. 소방 안전: 소화기, 비상구, 피난통로
7. 환기시설: 국소배기장치, 전체환기

관련 법규:
{regulations_text}

{base_response_format}

이미지에서 확인 가능한 모든 안전 위험요소를 분석하고, 심각도에 따라 분류하세요."""

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

# ==================== HTML Report Generator ====================

def generate_safety_report_html(analysis_result: dict, capture_image_base64: str, incident_time: str) -> tuple:
    """Generate HTML safety report from analysis result"""

    report_id = f"SL-{datetime.now().strftime('%Y%m%d-%H%M%S')}"

    # Get analysis type info
    analysis_type = analysis_result.get("analysis_type", "factory")
    analysis_type_label = analysis_result.get("analysis_type_label", "🏭 공장/제조시설 안전")

    type_titles = {
        "factory": "산업안전 분석 보고서",
        "building_plan": "건축법규 분석 보고서",
        "construction_site": "건설현장 안전 보고서"
    }
    report_title = type_titles.get(analysis_type, "안전 분석 보고서")

    risk_level = analysis_result.get("overall_risk_level", "high")
    risk_color = {"high": "#ff4444", "medium": "#ffaa00", "low": "#00ff88"}.get(risk_level, "#ff4444")
    pass_status = analysis_result.get("pass_status", False)
    status_text = "규정 준수" if pass_status else "개선 필요"
    status_color = "#00ff88" if pass_status else "#ff4444"

    violations = analysis_result.get("violations", [])
    compliant_items = analysis_result.get("compliant_items", [])
    immediate_actions = analysis_result.get("immediate_actions", [])
    recommendations = analysis_result.get("recommendations", [])

    violations_html = ""
    for v in violations:
        severity = v.get("severity", "medium")
        sev_color = {"high": "#ff4444", "medium": "#ffaa00", "low": "#00ff88"}.get(severity, "#ffaa00")
        violations_html += f"""
        <div class="item violation">
            <div class="item-header">
                <span class="category">[{v.get('category', '')}] {v.get('item', '')}</span>
                <span class="severity" style="background: {sev_color};">{severity.upper()}</span>
            </div>
            <div class="item-body">
                <p><strong>위반 내용:</strong> {v.get('description', '')}</p>
                <p><strong>관련 규정:</strong> {v.get('rule', 'N/A')}</p>
                <p><strong>개선 권고:</strong> {v.get('recommendation', 'N/A')}</p>
            </div>
        </div>
        """

    compliant_html = ""
    for c in compliant_items:
        compliant_html += f"""
        <div class="item compliant">
            <div class="item-header">
                <span class="category">[{c.get('category', '')}] {c.get('item', '')}</span>
            </div>
            <div class="item-body">
                <p>{c.get('description', '')}</p>
            </div>
        </div>
        """

    immediate_html = "".join([f"<li>{action}</li>" for action in immediate_actions]) if immediate_actions else "<li>없음</li>"
    recommendations_html = "".join([f"<li>{rec}</li>" for rec in recommendations]) if recommendations else "<li>없음</li>"

    html_content = f"""<!DOCTYPE html>
<html lang="ko">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Safety Lens Report - {report_id}</title>
    <style>
        * {{ margin: 0; padding: 0; box-sizing: border-box; }}
        body {{
            font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
            background: linear-gradient(135deg, #1a1a2e 0%, #16213e 100%);
            min-height: 100vh;
            color: #fff;
            padding: 40px;
        }}
        .container {{ max-width: 1200px; margin: 0 auto; }}
        .header {{
            text-align: center;
            padding: 30px;
            background: rgba(255,255,255,0.05);
            border-radius: 20px;
            margin-bottom: 30px;
            border: 1px solid rgba(255,255,255,0.1);
        }}
        .header h1 {{
            font-size: 2.5em;
            background: linear-gradient(90deg, #00d4ff, #7b2cbf);
            -webkit-background-clip: text;
            -webkit-text-fill-color: transparent;
            margin-bottom: 10px;
        }}
        .report-id {{ color: #888; font-size: 1.1em; }}
        .incident-time {{ color: #ff4444; font-size: 1.2em; margin-top: 10px; font-weight: bold; }}
        .main-grid {{ display: grid; grid-template-columns: 1fr 1fr; gap: 30px; }}
        @media (max-width: 900px) {{ .main-grid {{ grid-template-columns: 1fr; }} }}
        .card {{
            background: rgba(255,255,255,0.05);
            border-radius: 20px;
            padding: 25px;
            border: 1px solid rgba(255,255,255,0.1);
        }}
        .card h2 {{
            color: #00d4ff;
            margin-bottom: 20px;
            padding-bottom: 10px;
            border-bottom: 1px solid rgba(255,255,255,0.1);
        }}
        .capture-image {{
            width: 100%;
            border-radius: 15px;
            border: 3px solid #ff4444;
            box-shadow: 0 0 30px rgba(255, 68, 68, 0.3);
        }}
        .status-badge {{
            display: inline-flex;
            align-items: center;
            gap: 10px;
            padding: 15px 30px;
            border-radius: 30px;
            font-size: 1.3em;
            font-weight: bold;
            margin: 20px 0;
            background: rgba(255,68,68,0.2);
            color: {status_color};
            border: 2px solid {status_color};
        }}
        .risk-badge {{
            display: inline-block;
            padding: 8px 20px;
            border-radius: 20px;
            font-weight: bold;
            background: {risk_color};
            color: #fff;
            margin-left: 15px;
        }}
        .summary-grid {{ display: grid; grid-template-columns: repeat(3, 1fr); gap: 15px; margin: 20px 0; }}
        .summary-item {{
            text-align: center;
            padding: 20px;
            background: rgba(255,255,255,0.05);
            border-radius: 15px;
        }}
        .summary-number {{ font-size: 2.5em; font-weight: bold; }}
        .summary-label {{ color: #888; margin-top: 5px; }}
        .pass {{ color: #00ff88; }}
        .fail {{ color: #ff4444; }}
        .item {{
            background: rgba(255,255,255,0.05);
            padding: 20px;
            border-radius: 15px;
            margin-bottom: 15px;
            border-left: 4px solid #00d4ff;
        }}
        .item.violation {{ border-left-color: #ff4444; }}
        .item.compliant {{ border-left-color: #00ff88; }}
        .item-header {{
            display: flex;
            justify-content: space-between;
            align-items: center;
            margin-bottom: 10px;
        }}
        .category {{ font-weight: bold; color: #00d4ff; }}
        .severity {{
            padding: 4px 12px;
            border-radius: 12px;
            font-size: 0.8em;
            color: #000;
        }}
        .item-body {{ color: #ccc; line-height: 1.8; }}
        .item-body p {{ margin: 5px 0; }}
        .action-list {{ list-style: none; }}
        .action-list li {{
            background: rgba(255,68,68,0.1);
            padding: 12px 15px;
            border-radius: 10px;
            margin-bottom: 8px;
            border-left: 3px solid #ff4444;
        }}
        .rec-list {{ list-style: none; }}
        .rec-list li {{
            background: rgba(0,212,255,0.1);
            padding: 12px 15px;
            border-radius: 10px;
            margin-bottom: 8px;
            border-left: 3px solid #00d4ff;
        }}
        .footer {{
            text-align: center;
            margin-top: 40px;
            padding: 20px;
            color: #666;
            border-top: 1px solid rgba(255,255,255,0.1);
        }}
        .footer a {{ color: #00d4ff; text-decoration: none; }}
        @media print {{
            body {{ background: #fff; color: #000; }}
            .card {{ border: 1px solid #ddd; }}
        }}
    </style>
</head>
<body>
    <div class="container">
        <div class="header">
            <h1>Safety Lens - {report_title}</h1>
            <div style="background: rgba(0,212,255,0.2); display: inline-block; padding: 8px 20px; border-radius: 20px; margin: 10px 0; color: #00d4ff; font-size: 1.1em;">
                {analysis_type_label}
            </div>
            <div class="report-id">Report ID: {report_id}</div>
            <div class="incident-time">분석 시간: {incident_time}</div>
        </div>

        <div class="main-grid">
            <div class="card">
                <h2>📸 분석 대상 이미지</h2>
                <img src="data:image/jpeg;base64,{capture_image_base64}" class="capture-image" alt="분석 이미지">
                <div style="text-align: center; margin-top: 20px;">
                    <span class="status-badge">⚠️ {status_text}</span>
                    <span class="risk-badge">위험도: {risk_level.upper()}</span>
                </div>
            </div>

            <div class="card">
                <h2>📊 분석 요약</h2>
                <p style="margin-bottom: 15px;"><strong>작업장 유형:</strong> {analysis_result.get('workplace_type', 'N/A')}</p>
                <div class="summary-grid">
                    <div class="summary-item">
                        <div class="summary-number pass">{len(compliant_items)}</div>
                        <div class="summary-label">준수 항목</div>
                    </div>
                    <div class="summary-item">
                        <div class="summary-number fail">{len(violations)}</div>
                        <div class="summary-label">위반 항목</div>
                    </div>
                    <div class="summary-item">
                        <div class="summary-number" style="color: #00d4ff;">{len(compliant_items) + len(violations)}</div>
                        <div class="summary-label">총 검사</div>
                    </div>
                </div>

                <h3 style="color: #ff4444; margin: 20px 0 15px;">🚨 즉시 조치 필요</h3>
                <ul class="action-list">{immediate_html}</ul>
            </div>
        </div>

        <div class="card" style="margin-top: 30px;">
            <h2>⚠️ 규정 위반 사항</h2>
            {violations_html if violations_html else '<p style="color: #888;">위반 사항 없음</p>'}
        </div>

        <div class="card" style="margin-top: 30px;">
            <h2>✅ 규정 준수 사항</h2>
            {compliant_html if compliant_html else '<p style="color: #888;">준수 사항 없음</p>'}
        </div>

        <div class="card" style="margin-top: 30px;">
            <h2>💡 권고사항</h2>
            <ul class="rec-list">{recommendations_html}</ul>
        </div>

        <div class="footer">
            <p>Generated by <a href="#">Safety Lens</a> | {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}</p>
            <p style="margin-top: 10px;">Powered by SpoonOS + Gemini Vision AI</p>
        </div>
    </div>
</body>
</html>"""

    return html_content, report_id

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

# ==================== WebRTC Video Processor ====================

class SafetyVideoProcessor(VideoProcessorBase):
    """Video processor for real-time safety monitoring via WebRTC"""

    def __init__(self):
        self.detector = get_detector()
        self.prev_gray = None
        self.prev_hand_area = None
        self.prev_hand_centroid = None
        self.motion_buffer = 0
        self.squeeze_buffer = 0

        # Hazard counters
        self.pinch_frame_count = 0
        self.impact_frame_count = 0
        self.hazard_count_l = 0
        self.hazard_count_r = 0
        self.hazard_count_t = 0
        self.hazard_count_b = 0

        # State
        self.current_status = "SAFE"
        self.hazard_latched = False
        self.captured_frame = None
        self.captured_frame_base64 = None
        self.incident_time = None
        self.lock = threading.Lock()

    def get_hazard_state(self):
        """Get current hazard state (thread-safe)"""
        with self.lock:
            return {
                "status": self.current_status,
                "hazard_latched": self.hazard_latched,
                "captured_frame": self.captured_frame,
                "captured_frame_base64": self.captured_frame_base64,
                "incident_time": self.incident_time
            }

    def reset_state(self):
        """Reset hazard state"""
        with self.lock:
            self.pinch_frame_count = 0
            self.impact_frame_count = 0
            self.hazard_count_l = 0
            self.hazard_count_r = 0
            self.hazard_count_t = 0
            self.hazard_count_b = 0
            self.current_status = "SAFE"
            self.hazard_latched = False
            self.captured_frame = None
            self.captured_frame_base64 = None
            self.incident_time = None

    def recv(self, frame: av.VideoFrame) -> av.VideoFrame:
        img = frame.to_ndarray(format="bgr24")
        processed = self.process_frame(img)
        return av.VideoFrame.from_ndarray(processed, format="bgr24")

    def process_frame(self, frame):
        """Process a single frame for safety hazards"""
        h, w, c = frame.shape
        small_w, small_h = 320, 240
        frame_small = cv2.resize(frame, (small_w, small_h))
        gray_small = cv2.cvtColor(frame_small, cv2.COLOR_BGR2GRAY)

        # If hazard is latched, show warning overlay
        if self.hazard_latched:
            cv2.putText(frame, "SYSTEM HALTED", (50, 100), cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0,0,255), 4)
            cv2.putText(frame, "PRESS 'RESET' TO RESUME", (50, 160), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0,0,255), 3)
            cv2.rectangle(frame, (0,0), (w-1,h-1), (0,0,255), 20)
            return frame

        pinch_detected = False
        flow = None

        if self.prev_gray is not None:
            flow = cv2.calcOpticalFlowFarneback(self.prev_gray, gray_small, None, 0.5, 2, 10, 2, 5, 1.1, 0)
        self.prev_gray = gray_small

        # Hand Detection
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        hand_detected = False

        if self.detector:
            mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=frame_rgb)
            detection_result = self.detector.detect(mp_image)

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

                    # Compression Detection
                    current_raw_area = (x_max - x_min) * (y_max - y_min)
                    compression_detected = False
                    current_area = 0.5 * current_raw_area + 0.5 * self.prev_hand_area if self.prev_hand_area else current_raw_area

                    if self.prev_hand_area:
                        percent_change = (current_area - self.prev_hand_area) / self.prev_hand_area
                        if percent_change < -0.05:
                            compression_detected = True
                            cv2.putText(frame, "SQUEEZE!", (x_min, y_min-30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
                    self.prev_hand_area = current_area

                    # Optical Flow Analysis
                    if flow is not None:
                        s_x_min, s_x_max = int(x_min * sx), int(x_max * sx)
                        s_y_min, s_y_max = int(y_min * sy), int(y_max * sy)
                        roi_w = int(100 * sx)
                        velocity_thresh, pixel_thresh = 4.0, 400

                        current_centroid = ((x_min + x_max)//2, (y_min + y_max)//2)
                        suppress_hazards = False
                        if self.prev_hand_centroid:
                            dist = np.sqrt((current_centroid[0]-self.prev_hand_centroid[0])**2 + (current_centroid[1]-self.prev_hand_centroid[1])**2)
                            if dist > 15.0:
                                suppress_hazards = True
                        self.prev_hand_centroid = current_centroid

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
                        if any_motion_active: self.motion_buffer = 5
                        if compression_detected: self.squeeze_buffer = 5
                        self.motion_buffer = max(0, self.motion_buffer - 1)
                        self.squeeze_buffer = max(0, self.squeeze_buffer - 1)
                        buffered_trigger = (self.motion_buffer > 0) and (self.squeeze_buffer > 0)

                        strict_pincer = (left_active and right_active) or (top_active and bottom_active)
                        if strict_pincer: self.pinch_frame_count += 1
                        else: self.pinch_frame_count = 0

                        if buffered_trigger: self.impact_frame_count += 1
                        else: self.impact_frame_count = max(0, self.impact_frame_count - 1)

                        if left_active: self.hazard_count_l += 2
                        else: self.hazard_count_l = max(0, self.hazard_count_l - 1)
                        if right_active: self.hazard_count_r += 2
                        else: self.hazard_count_r = max(0, self.hazard_count_r - 1)
                        if top_active: self.hazard_count_t += 2
                        else: self.hazard_count_t = max(0, self.hazard_count_t - 1)
                        if bottom_active: self.hazard_count_b += 2
                        else: self.hazard_count_b = max(0, self.hazard_count_b - 1)

                        max_hazard = max(self.hazard_count_l, self.hazard_count_r, self.hazard_count_t, self.hazard_count_b)
                        if max_hazard >= 10: pinch_detected = True
                        if self.pinch_frame_count > 3: pinch_detected = True
                        if self.impact_frame_count > 12: pinch_detected = True

        # Update State
        if pinch_detected:
            with self.lock:
                self.current_status = "DANGER"
                self.hazard_latched = True
                self.incident_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                _, buffer = cv2.imencode('.jpg', frame)
                self.captured_frame = frame.copy()
                self.captured_frame_base64 = base64.b64encode(buffer).decode('utf-8')
            cv2.putText(frame, "CRUSH HAZARD!", (50, 100), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 4)
        elif hand_detected:
            self.current_status = "SAFE"
            cv2.putText(frame, "Hand Detected - Monitoring...", (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2)
        else:
            self.current_status = "SAFE"
            cv2.putText(frame, "ZONE CLEAR", (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (100, 255, 100), 2)

        return frame

# WebRTC Configuration for STUN/TURN servers
RTC_CONFIGURATION = RTCConfiguration(
    {"iceServers": [
        {"urls": ["stun:stun.l.google.com:19302"]},
        {"urls": ["turn:safetylens.store:3478"],
         "username": "webrtcuser",
         "credential": "webrtcpass"}
    ]}
)

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
    # Capture & Report State
    if 'captured_frame' not in st.session_state:
        st.session_state.captured_frame = None
    if 'captured_frame_base64' not in st.session_state:
        st.session_state.captured_frame_base64 = None
    if 'incident_analysis' not in st.session_state:
        st.session_state.incident_analysis = None
    if 'report_html' not in st.session_state:
        st.session_state.report_html = None
    if 'report_id' not in st.session_state:
        st.session_state.report_id = None
    if 'incident_time' not in st.session_state:
        st.session_state.incident_time = None
    if 'analyzing_incident' not in st.session_state:
        st.session_state.analyzing_incident = False
    if 'download_btn_rendered' not in st.session_state:
        st.session_state.download_btn_rendered = False
    if 'webrtc_processor' not in st.session_state:
        st.session_state.webrtc_processor = None

    st.title("🛡️ Real-time Safety Sentinel")
    st.markdown("### Real-time Factory Safety Monitoring")

    col_video, col_dashboard = st.columns([1, 1])

    with col_dashboard:
        # Reset button when hazard detected
        if st.session_state.hazard_latched:
            if st.button("⏹️ EMERGENCY STOP / RESET", type="secondary"):
                st.session_state.monitor_active = False
                st.session_state.current_status = "SAFE"
                st.session_state.hazard_latched = False
                st.session_state.captured_frame = None
                st.session_state.captured_frame_base64 = None
                st.session_state.incident_analysis = None
                st.session_state.report_html = None
                st.session_state.report_id = None
                st.session_state.incident_time = None
                st.session_state.analyzing_incident = False
                st.session_state.download_btn_rendered = False
                if st.session_state.webrtc_processor:
                    st.session_state.webrtc_processor.reset_state()
                st.rerun()

        st.markdown("---")
        st.markdown("#### 🏭 Machine Status")
        vehicle_status_ph = st.empty()
        machine_anim_ph = st.empty()

        st.markdown("#### 📋 Incident Log")
        log_ph = st.empty()

        # Captured Incident Section (shown when hazard detected)
        st.markdown("---")
        st.markdown("#### 📸 Incident Capture")
        capture_ph = st.empty()
        analysis_status_ph = st.empty()
        analysis_result_ph = st.empty()
        download_btn_ph = st.empty()

    with col_video:
        # WebRTC Streamer - 브라우저 카메라 사용
        ctx = webrtc_streamer(
            key="safety-sentinel",
            mode=WebRtcMode.SENDRECV,
            rtc_configuration=RTC_CONFIGURATION,
            video_processor_factory=SafetyVideoProcessor,
            media_stream_constraints={"video": True, "audio": False},
            async_processing=True,
        )

        # Store processor reference
        if ctx.video_processor:
            st.session_state.webrtc_processor = ctx.video_processor

    # Update UI based on processor state
    if ctx.state.playing and ctx.video_processor:
        processor = ctx.video_processor
        state = processor.get_hazard_state()

        if state["hazard_latched"]:
            st.session_state.hazard_latched = True
            st.session_state.current_status = "DANGER"

            vehicle_status_ph.markdown('<div class="status-badge status-danger">🚫 STOPPED</div>', unsafe_allow_html=True)
            machine_anim_ph.markdown('<div class="machine-container"><div class="gear-icon stopped">⚙️</div></div>', unsafe_allow_html=True)

            # Transfer captured frame to session state (only once)
            if state["captured_frame"] is not None and st.session_state.captured_frame is None:
                st.session_state.captured_frame = state["captured_frame"]
                st.session_state.captured_frame_base64 = state["captured_frame_base64"]
                st.session_state.incident_time = state["incident_time"]
                st.session_state.analyzing_incident = True

                # Log incident
                log_msg = trigger_spoon_agent_incident()
                st.session_state.incident_log.insert(0, log_msg)

            # Display captured frame
            if st.session_state.captured_frame is not None:
                capture_ph.image(
                    cv2.cvtColor(st.session_state.captured_frame, cv2.COLOR_BGR2RGB),
                    caption=f"Incident @ {st.session_state.incident_time}",
                    use_container_width=True
                )

            # Perform AI analysis if not done yet
            if st.session_state.analyzing_incident and st.session_state.incident_analysis is None:
                analysis_status_ph.info("🔍 AI 분석 중...")
                try:
                    regulations_fetcher = RegulationsFetcher()
                    analysis_service = SafetyAnalysisService(regulations_fetcher)
                    image_bytes = base64.b64decode(st.session_state.captured_frame_base64)
                    result = analysis_service.analyze_image_bytes(image_bytes, "incident_capture.jpg")
                    st.session_state.incident_analysis = result

                    # Generate HTML report
                    html_content, report_id = generate_safety_report_html(
                        result,
                        st.session_state.captured_frame_base64,
                        st.session_state.incident_time
                    )
                    st.session_state.report_html = html_content
                    st.session_state.report_id = report_id

                    # Auto-save report to history
                    report_manager = ReportManager()
                    report_manager.save_report(
                        report_id=report_id,
                        html_content=html_content,
                        analysis=result,
                        incident_time=st.session_state.incident_time,
                        capture_base64=st.session_state.captured_frame_base64
                    )

                    st.session_state.analyzing_incident = False
                    analysis_status_ph.success("✅ 분석 완료! 보고서가 자동 저장되었습니다.")
                except Exception as e:
                    analysis_status_ph.error(f"분석 오류: {str(e)}")
                    st.session_state.analyzing_incident = False

            # Display analysis result
            if st.session_state.incident_analysis:
                result = st.session_state.incident_analysis
                risk_level = result.get("overall_risk_level", "high")
                violations = result.get("violations", [])
                risk_color = {'high': '#ff4444', 'medium': '#ffaa00', 'low': '#00ff88'}.get(risk_level, '#ff4444')

                analysis_result_ph.markdown(f"""
                <div class="analysis-box" style="background: rgba(255,68,68,0.15); padding: 25px; border-radius: 15px; border: 2px solid {risk_color};">
                    <p style="font-size: 1.4rem; margin: 15px 0;"><strong style="color: #00d4ff;">위험도:</strong> <span style="color: {risk_color}; font-weight: bold; font-size: 1.6rem;">{risk_level.upper()}</span></p>
                    <p style="font-size: 1.4rem; margin: 15px 0;"><strong style="color: #00d4ff;">위반 항목:</strong> <span style="color: #ff4444; font-weight: bold; font-size: 1.5rem;">{len(violations)}건</span></p>
                    <p style="font-size: 1.4rem; margin: 15px 0;"><strong style="color: #00d4ff;">작업장:</strong> {result.get('workplace_type', 'N/A')}</p>
                </div>
                """, unsafe_allow_html=True)

            # Download button for report
            if st.session_state.report_html and not st.session_state.get('download_btn_rendered'):
                download_btn_ph.download_button(
                    label="📥 보고서 다운로드 (HTML)",
                    data=st.session_state.report_html,
                    file_name=f"safety_report_{st.session_state.report_id}.html",
                    mime="text/html",
                    type="primary",
                    key=f"download_report_{st.session_state.report_id}"
                )
                st.session_state.download_btn_rendered = True

        else:
            st.session_state.current_status = "SAFE"
            vehicle_status_ph.markdown('<div class="status-badge status-safe">OPERATIONAL</div>', unsafe_allow_html=True)
            machine_anim_ph.markdown('<div class="machine-container"><div class="gear-icon spinning">⚙️</div></div>', unsafe_allow_html=True)

        # Update log
        log_html = "<br>".join([f"• {msg}" for msg in st.session_state.incident_log[:5]])
        log_ph.markdown(f'<div class="log-area">{log_html}</div>', unsafe_allow_html=True)

    elif not ctx.state.playing:
        vehicle_status_ph.markdown('<div class="status-badge" style="border-color: #666; color: #666;">OFFLINE</div>', unsafe_allow_html=True)
        machine_anim_ph.markdown('<div class="machine-container"><div class="gear-icon stopped">⚙️</div></div>', unsafe_allow_html=True)

        # Show previous incident data if exists
        if st.session_state.captured_frame is not None:
            capture_ph.image(
                cv2.cvtColor(st.session_state.captured_frame, cv2.COLOR_BGR2RGB),
                caption=f"Incident @ {st.session_state.incident_time}",
                use_container_width=True
            )
        if st.session_state.incident_analysis:
            result = st.session_state.incident_analysis
            risk_level = result.get("overall_risk_level", "high")
            violations = result.get("violations", [])
            risk_color = {'high': '#ff4444', 'medium': '#ffaa00', 'low': '#00ff88'}.get(risk_level, '#ff4444')
            analysis_result_ph.markdown(f"""
            <div class="analysis-box" style="background: rgba(255,68,68,0.15); padding: 25px; border-radius: 15px; border: 2px solid {risk_color};">
                <p style="font-size: 1.4rem; margin: 15px 0;"><strong style="color: #00d4ff;">위험도:</strong> <span style="color: {risk_color}; font-weight: bold; font-size: 1.6rem;">{risk_level.upper()}</span></p>
                <p style="font-size: 1.4rem; margin: 15px 0;"><strong style="color: #00d4ff;">위반 항목:</strong> <span style="color: #ff4444; font-weight: bold; font-size: 1.5rem;">{len(violations)}건</span></p>
                <p style="font-size: 1.4rem; margin: 15px 0;"><strong style="color: #00d4ff;">작업장:</strong> {result.get('workplace_type', 'N/A')}</p>
            </div>
            """, unsafe_allow_html=True)
        if st.session_state.report_html:
            download_btn_ph.download_button(
                label="📥 보고서 다운로드 (HTML)",
                data=st.session_state.report_html,
                file_name=f"safety_report_{st.session_state.report_id}.html",
                mime="text/html",
                type="primary",
                key=f"download_stopped_{st.session_state.report_id or 'none'}"
            )

# ==================== Page 2: Image Analysis ====================

def render_image_analysis():
    load_css()

    st.title("🔍 Safety Regulation Analysis")
    st.markdown("### Upload an image to analyze safety compliance")

    # Initialize services
    regulations_fetcher = RegulationsFetcher()
    memory_store = MemoryStore()
    analysis_service = SafetyAnalysisService(regulations_fetcher)

    # Analysis Type Selector
    st.markdown("### 📋 분석 유형 선택")
    st.markdown("""
    <p style="font-size: 1.2rem; color: #b0b0b0; margin-bottom: 20px;">
        분석하려는 이미지의 유형을 선택하세요. 각 유형에 맞는 전문 법규로 분석됩니다.
    </p>
    """, unsafe_allow_html=True)

    analysis_types = {
        "🏭 공장/제조시설 안전": "factory",
        "📐 건축 도면/설계도": "building_plan",
        "🏗️ 건설현장 안전": "construction_site"
    }

    type_descriptions = {
        "🏭 공장/제조시설 안전": "산업안전보건법 기반 - 공장, 제조시설, 작업장의 안전 규정 위반 검사",
        "📐 건축 도면/설계도": "건축법 기반 - 건축 도면, 설계도의 법규 위반 여부 검토 (피난, 방화, 주차, 편의시설 등)",
        "🏗️ 건설현장 안전": "건설안전법 기반 - 건설현장 사진의 안전시설, PPE 착용, 추락방지 등 검사"
    }

    selected_type_label = st.selectbox(
        "분석 유형",
        options=list(analysis_types.keys()),
        index=0,
        key="analysis_type_selector"
    )

    st.markdown(f"""
    <div style="background: rgba(0,212,255,0.1); padding: 15px 20px; border-radius: 10px; margin: 15px 0; border-left: 4px solid #00d4ff;">
        <p style="font-size: 1.15rem; margin: 0; color: #e0e0e0;">
            ℹ️ {type_descriptions[selected_type_label]}
        </p>
    </div>
    """, unsafe_allow_html=True)

    selected_analysis_type = analysis_types[selected_type_label]

    st.markdown("---")

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

                    # Run analysis (synchronous) with selected type
                    result = analysis_service.analyze_image_bytes(
                        image_bytes,
                        uploaded_file.name,
                        analysis_type=selected_analysis_type
                    )
                    result["analysis_type"] = selected_analysis_type
                    result["analysis_type_label"] = selected_type_label

                    # Save to memory
                    memory_store.add_memory({
                        "filename": uploaded_file.name,
                        "file_path": str(file_path),
                        "analysis": result,
                        "analysis_type": selected_analysis_type
                    })

                    st.session_state.analysis_result = result

                    # Generate report for download
                    image_base64 = base64.b64encode(image_bytes).decode('utf-8')
                    analysis_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                    html_content, report_id = generate_safety_report_html(
                        result,
                        image_base64,
                        analysis_time
                    )
                    st.session_state.image_analysis_report_html = html_content
                    st.session_state.image_analysis_report_id = report_id
                    st.session_state.image_analysis_base64 = image_base64
                    st.session_state.image_analysis_time = analysis_time

                    # Auto-save to report history
                    report_manager = ReportManager()
                    report_manager.save_report(
                        report_id=report_id,
                        html_content=html_content,
                        analysis=result,
                        incident_time=analysis_time,
                        capture_base64=image_base64
                    )

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

                st.markdown(f"<p style='font-size: 1.4rem;'><strong style='color: #00d4ff;'>Workplace Type:</strong> {result.get('workplace_type', 'N/A')}</p>", unsafe_allow_html=True)

                # Violations
                violations = result.get("violations", [])
                if violations:
                    st.markdown("#### ⚠️ Violations")
                    for v in violations:
                        severity = v.get("severity", "medium")
                        severity_emoji = {"high": "🔴", "medium": "🟡", "low": "🟢"}.get(severity, "⚪")
                        severity_color = {"high": "#ff4444", "medium": "#ffaa00", "low": "#00ff88"}.get(severity, "#888")
                        st.markdown(f"""
                        <div style="background: rgba(255,68,68,0.1); padding: 25px; border-radius: 15px; margin: 20px 0; border-left: 5px solid {severity_color}; border: 1px solid rgba(255,255,255,0.1);">
                            <p style="font-size: 1.4rem; font-weight: bold; margin-bottom: 12px;">{severity_emoji} <span style="color: #00d4ff;">[{v.get('category', '')}]</span> {v.get('item', '')}</p>
                            <p style="font-size: 1.2rem; color: #ffaa00; margin: 10px 0;"><em>📜 Rule: {v.get('rule', 'N/A')}</em></p>
                            <p style="font-size: 1.2rem; margin: 10px 0;">{v.get('description', '')}</p>
                            <p style="font-size: 1.2rem; margin: 10px 0; padding: 12px; background: rgba(0,212,255,0.1); border-radius: 8px;"><strong style="color: #00d4ff;">💡 Recommendation:</strong> {v.get('recommendation', 'N/A')}</p>
                        </div>
                        """, unsafe_allow_html=True)

                # Compliant Items
                compliant = result.get("compliant_items", [])
                if compliant:
                    with st.expander("✅ Compliant Items"):
                        for c in compliant:
                            st.markdown(f"""
                            <p style="font-size: 1.25rem; margin: 12px 0; padding: 15px; background: rgba(0,255,136,0.1); border-radius: 8px; border-left: 4px solid #00ff88;">
                                <strong style="color: #00d4ff;">[{c.get('category', '')}] {c.get('item', '')}:</strong> {c.get('description', '')}
                            </p>
                            """, unsafe_allow_html=True)

                # Recommendations
                recommendations = result.get("recommendations", [])
                if recommendations:
                    with st.expander("💡 Recommendations"):
                        for r in recommendations:
                            st.markdown(f"""
                            <p style="font-size: 1.25rem; margin: 10px 0; padding: 15px; background: rgba(0,212,255,0.1); border-radius: 8px; border-left: 4px solid #00d4ff;">
                                {r}
                            </p>
                            """, unsafe_allow_html=True)

                # Download Report Button
                st.markdown("---")
                st.markdown("#### 📥 Download Report")
                if st.session_state.get('image_analysis_report_html'):
                    st.download_button(
                        label="📥 보고서 다운로드 (HTML)",
                        data=st.session_state.image_analysis_report_html,
                        file_name=f"safety_report_{st.session_state.image_analysis_report_id}.html",
                        mime="text/html",
                        type="primary",
                        key=f"download_image_analysis_{st.session_state.image_analysis_report_id}"
                    )
                    st.success(f"✅ 보고서가 Report History에 자동 저장되었습니다. (ID: {st.session_state.image_analysis_report_id})")

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

# ==================== Page 3: Report History (Admin) ====================

def render_report_history():
    load_css()

    st.title("📋 Report History")
    st.markdown("### Incident Report Management")

    report_manager = ReportManager()
    reports = report_manager.get_all_reports()

    # Stats
    col1, col2, col3, col4 = st.columns(4)
    total_reports = len(reports)
    high_risk = len([r for r in reports if r.get("risk_level") == "high"])
    medium_risk = len([r for r in reports if r.get("risk_level") == "medium"])
    low_risk = len([r for r in reports if r.get("risk_level") == "low"])

    with col1:
        st.metric("Total Reports", total_reports)
    with col2:
        st.metric("High Risk", high_risk, delta=None)
    with col3:
        st.metric("Medium Risk", medium_risk)
    with col4:
        st.metric("Low Risk", low_risk)

    st.markdown("---")

    if not reports:
        st.info("No incident reports saved yet. Reports will appear here when hazards are detected during real-time monitoring.")
        return

    # Filter options
    col_filter1, col_filter2 = st.columns([1, 3])
    with col_filter1:
        risk_filter = st.selectbox(
            "Filter by Risk Level",
            ["All", "High", "Medium", "Low"],
            key="risk_filter"
        )

    # Apply filter
    filtered_reports = reports
    if risk_filter != "All":
        filtered_reports = [r for r in reports if r.get("risk_level", "").lower() == risk_filter.lower()]

    # Report list
    st.markdown(f"### Showing {len(filtered_reports)} report(s)")

    for idx, report in enumerate(filtered_reports):
        risk_level = report.get("risk_level", "unknown")
        risk_color = {"high": "#ff4444", "medium": "#ffaa00", "low": "#00ff88"}.get(risk_level, "#888")
        risk_emoji = {"high": "🔴", "medium": "🟡", "low": "🟢"}.get(risk_level, "⚪")

        with st.expander(
            f"{risk_emoji} **{report['id']}** | {report.get('incident_time', 'N/A')} | "
            f"Violations: {report.get('violations_count', 0)}",
            expanded=False
        ):
            col_info, col_image = st.columns([2, 1])

            with col_info:
                st.markdown(f"""
                <div style="padding: 15px 0;">
                    <p style="font-size: 1.3rem; margin: 15px 0;"><strong style="color: #00d4ff;">Report ID:</strong> <code style="background: rgba(0,212,255,0.1); padding: 5px 12px; border-radius: 5px; color: #00d4ff; font-size: 1.2rem;">{report['id']}</code></p>
                    <p style="font-size: 1.3rem; margin: 15px 0;"><strong style="color: #00d4ff;">Incident Time:</strong> {report.get('incident_time', 'N/A')}</p>
                    <p style="font-size: 1.3rem; margin: 15px 0;"><strong style="color: #00d4ff;">Created At:</strong> {report.get('created_at', 'N/A')[:19]}</p>
                    <p style="font-size: 1.3rem; margin: 15px 0;"><strong style="color: #00d4ff;">Workplace Type:</strong> {report.get('workplace_type', 'N/A')}</p>
                    <p style="font-size: 1.3rem; margin: 15px 0;"><strong style="color: #00d4ff;">Risk Level:</strong> <span style="color: {risk_color}; font-weight: bold; font-size: 1.5rem;">{risk_level.upper()}</span></p>
                    <p style="font-size: 1.3rem; margin: 15px 0;"><strong style="color: #00d4ff;">Violations:</strong> <span style="color: #ff4444; font-weight: bold; font-size: 1.4rem;">{report.get('violations_count', 0)}</span> | <strong style="color: #00d4ff;">Compliant:</strong> <span style="color: #00ff88; font-weight: bold; font-size: 1.4rem;">{report.get('compliant_count', 0)}</span></p>
                </div>
                """, unsafe_allow_html=True)

                # Immediate actions
                summary = report.get("analysis_summary", {})
                immediate = summary.get("immediate_actions", [])
                if immediate:
                    st.markdown("**🚨 Immediate Actions Required:**")
                    for action in immediate[:3]:
                        st.markdown(f"<p style='font-size: 1.2rem; margin: 8px 0; padding: 10px 15px; border-left: 4px solid #ff4444; background: rgba(255,68,68,0.1); border-radius: 5px;'>{action}</p>", unsafe_allow_html=True)

            with col_image:
                capture_path = report.get("capture_path", "")
                if capture_path and Path(capture_path).exists():
                    st.image(capture_path, caption="Captured Frame", use_container_width=True)

            # Action buttons
            st.markdown("---")
            btn_col1, btn_col2, btn_col3 = st.columns(3)

            with btn_col1:
                html_content = report_manager.get_report_html(report['id'])
                if html_content:
                    st.download_button(
                        label="📥 Download HTML",
                        data=html_content,
                        file_name=f"safety_report_{report['id']}.html",
                        mime="text/html",
                        key=f"download_{report['id']}_{idx}"
                    )

            with btn_col2:
                if st.button("👁️ View Full Report", key=f"view_{report['id']}_{idx}"):
                    st.session_state.viewing_report = report['id']

            with btn_col3:
                if st.button("🗑️ Delete", key=f"delete_{report['id']}_{idx}", type="secondary"):
                    if report_manager.delete_report(report['id']):
                        st.success(f"Report {report['id']} deleted.")
                        st.rerun()
                    else:
                        st.error("Failed to delete report.")

    # Full report viewer modal
    if st.session_state.get('viewing_report'):
        st.markdown("---")
        st.markdown("### 📄 Full Report View")

        report_id = st.session_state.viewing_report
        html_content = report_manager.get_report_html(report_id)

        if html_content:
            # Display in iframe
            st.components.v1.html(html_content, height=800, scrolling=True)

            if st.button("Close Report View"):
                st.session_state.viewing_report = None
                st.rerun()
        else:
            st.error("Report file not found.")
            st.session_state.viewing_report = None

# ==================== Main App ====================

def main():
    load_css()

    st.sidebar.markdown("""
        <div style="text-align: center; padding: 20px 0;">
            <h1 style="font-size: 2rem; background: linear-gradient(90deg, #00d4ff, #7b2cbf); -webkit-background-clip: text; -webkit-text-fill-color: transparent; margin-bottom: 5px;">🛡️ Safety Lens</h1>
            <p style="color: #888; font-size: 0.95rem;">AI-Powered Safety Monitoring</p>
        </div>
    """, unsafe_allow_html=True)
    st.sidebar.markdown("---")

    page = st.sidebar.radio(
        "Select Page",
        ["🎥 Real-time Sentinel", "🔍 Image Analysis", "📋 Report History"],
        index=0
    )

    st.sidebar.markdown("---")
    st.sidebar.markdown("""
        <div style="padding: 15px; background: rgba(0,212,255,0.05); border-radius: 12px; border: 1px solid rgba(0,212,255,0.2);">
            <h4 style="color: #00d4ff; margin-bottom: 10px; font-size: 1.1rem;">About</h4>
            <p style="color: #aaa; font-size: 0.95rem; line-height: 1.6;">
                Safety Lens is a real-time factory safety monitoring system with AI-powered image analysis.
            </p>
            <div style="margin-top: 15px;">
                <span class="tech-badge" style="display: inline-block; background: rgba(0,212,255,0.15); color: #00d4ff; padding: 4px 12px; border-radius: 15px; font-size: 0.8rem; margin: 3px;">SpoonOS</span>
                <span class="tech-badge" style="display: inline-block; background: rgba(0,212,255,0.15); color: #00d4ff; padding: 4px 12px; border-radius: 15px; font-size: 0.8rem; margin: 3px;">Gemini AI</span>
                <span class="tech-badge" style="display: inline-block; background: rgba(0,212,255,0.15); color: #00d4ff; padding: 4px 12px; border-radius: 15px; font-size: 0.8rem; margin: 3px;">MediaPipe</span>
            </div>
        </div>
    """, unsafe_allow_html=True)

    if page == "🎥 Real-time Sentinel":
        render_realtime_sentinel()
    elif page == "🔍 Image Analysis":
        render_image_analysis()
    else:
        render_report_history()

if __name__ == "__main__":
    main()
