# 🛡️ Safety Lens: Active Sentinel

## 1. 프로젝트 개요
**Safety Lens**는 단순한 감시 도구가 아닌, Vision AI와 LLM Agent가 결합된 차세대 공장 안전 솔루션입니다. 실시간 공장 안전 모니터링 시스템으로서 컴퓨터 비전과 AI를 활용하여 작업자의 손을 감지하고 기계 압착 위험을 사전에 방지합니다. 

본 프로젝트는 모든 팀원이 AI 엔지니어링에 참여하여, 실시간성이 중요한 'Vision 파트'와 논리적 판단이 중요한 'Agent 파트'를 긴밀하게 통합한 공동 연구 결과물입니다.

---

## 2. 핵심 기능

### 🚨 Real-time Safety Sentinel (실시간 안전 감시)
| 기능 | 설명 | 기술 스택 |
| :--- | :--- | :--- |
| **손 감지** | MediaPipe를 이용한 실시간 손 위치 추적 | MediaPipe Hand Landmarker |
| **광학 흐름 분석** | 손 주변 물체 움직임 정밀 감지 | OpenCV Farneback |
| **압착 위험 감지** | 손 영역 축소 + 4방향 물체 접근 분석 | Leaky Bucket Algorithm |
| **긴급 정지** | 위험 감지 시 자동 시스템 정지 및 알람 | State Latching |

### 🔍 Image Analysis (이미지 정밀 분석)
| 기능 | 설명 | 기술 스택 |
| :--- | :--- | :--- |
| **공장 안전 분석** | 산업안전보건법 기준 위반 여부 검사 | Gemini Vision + OpenRouter |
| **건축 도면 분석** | 도면 내 설계 오류 및 법규 위반 검사 | Gemini Vision |
| **건설현장 분석** | 현장 사진 기반 위험 요소 자동 식별 | Gemini Vision |

### 📊 Report Management (보고서 관리)
| 기능 | 설명 |
| :--- | :--- |
| **HTML 보고서** | 분석 결과를 시각적 대시보드 형태로 저장 |
| **PDF 리포트** | 공식 제출용 안전 진단 보고서 자동 생성 |
| **이력 관리** | 모든 분석 로그 및 결과 데이터베이스화 |

---

## 3. Team R&D 역할 분담 (AI Co-Work)

| 담당 영역 | 주요 기여 내용 (Role) | 협업 포인트 |
| :--- | :--- | :--- |
| **Vision AI (Real-time)** | • MediaPipe & OpenCV 파이프라인 구축<br>• Hand Landmark와 Optical Flow를 결합한 하이브리드 감지 로직 설계<br>• 프레임 드랍 방지를 위한 경량화 모델 적용 | Agent 영역에 정확한 트리거(Trigger) 신호를 보내기 위해 오탐지율(False Positive) 최소화에 주력 |
| **Agent AI (SpoonOS)** | • SpoonOS 기반 LLM Agent 설계<br>• 산업안전보건법 기반의 System Prompt 엔지니어링<br>• Vision 분석 결과를 받아 법규 위반 여부를 판단하는 추론 로직 구현 | Vision 영역의 데이터를 LLM이 이해하기 쉬운 JSON/Text 포맷으로 변환하는 프로토콜 공동 설계 |
| **System Integration** | • Streamlit 비동기(Async) 아키텍처 설계<br>• 실시간 영상 루프와 LLM 응답 대기 시간의 병목 현상 해결<br>• Leaky Bucket Algorithm을 통한 상태 관리 로직 구현 | Vision의 속도와 Agent의 정확도 사이 균형을 맞추기 위해 전체 데이터 파이프라인 최적화 |

---

## 4. 기술 아키텍처 및 데이터 파이프라인

협업을 통해 **실시간 처리(Fast Path)**와 **정밀 분석(Slow Path)**이 동시에 돌아가는 듀얼 트랙 구조를 완성했습니다.

```mermaid
graph LR
    %% 스타일 정의
    classDef user fill:#333,stroke:#fff,stroke-width:2px,color:#fff;
    classDef ui fill:#5D3FD3,stroke:#fff,stroke-width:2px,color:#fff;
    classDef vision fill:#E6F7FF,stroke:#1890FF,stroke-width:2px,color:#000;
    classDef spoon fill:#FFF7E6,stroke:#FA8C16,stroke-width:2px,color:#000;
    classDef ext fill:#F6FFED,stroke:#52C41A,stroke-width:2px,color:#000;

    %% 1. 사용자 입력 (왼쪽)
    User((User / Webcam)):::user --> UI[🖥️ Streamlit App]:::ui

    %% 2. 프로세스 그룹 (중앙)
    subgraph Core_System ["Safety Lens Core System"]
        direction TB
        
        %% A. Vision 파이프라인 (위)
        subgraph Vision_AI ["👁️ Vision Pipeline - Realtime"]
            direction LR
            MP(Hand Detect) --> OF(Optical Flow)
            OF --> LB{Leaky Bucket}
            LB -->|Danger| Trigger(🚨 Auto Stop)
        end

        %% B. SpoonOS 에이전트 (아래)
        subgraph Spoon_Agent ["🧠 SpoonOS Agent - Analysis"]
            direction LR
            Agent(Spoon Agent) <--> Tools[[🛠️ Custom Tools]]
            Agent <--> Wrapper(ChatBot)
        end
    end

    %% 3. 외부 LLM (오른쪽)
    LLM(Gemini 2.0 Flash):::ext

    %% 데이터 연결
    UI -->|1. Video Stream| Vision_AI
    UI -->|2. Image Upload| Spoon_Agent
    
    Trigger -.->|Signal| UI
    Wrapper <-->|API Call| LLM
    Spoon_Agent -.->|Safety Report| UI

    %% 스타일 적용
    class MP,OF,LB,Trigger vision;
    class Agent,Wrapper,Tools spoon;
