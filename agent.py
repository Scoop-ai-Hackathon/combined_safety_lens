import asyncio
import json
import base64
import httpx
import os
import re
from pathlib import Path
from datetime import datetime
from fpdf import FPDF
from spoon_ai import ChatBot
from spoon_ai.schema import Message


# 마지막 분석 결과 저장용

last_analysis_result = {
    "image_path": None,
    "analysis": None,
    "timestamp": None
}


# 도구 정의
TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "calculate",
            "description": "수학 계산을 수행합니다.",
            "parameters": {
                "type": "object",
                "properties": {
                    "expression": {
                        "type": "string",
                        "description": "계산할 수식 (예: '2 + 3', '10 * 5')"
                    }
                },
                "required": ["expression"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "get_weather",
            "description": "특정 도시의 현재 날씨 정보를 가져옵니다.",
            "parameters": {
                "type": "object",
                "properties": {
                    "city": {
                        "type": "string",
                        "description": "날씨를 조회할 도시 이름"
                    }
                },
                "required": ["city"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "get_current_time",
            "description": "현재 시간을 가져옵니다.",
            "parameters": {
                "type": "object",
                "properties": {},
                "required": []
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "analyze_factory_safety",
            "description": "공장 이미지를 분석하여 한국 산업안전보건 기준에 따른 규정 준수 여부를 검사합니다.",
            "parameters": {
                "type": "object",
                "properties": {
                    "image_path": {
                        "type": "string",
                        "description": "분석할 공장 이미지 파일 경로"
                    },
                    "check_categories": {
                        "type": "array",
                        "items": {"type": "string"},
                        "description": "검사할 카테고리 목록 (예: ['보호구', '통로', '소화설비', '전기안전', '위험물']). 비워두면 전체 검사."
                    }
                },
                "required": ["image_path"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "search_safety_regulations",
            "description": "한국 산업안전보건 규정을 검색합니다.",
            "parameters": {
                "type": "object",
                "properties": {
                    "keyword": {
                        "type": "string",
                        "description": "검색할 키워드 (예: '안전모', '통로 폭', '소화기')"
                    }
                },
                "required": ["keyword"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "generate_pdf_report",
            "description": "마지막 안전 분석 결과를 PDF 리포트로 저장합니다.",
            "parameters": {
                "type": "object",
                "properties": {
                    "output_path": {
                        "type": "string",
                        "description": "저장할 PDF 파일 경로. 지정하지 않으면 자동 생성됩니다."
                    },
                    "include_image": {
                        "type": "boolean",
                        "description": "분석한 이미지를 리포트에 포함할지 여부 (기본값: true)"
                    }
                },
                "required": []
            }
        }
    }
]


# 한국 산업안전보건 기준 데이터베이스
KOREAN_SAFETY_REGULATIONS = {
    "보호구": {
        "안전모": "산업안전보건기준에 관한 규칙 제32조: 물체가 떨어지거나 날아올 위험이 있는 작업장에서는 안전모를 착용해야 함",
        "안전화": "산업안전보건기준에 관한 규칙 제32조: 물체의 낙하, 충격, 감전 위험이 있는 작업시 안전화 착용 필수",
        "보안경": "산업안전보건기준에 관한 규칙 제32조: 비산물, 분진 등이 발생하는 작업시 보안경 착용",
        "안전장갑": "유해물질 취급 또는 날카로운 물체 취급시 적절한 보호장갑 착용 필수"
    },
    "통로": {
        "폭": "산업안전보건기준에 관한 규칙 제17조: 통로 폭은 최소 80cm 이상 확보",
        "표시": "통로는 황색 선으로 명확히 표시해야 함",
        "장애물": "통로에 장애물을 방치해서는 안 됨",
        "조명": "통로는 75럭스 이상의 조도 유지"
    },
    "소화설비": {
        "소화기": "산업안전보건기준에 관한 규칙 제241조: 각 층마다 보행거리 20m 이내에 소화기 비치",
        "소화전": "옥내소화전은 호스 접결구로부터 25m 이내에 설치",
        "스프링클러": "위험물 저장소는 스프링클러 설치 필수",
        "비상구": "비상구 표지는 녹색 바탕에 백색 문자, 상시 점등"
    },
    "전기안전": {
        "배전반": "배전반 전면 80cm 이상 공간 확보, 잠금장치 필수",
        "접지": "모든 전기기기는 접지 필수",
        "누전차단기": "전동기계기구에는 누전차단기 설치",
        "전선관리": "전선은 피복 손상 없이 정리, 과부하 금지"
    },
    "위험물": {
        "표지": "위험물 저장소에는 경고표지 부착 필수",
        "환기": "인화성 물질 취급장소는 환기설비 설치",
        "저장": "위험물은 지정 장소에 보관, 혼합 저장 금지",
        "MSDS": "화학물질 취급장소에 물질안전보건자료(MSDS) 비치"
    },
    "작업환경": {
        "조도": "정밀작업 300럭스, 보통작업 150럭스, 기타 75럭스 이상",
        "환기": "유해물질 발생 작업장은 국소배기장치 설치",
        "소음": "소음 90dB 이상 작업장은 청력보호구 착용",
        "온도": "고열작업장은 휴식공간 및 냉방설비 확보"
    },
    "기계안전": {
        "방호장치": "회전체, 벨트, 풀리 등에는 덮개 또는 울 설치",
        "비상정지": "기계에는 비상정지장치 설치 필수",
        "안전거리": "위험 기계와 통로 사이 80cm 이상 거리 유지",
        "점검": "기계 점검 시 전원 차단 및 잠금표지(LOTO) 부착"
    }
}


class SafetyReportPDF(FPDF):
    """안전 분석 리포트용 PDF 클래스"""

    def __init__(self):
        super().__init__()
        # macOS 한글 폰트 경로
        self.add_font("AppleGothic", "", "/System/Library/Fonts/AppleSDGothicNeo.ttc")
        self.set_auto_page_break(auto=True, margin=15)

    def header(self):
        self.set_font("AppleGothic", "", 12)
        self.set_text_color(100, 100, 100)
        self.cell(0, 10, "Factory Safety Analysis Report", 0, 1, "R")
        self.line(10, 20, 200, 20)
        self.ln(5)

    def footer(self):
        self.set_y(-15)
        self.set_font("AppleGothic", "", 8)
        self.set_text_color(128, 128, 128)
        self.cell(0, 10, f"Page {self.page_no()}", 0, 0, "C")

    def add_title(self, title: str):
        self.set_font("AppleGothic", "", 24)
        self.set_text_color(0, 51, 102)
        self.cell(0, 15, title, 0, 1, "C")
        self.ln(5)

    def add_section(self, title: str, content: str):
        # 섹션 제목
        self.set_font("AppleGothic", "", 14)
        self.set_text_color(0, 102, 153)
        self.cell(0, 10, title, 0, 1, "L")
        self.line(10, self.get_y(), 200, self.get_y())
        self.ln(3)

        # 섹션 내용
        self.set_font("AppleGothic", "", 10)
        self.set_text_color(0, 0, 0)
        self.multi_cell(0, 6, content)
        self.ln(5)

    def add_image_section(self, image_path: str):
        if Path(image_path).exists():
            self.set_font("AppleGothic", "", 14)
            self.set_text_color(0, 102, 153)
            self.cell(0, 10, "Analyzed Image", 0, 1, "L")
            self.line(10, self.get_y(), 200, self.get_y())
            self.ln(3)

            # 이미지 크기 조정
            try:
                self.image(image_path, x=10, w=190)
            except Exception as e:
                self.set_font("AppleGothic", "", 10)
                self.cell(0, 10, f"Image loading error: {str(e)}", 0, 1)
            self.ln(10)


def generate_pdf_report(output_path: str = None, include_image: bool = True) -> str:
    """PDF 리포트 생성"""
    global last_analysis_result

    if not last_analysis_result["analysis"]:
        return "Error: No analysis result. Please run analyze_factory_safety first."

    # 출력 경로 설정
    if not output_path:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_path = f"/Users/ichaeeun/hackathon/safety_report_{timestamp}.pdf"

    try:
        pdf = SafetyReportPDF()
        pdf.add_page()

        # 제목
        pdf.add_title("Factory Safety Analysis Report")

        # 메타 정보
        pdf.set_font("AppleGothic", "", 10)
        pdf.set_text_color(100, 100, 100)
        pdf.cell(0, 6, f"Analysis Date: {last_analysis_result['timestamp']}", 0, 1)
        if last_analysis_result["image_path"]:
            pdf.cell(0, 6, f"Image: {last_analysis_result['image_path']}", 0, 1)
        pdf.ln(10)

        # 분석된 이미지 추가
        if include_image and last_analysis_result["image_path"]:
            pdf.add_image_section(last_analysis_result["image_path"])

        # 분석 결과 파싱 및 섹션별 추가
        analysis = last_analysis_result["analysis"]

        # 마크다운 헤더를 기준으로 섹션 분리
        sections = re.split(r'###\s*', analysis)

        for section in sections:
            if not section.strip():
                continue

            lines = section.strip().split('\n', 1)
            if len(lines) >= 1:
                title = lines[0].strip()
                content = lines[1].strip() if len(lines) > 1 else ""

                # 마크다운 기호 정리
                content = content.replace('**', '')
                content = content.replace('- ', '• ')
                content = re.sub(r'\*\*(.+?)\*\*', r'\1', content)

                if title and content:
                    pdf.add_section(title, content)

        # 적용 규정 요약
        pdf.add_page()
        pdf.add_section("Applied Regulations (Korean Industrial Safety Standards)",
            "\n".join([
                "• Occupational Safety and Health Standards (KOSHA)",
                "• Article 32: Personal Protective Equipment",
                "• Article 17: Passage Width Requirements",
                "• Article 241: Fire Safety Equipment",
                "• Electrical Safety Standards",
                "• Hazardous Materials Handling Standards",
                "• Machinery Safety Standards"
            ])
        )

        # PDF 저장
        pdf.output(output_path)

        return f"PDF report saved: {output_path}"

    except Exception as e:
        return f"PDF generation error: {str(e)}"


def execute_tool(tool_name: str, arguments: dict) -> str:
    """도구 실행"""
    if tool_name == "calculate":
        try:
            expression = arguments.get("expression", "")
            allowed = set("0123456789+-*/(). ")
            if all(c in allowed for c in expression):
                result = eval(expression)
                return f"Result: {expression} = {result}"
            else:
                return "Error: Invalid characters in expression."
        except Exception as e:
            return f"Calculation error: {str(e)}"

    elif tool_name == "get_weather":
        city = arguments.get("city", "Unknown")
        weather_data = {
            "Seoul": {"temp": "5°C", "condition": "Clear", "humidity": "45%"},
            "New York": {"temp": "2°C", "condition": "Cloudy", "humidity": "60%"},
            "Tokyo": {"temp": "8°C", "condition": "Rainy", "humidity": "75%"},
        }
        if city in weather_data:
            w = weather_data[city]
            return f"{city} weather: {w['temp']}, {w['condition']}, Humidity {w['humidity']}"
        else:
            return f"Weather data not found for {city}."

    elif tool_name == "get_current_time":
        now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        return f"Current time: {now}"

    elif tool_name == "search_safety_regulations":
        keyword = arguments.get("keyword", "")
        results = []
        for category, regulations in KOREAN_SAFETY_REGULATIONS.items():
            for item, rule in regulations.items():
                if keyword.lower() in item.lower() or keyword.lower() in rule.lower():
                    results.append(f"[{category}/{item}] {rule}")

        if results:
            return "Related regulations:\n" + "\n".join(results)
        else:
            return f"No regulations found for '{keyword}'."

    elif tool_name == "generate_pdf_report":
        output_path = arguments.get("output_path", None)
        include_image = arguments.get("include_image", True)
        return generate_pdf_report(output_path, include_image)

    elif tool_name == "analyze_factory_safety":
        return "ASYNC_IMAGE_ANALYSIS_REQUIRED"

    else:
        return f"Unknown tool: {tool_name}"


async def analyze_image_with_vision(image_path: str, categories: list = None) -> str:
    """Vision 모델을 사용하여 공장 이미지 안전 분석"""
    global last_analysis_result

    path = Path(image_path)
    if not path.exists():
        return f"Error: Image file not found: {image_path}"

    with open(path, "rb") as f:
        image_data = base64.b64encode(f.read()).decode("utf-8")

    extension = path.suffix.lower()
    mime_types = {
        ".jpg": "image/jpeg",
        ".jpeg": "image/jpeg",
        ".png": "image/png",
        ".gif": "image/gif",
        ".webp": "image/webp"
    }
    mime_type = mime_types.get(extension, "image/jpeg")

    if not categories:
        categories = list(KOREAN_SAFETY_REGULATIONS.keys())

    relevant_regulations = []
    for cat in categories:
        if cat in KOREAN_SAFETY_REGULATIONS:
            for item, rule in KOREAN_SAFETY_REGULATIONS[cat].items():
                relevant_regulations.append(f"- [{cat}] {item}: {rule}")

    regulations_text = "\n".join(relevant_regulations)

    analysis_prompt = f"""당신은 한국 산업안전보건 전문가입니다.
이 공장/작업장 이미지를 분석하여 다음 한국 산업안전보건 기준에 따라 규정 준수 여부를 상세히 검사해주세요.

## 적용 규정:
{regulations_text}

## 분석 요청:
이미지를 면밀히 분석하여 다음 형식으로 보고서를 작성해주세요:

### 1. 이미지 개요
- 작업장 유형 및 환경 설명

### 2. 규정 준수 사항
- 준수하고 있는 안전 규정들을 구체적으로 나열

### 3. 규정 위반 또는 개선 필요 사항
- 위반 사항 또는 위험 요소를 구체적으로 지적
- 각 항목에 대해 관련 규정 명시
- 개선 방안 제시

### 4. 위험도 평가
- 전체 위험도: (높음/중간/낮음)
- 즉시 조치가 필요한 사항

### 5. 권고사항
- 안전 개선을 위한 우선순위별 권고사항
"""

    api_key = os.getenv("OPENROUTER_API_KEY")

    async with httpx.AsyncClient(timeout=60.0) as client:
        response = await client.post(
            "https://openrouter.ai/api/v1/chat/completions",
            headers={
                "Authorization": f"Bearer {api_key}",
                "Content-Type": "application/json"
            },
            json={
                "model": "google/gemini-2.0-flash-001",
                "messages": [
                    {
                        "role": "user",
                        "content": [
                            {"type": "text", "text": analysis_prompt},
                            {"type": "image_url", "image_url": {"url": f"data:{mime_type};base64,{image_data}"}}
                        ]
                    }
                ],
                "max_tokens": 4096
            }
        )

        if response.status_code == 200:
            result = response.json()
            analysis = result["choices"][0]["message"]["content"]

            # 분석 결과 저장
            last_analysis_result["image_path"] = image_path
            last_analysis_result["analysis"] = analysis
            last_analysis_result["timestamp"] = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

            return analysis
        else:
            return f"Image analysis error: {response.status_code} - {response.text}"


class Agent:
    """SpoonOS 통합 프로토콜 레이어를 통해 LLM을 호출하는 에이전트"""

    def __init__(self, llm_provider: str = "openrouter", model_name: str = None):
        self.chatbot = ChatBot(llm_provider=llm_provider, model_name=model_name)
        self.conversation_history: list[Message] = []
        self.system_prompt = """You are a helpful assistant specialized in Korean industrial safety regulations.
You can analyze factory images for safety compliance, search Korean safety regulations, and generate PDF reports.
When users ask for a PDF report after analysis, use the generate_pdf_report tool.
Always respond in Korean."""
        self.tools = TOOLS

    async def chat(self, user_input: str) -> str:
        """일반 채팅 (도구 없음)"""
        self.conversation_history.append(Message(role="user", content=user_input))
        response = await self.chatbot.ask(messages=self.conversation_history, system_msg=self.system_prompt)
        self.conversation_history.append(Message(role="assistant", content=response))
        return response

    async def chat_with_tools(self, user_input: str) -> str:
        """도구를 사용하는 채팅"""
        self.conversation_history.append(Message(role="user", content=user_input))

        response = await self.chatbot.ask_tool(
            messages=self.conversation_history,
            tools=self.tools,
            tool_choice="auto",
            system_msg=self.system_prompt
        )

        if response.tool_calls:
            self.conversation_history.append(
                Message(role="assistant", content=response.content, tool_calls=response.tool_calls)
            )

            for tool_call in response.tool_calls:
                tool_name = tool_call.function.name
                arguments = tool_call.function.get_arguments_dict()

                print(f"  [Tool Call] {tool_name}({arguments})")

                if tool_name == "analyze_factory_safety":
                    image_path = arguments.get("image_path", "")
                    categories = arguments.get("check_categories", [])
                    result = await analyze_image_with_vision(image_path, categories)
                else:
                    result = execute_tool(tool_name, arguments)

                print(f"  [Tool Result] {result[:100]}..." if len(result) > 100 else f"  [Tool Result] {result}")

                self.conversation_history.append(
                    Message(role="tool", content=result, tool_call_id=tool_call.id)
                )

            final_response = await self.chatbot.ask(
                messages=self.conversation_history,
                system_msg=self.system_prompt
            )

            self.conversation_history.append(Message(role="assistant", content=final_response))
            return final_response

        else:
            self.conversation_history.append(Message(role="assistant", content=response.content))
            return response.content

    def clear_history(self):
        """대화 기록 초기화"""
        self.conversation_history = []


async def main():
    agent = Agent(
        llm_provider="openrouter",
        model_name="google/gemini-2.0-flash-001"
    )

    print("=== 공장 안전 규정 분석 에이전트 ===")
    print("사용 가능한 도구:")
    print("  - analyze_factory_safety: 공장 이미지 안전 분석")
    print("  - generate_pdf_report: 분석 결과 PDF 저장")
    print("  - search_safety_regulations: 안전 규정 검색")
    print("\n사용 예시:")
    print("  - '이 이미지 분석해줘: /path/to/image.jpg'")
    print("  - 'PDF 리포트로 저장해줘'")
    print("\n종료하려면 'exit'를 입력하세요.\n")

    while True:
        try:
            user_input = input("You: ").strip()

            if not user_input:
                continue

            if user_input.lower() in ["exit", "quit", "종료"]:
                print("채팅을 종료합니다.")
                break

            response = await agent.chat_with_tools(user_input)
            print(f"Assistant: {response}\n")

        except KeyboardInterrupt:
            print("\n채팅을 종료합니다.")
            break


if __name__ == "__main__":
    asyncio.run(main())