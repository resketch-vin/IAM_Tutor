import base64
from typing import Optional

from langchain import hub
from langchain.agents import AgentExecutor, create_react_agent
from langchain_core.messages import HumanMessage
from langchain_core.prompts import PromptTemplate
from langchain_google_genai import ChatGoogleGenerativeAI

from rag.config import LLM_MODEL_NAME

from .tools import (
    analyze_chord_progression,
    detect_advanced_techniques,
    suggest_playing_guide,
)

# ─────────────────────────────────────────────────────────────────────────────
# 상수 (Single Source of Truth)
# ─────────────────────────────────────────────────────────────────────────────
LLM_TEMPERATURE = 0.1
LLM_MAX_RETRIES = 5
AGENT_MAX_ITERATIONS = 10
HISTORY_TURNS = 3
HISTORY_PREVIEW_CHARS = 100

DEFAULT_LEVEL_KEY = "초급"

LEVEL_MAP = {
    "완전 입문": {
        "desc": "기타 입문자입니다. 줄/프렛 개념부터 아주 쉽게 설명하고 ASCII 다이어그램 읽는 법을 포함하세요.",
        "val": 1,
    },
    "초급": {
        "desc": "오픈 코드를 아는 수준입니다. 기본기와 깨끗한 소리, 기초 리듬(칼립소 등)에 집중하세요.",
        "val": 2,
    },
    "중급": {
        "desc": "하이 코드와 다이아토닉을 익히는 중입니다. 펜타토닉, 코드 원리를 설명하되 기본기를 강조하세요.",
        "val": 3,
    },
    "상급": {
        "desc": "심화 이론(텐션, 모드)에 관심이 많습니다. 실질적인 연주 적용법과 톤 메이킹을 조언하세요.",
        "val": 4,
    },
    "전문/마스터": {
        "desc": "프로 수준입니다. 모달 인터체인지 등 최고 난이도 지식을 간결하고 핵심적으로 전달하세요.",
        "val": 5,
    },
}

SYSTEM_MESSAGE = (
    "당신은 'IAM Tutor'라는 친절하고 전문적인 기타 화성학 교육 에이전트입니다.\n"
    "당신의 주 임무는 사용자가 입력한 **코드 진행(Progression)**을 심층 분석하여 음악적 원리와 연주 가이드를 제공하는 것입니다.\n"
    "상황에 따라 'analyze_chord_progression', 'detect_advanced_techniques', 'suggest_playing_guide' 도구를 순차적 혹은 적절히 호출하여 답변을 구성하세요.\n"
    "*** 분석 원칙 ***\n"
    "1. 사용자의 숙련도(Level)에 맞춰 전문 용어의 깊이를 조절하세요.\n"
    "2. 반드시 RAG 검색 결과를 바탕으로 답변의 근거(예: RAG 화성학 07번)를 명시하세요.\n"
    "3. 코드의 단순 나열이 아닌 '왜(Why)' 이런 소리가 나는지 정서적/이론적 효과를 설명하세요.\n"
    "항상 한국어로 답변하며, 답변 마지막에 'Next Questions:'와 함께 후속 질문 3개를 목록으로 제시하세요."
)

VISION_EXTRACTION_PROMPT = (
    "당신은 악보 이미지에서 코드 진행을 추출하는 전문가입니다.\n"
    "다음 규칙을 엄격히 지키세요:\n"
    "1. 이미지에서 보이는 코드 네임만 추출합니다 (예: C, G7, Am, F#m7b5, D/F#).\n"
    "2. 코드 사이는 ' - '로 구분하고, 마디 경계는 ' | '로 표시합니다.\n"
    "3. 코드가 식별되지 않으면 정확히 'NO_CHORDS_FOUND' 만 출력하세요.\n"
    "4. 코드 외 어떤 설명, 머리말, 마침표도 출력하지 않습니다.\n"
    "출력 예시: C - G | Am - F | C - G - Am - F"
)

REACT_FALLBACK_TEMPLATE = """Answer the following questions as best you can. You have access to the following tools:
{tools}
Use the following format:
Question: the input question you must answer
Thought: you should always think about what to do
Action: the action to take, should be one of [{tool_names}]
Action Input: the input to the action
Observation: the result of the action
... (this Thought/Action/Action Input/Observation can repeat N times)
Thought: I now know the final answer
Final Answer: the final answer to the original input question

Begin!
Question: {input}
Thought:{agent_scratchpad}"""

VISION_FAILURE_SENTINEL = "NO_CHORDS_FOUND"
DEFAULT_IMAGE_MIME = "image/png"


class IAMAgent:
    """IAM Tutor ReAct 에이전트 핵심 클래스."""

    def __init__(self):
        self.llm = ChatGoogleGenerativeAI(
            model=LLM_MODEL_NAME,
            temperature=LLM_TEMPERATURE,
            streaming=True,
            max_retries=LLM_MAX_RETRIES,
        )

        self.tools = [
            analyze_chord_progression,
            detect_advanced_techniques,
            suggest_playing_guide,
        ]

        self.prompt = self._load_prompt()

        self.agent = create_react_agent(self.llm, self.tools, self.prompt)
        self.agent_executor = AgentExecutor(
            agent=self.agent,
            tools=self.tools,
            verbose=True,
            handle_parsing_errors=True,
            max_iterations=AGENT_MAX_ITERATIONS,
            early_stopping_method="generate",
        )

    @staticmethod
    def _load_prompt():
        """LangChain Hub에서 ReAct 프롬프트를 로드하되, 실패 시 안전한 폴백을 사용한다."""
        try:
            return hub.pull("hwchase17/react")
        except Exception:
            return PromptTemplate.from_template(REACT_FALLBACK_TEMPLATE)

    def _build_level_instructions(self, user_level: str) -> str:
        """수준 키를 받아 시스템 프롬프트에 주입할 지침 문자열을 만든다."""
        level_info = LEVEL_MAP.get(user_level, LEVEL_MAP[DEFAULT_LEVEL_KEY])
        return (
            f"*** [현재 사용자 수준: {user_level}] ***\n"
            f"지침: {level_info['desc']}\n"
            f"중요: 도구 사용 시 'user_level' 파라미터에 반드시 {level_info['val']}을 입력하여 수준에 맞는 정보를 검색하세요."
        )

    @staticmethod
    def _format_history(chat_history: Optional[list]) -> str:
        if not chat_history:
            return ""
        recent = chat_history[-HISTORY_TURNS:]
        lines = [
            f"{'사용자' if m['role'] == 'user' else '튜터'}: {m['content'][:HISTORY_PREVIEW_CHARS]}..."
            for m in recent
        ]
        return "\n[이전 대화]\n" + "\n".join(lines)

    def _extract_chords_from_image(self, image) -> str:
        """Gemini Vision으로 악보 이미지에서 코드 진행만 추출한다.

        반환값:
            정상 추출 시: "[이미지에서 추출된 코드 진행]\\n<코드>\\n"
            추출 실패 시: 실패 사실을 알리는 안내 문자열.
        """
        try:
            if hasattr(image, "seek"):
                image.seek(0)
            img_bytes = image.read()
            if hasattr(image, "seek"):
                image.seek(0)

            mime = getattr(image, "type", None) or DEFAULT_IMAGE_MIME
            img_b64 = base64.b64encode(img_bytes).decode("utf-8")

            vision_msg = HumanMessage(
                content=[
                    {"type": "text", "text": VISION_EXTRACTION_PROMPT},
                    {
                        "type": "image_url",
                        "image_url": f"data:{mime};base64,{img_b64}",
                    },
                ]
            )
            response = self.llm.invoke([vision_msg])
            extracted = (response.content or "").strip()

            if not extracted or VISION_FAILURE_SENTINEL in extracted:
                return (
                    "\n[이미지 분석 결과]\n"
                    "이미지에서 코드 진행을 식별하지 못했습니다. 사용자에게 코드 진행을 텍스트로 직접 입력해 달라고 요청하세요.\n"
                )
            return f"\n[이미지에서 추출된 코드 진행]\n{extracted}\n"
        except Exception as e:
            return (
                "\n[이미지 분석 결과]\n"
                f"이미지 처리 중 오류가 발생했습니다: {str(e)}. 사용자에게 코드 진행을 텍스트로 입력해 달라고 안내하세요.\n"
            )

    def ask(
        self,
        query: str,
        chat_history: Optional[list] = None,
        user_level: str = DEFAULT_LEVEL_KEY,
        image=None,
    ):
        """에이전트에게 질의를 보낸다. image가 주어지면 Vision으로 코드를 먼저 추출한다."""
        level_instructions = self._build_level_instructions(user_level)
        history_text = self._format_history(chat_history)

        image_context = ""
        if image is not None:
            image_context = self._extract_chords_from_image(image)

        full_query = (
            f"{SYSTEM_MESSAGE}\n\n"
            f"{level_instructions}\n\n"
            f"{history_text}"
            f"{image_context}\n\n"
            f"사용자 질문: {query}"
        )
        return self.agent_executor.stream({"input": full_query})
