import re

import streamlit as st

from agent.core import IAMAgent

# ─────────────────────────────────────────────────────────────────────────────
# 상수 정의 (Single Source of Truth)
# ─────────────────────────────────────────────────────────────────────────────
PAGE_TITLE = "IAM Tutor - 지능형 화성학 AI 튜터"
PAGE_ICON = "🎸"

LEVEL_LABELS = {
    "완전 입문": "기타를 처음 잡아봐요 (완전 입문)",
    "초급": "기본 코드는 조금 알아요 (초급)",
    "중급": "하이 코드와 간단한 스케일을 연습 중이에요 (중급)",
    "상급": "화성학 원리를 이해하고 자유롭게 연주해요 (상급)",
    "전문/마스터": "프로 연주자이거나 전문적인 지식을 갖추고 있어요 (전문/마스터)",
}
DEFAULT_LEVEL = "초급"

GOAL_OPTIONS = [
    "코드 진행 분석",
    "화성학적 원리 이해",
    "고급 화성 기법 탐지",
    "수준별 연주 팁",
    "악보 이미지 코드 추출",
]

IMAGE_ANALYSIS_QUERY = "업로드한 악보 이미지에 포함된 코드 진행을 정확히 추출한 뒤, 그 진행을 화성학적으로 분석하고 연주 가이드를 제시해줘."
SUPPORTED_IMAGE_TYPES = ["png", "jpg", "jpeg"]
NEXT_QUESTIONS_MARKER = "Next Questions:"
MAX_SUGGESTIONS = 3

# ─────────────────────────────────────────────────────────────────────────────
# 페이지 설정 및 세션 상태 초기화
# ─────────────────────────────────────────────────────────────────────────────
st.set_page_config(page_title=PAGE_TITLE, page_icon=PAGE_ICON, layout="wide")

if "agent" not in st.session_state:
    with st.spinner("🎸 튜터가 지식 베이스를 점검 중입니다... 잠시만 기다려주세요."):
        st.session_state.agent = IAMAgent()

if "messages" not in st.session_state:
    st.session_state.messages = []

if "user_level" not in st.session_state:
    st.session_state.user_level = None

if "diagnosed" not in st.session_state:
    st.session_state.diagnosed = False

if "quick_action" not in st.session_state:
    st.session_state.quick_action = None


# ─────────────────────────────────────────────────────────────────────────────
# 공용 헬퍼: 사용자 입력 처리 (NameError 방지를 위해 최상단에 정의)
# ─────────────────────────────────────────────────────────────────────────────
def process_user_input(user_query: str, image=None) -> None:
    """사용자 질의(텍스트/이미지)를 에이전트에 전달하고 응답을 스트리밍 표시한다."""
    st.session_state.messages.append({"role": "user", "content": user_query})
    with st.chat_message("user"):
        st.markdown(user_query)
        if image is not None:
            st.image(image, caption="분석에 사용된 악보", use_container_width=True)

    with st.chat_message("assistant"):
        message_placeholder = st.empty()
        full_response = ""

        try:
            with st.status("🎸 튜터가 화성학적 원리를 분석 중입니다...", expanded=True) as status:
                if image is not None:
                    st.write("🖼️ **악보 이미지에서 코드 추출 중...**")

                stream = st.session_state.agent.ask(
                    user_query,
                    st.session_state.messages[:-1],
                    user_level=st.session_state.user_level,
                    image=image,
                )

                for chunk in stream:
                    if "actions" in chunk:
                        for action in chunk["actions"]:
                            st.write(f"🔍 **분석 도구 사용:** {action.tool}")

                    if "output" in chunk:
                        status.update(label="🎸 분석을 완료했습니다!", state="complete", expanded=False)
                        full_response += chunk["output"]
                        message_placeholder.markdown(full_response + "▌")

                message_placeholder.markdown(full_response)
        except Exception as e:
            full_response = f"죄송합니다. 분석 중 오류가 발생했습니다: {str(e)}"
            message_placeholder.markdown(full_response)

    st.session_state.messages.append({"role": "assistant", "content": full_response})
    st.rerun()


def _resolve_level_from_experience(experience: str) -> str:
    """라디오 라벨에서 내부 수준 키를 안전하게 추출한다."""
    for level_key, label in LEVEL_LABELS.items():
        if label == experience:
            return level_key
    return DEFAULT_LEVEL


def _extract_next_questions(content: str) -> list:
    """응답 본문 끝의 'Next Questions:' 영역에서 후속 질문 후보를 정제하여 추출한다."""
    if NEXT_QUESTIONS_MARKER not in content:
        return []
    raw = content.split(NEXT_QUESTIONS_MARKER, maxsplit=1)[1].strip()
    suggestions = []
    for line in raw.split("\n"):
        clean = re.sub(r"^[\d\.\s\-\*\+]+", "", line.strip())
        clean = clean.replace("**", "").replace("*", "").strip()
        if clean and len(clean) >= 2:
            suggestions.append(clean)
    return suggestions[:MAX_SUGGESTIONS]


# ─────────────────────────────────────────────────────────────────────────────
# 사이드바
# ─────────────────────────────────────────────────────────────────────────────
with st.sidebar:
    st.title("🎸 IAM Tutor")
    st.markdown("---")

    if st.session_state.user_level:
        st.info(f"**현재 설정된 수준:** {st.session_state.user_level}")
        if st.button("수준 다시 설정하기"):
            st.session_state.diagnosed = False
            st.session_state.user_level = None
            st.session_state.messages = []
            st.rerun()

    st.markdown("---")
    st.info(
        """
        **지원하는 도움:**
        - 🎼 **코드 진행 분석**: 기능적 화성학 해설
        - ✨ **고급 기법 탐지**: 세컨더리 도미넌트 등 탐지
        - 🎸 **연주 가이드**: 수준별 운지 및 팁 제공
        - 🖼️ **악보 이미지 분석**: 이미지에서 코드 추출
        """
    )

    if st.button("대화 초기화"):
        st.session_state.messages = [
            {
                "role": "assistant",
                "content": f"안녕하세요! {st.session_state.user_level} 수준에 맞춰 새로운 마음으로 다시 시작해볼까요? 분석하고 싶은 코드 진행을 알려주세요! 🎸",
            }
        ]
        st.rerun()

# ─────────────────────────────────────────────────────────────────────────────
# 메인 UI 로직
# ─────────────────────────────────────────────────────────────────────────────
if not st.session_state.diagnosed:
    st.title("🎸 IAM Tutor: 맞춤형 진단")
    st.subheader("당신에게 꼭 맞는 튜터링을 위해 몇 가지만 여쭤볼게요!")

    with st.form("diagnosis_form"):
        experience = st.radio(
            "1. 기타를 연주하신 지 얼마나 되셨나요?",
            list(LEVEL_LABELS.values()),
        )
        goal = st.multiselect(
            "2. 현재 어떤 도움이 가장 필요하신가요? (중복 선택 가능)",
            GOAL_OPTIONS,
        )

        submitted = st.form_submit_button("진단 완료 및 시작하기")

        if submitted:
            st.session_state.user_level = _resolve_level_from_experience(experience)
            st.session_state.diagnosed = True

            welcome_msg = (
                f"반갑습니다! **{st.session_state.user_level}** 수준에 맞춰 정교한 화성학 분석을 제공해 드릴게요. "
            )
            st.session_state.messages.append({"role": "assistant", "content": welcome_msg})
            st.rerun()

else:
    # 대화 인터페이스
    st.title("🎼 AI Guitar Tutor: IAM Tutor")
    st.caption(f"RAG 기반 지능형 화성학 분석 | **현재 수준: {st.session_state.user_level}**")

    # 사용법 가이드
    with st.expander("📖 **IAM Tutor 사용법 가이드**", expanded=False):
        st.markdown(
            f"""
            1. **코드 진행 입력**: 아래 입력창에 분석하고 싶은 코드 진행을 적어주세요.
               *   예: "C - G - Am - F" 또는 "키: G major, 진행: G-D-Em-C"
            2. **악보 이미지 분석**: 아래 업로드 섹션에 악보 이미지를 올리면 코드를 추출해 드립니다.
            3. **맞춤형 분석**: 현재 **{st.session_state.user_level}** 수준에 맞춰 화성학적 원리와 연주 팁을 알려드립니다.
            """
        )

    # 악보 이미지 업로드 섹션
    with st.expander("🖼️ **악보 이미지로 분석하기 (Beta)**", expanded=False):
        uploaded_file = st.file_uploader(
            "코드 악보 이미지를 업로드하세요",
            type=SUPPORTED_IMAGE_TYPES,
        )
        if uploaded_file:
            st.image(uploaded_file, caption="업로드된 악보", use_container_width=True)
            if st.button("이미지에서 코드 추출 및 분석 시작"):
                process_user_input(IMAGE_ANALYSIS_QUERY, image=uploaded_file)

    # 대화 기록 표시
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            display_content = message["content"].split(NEXT_QUESTIONS_MARKER)[0].strip()
            st.markdown(display_content)

    # 마지막 어시스턴트 메시지에서 후속 질문 버튼 노출
    if st.session_state.messages and st.session_state.messages[-1]["role"] == "assistant":
        suggestions = _extract_next_questions(st.session_state.messages[-1]["content"])
        if suggestions:
            st.markdown("---")
            st.caption("💡 **이어서 이런 질문은 어떠신가요?**")
            cols = st.columns(len(suggestions))
            for idx, suggestion in enumerate(suggestions):
                with cols[idx]:
                    if st.button(suggestion, key=f"suggest_btn_{idx}", use_container_width=True):
                        st.session_state.quick_action = suggestion
                        st.rerun()

    # 퀵 액션 버튼 (화성학 분석 중심)
    st.markdown("---")
    quick_cols = st.columns(4)
    with quick_cols[0]:
        if st.button("🎼 1-4-5 진행 분석"):
            st.session_state.quick_action = "가장 기본적인 C-F-G 1-4-5 진행의 화성학적 원리를 설명해줘."
    with quick_cols[1]:
        if st.button("✨ 세컨더리 도미넌트"):
            st.session_state.quick_action = "C - E7 - Am 진행에서 E7의 역할(세컨더리 도미넌트)이 뭐야?"
    with quick_cols[2]:
        if st.button("아련한 IVm 분석"):
            st.session_state.quick_action = "C - F - Fm - C 진행에서 Fm(모달 인터체인지)은 왜 쓰이는 거야?"
    with quick_cols[3]:
        if st.button("🎸 연주 가이드 요청"):
            st.session_state.quick_action = "G - D/F# - Em - C 진행을 부드럽게 연주하는 팁을 알려줘."

    prompt = st.chat_input("분석하고 싶은 코드 진행을 입력하세요 (예: C-G-Am-F)")
    user_query = prompt or st.session_state.quick_action

    if user_query:
        st.session_state.quick_action = None
        process_user_input(user_query)
