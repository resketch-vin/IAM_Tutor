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
DEFAULT_IMAGE_MIME = "image/png"
NEXT_QUESTIONS_MARKER = "Next Questions:"
MAX_SUGGESTIONS = 3

# 세션 상태 키
SS_AGENT = "agent"
SS_MESSAGES = "messages"
SS_USER_LEVEL = "user_level"
SS_DIAGNOSED = "diagnosed"
SS_QUICK_ACTION = "quick_action"
SS_PENDING_IMAGE = "pending_image"  # {"bytes": ..., "mime": ..., "name": ...}

# ─────────────────────────────────────────────────────────────────────────────
# 페이지 설정 및 세션 상태 초기화
# ─────────────────────────────────────────────────────────────────────────────
st.set_page_config(page_title=PAGE_TITLE, page_icon=PAGE_ICON, layout="wide")

if SS_AGENT not in st.session_state:
    with st.spinner("🎸 튜터가 지식 베이스를 점검 중입니다... 잠시만 기다려주세요."):
        st.session_state[SS_AGENT] = IAMAgent()

if SS_MESSAGES not in st.session_state:
    st.session_state[SS_MESSAGES] = []

if SS_USER_LEVEL not in st.session_state:
    st.session_state[SS_USER_LEVEL] = None

if SS_DIAGNOSED not in st.session_state:
    st.session_state[SS_DIAGNOSED] = False

if SS_QUICK_ACTION not in st.session_state:
    st.session_state[SS_QUICK_ACTION] = None

if SS_PENDING_IMAGE not in st.session_state:
    st.session_state[SS_PENDING_IMAGE] = None


# ─────────────────────────────────────────────────────────────────────────────
# 공용 헬퍼
# ─────────────────────────────────────────────────────────────────────────────
def process_user_input(user_query: str, image_payload: dict = None) -> None:
    """사용자 질의를 에이전트에 전달하고 응답을 같은 실행 사이클에 표시한다.

    image_payload: {"bytes": <bytes>, "mime": <str>, "name": <str>} 또는 None
    """
    st.session_state[SS_MESSAGES].append({"role": "user", "content": user_query})

    with st.chat_message("user"):
        st.markdown(user_query)
        if image_payload is not None:
            st.image(image_payload["bytes"], caption=image_payload.get("name", "분석에 사용된 악보"), use_container_width=True)

    with st.chat_message("assistant"):
        message_placeholder = st.empty()
        full_response = ""

        try:
            with st.status("🎸 튜터가 화성학적 원리를 분석 중입니다...", expanded=True) as status:
                if image_payload is not None:
                    st.write("🖼️ **악보 이미지에서 코드 추출 중...** (Gemini Vision 호출)")

                stream = st.session_state[SS_AGENT].ask(
                    user_query,
                    st.session_state[SS_MESSAGES][:-1],
                    user_level=st.session_state[SS_USER_LEVEL],
                    image_bytes=(image_payload["bytes"] if image_payload else None),
                    image_mime=(image_payload["mime"] if image_payload else None),
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
            st.error(full_response)

    st.session_state[SS_MESSAGES].append({"role": "assistant", "content": full_response})
    # NOTE: st.rerun()은 호출하지 않는다. 같은 실행 사이클 안에서 모든 출력을 끝까지 그린다.


def _resolve_level_from_experience(experience: str) -> str:
    for level_key, label in LEVEL_LABELS.items():
        if label == experience:
            return level_key
    return DEFAULT_LEVEL


def _extract_next_questions(content: str) -> list:
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


def _consume_pending_image() -> dict:
    """대기 중인 이미지를 꺼내고 세션 상태를 비운다 (1회용)."""
    payload = st.session_state[SS_PENDING_IMAGE]
    st.session_state[SS_PENDING_IMAGE] = None
    return payload


# ─────────────────────────────────────────────────────────────────────────────
# 사이드바
# ─────────────────────────────────────────────────────────────────────────────
with st.sidebar:
    st.title("🎸 IAM Tutor")
    st.markdown("---")

    if st.session_state[SS_USER_LEVEL]:
        st.info(f"**현재 설정된 수준:** {st.session_state[SS_USER_LEVEL]}")
        if st.button("수준 다시 설정하기"):
            st.session_state[SS_DIAGNOSED] = False
            st.session_state[SS_USER_LEVEL] = None
            st.session_state[SS_MESSAGES] = []
            st.session_state[SS_PENDING_IMAGE] = None
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
        st.session_state[SS_MESSAGES] = [
            {
                "role": "assistant",
                "content": f"안녕하세요! {st.session_state[SS_USER_LEVEL]} 수준에 맞춰 새로운 마음으로 다시 시작해볼까요? 분석하고 싶은 코드 진행을 알려주세요! 🎸",
            }
        ]
        st.session_state[SS_PENDING_IMAGE] = None
        st.rerun()

# ─────────────────────────────────────────────────────────────────────────────
# 메인 UI 로직
# ─────────────────────────────────────────────────────────────────────────────
if not st.session_state[SS_DIAGNOSED]:
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
            st.session_state[SS_USER_LEVEL] = _resolve_level_from_experience(experience)
            st.session_state[SS_DIAGNOSED] = True

            welcome_msg = (
                f"반갑습니다! **{st.session_state[SS_USER_LEVEL]}** 수준에 맞춰 정교한 화성학 분석을 제공해 드릴게요. "
            )
            st.session_state[SS_MESSAGES].append({"role": "assistant", "content": welcome_msg})
            st.rerun()

else:
    # 대화 인터페이스
    st.title("🎼 AI Guitar Tutor: IAM Tutor")
    st.caption(f"RAG 기반 지능형 화성학 분석 | **현재 수준: {st.session_state[SS_USER_LEVEL]}**")

    with st.expander("📖 **IAM Tutor 사용법 가이드**", expanded=False):
        st.markdown(
            f"""
            1. **코드 진행 입력**: 아래 입력창에 분석하고 싶은 코드 진행을 적어주세요.
               *   예: "C - G - Am - F" 또는 "키: G major, 진행: G-D-Em-C"
            2. **악보 이미지 분석**: 아래 업로드 섹션에 악보 이미지를 올리면 코드를 추출해 드립니다.
            3. **맞춤형 분석**: 현재 **{st.session_state[SS_USER_LEVEL]}** 수준에 맞춰 화성학적 원리와 연주 팁을 알려드립니다.
            """
        )

    # ── 악보 이미지 업로드: 버튼 클릭 시 분석을 직접 호출하지 않고
    #    pending_image에 바이트만 저장한 뒤 rerun. 본문 흐름에서 처리한다.
    with st.expander("🖼️ **악보 이미지로 분석하기 (Beta)**", expanded=False):
        uploaded_file = st.file_uploader(
            "코드 악보 이미지를 업로드하세요",
            type=SUPPORTED_IMAGE_TYPES,
            key="image_uploader",
        )
        if uploaded_file is not None:
            st.image(uploaded_file, caption="업로드된 악보", use_container_width=True)
            if st.button("이미지에서 코드 추출 및 분석 시작", key="btn_image_analyze"):
                st.session_state[SS_PENDING_IMAGE] = {
                    "bytes": uploaded_file.getvalue(),
                    "mime": uploaded_file.type or DEFAULT_IMAGE_MIME,
                    "name": uploaded_file.name,
                }
                st.session_state[SS_QUICK_ACTION] = IMAGE_ANALYSIS_QUERY
                st.rerun()

    # 대화 기록 표시
    for message in st.session_state[SS_MESSAGES]:
        with st.chat_message(message["role"]):
            display_content = message["content"].split(NEXT_QUESTIONS_MARKER)[0].strip()
            st.markdown(display_content)

    # 후속 질문 버튼
    if st.session_state[SS_MESSAGES] and st.session_state[SS_MESSAGES][-1]["role"] == "assistant":
        suggestions = _extract_next_questions(st.session_state[SS_MESSAGES][-1]["content"])
        if suggestions:
            st.markdown("---")
            st.caption("💡 **이어서 이런 질문은 어떠신가요?**")
            cols = st.columns(len(suggestions))
            for idx, suggestion in enumerate(suggestions):
                with cols[idx]:
                    if st.button(suggestion, key=f"suggest_btn_{idx}", use_container_width=True):
                        st.session_state[SS_QUICK_ACTION] = suggestion
                        st.rerun()

    # 퀵 액션 버튼
    st.markdown("---")
    quick_cols = st.columns(4)
    with quick_cols[0]:
        if st.button("🎼 1-4-5 진행 분석"):
            st.session_state[SS_QUICK_ACTION] = "가장 기본적인 C-F-G 1-4-5 진행의 화성학적 원리를 설명해줘."
            st.rerun()
    with quick_cols[1]:
        if st.button("✨ 세컨더리 도미넌트"):
            st.session_state[SS_QUICK_ACTION] = "C - E7 - Am 진행에서 E7의 역할(세컨더리 도미넌트)이 뭐야?"
            st.rerun()
    with quick_cols[2]:
        if st.button("아련한 IVm 분석"):
            st.session_state[SS_QUICK_ACTION] = "C - F - Fm - C 진행에서 Fm(모달 인터체인지)은 왜 쓰이는 거야?"
            st.rerun()
    with quick_cols[3]:
        if st.button("🎸 연주 가이드 요청"):
            st.session_state[SS_QUICK_ACTION] = "G - D/F# - Em - C 진행을 부드럽게 연주하는 팁을 알려줘."
            st.rerun()

    prompt = st.chat_input("분석하고 싶은 코드 진행을 입력하세요 (예: C-G-Am-F)")

    # ── 실제 분석 트리거: chat_input > quick_action 순으로 우선
    user_query = prompt or st.session_state[SS_QUICK_ACTION]
    if user_query:
        st.session_state[SS_QUICK_ACTION] = None
        image_payload = _consume_pending_image()
        process_user_input(user_query, image_payload=image_payload)
