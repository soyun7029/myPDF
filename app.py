"""PDF 분석기 - Streamlit 진입점."""

from datetime import datetime

import streamlit as st

from llm import (
    ANALYSIS_TYPES,
    demote_headings,
    friendly_error,
    get_client,
    get_settings,
    stream_analysis,
    stream_answer,
)
from pdf_utils import Document, PdfExtractionError, extract_text

LANGUAGES = ["한국어", "English", "日本語"]
PREVIEW_CHARS = 1_000
EXAMPLE_QUESTIONS = [
    "이 문서의 핵심 내용을 세 줄로 알려줘",
    "문서에 나온 중요한 날짜나 숫자를 정리해줘",
    "이 문서에서 결론이나 권고 사항은 뭐야?",
]
ANALYSIS_DESCRIPTIONS = {
    "요약": "파일별 요약과 통합 요약을 만들어요.",
    "핵심 포인트": "파일별 핵심 내용을 불릿으로 정리해요.",
    "키워드": "주요 키워드와 짧은 설명을 뽑아요.",
    "문서 비교": "공통점과 차이점을 비교표로 정리해요.",
}



def md_safe(text: str) -> str:
    """Streamlit 마크다운이 $...$ 를 수식으로 바꾸지 않도록 $ 를 이스케이프한다."""
    return text.replace("$", r"\$")


def escaped_stream(stream, raw: list[str]):
    """화면에는 이스케이프한 텍스트를 내보내고, 원문은 raw에 모은다."""
    for piece in stream:
        raw.append(piece)
        yield md_safe(piece)


st.set_page_config(page_title="PDF 분석기", page_icon="📄", layout="wide")

st.title("📄 PDF 분석기")
st.caption("PDF를 올리면 요약·분석하고 질문에 답해 드려요")

settings = get_settings()
if not settings.api_key:
    st.error(
        "OpenAI API 키가 설정되지 않았어요.\n\n"
        "- **로컬 실행**: 프로젝트 폴더의 `.env` 파일에 `OPENAI_API_KEY=sk-...`를 넣어 주세요.\n"
        "- **Streamlit Cloud**: 앱의 Settings → Secrets에 `OPENAI_API_KEY = \"sk-...\"`를 넣어 주세요."
    )
    st.stop()

client = get_client(settings.api_key)

# ---------------------------------------------------------------------------
# 세션 상태
# ---------------------------------------------------------------------------
st.session_state.setdefault("doc_key", ())
st.session_state.setdefault("results", {})  # (분석 유형, 언어) -> 결과 텍스트
st.session_state.setdefault("messages", [])  # [{"role": ..., "content": ...}]
st.session_state.setdefault("pending_question", None)  # 예시 질문 클릭 시

# ---------------------------------------------------------------------------
# 사이드바 1: 파일 업로드
# ---------------------------------------------------------------------------
with st.sidebar:
    st.subheader("1. 파일 업로드")
    uploaded_files = st.file_uploader(
        "PDF 파일을 끌어다 놓거나 선택하세요",
        type="pdf",
        accept_multiple_files=True,
    )

# PDF 처리 (메모리에서만 처리하고 디스크에 저장하지 않음)
docs: list[Document] = []
failed: list[tuple[str, str]] = []
duplicates: list[tuple[str, str]] = []  # (중복 파일명, 먼저 올린 파일명)
seen: dict[str, str] = {}  # 파일 해시 -> 파일명
for f in uploaded_files or []:
    try:
        doc = extract_text(f.name, f.getvalue())
    except PdfExtractionError as e:
        failed.append((f.name, str(e)))
        continue
    if doc.file_hash in seen:
        duplicates.append((f.name, seen[doc.file_hash]))
        continue
    seen[doc.file_hash] = f.name
    docs.append(doc)

usable_docs = [d for d in docs if d.has_text]

# 업로드한 파일 구성이 바뀌면 이전 분석 결과와 대화를 지운다.
doc_key = tuple(d.file_hash for d in usable_docs)
if doc_key != st.session_state.doc_key:
    st.session_state.doc_key = doc_key
    st.session_state.results = {}
    st.session_state.messages = []

# ---------------------------------------------------------------------------
# 사이드바 2~3: 분석 설정, 기타
# ---------------------------------------------------------------------------
with st.sidebar:
    st.divider()
    st.subheader("2. 분석 설정")
    # 선택지를 항상 고정해야 파일을 추가·삭제해도 선택이 초기화되지 않는다.
    kind = st.radio(
        "분석 유형",
        ANALYSIS_TYPES,
        captions=[ANALYSIS_DESCRIPTIONS[k] for k in ANALYSIS_TYPES],
        key="kind",
    )
    compare_unavailable = kind == "문서 비교" and len(usable_docs) < 2
    if len(usable_docs) < 2:
        st.caption("💡 문서 비교는 PDF를 2개 이상 올려야 실행할 수 있어요.")
    language = st.selectbox("응답 언어", LANGUAGES)

    st.divider()
    st.subheader("3. 기타")
    if st.button("대화 초기화", width="stretch", disabled=not st.session_state.messages):
        st.session_state.messages = []
        st.rerun()
    st.caption(f"모델: `{settings.model}`")

# ---------------------------------------------------------------------------
# 빈 상태: 파일이 없을 때 사용 방법 안내
# ---------------------------------------------------------------------------
if not uploaded_files:
    st.write("")
    cols = st.columns(3)
    steps = [
        ("📤", "1. PDF 업로드", "왼쪽 사이드바에 PDF 파일을 하나 이상 올려 주세요."),
        ("⚙️", "2. 분석 설정", "요약·핵심 포인트·키워드·문서 비교 중 원하는 분석을 고르세요."),
        ("💬", "3. 결과 확인·질문", "분석 결과를 확인하고, 문서에 대해 자유롭게 질문하세요."),
    ]
    for col, (icon, title, desc) in zip(cols, steps):
        with col, st.container(border=True):
            st.markdown(f"### {icon}")
            st.markdown(f"**{title}**")
            st.caption(desc)
    st.info("모바일에서는 왼쪽 위 **»** 버튼을 눌러 사이드바를 열 수 있어요.")
    st.stop()

tab_info, tab_analysis, tab_chat = st.tabs(["📑 문서 정보", "📊 분석 결과", "💬 질문하기"])

# ---------------------------------------------------------------------------
# 탭 1: 문서 정보
# ---------------------------------------------------------------------------
with tab_info:
    for name, message in failed:
        st.error(f"**{name}** · {message}", icon="🚫")
    for name, original in duplicates:
        st.info(f"**{name}**은(는) **{original}**과(와) 같은 파일이라 한 번만 분석해요.", icon="ℹ️")

    if docs:
        total_pages = sum(d.page_count for d in usable_docs)
        st.caption(
            f"분석 가능한 문서 {len(usable_docs)}개 · 총 {total_pages:,}페이지"
        )

    cols = st.columns(2)
    for i, doc in enumerate(docs):
        with cols[i % 2], st.container(border=True):
            st.markdown(f"**📄 {doc.name}**")
            m1, m2 = st.columns(2)
            m1.metric("페이지 수", f"{doc.page_count:,}")
            m2.metric("글자 수", f"{doc.char_count:,}")
            if not doc.has_text:
                st.warning(
                    "텍스트를 추출할 수 없어요. 이미지(스캔) PDF일 수 있어서 분석에서 제외했어요.",
                    icon="⚠️",
                )
            elif doc.empty_pages:
                pages = ", ".join(map(str, doc.empty_pages[:20]))
                more = " 외" if len(doc.empty_pages) > 20 else ""
                st.warning(
                    f"텍스트가 없는 페이지가 있어요(이미지 PDF일 수 있음): p.{pages}{more}",
                    icon="⚠️",
                )
            with st.expander("텍스트 미리보기"):
                preview = doc.tagged_text()[:PREVIEW_CHARS]
                st.text(preview if preview else "(추출된 텍스트 없음)")

# ---------------------------------------------------------------------------
# 탭 2: 분석 결과
# ---------------------------------------------------------------------------
with tab_analysis:
    if not usable_docs:
        st.info("분석할 수 있는 PDF가 없어요. 텍스트가 있는 PDF를 업로드해 주세요.")
    else:
        result_key = (kind, language)
        head_left, head_right = st.columns([3, 1], vertical_alignment="center")
        head_left.markdown(
            f"**{kind}** · 문서 {len(usable_docs)}개 · {language}  \n"
            f"<span style='opacity:0.7'>{ANALYSIS_DESCRIPTIONS[kind]}</span>",
            unsafe_allow_html=True,
        )
        has_result = result_key in st.session_state.results
        run = head_right.button(
            "다시 분석" if has_result else "분석 실행",
            type="primary",
            width="stretch",
            disabled=compare_unavailable,
        )

        result_box = st.container(border=True)
        if compare_unavailable:
            result_box.info("문서 비교는 텍스트가 있는 PDF가 2개 이상 있어야 실행할 수 있어요.")
        elif run:
            with result_box:
                status = st.empty()
                raw: list[str] = []
                try:
                    with st.spinner("분석 중이에요..."):
                        stream = stream_analysis(
                            client,
                            settings.model,
                            usable_docs,
                            kind,
                            language,
                            on_progress=lambda msg: status.caption(msg),
                        )
                        st.write_stream(escaped_stream(stream, raw))
                    status.empty()
                    result = demote_headings("".join(raw))
                    if result.strip():
                        st.session_state.results[result_key] = result
                        # 저장된 결과로 다시 그려 버튼 문구를 '다시 분석'으로 바꾼다 (API 재호출 없음).
                        st.rerun()
                    else:
                        st.warning("모델이 빈 응답을 돌려줬어요. 다시 시도해 주세요.")
                except Exception as e:
                    status.empty()
                    st.error(friendly_error(e), icon="🚫")
        elif has_result:
            result_box.markdown(md_safe(st.session_state.results[result_key]))
        else:
            result_box.caption("**분석 실행** 버튼을 누르면 결과가 여기에 표시돼요.")

        if result_key in st.session_state.results:
            st.download_button(
                "📥 결과 다운로드 (.md)",
                data=st.session_state.results[result_key].encode("utf-8"),
                file_name=f"pdf_analysis_{kind}_{datetime.now():%Y%m%d_%H%M}.md",
                mime="text/markdown",
            )

# ---------------------------------------------------------------------------
# 탭 3: 질문하기
# ---------------------------------------------------------------------------
with tab_chat:
    if not usable_docs:
        st.info("질문하려면 먼저 텍스트가 있는 PDF를 업로드해 주세요.")
    else:
        # 이번 실행에서 처리할 질문 (입력창 값은 스크립트 시작 시점에 이미 session_state에 있다)
        question = st.session_state.get("chat_input") or st.session_state.pending_question
        st.session_state.pending_question = None

        # 새 메시지도 입력창 위에 쌓이도록 대화 영역을 입력창보다 먼저 만든다.
        chat_box = st.container()
        with chat_box:
            if not st.session_state.messages and not question:
                st.caption("이런 질문을 해 보세요")
                for q in EXAMPLE_QUESTIONS:
                    if st.button(f"💡 {q}", key=f"example_{q}"):
                        st.session_state.pending_question = q
                        st.rerun()
            for msg in st.session_state.messages:
                with st.chat_message(msg["role"]):
                    st.markdown(md_safe(msg["content"]))

        st.chat_input("문서에 대해 질문해 보세요", key="chat_input")
        if question:
            with chat_box, st.chat_message("user"):
                st.markdown(md_safe(question))
            with chat_box, st.chat_message("assistant"):
                raw: list[str] = []
                try:
                    st.write_stream(
                        escaped_stream(
                            stream_answer(
                                client,
                                settings.model,
                                usable_docs,
                                st.session_state.messages,
                                question,
                                language,
                            ),
                            raw,
                        )
                    )
                except Exception as e:
                    st.error(friendly_error(e), icon="🚫")
                else:
                    answer = demote_headings("".join(raw))
                    if answer.strip():
                        st.session_state.messages.append({"role": "user", "content": question})
                        st.session_state.messages.append({"role": "assistant", "content": answer})
                        # 예시 질문 버튼을 숨기기 위해 다시 그린다.
                        st.rerun()
