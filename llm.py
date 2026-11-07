"""OpenAI 클라이언트, 설정 로딩, 프롬프트, 분석·Q&A 함수.

설정 로딩 순서 (PRD 6.3):
1. st.secrets (Streamlit Cloud)
2. .env / os.environ (로컬)
3. 둘 다 없으면 api_key=None → app.py에서 안내 후 st.stop()
"""

import os
import re
from collections.abc import Callable, Iterator
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass

import openai
import streamlit as st
from dotenv import load_dotenv
from openai import OpenAI

from pdf_utils import DEFAULT_CHUNK_CHARS, Document, chunk_text

DEFAULT_MODEL = "gpt-6.1-terra"

# 문서 하나를 최종 프롬프트에 넣을 때의 최대 글자 수. 넘으면 map-reduce로 줄인다.
DOC_MATERIAL_LIMIT = DEFAULT_CHUNK_CHARS
# 최종 프롬프트에 넣을 모든 문서 자료의 합계 최대 글자 수.
TOTAL_MATERIAL_LIMIT = 60_000
# 부분 요약(map) 단계에서 동시에 보낼 API 요청 수.
MAP_WORKERS = 4
# 부분 요약을 반복하는 최대 횟수. 넘으면 남은 분량을 잘라내고 사용자에게 알린다.
MAX_CONDENSE_ROUNDS = 3
# Q&A에서 한 번에 넣을 문서 컨텍스트 최대 글자 수.
QA_CONTEXT_LIMIT = 60_000
# Q&A에서 문서를 나누는 청크 크기.
QA_CHUNK_CHARS = 4_000
# Q&A에서 모델에 보낼 최근 대화 메시지 수.
QA_HISTORY_MESSAGES = 10

TRUNCATION_NOTE = (
    "> ⚠️ 문서가 매우 길어 일부 내용이 분석에서 빠졌을 수 있어요. "
    "중요한 부분은 해당 페이지만 따로 PDF로 만들어 분석해 보세요.\n\n"
)

load_dotenv()


# ---------------------------------------------------------------------------
# 설정 / 클라이언트
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class Settings:
    api_key: str | None
    model: str


def _read_secret(name: str) -> str | None:
    """st.secrets에서 값을 읽는다. secrets.toml이 없을 때 생기는 예외는 무시한다."""
    try:
        value = st.secrets.get(name)
    except Exception:
        return None
    return str(value).strip() if value else None


def _read_setting(name: str) -> str | None:
    value = _read_secret(name)
    if value:
        return value
    env_value = os.environ.get(name, "").strip()
    return env_value or None


def get_settings() -> Settings:
    """API 키와 모델명을 불러온다. 키가 없으면 api_key는 None이다."""
    return Settings(
        api_key=_read_setting("OPENAI_API_KEY"),
        model=_read_setting("OPENAI_MODEL") or DEFAULT_MODEL,
    )


@st.cache_resource
def get_client(api_key: str) -> OpenAI:
    """OpenAI 클라이언트를 만든다. 같은 키에 대해서는 한 번만 생성한다."""
    return OpenAI(api_key=api_key, timeout=120)


def friendly_error(e: Exception) -> str:
    """OpenAI 예외를 사용자용 한국어 메시지로 바꾼다."""
    if isinstance(e, openai.AuthenticationError):
        return "API 키가 올바르지 않아요. `.env` 또는 Streamlit Secrets의 `OPENAI_API_KEY`를 확인해 주세요."
    if isinstance(e, openai.PermissionDeniedError):
        return "이 API 키로는 해당 모델을 사용할 권한이 없어요."
    if isinstance(e, openai.NotFoundError):
        return "모델을 찾을 수 없어요. `OPENAI_MODEL` 설정을 확인해 주세요."
    if isinstance(e, openai.RateLimitError):
        return "요청 한도를 초과했거나 크레딧이 부족해요. 잠시 후 다시 시도해 주세요."
    if isinstance(e, openai.APITimeoutError):
        return "응답 시간이 초과됐어요. 잠시 후 다시 시도해 주세요."
    if isinstance(e, openai.APIConnectionError):
        return "OpenAI 서버에 연결할 수 없어요. 인터넷 연결을 확인해 주세요."
    if isinstance(e, openai.BadRequestError):
        return "요청이 거부됐어요. 문서가 너무 길거나 요청 형식에 문제가 있을 수 있어요."
    if isinstance(e, openai.APIError):
        return "OpenAI API 오류가 발생했어요. 잠시 후 다시 시도해 주세요."
    return f"알 수 없는 오류가 발생했어요: {type(e).__name__}"


# ---------------------------------------------------------------------------
# 프롬프트
# ---------------------------------------------------------------------------

SYSTEM_PROMPT = (
    "당신은 PDF 문서 분석 도우미입니다. 업로드된 문서 내용만 근거로 답하고, "
    "문서에 없는 내용은 문서에 없다고 말하세요. 추측하지 마세요. "
    "가능하면 [파일명 - p.N] 형식으로 근거를 밝히세요. "
    "답변은 마크다운으로 깔끔하게 작성하세요. "
    "제목이 필요하면 ### 또는 #### 수준만 사용하고, # 이나 ## 제목은 쓰지 마세요."
)

ANALYSIS_TYPES = ["요약", "핵심 포인트", "키워드", "문서 비교"]

ANALYSIS_TASKS = {
    "요약": (
        "각 문서를 파일별로 요약한 뒤(파일명을 소제목으로), "
        "문서가 2개 이상이면 마지막에 '통합 요약' 섹션을 추가하세요."
    ),
    "핵심 포인트": (
        "각 문서의 핵심 내용을 파일별로 5~10개의 불릿으로 정리하세요. "
        "각 불릿 끝에 [파일명 - p.N] 근거를 붙이세요."
    ),
    "키워드": (
        "각 문서의 주요 키워드 10개 내외를 파일별로 뽑고, "
        "키워드마다 한 줄 설명과 [파일명 - p.N] 근거를 붙이세요. 표 형식을 사용해도 좋습니다."
    ),
    "문서 비교": (
        "문서들을 비교하세요. '공통점', '차이점', '문서별 고유 내용' 섹션으로 나누고, "
        "가능하면 비교표를 포함하세요. 각 항목에 [파일명 - p.N] 근거를 붙이세요."
    ),
}

MAP_PROMPT = (
    "다음은 문서 '{name}'의 일부({index}/{total})입니다. 이후 '{kind}' 작업에 필요한 "
    "핵심 정보를 압축해 정리하세요. 결과는 {target:,}자 이내로 작성하세요. "
    "원문의 [파일명 - p.N] 표시는 해당 내용 옆에 그대로 유지하세요.\n\n"
    "---\n{chunk}"
)

_BIG_HEADING_RE = re.compile(r"^#{1,2}(?=\s)", re.MULTILINE)


def demote_headings(text: str) -> str:
    """모델이 # / ## 제목을 쓰면 ### 로 낮춘다 (앱 제목보다 커지지 않게)."""
    return _BIG_HEADING_RE.sub("###", text)


def _language_instruction(language: str) -> str:
    return f"\n\n응답 언어: {language}"


# ---------------------------------------------------------------------------
# OpenAI 호출
# ---------------------------------------------------------------------------

def _complete(client: OpenAI, model: str, messages: list[dict]) -> str:
    response = client.chat.completions.create(model=model, messages=messages)
    return response.choices[0].message.content or ""


def _stream(client: OpenAI, model: str, messages: list[dict]) -> Iterator[str]:
    stream = client.chat.completions.create(model=model, messages=messages, stream=True)
    for event in stream:
        if event.choices and event.choices[0].delta.content:
            yield event.choices[0].delta.content


def _map_chunks(
    client: OpenAI,
    model: str,
    name: str,
    chunks: list[str],
    kind: str,
    target: int,
    on_progress: Callable[[str], None] | None,
) -> list[str]:
    """청크들을 병렬로 부분 요약한다. 결과는 원래 순서대로 돌려준다.

    on_progress는 Streamlit 요소를 건드리므로 반드시 메인 스레드에서만 호출한다.
    """
    def summarize(index: int, chunk: str) -> str:
        prompt = MAP_PROMPT.format(
            name=name, index=index + 1, total=len(chunks), kind=kind, target=target, chunk=chunk
        )
        return _complete(
            client,
            model,
            [{"role": "system", "content": SYSTEM_PROMPT}, {"role": "user", "content": prompt}],
        )

    results: list[str] = [""] * len(chunks)
    if on_progress:
        on_progress(f"'{name}' 부분 분석 중... (0/{len(chunks)})")
    with ThreadPoolExecutor(max_workers=MAP_WORKERS) as pool:
        futures = {pool.submit(summarize, i, c): i for i, c in enumerate(chunks)}
        for done, future in enumerate(as_completed(futures), start=1):
            results[futures[future]] = future.result()
            if on_progress:
                on_progress(f"'{name}' 부분 분석 중... ({done}/{len(chunks)})")
    return results


def _condense(
    client: OpenAI,
    model: str,
    name: str,
    text: str,
    kind: str,
    limit: int,
    on_progress: Callable[[str], None] | None,
    first_chunks: list[str] | None = None,
) -> tuple[str, bool]:
    """text가 limit보다 길면 부분 요약(map)을 반복해 limit 이하로 줄인다.

    Returns:
        (줄인 텍스트, 내용 일부를 잘라냈는지 여부)
    """
    chunks = first_chunks
    for _ in range(MAX_CONDENSE_ROUNDS):
        if len(text) <= limit:
            return text, False
        if chunks is None:
            chunks = chunk_text(text, DEFAULT_CHUNK_CHARS)
        target = max(limit // len(chunks), 300)
        new_text = "\n\n".join(_map_chunks(client, model, name, chunks, kind, target, on_progress))
        chunks = None
        if len(new_text) >= len(text):
            break  # 더 줄어들지 않으면 반복해도 소용없다.
        text = new_text
    if len(text) <= limit:
        return text, False
    return text[:limit], True


def stream_analysis(
    client: OpenAI,
    model: str,
    docs: list[Document],
    kind: str,
    language: str,
    on_progress: Callable[[str], None] | None = None,
) -> Iterator[str]:
    """분석을 실행하고 최종 결과를 스트리밍으로 돌려준다.

    1. 문서마다 DOC_MATERIAL_LIMIT 이하로 줄인다 (긴 문서는 부분 요약).
    2. 전체 합계가 TOTAL_MATERIAL_LIMIT를 넘으면 문서별 할당량까지 한 번 더 줄인다.
    3. 모은 자료로 최종 결과(reduce)를 스트리밍한다.
    내용 일부를 잘라냈다면 결과 맨 앞에 안내 문구를 붙인다.
    """
    truncated = False
    materials: list[str] = []
    for doc in docs:
        text, cut = _condense(
            client, model, doc.name, doc.tagged_text(), kind, DOC_MATERIAL_LIMIT,
            on_progress, first_chunks=doc.tagged_chunks(DEFAULT_CHUNK_CHARS),
        )
        truncated |= cut
        materials.append(text)

    if sum(len(m) for m in materials) > TOTAL_MATERIAL_LIMIT:
        budget = TOTAL_MATERIAL_LIMIT // len(docs)
        for i, doc in enumerate(docs):
            if len(materials[i]) > budget:
                materials[i], cut = _condense(
                    client, model, doc.name, materials[i], kind, budget, on_progress
                )
                truncated |= cut

    if on_progress:
        on_progress("최종 결과 작성 중...")
    body = "\n\n".join(
        f"### 문서: {doc.name} ({doc.page_count}페이지)\n{material}"
        for doc, material in zip(docs, materials)
    )
    user_prompt = (
        f"작업: {ANALYSIS_TASKS[kind]}{_language_instruction(language)}\n\n"
        "아래는 업로드된 문서 내용입니다.\n\n" + body
    )
    if truncated:
        yield TRUNCATION_NOTE
    yield from _stream(
        client,
        model,
        [{"role": "system", "content": SYSTEM_PROMPT}, {"role": "user", "content": user_prompt}],
    )


# ---------------------------------------------------------------------------
# Q&A
# ---------------------------------------------------------------------------

_WORD_RE = re.compile(r"[0-9a-z가-힣]+")


def _bigrams(text: str) -> set[str]:
    """단어별 두 글자 조각 집합. 조사가 붙어도('보안정책은' vs '보안정책') 대부분 겹친다."""
    grams: set[str] = set()
    for word in _WORD_RE.findall(text.lower()):
        if len(word) == 1:
            continue
        grams.update(word[i : i + 2] for i in range(len(word) - 1))
    return grams


def build_qa_context(docs: list[Document], question: str) -> str:
    """질문에 쓸 문서 컨텍스트를 만든다.

    전체 텍스트가 QA_CONTEXT_LIMIT 이하면 전부 넣는다. 넘으면 페이지 표시가 유지된
    청크로 나눈 뒤, 질문과 두 글자 조각이 많이 겹치는 청크부터 한도까지 넣는다
    (출력은 원래 순서 유지).
    """
    full = "\n\n".join(doc.tagged_text() for doc in docs)
    if len(full) <= QA_CONTEXT_LIMIT:
        return full

    chunks: list[str] = []
    for doc in docs:
        chunks.extend(doc.tagged_chunks(QA_CHUNK_CHARS))

    q_grams = _bigrams(question)
    scores = [len(q_grams & _bigrams(c)) for c in chunks]
    ranked = sorted(range(len(chunks)), key=lambda i: scores[i], reverse=True)
    selected: list[int] = []
    total = 0
    for i in ranked:
        if total + len(chunks[i]) > QA_CONTEXT_LIMIT:
            continue
        selected.append(i)
        total += len(chunks[i])
    return "\n\n".join(chunks[i] for i in sorted(selected))


def stream_answer(
    client: OpenAI,
    model: str,
    docs: list[Document],
    history: list[dict],
    question: str,
    language: str,
) -> Iterator[str]:
    """대화 내역과 문서 컨텍스트를 바탕으로 질문에 대한 답을 스트리밍한다."""
    context = build_qa_context(docs, question)
    messages = [
        {
            "role": "system",
            "content": f"{SYSTEM_PROMPT}{_language_instruction(language)}\n\n[문서 내용]\n{context}",
        },
        *history[-QA_HISTORY_MESSAGES:],
        {"role": "user", "content": question},
    ]
    yield from _stream(client, model, messages)
