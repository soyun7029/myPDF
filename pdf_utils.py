"""PDF 텍스트 추출과 청크 분할."""

import hashlib
import io
from dataclasses import dataclass

import streamlit as st
from pypdf import PdfReader
from pypdf.errors import PdfReadError

DEFAULT_CHUNK_CHARS = 12_000


class PdfExtractionError(Exception):
    """PDF를 열거나 텍스트를 추출할 수 없을 때 발생한다. 메시지는 사용자에게 그대로 보여준다."""


@dataclass(frozen=True)
class Document:
    name: str
    file_hash: str
    pages: tuple[str, ...]

    @property
    def page_count(self) -> int:
        return len(self.pages)

    @property
    def char_count(self) -> int:
        return sum(len(p) for p in self.pages)

    @property
    def empty_pages(self) -> list[int]:
        """텍스트가 없는 페이지 번호(1부터)."""
        return [i for i, p in enumerate(self.pages, start=1) if not p.strip()]

    @property
    def has_text(self) -> bool:
        return self.char_count > 0 and len(self.empty_pages) < self.page_count

    def page_tag(self, page_no: int) -> str:
        return f"[{self.name} - p.{page_no}]"

    def tagged_text(self) -> str:
        """각 페이지 앞에 [파일명 - p.N] 표시를 붙인 전체 텍스트."""
        return "\n\n".join(
            f"{self.page_tag(i)}\n{text.strip()}"
            for i, text in enumerate(self.pages, start=1)
            if text.strip()
        )

    def tagged_chunks(self, max_chars: int) -> list[str]:
        """페이지 표시를 유지한 채 max_chars 이하의 청크로 나눈다.

        모든 청크는 [파일명 - p.N] 표시로 시작한다. 한 페이지가 max_chars보다 길면
        여러 조각으로 자르고, 조각마다 같은 페이지 표시를 다시 붙인다.
        짧은 페이지들은 순서대로 한 청크에 모은다.
        """
        pieces: list[str] = []
        for i, text in enumerate(self.pages, start=1):
            body = text.strip()
            if not body:
                continue
            tag = self.page_tag(i)
            budget = max(max_chars - len(tag) - 1, 1)
            for part in chunk_text(body, budget):
                pieces.append(f"{tag}\n{part}")

        chunks: list[str] = []
        current = ""
        for piece in pieces:
            candidate = f"{current}\n\n{piece}" if current else piece
            if len(candidate) <= max_chars:
                current = candidate
            else:
                if current:
                    chunks.append(current)
                current = piece
        if current:
            chunks.append(current)
        return chunks


def file_hash(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


@st.cache_data(show_spinner=False, max_entries=50)
def _extract_pages_cached(digest: str, _data: bytes) -> tuple[str, ...]:
    # _data는 캐시 키에서 제외되고, 파일 내용의 해시(digest)가 키 역할을 한다.
    try:
        reader = PdfReader(io.BytesIO(_data))
        if reader.is_encrypted:
            # 비밀번호 없이 열리는 암호화 PDF도 있으므로 빈 비밀번호로 먼저 시도한다.
            try:
                if not reader.decrypt(""):
                    raise PdfExtractionError("암호로 보호된 PDF라서 열 수 없어요.")
            except PdfExtractionError:
                raise
            except Exception as e:
                raise PdfExtractionError("암호로 보호된 PDF라서 열 수 없어요.") from e
        return tuple((page.extract_text() or "") for page in reader.pages)
    except PdfExtractionError:
        raise
    except (PdfReadError, ValueError, OSError, KeyError, TypeError) as e:
        raise PdfExtractionError("PDF 파일이 손상되었거나 올바른 PDF 형식이 아니에요.") from e
    except Exception as e:
        raise PdfExtractionError(f"PDF를 읽는 중 오류가 발생했어요: {type(e).__name__}") from e


def extract_text(name: str, data: bytes) -> Document:
    """PDF 바이트에서 페이지별 텍스트를 추출해 Document로 돌려준다.

    결과는 파일 내용 해시 기준으로 캐싱되므로, 같은 파일을 다시 처리하지 않는다.
    실패하면 PdfExtractionError를 던진다.
    """
    digest = file_hash(data)
    pages = _extract_pages_cached(digest, data)
    return Document(name=name, file_hash=digest, pages=pages)


def chunk_text(text: str, max_chars: int = DEFAULT_CHUNK_CHARS) -> list[str]:
    """긴 텍스트를 max_chars 이하의 청크로 나눈다 (map-reduce 요약용).

    문단(빈 줄) 경계를 우선으로 나누고, 한 문단이 너무 길면 줄 단위,
    그래도 길면 글자 단위로 자른다. 텍스트는 빠짐없이 순서대로 포함된다.
    """
    if max_chars <= 0:
        raise ValueError("max_chars는 1 이상이어야 해요.")
    if len(text) <= max_chars:
        return [text] if text else []

    pieces: list[str] = []
    for para in text.split("\n\n"):
        if len(para) <= max_chars:
            pieces.append(para)
            continue
        for line in para.split("\n"):
            while len(line) > max_chars:
                pieces.append(line[:max_chars])
                line = line[max_chars:]
            pieces.append(line)

    chunks: list[str] = []
    current = ""
    for piece in pieces:
        candidate = f"{current}\n\n{piece}" if current else piece
        if len(candidate) <= max_chars:
            current = candidate
        else:
            if current:
                chunks.append(current)
            current = piece
    if current:
        chunks.append(current)
    return chunks
