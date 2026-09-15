"""`.env` 설정 로드. 이 프로젝트에서 ``os.getenv``를 호출하는 유일한 모듈이다."""

from __future__ import annotations

import os
from dataclasses import dataclass, replace
from functools import lru_cache
from pathlib import Path
from typing import Any

from dotenv import load_dotenv

PROJECT_ROOT = Path(__file__).resolve().parent.parent
STORAGE_DIR = PROJECT_ROOT / "storage"
INDEX_PATH = STORAGE_DIR / "index.faiss"
METADATA_PATH = STORAGE_DIR / "metadata.pkl"
MANIFEST_PATH = STORAGE_DIR / "manifest.json"


class ConfigError(RuntimeError):
    """`.env` 값이 없거나 해석할 수 없을 때."""


@dataclass(frozen=True)
class Config:
    """실행 시점에 확정된 설정값."""

    openai_api_key: str
    embedding_model: str
    chat_model: str
    top_k: int
    final_k: int
    chunk_size: int
    chunk_overlap: int
    min_chunk_chars: int
    similarity_threshold: float
    relative_cutoff: float
    mmr_lambda: float
    history_turns: int
    max_context_chars: int


def _get_int(name: str, default: int) -> int:
    raw = os.getenv(name)
    if raw is None or raw.strip() == "":
        return default
    try:
        return int(raw)
    except ValueError as exc:
        raise ConfigError(f".env의 {name} 값이 정수가 아닙니다: {raw!r}") from exc


def _get_float(name: str, default: float) -> float:
    raw = os.getenv(name)
    if raw is None or raw.strip() == "":
        return default
    try:
        return float(raw)
    except ValueError as exc:
        raise ConfigError(f".env의 {name} 값이 실수가 아닙니다: {raw!r}") from exc


# UI에서 런타임에 바꾼 값. `.env`를 덮어쓰되 파일은 건드리지 않는다.
_overrides: dict[str, Any] = {}


def set_overrides(**values: Any) -> None:
    """UI 위젯이 바꾼 설정을 반영한다. 값이 그대로면 아무 일도 하지 않는다."""
    global _overrides
    if values == _overrides:
        return
    _overrides = dict(values)
    get_config.cache_clear()


@lru_cache(maxsize=1)
def get_config() -> Config:
    """`.env`를 읽어 설정을 반환한다. 설정이 바뀌기 전까지 한 번만 수행된다."""
    load_dotenv(PROJECT_ROOT / ".env")

    api_key = os.getenv("OPENAI_API_KEY", "").strip()
    if not api_key:
        raise ConfigError(
            f"OPENAI_API_KEY가 없습니다. {PROJECT_ROOT / '.env'} 파일에 "
            "OPENAI_API_KEY=sk-... 형식으로 추가하세요."
        )

    config = Config(
        openai_api_key=api_key,
        embedding_model=os.getenv("EMBEDDING_MODEL", "text-embedding-3-small"),
        chat_model=os.getenv("CHAT_MODEL", "gpt-4o-mini"),
        top_k=_get_int("TOP_K", 20),
        final_k=_get_int("FINAL_K", 8),
        chunk_size=_get_int("CHUNK_SIZE", 800),
        chunk_overlap=_get_int("CHUNK_OVERLAP", 150),
        min_chunk_chars=_get_int("MIN_CHUNK_CHARS", 50),
        # 1단계 실측: 한국어 질문 → 영어 문서에서 정답 청크가 0.263이었다.
        # PRD 초안의 0.30은 정답을 걸러낸다. 절대 하한은 낮게 두고, 실제
        # 선별은 아래 relative_cutoff(최고점 대비 비율)에 맡긴다.
        similarity_threshold=_get_float("SIMILARITY_THRESHOLD", 0.15),
        relative_cutoff=_get_float("RELATIVE_CUTOFF", 0.45),
        mmr_lambda=_get_float("MMR_LAMBDA", 0.7),
        history_turns=_get_int("HISTORY_TURNS", 3),
        max_context_chars=_get_int("MAX_CONTEXT_CHARS", 6000),
    )
    if _overrides:
        config = replace(config, **_overrides)

    if config.chunk_overlap >= config.chunk_size:
        raise ConfigError(
            f"CHUNK_OVERLAP({config.chunk_overlap})이 CHUNK_SIZE({config.chunk_size}) "
            "이상이면 청킹이 무한 루프에 빠집니다."
        )
    if config.final_k > config.top_k:
        raise ConfigError(
            f"FINAL_K({config.final_k})는 TOP_K({config.top_k})보다 클 수 없습니다."
        )
    if not 0.0 <= config.mmr_lambda <= 1.0:
        raise ConfigError(f"MMR_LAMBDA는 0~1이어야 합니다: {config.mmr_lambda}")
    if not 0.0 <= config.relative_cutoff <= 1.0:
        raise ConfigError(f"RELATIVE_CUTOFF는 0~1이어야 합니다: {config.relative_cutoff}")
    return config
