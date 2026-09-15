"""PDF 문서 챗봇 UI (F6).

실행:
    .venv/Scripts/python.exe -m streamlit run app.py
"""

from __future__ import annotations

import tempfile
from pathlib import Path

import streamlit as st

from core.config import ConfigError, get_config, set_overrides
from core.rag import (
    NO_ANSWER,
    ContextBlock,
    Turn,
    build_messages,
    recent_turns,
    rewrite_query,
    search_context,
    stream_completion,
)
from core.vectorstore import VectorStore, VectorStoreError, clear_storage, storage_exists
from ingest import ingest_paths

MODEL_CHOICES = ["gpt-4o-mini", "gpt-4o"]

st.set_page_config(
    page_title="문서 챗봇",
    page_icon=":material/menu_book:",
    layout="centered",
)


@st.cache_resource(show_spinner=False)
def load_store() -> tuple[VectorStore | None, str | None]:
    """인덱스를 한 번만 읽어 재사용한다 (F6.4).

    모델 불일치 같은 실패는 예외로 던지지 않고 메시지로 돌려준다. 캐시된
    함수가 매번 예외를 던지면 화면 전체가 멈춰 되돌릴 방법이 없어진다.
    """
    if not storage_exists():
        return None, None
    try:
        return VectorStore.load(), None
    except VectorStoreError as exc:
        return None, str(exc)


@st.cache_resource(show_spinner=False)
def upload_dir() -> Path:
    """업로드된 PDF를 잠시 두는 곳. 파일명을 보존해야 출처 표기가 맞는다."""
    return Path(tempfile.mkdtemp(prefix="pdf-rag-"))


def render_sources(blocks: list[ContextBlock]) -> None:
    """답변 아래 접이식 출처 패널 (F5.8, F6.3)."""
    if not blocks:
        return
    labels = ", ".join(block.label for block in blocks)
    with st.expander(f":material/description: 출처 {len(blocks)}건 · {labels}"):
        for number, block in enumerate(blocks, start=1):
            with st.container(border=True):
                st.markdown(f"**[{number}] {block.label}** · 유사도 {block.score:.3f}")
                if len(block.sources) > 1:
                    pages = ", ".join(
                        f"p.{source.chunk.page}" for source in block.sources
                    )
                    st.caption(f"이어 붙인 청크 {len(block.sources)}개 · {pages}")
                st.text(block.text)


def render_sidebar() -> VectorStore | None:
    """업로드, 인덱싱, 현황, 설정 (F6.1)."""
    store, load_error = load_store()

    with st.sidebar:
        st.subheader("문서")
        uploads = st.file_uploader(
            "PDF 업로드",
            type="pdf",
            accept_multiple_files=True,
            help="여러 개를 한 번에 올릴 수 있습니다.",
        )
        if st.button(
            "인덱싱",
            icon=":material/database_upload:",
            width="stretch",
            disabled=not uploads,
        ):
            run_indexing(uploads)
        render_index_report()

        if load_error:
            st.error(load_error, icon=":material/error:")
        elif store is None:
            st.info("아직 인덱싱된 문서가 없습니다.", icon=":material/info:")
        else:
            files = store.indexed_files
            st.metric("등록된 문서", f"{len(files)}건")
            st.metric("총 청크", f"{store.ntotal:,}개")
            st.caption(f"임베딩 모델 · {store.manifest.embedding_model}")
            with st.expander(f"문서 목록 {len(files)}건"):
                for item in files:
                    st.markdown(f"- {item.name} · 청크 {item.chunks}개")

        st.subheader("설정")
        config = get_config()
        top_k = st.slider(
            "검색할 청크 수",
            min_value=10,
            max_value=20,
            value=min(max(config.top_k, 10), 20),
            help="1차로 회수할 개수입니다. 이 중 일부만 답변 생성에 쓰입니다.",
        )
        model = st.selectbox(
            "답변 모델",
            options=_model_options(config.chat_model),
            index=0,
        )
        # 슬라이더를 최소로 내려도 FINAL_K가 TOP_K를 넘지 않게 맞춘다.
        set_overrides(top_k=top_k, final_k=min(config.final_k, top_k), chat_model=model)

        render_reset_controls(store is not None)

    return store


def _model_options(current: str) -> list[str]:
    """`.env`에 적힌 모델을 항상 첫 번째 선택지로 둔다."""
    return [current] + [name for name in MODEL_CHOICES if name != current]


def render_reset_controls(has_index: bool) -> None:
    """대화 초기화와 인덱스 초기화 (F4.5, F6.5)."""
    st.subheader("초기화")
    if st.button("대화 초기화", icon=":material/refresh:", width="stretch"):
        st.session_state.messages = []
        st.rerun()

    if not has_index:
        return

    if st.session_state.get("confirm_clear"):
        st.warning(
            "인덱스를 지우면 PDF를 다시 올려 임베딩해야 합니다. 되돌릴 수 없습니다.",
            icon=":material/warning:",
        )
        confirm, cancel = st.columns(2)
        if confirm.button("삭제", type="primary", width="stretch"):
            clear_storage()
            load_store.clear()
            st.session_state.messages = []
            st.session_state.confirm_clear = False
            st.rerun()
        if cancel.button("취소", width="stretch"):
            st.session_state.confirm_clear = False
            st.rerun()
    elif st.button("인덱스 초기화", icon=":material/delete:", width="stretch"):
        st.session_state.confirm_clear = True
        st.rerun()


def run_indexing(uploads: list) -> None:
    """업로드된 PDF를 디스크에 옮기고 인덱싱한다 (F1.1, F3.6)."""
    target = upload_dir()
    paths: list[str | Path] = []
    for upload in uploads:
        path = target / upload.name
        path.write_bytes(upload.getvalue())
        paths.append(path)

    progress = st.progress(0.0, text="PDF를 읽는 중...")

    def on_progress(done: int, total: int) -> None:
        progress.progress(done / total, text=f"임베딩 {done}/{total}")

    # 결과를 바로 그리지 않고 넘겨둔다. 인덱싱 뒤에는 사이드바 통계를 새로
    # 그려야 해서 rerun이 필요한데, rerun은 지금 그린 메시지를 지워버린다.
    report: dict = {
        "error": None,
        "added": 0,
        "total": 0,
        "duplicates": [],
        "skipped": [],
    }
    try:
        result = ingest_paths(paths, on_progress=on_progress)
    except (ConfigError, VectorStoreError, RuntimeError) as exc:
        report["error"] = str(exc)
    else:
        report.update(
            added=result.added_chunks,
            total=result.total_chunks,
            duplicates=result.duplicates,
            skipped=[item.reason for item in result.skipped],
        )
        load_store.clear()

    progress.empty()
    st.session_state.index_report = report
    st.rerun()


def render_index_report() -> None:
    """직전 인덱싱 결과를 한 번만 보여준다."""
    report = st.session_state.pop("index_report", None)
    if not report:
        return
    if report["error"]:
        st.error(report["error"], icon=":material/error:")
    if report["added"]:
        st.success(
            f"청크 {report['added']}개 추가 (전체 {report['total']}개)",
            icon=":material/check_circle:",
        )
    elif not any((report["error"], report["skipped"], report["duplicates"])):
        st.info("새로 인덱싱된 내용이 없습니다.", icon=":material/info:")
    for item in report["duplicates"]:
        st.info(f"건너뜀 · {item}", icon=":material/content_copy:")
    for reason in report["skipped"]:
        st.warning(reason, icon=":material/warning:")


def history_turns() -> list[Turn]:
    """지금까지의 대화를 검색·프롬프트에 쓸 형태로 바꾼다."""
    messages = st.session_state.messages
    turns: list[Turn] = []
    for user, assistant in zip(messages[::2], messages[1::2]):
        if user["role"] == "user" and assistant["role"] == "assistant":
            turns.append(Turn(question=user["content"], answer=assistant["content"]))
    return recent_turns(turns)


def respond(store: VectorStore, question: str) -> None:
    """검색 과정을 보여주고 답변을 스트리밍한다 (F5.9)."""
    history = history_turns()

    with st.status(":shimmer[근거를 찾는 중]", type="compact") as status:
        search_query = question
        if history:
            with st.status("질문 정리", type="step"):
                search_query = rewrite_query(question, history)
                st.write(search_query)
        with st.status("문서 검색", type="step"):
            blocks = search_context(store, search_query)
            st.write(f"관련 블록 {len(blocks)}개")
        status.update(
            label=f"근거 {len(blocks)}건" if blocks else "관련 근거 없음",
            state="complete",
        )

    if not blocks:
        text = NO_ANSWER
        st.markdown(text)
    else:
        text = st.write_stream(
            stream_completion(build_messages(question, blocks, history))
        )

    # 근거를 찾지 못했다고 답하면서 출처를 나열하면 사용자를 오도한다.
    if text.startswith(NO_ANSWER):
        blocks = []
    render_sources(blocks)

    st.session_state.messages.append(
        {"role": "assistant", "content": text, "blocks": blocks}
    )


def main() -> None:
    if "messages" not in st.session_state:
        st.session_state.messages = []

    st.title("문서 챗봇")

    try:
        get_config()
    except ConfigError as exc:
        st.error(str(exc), icon=":material/key_off:")
        st.stop()

    store = render_sidebar()

    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])
            render_sources(message.get("blocks") or [])

    if store is None:
        st.info(
            "왼쪽에서 PDF를 올리고 **인덱싱**을 누르면 질문할 수 있습니다.",
            icon=":material/upload_file:",
        )

    question = st.chat_input(
        "문서에 대해 질문하세요",
        disabled=store is None,
        submit_mode="disable",
    )
    if not question:
        return

    st.session_state.messages.append({"role": "user", "content": question})
    with st.chat_message("user"):
        st.markdown(question)

    with st.chat_message("assistant"):
        try:
            respond(store, question)
        except (VectorStoreError, ValueError, RuntimeError) as exc:
            st.error(str(exc), icon=":material/error:")
            st.session_state.messages.pop()  # 답변 없는 질문만 남기지 않는다


main()
