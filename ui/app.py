"""
Gradio 챗봇 UI (HuggingFace Spaces 배포 대상).
- Claude 스타일 채팅: 유저 말풍선 우측, 어시스턴트는 플레인 텍스트, 하단 pill 입력창
- 스트리밍 답변 (검색 → 답변 생성 단계별 표시)
- 우측 패널: 도메인 분석 카드 + 근거 판례 카드
"""
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import html
import time
import logging

import gradio as gr

from graph.pipeline import run_retrieval
from agents.generator import stream_answer
from logs.db import log_query, update_feedback

logger = logging.getLogger(__name__)

DOMAIN_COLORS = {
    "형사": "#dc2626",
    "민사": "#2563eb",
    "가족법": "#db2777",
    "행정": "#7c3aed",
    "노동": "#ea580c",
    "부동산": "#059669",
    "기타": "#64748b",
}

CSS = """
.gradio-container { width: 100% !important; max-width: 1240px !important; margin: 0 auto !important; }
footer { display: none !important; }

/* 헤더 */
#app-header { text-align: center; padding: 4px 0 0 0; }
#app-header h1 { font-size: 1.6rem; margin-bottom: 0; letter-spacing: -0.5px; }
#app-header p { color: var(--body-text-color-subdued); margin: 2px 0 0 0; font-size: 0.86rem; }
#disclaimer-banner {
    font-size: 0.75rem; color: var(--body-text-color-subdued);
    text-align: center; margin: 4px 0 6px 0;
}

/* 메인 2컬럼 고정 (Gradio 기본 wrap 방지) */
#main-row { flex-wrap: nowrap !important; }
#main-row > .column:first-child { flex: 3 1 0 !important; min-width: 0 !important; }
#main-row > .column:last-child { flex: 2 1 0 !important; min-width: 0 !important; }

/* ── Claude 스타일 채팅 ── */
#chat-window {
    border: none !important; background: transparent !important;
    box-shadow: none !important;
}
#chat-window .bubble-wrap { background: transparent !important; }
#chat-window .message { border: none !important; font-size: 0.95rem; line-height: 1.65; }
/* 유저: 우측 말풍선 */
#chat-window .user-row .message, #chat-window .message.user {
    background: color-mix(in srgb, var(--color-accent) 12%, var(--background-fill-secondary)) !important;
    border-radius: 18px 18px 6px 18px !important;
    padding: 10px 16px !important;
}
/* 어시스턴트: 배경 없는 플레인 텍스트 */
#chat-window .bot-row .message, #chat-window .message.bot {
    background: transparent !important;
    padding: 4px 2px !important;
}

/* 하단 pill 입력창 */
#input-row {
    border: 1px solid var(--border-color-primary);
    border-radius: 26px;
    background: var(--background-fill-primary);
    box-shadow: 0 2px 14px rgba(0,0,0,0.07);
    padding: 4px 6px 4px 16px;
    gap: 4px;
    align-items: center;
}
#input-row .form, #input-row .block { border: none !important; box-shadow: none !important; background: transparent !important; }
#input-row textarea {
    border: none !important; box-shadow: none !important;
    background: transparent !important; resize: none;
}
#send-btn {
    border-radius: 50% !important;
    min-width: 42px !important; max-width: 42px !important; height: 42px !important;
    padding: 0 !important; font-size: 1.15rem !important; flex: none !important;
}

/* 피드백 */
#feedback-row { justify-content: flex-end !important; gap: 0; margin-top: 2px; }
#feedback-row > div, #feedback-row .form { flex: none !important; min-width: 0 !important; }
#feedback-row button {
    background: transparent !important; border: none !important; box-shadow: none !important;
    color: var(--body-text-color-subdued) !important;
    font-size: 0.78rem !important; min-width: 0 !important; width: auto !important;
    flex: none !important; padding: 2px 10px !important;
}
#feedback-row button:hover { color: var(--body-text-color) !important; }

/* ── 사이드 패널 ── */
.panel-card {
    border: 1px solid var(--border-color-primary);
    border-radius: 14px; padding: 14px 16px; margin-bottom: 10px;
    background: var(--background-fill-secondary);
}
.panel-card .card-label {
    font-size: 0.72rem; font-weight: 700; text-transform: uppercase;
    letter-spacing: 0.08em; color: var(--body-text-color-subdued); margin-bottom: 8px;
}
.domain-badge {
    display: inline-block; padding: 3px 12px; border-radius: 999px;
    color: #fff; font-weight: 700; font-size: 0.9rem;
}
.conf-track {
    height: 6px; border-radius: 999px; margin-top: 10px;
    background: var(--border-color-primary); overflow: hidden;
}
.conf-fill { height: 100%; border-radius: 999px; }
.rewritten-query {
    margin-top: 10px; font-size: 0.86rem; color: var(--body-text-color);
    background: var(--background-fill-primary); border-radius: 8px; padding: 8px 10px;
    border: 1px dashed var(--border-color-primary);
}

/* 판례 카드 */
.prec-card {
    border: 1px solid var(--border-color-primary); border-radius: 14px;
    padding: 12px 14px; margin-bottom: 10px; background: var(--background-fill-secondary);
}
.prec-card .prec-title { font-weight: 700; font-size: 0.92rem; line-height: 1.35; margin-bottom: 6px; }
.prec-chips { display: flex; flex-wrap: wrap; gap: 4px; margin-bottom: 8px; }
.prec-chip {
    font-size: 0.72rem; padding: 2px 8px; border-radius: 999px;
    background: var(--background-fill-primary); border: 1px solid var(--border-color-primary);
    color: var(--body-text-color-subdued);
}
.prec-chip.domain { color: #fff; border: none; font-weight: 600; }
.prec-statutes { font-size: 0.8rem; margin-bottom: 6px; }
.prec-statutes b { color: var(--color-accent); }
.prec-summary { font-size: 0.84rem; line-height: 1.55; color: var(--body-text-color); }
.prec-verdict {
    margin-top: 8px; font-size: 0.8rem; padding: 7px 10px; border-radius: 8px;
    background: var(--background-fill-primary); border-left: 3px solid var(--color-accent);
    color: var(--body-text-color-subdued);
}
details.prec-verdict summary { cursor: pointer; user-select: none; }
details.prec-verdict summary::marker { color: var(--color-accent); }
details.prec-verdict .verdict-body { margin-top: 6px; line-height: 1.6; }
.placeholder-note {
    color: var(--body-text-color-subdued); font-size: 0.85rem; text-align: center;
    padding: 26px 10px; border: 1px dashed var(--border-color-primary); border-radius: 14px;
}
"""


def _esc(s) -> str:
    return html.escape(str(s or ""))


def _analysis_card(domain: str, confidence: float, rewritten_query: str) -> str:
    color = DOMAIN_COLORS.get(domain, DOMAIN_COLORS["기타"])
    pct = max(0, min(100, round(confidence * 100)))
    return f"""
<div class="panel-card">
  <div class="card-label">법적 영역 분석</div>
  <span class="domain-badge" style="background:{color}">{_esc(domain)}</span>
  <span style="font-size:0.82rem;color:var(--body-text-color-subdued);margin-left:8px">확신도 {pct}%</span>
  <div class="conf-track"><div class="conf-fill" style="width:{pct}%;background:{color}"></div></div>
  <div class="rewritten-query">🔍 {_esc(rewritten_query)}</div>
</div>"""


def _verdict_block(verdict: str) -> str:
    """주문 표시. 길면 접기/펼치기로 전문 제공 (잘라내지 않음)."""
    v = _esc(verdict)
    if len(verdict) <= 90:
        return f'<div class="prec-verdict"><b>주문</b> {v}</div>'
    preview = _esc(verdict[:60])
    return (
        f'<details class="prec-verdict"><summary><b>주문</b> {preview}… <u>전체 보기</u></summary>'
        f'<div class="verdict-body">{v}</div></details>'
    )


def _precedent_cards(summaries: list[dict]) -> str:
    if not summaries:
        return '<div class="placeholder-note">관련 판례를 찾지 못했습니다.</div>'

    cards = []
    for i, s in enumerate(summaries, 1):
        color = DOMAIN_COLORS.get(s.get("domain", "기타"), DOMAIN_COLORS["기타"])
        chips = [f'<span class="prec-chip domain" style="background:{color}">{_esc(s.get("domain", "기타"))}</span>']
        if s.get("court"):
            chips.append(f'<span class="prec-chip">🏛 {_esc(s["court"])}</span>')
        if s.get("date"):
            chips.append(f'<span class="prec-chip">📅 {_esc(s["date"])}</span>')
        if s.get("source"):
            chips.append(f'<span class="prec-chip">{_esc(s["source"])}</span>')

        statutes = s.get("key_statutes", "")
        statutes_html = (
            f'<div class="prec-statutes"><b>참조법조</b> · {_esc(statutes)}</div>'
            if statutes and statutes != "명시 없음" else ""
        )
        verdict_html = _verdict_block(s["verdict_snippet"]) if s.get("verdict_snippet") else ""
        cards.append(f"""
<div class="prec-card">
  <div class="prec-title">[{i}] {_esc(s.get("case_name", "사건명 없음"))}</div>
  <div class="prec-chips">{"".join(chips)}</div>
  {statutes_html}
  <div class="prec-summary">{_esc(s.get("summary", ""))}</div>
  {verdict_html}
</div>""")
    return "\n".join(cards)


_EMPTY_ANALYSIS = '<div class="placeholder-note">상담을 시작하면 법적 영역 분석이 표시됩니다.</div>'
_EMPTY_PRECEDENTS = '<div class="placeholder-note">답변의 근거가 된 판례가 여기에 표시됩니다.</div>'


def chat(user_message: str, history: list[dict]):
    """검색 → 스트리밍 답변. (chatbot, 분석 카드, 판례 카드, query_id) 순으로 yield."""
    user_message = (user_message or "").strip()
    if not user_message:
        yield history, gr.update(), gr.update(), gr.update()
        return

    history = history + [
        {"role": "user", "content": user_message},
        {"role": "assistant", "content": "⏳ 관련 판례를 검색하고 있습니다..."},
    ]
    yield history, gr.update(), gr.update(), ""

    start = time.time()
    try:
        result = run_retrieval(user_message)
    except Exception:
        logger.exception("검색 파이프라인 실패")
        history[-1]["content"] = "처리 중 오류가 발생했습니다. 잠시 후 다시 시도해주세요."
        yield history, gr.update(), gr.update(), ""
        return

    analysis = _analysis_card(result["domain"], result["domain_confidence"], result["rewritten_query"])
    cards = _precedent_cards(result["summaries"])
    history[-1]["content"] = "✍️ 판례를 바탕으로 답변을 작성하고 있습니다..."
    yield history, analysis, cards, ""

    answer = ""
    for chunk in stream_answer(user_message, result["context"]):
        answer += chunk
        history[-1]["content"] = answer
        yield history, analysis, cards, ""

    query_id = ""
    try:
        query_id = log_query(
            user_query=user_message,
            domain=result["domain"],
            rewritten_query=result["rewritten_query"],
            retrieved_count=len(result["retrieved_chunks"]),
            summaries=result["summaries"],
            final_answer=answer,
            latency_ms=(time.time() - start) * 1000,
        )
    except Exception:
        logger.exception("쿼리 로깅 실패")

    yield history, analysis, cards, str(query_id)


def handle_feedback(helpful: bool, query_id: str):
    if not query_id:
        gr.Warning("피드백을 남길 상담 내역이 없습니다.")
        return
    try:
        update_feedback(int(query_id), helpful)
        gr.Info("피드백이 저장되었습니다. 감사합니다!")
    except Exception:
        logger.exception("피드백 저장 실패")
        gr.Warning("피드백 저장에 실패했습니다.")


theme = gr.themes.Soft(
    primary_hue="orange",
    neutral_hue="stone",
    font=[gr.themes.GoogleFont("Noto Sans KR"), "system-ui", "sans-serif"],
).set(
    block_shadow="none",
)

with gr.Blocks(title="PanRye.ai — 판례 기반 법률 상담", theme=theme, css=CSS) as demo:
    gr.HTML("""
<div id="app-header">
  <h1>⚖️ PanRye.ai</h1>
  <p>상황을 설명하면 실제 판례를 근거로 답변합니다</p>
</div>
<div id="disclaimer-banner">
  AI가 공개 판례 기반으로 생성한 참고 정보입니다 · 정식 법률 자문이 아니며 실제 판단은 변호사와 상담하세요
</div>""")

    query_id_state = gr.State("")

    with gr.Row(equal_height=False, elem_id="main-row"):
        with gr.Column(scale=3):
            chatbot = gr.Chatbot(
                elem_id="chat-window",
                height=560,
                type="messages",
                show_label=False,
                show_copy_button=True,
                placeholder=(
                    "### 어떤 상황이신가요?\n\n"
                    "예) *전세 계약이 끝났는데 집주인이 보증금을 안 돌려줘요*"
                ),
            )
            with gr.Row(elem_id="input-row"):
                msg_input = gr.Textbox(
                    placeholder="상황을 자세히 설명할수록 정확한 판례를 찾습니다",
                    show_label=False,
                    lines=1,
                    max_lines=6,
                    scale=5,
                    container=False,
                )
                submit_btn = gr.Button("↑", elem_id="send-btn", variant="primary", scale=0)

            with gr.Row(elem_id="feedback-row"):
                helpful_btn = gr.Button("👍 도움됨", size="sm")
                not_helpful_btn = gr.Button("👎 아쉬움", size="sm")

        with gr.Column(scale=2):
            analysis_panel = gr.HTML(_EMPTY_ANALYSIS)
            gr.Markdown("#### 📚 근거 판례")
            precedents_panel = gr.HTML(_EMPTY_PRECEDENTS)

    chat_outputs = [chatbot, analysis_panel, precedents_panel, query_id_state]

    submit_btn.click(
        fn=chat, inputs=[msg_input, chatbot], outputs=chat_outputs,
    ).then(lambda: "", outputs=msg_input)

    msg_input.submit(
        fn=chat, inputs=[msg_input, chatbot], outputs=chat_outputs,
    ).then(lambda: "", outputs=msg_input)

    helpful_btn.click(fn=lambda qid: handle_feedback(True, qid), inputs=query_id_state)
    not_helpful_btn.click(fn=lambda qid: handle_feedback(False, qid), inputs=query_id_state)


if __name__ == "__main__":
    # 포트는 GRADIO_SERVER_PORT 환경변수로 재정의 가능 (기본 7860)
    demo.queue().launch(server_name="0.0.0.0", share=False)
