/* 판례.ai SPA — 프레임워크 없는 ES 모듈.
   SSE 계약은 src/panrye/api/sse.py 미러. */

const EVENT = {
  STAGE: "stage",
  DOMAIN: "domain",
  CONTEXT: "context",
  ANSWER_CHUNK: "answer_chunk",
  DONE: "done",
  ERROR: "error",
};

const STAGES = ["classify", "reformulate", "retrieve", "summarize", "generate"];

const el = {
  chat: document.getElementById("chat"),
  welcome: document.getElementById("welcome"),
  stepper: document.getElementById("stepper"),
  form: document.getElementById("form"),
  input: document.getElementById("input"),
  send: document.getElementById("send"),
  chips: document.getElementById("chips"),
};

/* ---------------- 상태 ----------------
   idle → streaming(검색 단계) → answering(토큰 수신) → done | error */
let state = "idle";
let es = null; // 활성 EventSource

/* ---------------- 마크다운 (escape-first 화이트리스트) ---------------- */

function escapeHtml(s) {
  return s
    .replaceAll("&", "&amp;")
    .replaceAll("<", "&lt;")
    .replaceAll(">", "&gt;")
    .replaceAll('"', "&quot;");
}

function inlineMd(s) {
  return s
    .replace(/\*\*([^*]+)\*\*/g, "<strong>$1</strong>")
    .replace(/`([^`]+)`/g, "<code>$1</code>");
}

/** 지원: **굵게**, `코드`, - 목록, --- 구분선, 문단. 그 외 전부 텍스트. */
function renderMarkdown(raw) {
  const lines = escapeHtml(raw).split("\n");
  const out = [];
  let list = null; // "ul" | null
  let afterHr = false;

  const closeList = () => {
    if (list) { out.push("</ul>"); list = null; }
  };

  for (const line of lines) {
    const t = line.trim();
    if (t === "---" || t === "***") {
      closeList();
      out.push("<hr>");
      afterHr = true;
      continue;
    }
    if (t.startsWith("- ") || t.startsWith("* ")) {
      if (!list) { out.push("<ul>"); list = "ul"; }
      out.push(`<li>${inlineMd(t.slice(2))}</li>`);
      continue;
    }
    closeList();
    if (t === "") continue;
    // 면책 고지(--- 이후)는 작은 글씨
    const cls = afterHr ? ' class="fineprint"' : "";
    out.push(`<p${cls}>${inlineMd(t)}</p>`);
  }
  closeList();
  return out.join("");
}

/* ---------------- 스테퍼 ---------------- */

function resetStepper() {
  el.stepper.hidden = false;
  for (const li of el.stepper.children) {
    li.classList.remove("active", "done", "error");
    li.querySelector(".stage-ms").textContent = "";
  }
}

function stepperItem(stage) {
  return el.stepper.querySelector(`[data-stage="${stage}"]`);
}

function onStageEvent({ stage, status, ms }) {
  const li = stepperItem(stage);
  if (!li) return;
  if (status === "start") {
    li.classList.add("active");
  } else if (status === "done") {
    li.classList.remove("active");
    li.classList.add("done");
    if (ms > 0) {
      li.querySelector(".stage-ms").textContent =
        ms >= 1000 ? `${(ms / 1000).toFixed(1)}s` : `${Math.round(ms)}ms`;
    }
  }
}

function markStepperError(stage) {
  const li = stepperItem(stage);
  if (li) { li.classList.remove("active"); li.classList.add("error"); }
}

/* ---------------- 렌더러 ---------------- */

const DOMAIN_LABEL = { 형사: "형사", 민사: "민사", 가족법: "가족법", 행정: "행정", 노동: "노동", 부동산: "부동산", 기타: "일반" };

function addUserMessage(text) {
  const div = document.createElement("div");
  div.className = "msg-user";
  div.textContent = text;
  el.chat.appendChild(div);
}

function createAnswerCard() {
  const card = document.createElement("article");
  card.className = "answer-card";
  card.innerHTML = `
    <div class="answer-meta">
      <span class="answer-status">판례 분석 중</span>
    </div>
    <div class="answer-body"></div>
    <div class="evidence-slot"></div>
    <div class="foot-slot"></div>`;
  el.chat.appendChild(card);
  return {
    root: card,
    meta: card.querySelector(".answer-meta"),
    body: card.querySelector(".answer-body"),
    evidenceSlot: card.querySelector(".evidence-slot"),
    footSlot: card.querySelector(".foot-slot"),
  };
}

function renderDomainBadge(card, { domain, confidence }) {
  const pct = Math.round((confidence || 0) * 100);
  card.meta.innerHTML = `
    <span class="domain-badge">◈ ${escapeHtml(DOMAIN_LABEL[domain] || domain || "일반")}</span>
    <span class="confidence" title="분류 신뢰도">
      <span class="confidence-bar"><i style="width:${pct}%"></i></span>${pct}%
    </span>
    <span class="answer-status">판례 검색 중</span>`;
}

function setCardStatus(card, text) {
  const s = card.meta.querySelector(".answer-status");
  if (!s) return;
  if (text === null) s.remove();
  else s.textContent = text;
}

function renderEvidence(card, { cards }) {
  if (!cards || cards.length === 0) return;
  const wrap = document.createElement("div");
  wrap.innerHTML = `<div class="evidence-divider">근거 판례 ${cards.length}건</div>`;
  const list = document.createElement("div");
  list.className = "evidence-list";

  const maxScore = Math.max(...cards.map((c) => c.score || 0), 0.0001);
  cards.forEach((c, i) => {
    const item = document.createElement("div");
    item.className = "case-card" + (i === 0 ? " top" : "");
    const statutes = (c.statutes && c.statutes !== "명시 없음")
      ? c.statutes.split(",").slice(0, 3).map(
          (s) => `<span class="statute-chip">${escapeHtml(s.trim())}</span>`
        ).join("")
      : "";
    const rel = Math.max(6, Math.round(((c.score || 0) / maxScore) * 100));
    item.innerHTML = `
      <div class="case-head">
        <span class="case-name">${escapeHtml(c.case_name || "판례")}</span>
        ${c.court ? `<span class="court-pill">${escapeHtml(c.court)}</span>` : ""}
        ${c.date ? `<span class="case-date">${escapeHtml(c.date)} 선고</span>` : ""}
      </div>
      <div class="case-sub">
        ${c.case_number ? `<span class="case-number">${escapeHtml(c.case_number)}</span>` : ""}
        ${statutes}
      </div>
      ${c.summary ? `<p class="case-summary">${escapeHtml(c.summary)}</p>` : ""}
      ${c.verdict ? `<div class="case-verdict">${escapeHtml(c.verdict)}</div>` : ""}
      ${c.full_text_snippet ? `
        <details>
          <summary>판결문 발췌 보기</summary>
          <div class="case-fulltext">${escapeHtml(c.full_text_snippet)}</div>
        </details>` : ""}
      <div class="case-score" aria-label="관련도">
        관련도 <span class="score-bar"><i style="width:${rel}%"></i></span>
      </div>`;
    list.appendChild(item);
  });

  wrap.appendChild(list);
  card.evidenceSlot.appendChild(wrap);
}

function renderFoot(card, { query_id, latency_ms }) {
  const foot = document.createElement("div");
  foot.className = "answer-foot";
  const sec = latency_ms ? (latency_ms / 1000).toFixed(1) : null;
  foot.innerHTML = `
    <span class="latency-note">${sec ? `${sec}초 소요` : ""}</span>
    <span class="feedback">
      <button class="fb-btn" data-v="1" type="button" aria-label="도움됨">👍 도움됐어요</button>
      <button class="fb-btn" data-v="0" type="button" aria-label="도움 안 됨">👎 아쉬워요</button>
    </span>`;
  card.footSlot.appendChild(foot);

  if (query_id == null) return;
  foot.querySelectorAll(".fb-btn").forEach((btn) => {
    btn.addEventListener("click", async () => {
      foot.querySelectorAll(".fb-btn").forEach((b) => { b.disabled = true; });
      btn.classList.add("selected");
      try {
        await fetch("/api/feedback", {
          method: "POST",
          headers: { "Content-Type": "application/json" },
          body: JSON.stringify({ query_id, helpful: btn.dataset.v === "1" }),
        });
        const thanks = document.createElement("span");
        thanks.className = "fb-thanks";
        thanks.textContent = "의견 감사합니다";
        foot.querySelector(".feedback").appendChild(thanks);
      } catch { /* 피드백 실패는 조용히 무시 */ }
    });
  });
}

function renderError(query, stage, message) {
  const STAGE_KO = {
    classify: "영역 분류", reformulate: "질의 재작성", retrieve: "판례 검색",
    summarize: "판례 요약", generate: "답변 생성",
  };
  const div = document.createElement("div");
  div.className = "error-card";
  div.innerHTML = `
    <strong>${escapeHtml(STAGE_KO[stage] || "처리")} 단계에서 문제가 발생했습니다.</strong>
    <div>${escapeHtml(message || "잠시 후 다시 시도해 주세요.")}</div>
    <button class="retry-btn" type="button">다시 시도</button>`;
  div.querySelector(".retry-btn").addEventListener("click", () => {
    div.remove();
    submitQuery(query);
  });
  el.chat.appendChild(div);
}

function scrollToBottom() {
  window.scrollTo({ top: document.body.scrollHeight, behavior: "smooth" });
}

/* ---------------- 쿼리 실행 ---------------- */

function setBusy(busy) {
  el.send.disabled = busy;
  el.input.disabled = busy;
}

function submitQuery(query) {
  if (state === "streaming" || state === "answering") return;
  state = "streaming";
  setBusy(true);
  el.welcome?.remove();
  el.chips.classList.add("hidden");

  addUserMessage(query);
  resetStepper();
  const card = createAnswerCard();
  let answerBuf = "";
  let sawError = false;
  let currentStage = "classify";
  scrollToBottom();

  es = new EventSource(`/api/stream?query=${encodeURIComponent(query)}`);

  const STATUS_TEXT = {
    classify: "법률 영역 분석 중",
    reformulate: "질의 재작성 중",
    retrieve: "판례 검색 중",
    summarize: "판례 요약 중",
    generate: "답변 작성 중",
  };

  es.addEventListener(EVENT.STAGE, (e) => {
    const data = JSON.parse(e.data);
    currentStage = data.stage;
    onStageEvent(data);
    if (data.status === "start") setCardStatus(card, STATUS_TEXT[data.stage] || "분석 중");
    if (data.stage === "generate" && data.status === "start") {
      state = "answering";
      card.body.classList.add("streaming");
    }
  });

  es.addEventListener(EVENT.DOMAIN, (e) => {
    renderDomainBadge(card, JSON.parse(e.data));
  });

  es.addEventListener(EVENT.CONTEXT, (e) => {
    renderEvidence(card, JSON.parse(e.data));
    scrollToBottom();
  });

  es.addEventListener(EVENT.ANSWER_CHUNK, (e) => {
    if (answerBuf === "") setCardStatus(card, null);
    answerBuf += JSON.parse(e.data).text;
    card.body.innerHTML = renderMarkdown(answerBuf);
  });

  es.addEventListener(EVENT.DONE, (e) => {
    finishStream();
    card.body.classList.remove("streaming");
    renderFoot(card, JSON.parse(e.data));
    state = "done";
    scrollToBottom();
  });

  es.addEventListener(EVENT.ERROR, (e) => {
    sawError = true;
    finishStream();
    card.body.classList.remove("streaming");
    setCardStatus(card, null);
    const data = JSON.parse(e.data);
    markStepperError(data.stage);
    renderError(query, data.stage, data.message);
    state = "error";
  });

  es.onerror = () => {
    if (state === "done" || sawError) return;
    sawError = true;
    finishStream();
    card.body.classList.remove("streaming");
    setCardStatus(card, null);
    markStepperError(currentStage);
    renderError(query, currentStage, "서버와의 연결이 끊어졌습니다.");
    state = "error";
  };

  function finishStream() {
    es?.close();
    es = null;
    setBusy(false);
    el.input.focus();
  }
}

/* ---------------- 입력 ---------------- */

el.form.addEventListener("submit", (e) => {
  e.preventDefault();
  const q = el.input.value.trim();
  if (q.length < 5) { el.input.reportValidity(); return; }
  el.input.value = "";
  autosize();
  submitQuery(q);
});

el.input.addEventListener("keydown", (e) => {
  if (e.key === "Enter" && !e.shiftKey && !e.isComposing) {
    e.preventDefault();
    el.form.requestSubmit();
  }
});

function autosize() {
  el.input.style.height = "auto";
  el.input.style.height = `${Math.min(el.input.scrollHeight, 132)}px`;
}
el.input.addEventListener("input", autosize);

el.chips.addEventListener("click", (e) => {
  const chip = e.target.closest(".chip");
  if (!chip) return;
  el.input.value = chip.textContent.trim();
  autosize();
  el.form.requestSubmit();
});

el.input.focus();
