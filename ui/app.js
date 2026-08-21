"use strict";

const STORAGE_KEY = "conquer-health-ui-state-v1";
const THEME_KEY = "conquer-health-theme";

const el = {
  chatScroll: document.querySelector("#chat-scroll"),
  emptyState: document.querySelector("#empty-state"),
  messageList: document.querySelector("#message-list"),
  composer: document.querySelector("#composer"),
  input: document.querySelector("#message-input"),
  send: document.querySelector("#send-button"),
  charCount: document.querySelector("#char-count"),
  newChat: document.querySelector("#new-chat-button"),
  regenerate: document.querySelector("#regenerate-button"),
  export: document.querySelector("#export-button"),
  retry: document.querySelector("#retry-button"),
  errorBanner: document.querySelector("#error-banner"),
  errorMessage: document.querySelector("#error-message"),
  turnCount: document.querySelector("#turn-count"),
  turnMeter: document.querySelector("#turn-meter"),
  messageCount: document.querySelector("#message-count"),
  connection: document.querySelector("#connection-status"),
  configGrid: document.querySelector("#config-grid"),
  configNote: document.querySelector("#config-note"),
  refreshConfig: document.querySelector("#refresh-config-button"),
  experimentName: document.querySelector("#experiment-name"),
  theme: document.querySelector("#theme-button"),
  toast: document.querySelector("#toast"),
  metricRoute: document.querySelector("#metric-route"),
  metricLatency: document.querySelector("#metric-latency"),
  metricEvidence: document.querySelector("#metric-evidence"),
  metricStatus: document.querySelector("#metric-status"),
  standaloneQuery: document.querySelector("#standalone-query"),
  retrievalQuery: document.querySelector("#retrieval-query"),
  traceCount: document.querySelector("#trace-count"),
  evidenceCount: document.querySelector("#evidence-count"),
  traceList: document.querySelector("#trace-list"),
  evidenceList: document.querySelector("#evidence-list"),
  tracePlaceholder: document.querySelector("#trace-placeholder"),
  evidencePlaceholder: document.querySelector("#evidence-placeholder"),
  overviewPlaceholder: document.querySelector("#overview-placeholder"),
  historyJson: document.querySelector("#history-json"),
  copyHistory: document.querySelector("#copy-history-button"),
  routeDiagram: document.querySelector(".route-diagram"),
  routeGeneration: document.querySelector("#route-generation"),
  routeRetrieval: document.querySelector("#route-retrieval"),
  routeAnswer: document.querySelector("#route-answer"),
};

const state = loadState();
let activeMessageIndex = latestAssistantIndex();
let activeConfig = null;
let toastTimer = null;

function loadState() {
  try {
    const saved = JSON.parse(localStorage.getItem(STORAGE_KEY));
    if (saved && Array.isArray(saved.messages) && Array.isArray(saved.runs)) {
      return { messages: saved.messages, runs: saved.runs, pending: false, error: "" };
    }
  } catch (_) {
    // A malformed local draft should never prevent the review UI from loading.
  }
  return { messages: [], runs: [], pending: false, error: "" };
}

function saveState() {
  localStorage.setItem(STORAGE_KEY, JSON.stringify({
    messages: state.messages,
    runs: state.runs,
  }));
}

function latestAssistantIndex() {
  for (let index = state.messages.length - 1; index >= 0; index -= 1) {
    if (state.messages[index].role === "assistant") return index;
  }
  return -1;
}

function runForMessage(messageIndex) {
  return state.runs.find((run) => run.messageIndex === messageIndex) || null;
}

function setConnection(status, label) {
  el.connection.classList.remove("online", "offline");
  if (status) el.connection.classList.add(status);
  el.connection.querySelector("span:last-child").textContent = label;
}

function showToast(message) {
  window.clearTimeout(toastTimer);
  el.toast.textContent = message;
  el.toast.classList.add("show");
  toastTimer = window.setTimeout(() => el.toast.classList.remove("show"), 1800);
}

async function copyText(text) {
  try {
    await navigator.clipboard.writeText(text);
  } catch (_) {
    const area = document.createElement("textarea");
    area.value = text;
    area.style.position = "fixed";
    area.style.opacity = "0";
    document.body.append(area);
    area.select();
    document.execCommand("copy");
    area.remove();
  }
  showToast("클립보드에 복사했습니다.");
}

function appendInline(parent, value) {
  const pattern = /(\*\*[^*]+\*\*|`[^`]+`|\[\d+(?:\s*,\s*\d+)*\])/g;
  let cursor = 0;
  for (const match of value.matchAll(pattern)) {
    if (match.index > cursor) parent.append(document.createTextNode(value.slice(cursor, match.index)));
    const token = match[0];
    if (token.startsWith("**")) {
      const strong = document.createElement("strong");
      strong.textContent = token.slice(2, -2);
      parent.append(strong);
    } else if (token.startsWith("`")) {
      const code = document.createElement("code");
      code.textContent = token.slice(1, -1);
      parent.append(code);
    } else {
      const citation = document.createElement("button");
      citation.type = "button";
      citation.className = "citation-link";
      citation.textContent = token;
      citation.dataset.marker = token.match(/\d+/)?.[0] || "";
      citation.title = "선택된 근거 보기";
      parent.append(citation);
    }
    cursor = match.index + token.length;
  }
  if (cursor < value.length) parent.append(document.createTextNode(value.slice(cursor)));
}

function renderRichText(value) {
  const root = document.createElement("div");
  root.className = "rich-text";
  const lines = String(value || "").replace(/\r\n/g, "\n").split("\n");
  let paragraph = [];

  const flushParagraph = () => {
    if (!paragraph.length) return;
    const p = document.createElement("p");
    paragraph.forEach((line, index) => {
      if (index) p.append(document.createElement("br"));
      appendInline(p, line);
    });
    root.append(p);
    paragraph = [];
  };

  for (let index = 0; index < lines.length; index += 1) {
    const line = lines[index];
    if (line.trim().startsWith("```")) {
      flushParagraph();
      const codeLines = [];
      index += 1;
      while (index < lines.length && !lines[index].trim().startsWith("```")) {
        codeLines.push(lines[index]);
        index += 1;
      }
      const pre = document.createElement("pre");
      pre.textContent = codeLines.join("\n");
      root.append(pre);
      continue;
    }
    if (!line.trim()) {
      flushParagraph();
      continue;
    }
    const heading = line.match(/^(#{1,3})\s+(.+)$/);
    if (heading) {
      flushParagraph();
      const h = document.createElement(`h${Math.min(heading[1].length + 1, 4)}`);
      appendInline(h, heading[2]);
      root.append(h);
      continue;
    }
    const unordered = line.match(/^\s*[-*]\s+(.+)$/);
    const ordered = line.match(/^\s*\d+[.)]\s+(.+)$/);
    if (unordered || ordered) {
      flushParagraph();
      const list = document.createElement(unordered ? "ul" : "ol");
      let current = line;
      while (index < lines.length) {
        const item = unordered
          ? current.match(/^\s*[-*]\s+(.+)$/)
          : current.match(/^\s*\d+[.)]\s+(.+)$/);
        if (!item) break;
        const li = document.createElement("li");
        appendInline(li, item[1]);
        list.append(li);
        index += 1;
        current = lines[index] || "";
      }
      index -= 1;
      root.append(list);
      continue;
    }
    if (line.trim().startsWith(">")) {
      flushParagraph();
      const quote = document.createElement("blockquote");
      appendInline(quote, line.trim().replace(/^>\s?/, ""));
      root.append(quote);
      continue;
    }
    paragraph.push(line);
  }
  flushParagraph();
  return root;
}

function createMessageRow(message, index) {
  const row = document.createElement("article");
  row.className = `message-row ${message.role}`;
  row.dataset.messageIndex = String(index);
  if (index === activeMessageIndex) row.classList.add("selected");

  const avatar = document.createElement("div");
  avatar.className = "message-avatar";
  avatar.textContent = message.role === "assistant" ? "L2" : "YOU";

  const stack = document.createElement("div");
  stack.className = "message-stack";
  const role = document.createElement("p");
  role.className = "message-role";
  role.textContent = message.role === "assistant" ? "ASSISTANT · LUNIT L2" : "USER";
  const bubble = document.createElement("div");
  bubble.className = "message-bubble";
  bubble.append(renderRichText(message.content));
  stack.append(role, bubble);

  const run = runForMessage(index);
  if (message.role === "assistant" && run) {
    const meta = document.createElement("div");
    meta.className = "message-meta";
    const route = document.createElement("span");
    route.className = "meta-route";
    route.textContent = run.meta?.route || "direct";
    const latency = document.createElement("span");
    latency.textContent = formatLatency(run.latency_ms);
    const evidence = document.createElement("span");
    evidence.textContent = `${run.meta?.n_evidence || 0} evidence`;
    const copy = document.createElement("button");
    copy.type = "button";
    copy.textContent = "답변 복사";
    copy.addEventListener("click", (event) => {
      event.stopPropagation();
      copyText(message.content);
    });
    meta.append(route, latency, evidence, copy);
    stack.append(meta);
    row.tabIndex = 0;
    row.title = "이 턴의 실행 정보 보기";
    row.addEventListener("click", () => selectMessage(index));
    row.addEventListener("keydown", (event) => {
      if (event.key === "Enter" || event.key === " ") selectMessage(index);
    });
  }
  row.append(avatar, stack);
  return row;
}

function createTypingRow() {
  const row = document.createElement("article");
  row.className = "message-row assistant";
  row.innerHTML = `
    <div class="message-avatar">L2</div>
    <div class="message-stack">
      <p class="message-role">ASSISTANT · THINKING</p>
      <div class="message-bubble typing-bubble"><i></i><i></i><i></i></div>
    </div>`;
  return row;
}

function formatLatency(ms) {
  if (typeof ms !== "number") return "—";
  if (ms >= 1000) return `${(ms / 1000).toFixed(ms >= 10000 ? 1 : 2)} s`;
  return `${ms} ms`;
}

function renderConversation({ scroll = false } = {}) {
  el.emptyState.hidden = state.messages.length > 0 || state.pending;
  el.messageList.replaceChildren(...state.messages.map(createMessageRow));
  if (state.pending) el.messageList.append(createTypingRow());

  const assistantTurns = state.messages.filter((message) => message.role === "assistant").length;
  el.turnCount.textContent = `${assistantTurns} / 3 TURNS`;
  [...el.turnMeter.children].forEach((node, index) => node.classList.toggle("active", index < assistantTurns));
  el.messageCount.textContent = `${state.messages.length} message${state.messages.length === 1 ? "" : "s"}`;
  el.historyJson.textContent = JSON.stringify(state.messages, null, 2);
  el.regenerate.disabled = state.pending || !state.messages.some((message) => message.role === "user");
  el.export.disabled = state.messages.length === 0;
  el.input.disabled = state.pending;
  updateSendState();
  showError(state.error);
  if (scroll) requestAnimationFrame(() => { el.chatScroll.scrollTop = el.chatScroll.scrollHeight; });
}

function showError(message) {
  el.errorBanner.hidden = !message;
  el.errorMessage.textContent = message || "";
}

function updateSendState() {
  const length = el.input.value.length;
  el.charCount.textContent = `${length} / 6000`;
  el.send.disabled = state.pending || !el.input.value.trim();
}

function resizeInput() {
  el.input.style.height = "auto";
  el.input.style.height = `${Math.min(el.input.scrollHeight, 160)}px`;
}

function selectMessage(index) {
  if (!runForMessage(index)) return;
  activeMessageIndex = index;
  renderConversation();
  renderInspector(runForMessage(index));
}

function statusLabel(value) {
  if (!value) return "DIRECT";
  return String(value).replaceAll("_", " ").toUpperCase();
}

function setQueryBox(node, value, fallback) {
  node.textContent = value || fallback;
  node.classList.toggle("empty", !value);
}

function renderInspector(run) {
  const meta = run?.meta || {};
  const trace = Array.isArray(meta.trace) ? meta.trace : [];
  const evidence = Array.isArray(meta.evidence) ? meta.evidence : [];
  const route = meta.route || "direct";

  el.metricRoute.textContent = run ? route.toUpperCase() : "—";
  el.metricLatency.textContent = run ? formatLatency(run.latency_ms) : "—";
  el.metricEvidence.textContent = run ? String(meta.n_evidence || 0) : "—";
  el.metricStatus.textContent = run ? statusLabel(meta.status) : "IDLE";
  setQueryBox(el.standaloneQuery, run && meta.standalone_query, "아직 실행된 질문이 없습니다.");
  setQueryBox(el.retrievalQuery, run && meta.retrieval_query, "Retrieval이 실행되면 여기에 표시됩니다.");
  el.overviewPlaceholder.hidden = Boolean(run);

  el.traceCount.textContent = String(trace.length);
  el.evidenceCount.textContent = String(evidence.length);
  renderTrace(trace);
  renderEvidence(evidence);

  el.routeGeneration.classList.toggle("active", Boolean(run));
  el.routeRetrieval.classList.toggle("active", Boolean(run) && route === "retrieve");
  el.routeAnswer.classList.toggle("active", Boolean(run));
  el.routeDiagram.classList.toggle("retrieved", Boolean(run) && route === "retrieve");
  el.routeDiagram.classList.toggle("complete", Boolean(run));
}

function renderTrace(trace) {
  el.traceList.replaceChildren();
  el.tracePlaceholder.hidden = trace.length > 0;
  trace.forEach((entry, index) => {
    const text = String(entry);
    const opening = text.indexOf("(");
    const name = opening > 0 ? text.slice(0, opening) : text;
    const args = opening > 0 ? text.slice(opening + 1, text.endsWith(")") ? -1 : undefined) : "";
    const item = document.createElement("article");
    item.className = "trace-item";
    const number = document.createElement("span");
    number.className = "trace-number";
    number.textContent = String(index + 1).padStart(2, "0");
    const content = document.createElement("div");
    content.className = "trace-content";
    const title = document.createElement("strong");
    title.textContent = name;
    const detail = document.createElement("pre");
    detail.textContent = args || "no arguments";
    content.append(title, detail);
    item.append(number, content);
    el.traceList.append(item);
  });
}

function safeSourceUrl(value) {
  try {
    const url = new URL(value);
    return ["http:", "https:"].includes(url.protocol) ? url.href : "";
  } catch (_) {
    return "";
  }
}

function renderEvidence(evidence) {
  el.evidenceList.replaceChildren();
  el.evidencePlaceholder.hidden = evidence.length > 0;
  evidence.forEach((source, index) => {
    const marker = source.marker || index + 1;
    const card = document.createElement("article");
    card.className = "evidence-card";
    card.id = `evidence-${marker}`;
    const head = document.createElement("div");
    head.className = "evidence-head";
    const markerNode = document.createElement("span");
    markerNode.className = "evidence-marker";
    markerNode.textContent = `[${marker}]`;
    const title = document.createElement("div");
    title.className = "evidence-title";
    const strong = document.createElement("strong");
    strong.textContent = source.title || "Untitled source";
    const type = document.createElement("span");
    const score = typeof source.relevance_score === "number" ? ` · score ${source.relevance_score.toFixed(2)}` : "";
    type.textContent = `${source.source_type || "unknown"}${score} · ${source.cite_uid || "no cite_uid"}`;
    title.append(strong, type);
    head.append(markerNode, title);
    const href = safeSourceUrl(source.url);
    if (href) {
      const link = document.createElement("a");
      link.className = "source-link";
      link.href = href;
      link.target = "_blank";
      link.rel = "noreferrer noopener";
      link.title = "원문 열기";
      link.innerHTML = `<svg viewBox="0 0 24 24" aria-hidden="true"><path d="M14 4h6v6m0-6-9 9M19 13v6a1 1 0 0 1-1 1H5a1 1 0 0 1-1-1V6a1 1 0 0 1 1-1h6"/></svg>`;
      head.append(link);
    }
    const content = document.createElement("pre");
    content.className = "evidence-content";
    content.textContent = source.content || "본문이 제공되지 않았습니다.";
    card.append(head, content);
    el.evidenceList.append(card);
  });
}

function switchTab(name) {
  document.querySelectorAll(".tab-button").forEach((button) => {
    const active = button.dataset.tab === name;
    button.classList.toggle("active", active);
    button.setAttribute("aria-selected", String(active));
  });
  document.querySelectorAll(".tab-panel").forEach((panel) => {
    const active = panel.dataset.panel === name;
    panel.classList.toggle("active", active);
    panel.hidden = !active;
  });
}

async function loadConfig() {
  setConnection("", "연결 확인 중");
  try {
    const response = await fetch("/debug/config", { headers: { Accept: "application/json" } });
    if (!response.ok) throw new Error(`HTTP ${response.status}`);
    activeConfig = await response.json();
    renderConfig(activeConfig);
    setConnection("online", "L2 HARNESS ONLINE");
  } catch (error) {
    setConnection("offline", "HARNESS OFFLINE");
    el.configGrid.replaceChildren();
    el.configNote.textContent = `설정을 불러오지 못했습니다: ${error.message}`;
  }
}

function renderConfig(payload) {
  const config = payload.config || {};
  const rows = [
    ["model", config.model],
    ["retrieval", config.retrieval],
    ["prompt", config.prompt],
    ["rewrite", config.rewrite],
    ["summary", config.case_summary],
    ["MCP limit", config.max_mcp_calls],
    ["thinking", config.generation_thinking],
    ["candidates", config.num_candidates],
  ];
  el.configGrid.replaceChildren(...rows.map(([key, value]) => {
    const chip = document.createElement("div");
    chip.className = "config-chip";
    if (typeof value === "boolean") chip.classList.add(value ? "on" : "off");
    const label = document.createElement("span");
    label.textContent = key;
    const result = document.createElement("strong");
    result.textContent = typeof value === "boolean" ? (value ? "ON" : "OFF") : String(value ?? "—");
    result.title = String(value ?? "");
    chip.append(label, result);
    return chip;
  }));

  const enabled = [];
  if (config.retrieval) enabled.push("RAG");
  if (config.rewrite) enabled.push("REWRITE");
  if (config.case_summary) enabled.push("SUMMARY");
  if (config.critic_pass) enabled.push("CRITIC");
  if (config.prompt && config.prompt !== "none") enabled.push(String(config.prompt).toUpperCase());
  if (!enabled.length) {
    el.experimentName.textContent = "B0 · RAW L2";
    el.configNote.textContent = "개선 기능이 모두 꺼진 원시 기준점입니다. 설정값은 읽기 전용입니다.";
  } else {
    el.experimentName.textContent = `L2 · ${enabled.join(" + ")}`;
    el.configNote.textContent = "현재 서버 프로세스에 적용된 설정입니다. config.py 변경 후 서버를 재시작하세요.";
  }
}

async function requestAnswer() {
  if (state.pending || !state.messages.length || state.messages.at(-1).role !== "user") return;
  state.pending = true;
  state.error = "";
  renderConversation({ scroll: true });

  try {
    const response = await fetch("/debug/chat", {
      method: "POST",
      headers: { "Content-Type": "application/json", Accept: "application/json" },
      body: JSON.stringify({ messages: state.messages.map(({ role, content }) => ({ role, content })) }),
    });
    let payload;
    try {
      payload = await response.json();
    } catch (_) {
      payload = {};
    }
    if (!response.ok) throw new Error(payload.error?.message || `HTTP ${response.status}`);
    const content = payload.message?.content?.trim();
    if (!content) throw new Error("모델이 빈 답변을 반환했습니다.");

    state.messages.push({ role: "assistant", content });
    const messageIndex = state.messages.length - 1;
    state.runs.push({
      messageIndex,
      latency_ms: payload.latency_ms,
      meta: payload.meta || {},
    });
    activeMessageIndex = messageIndex;
    state.pending = false;
    saveState();
    renderConversation({ scroll: true });
    renderInspector(runForMessage(messageIndex));
  } catch (error) {
    state.pending = false;
    state.error = error.message || String(error);
    renderConversation({ scroll: true });
  }
}

function submitUserMessage(value) {
  const content = value.trim();
  if (!content || state.pending) return;
  state.messages.push({ role: "user", content });
  state.error = "";
  el.input.value = "";
  resizeInput();
  saveState();
  renderConversation({ scroll: true });
  requestAnswer();
}

function newConversation() {
  if (state.pending) return;
  if (state.messages.length && !window.confirm("현재 대화와 실행 기록을 모두 지울까요?")) return;
  state.messages = [];
  state.runs = [];
  state.error = "";
  activeMessageIndex = -1;
  saveState();
  renderConversation();
  renderInspector(null);
  switchTab("overview");
  el.input.focus();
}

function regenerate() {
  if (state.pending || !state.messages.length) return;
  if (state.messages.at(-1).role === "assistant") {
    const removedIndex = state.messages.length - 1;
    state.messages.pop();
    state.runs = state.runs.filter((run) => run.messageIndex !== removedIndex);
  }
  if (!state.messages.length || state.messages.at(-1).role !== "user") return;
  state.error = "";
  activeMessageIndex = latestAssistantIndex();
  saveState();
  renderConversation({ scroll: true });
  renderInspector(activeMessageIndex >= 0 ? runForMessage(activeMessageIndex) : null);
  requestAnswer();
}

function exportConversation() {
  if (!state.messages.length) return;
  const payload = {
    exported_at: new Date().toISOString(),
    config: activeConfig?.config || null,
    messages: state.messages,
    runs: state.runs,
  };
  const blob = new Blob([JSON.stringify(payload, null, 2)], { type: "application/json" });
  const link = document.createElement("a");
  link.href = URL.createObjectURL(blob);
  link.download = `conquer-health-${new Date().toISOString().replaceAll(":", "-")}.json`;
  link.click();
  URL.revokeObjectURL(link.href);
  showToast("대화 기록을 내보냈습니다.");
}

function initializeTheme() {
  const saved = localStorage.getItem(THEME_KEY);
  const preferred = window.matchMedia?.("(prefers-color-scheme: dark)").matches ? "dark" : "light";
  document.documentElement.dataset.theme = saved || preferred;
}

function toggleTheme() {
  const next = document.documentElement.dataset.theme === "dark" ? "light" : "dark";
  document.documentElement.dataset.theme = next;
  localStorage.setItem(THEME_KEY, next);
}

el.composer.addEventListener("submit", (event) => {
  event.preventDefault();
  submitUserMessage(el.input.value);
});

el.input.addEventListener("input", () => {
  resizeInput();
  updateSendState();
});

el.input.addEventListener("keydown", (event) => {
  if (event.key === "Enter" && !event.shiftKey && !event.isComposing) {
    event.preventDefault();
    submitUserMessage(el.input.value);
  }
});

el.newChat.addEventListener("click", newConversation);
el.regenerate.addEventListener("click", regenerate);
el.export.addEventListener("click", exportConversation);
el.retry.addEventListener("click", requestAnswer);
el.refreshConfig.addEventListener("click", loadConfig);
el.theme.addEventListener("click", toggleTheme);
el.copyHistory.addEventListener("click", () => copyText(JSON.stringify(state.messages, null, 2)));

document.querySelectorAll(".preset-card").forEach((button) => {
  button.addEventListener("click", () => {
    el.input.value = button.dataset.prompt || "";
    resizeInput();
    updateSendState();
    el.input.focus();
  });
});

document.querySelectorAll(".tab-button").forEach((button) => {
  button.addEventListener("click", () => switchTab(button.dataset.tab));
});

document.querySelectorAll("[data-copy-target]").forEach((button) => {
  button.addEventListener("click", () => {
    const target = document.querySelector(`#${button.dataset.copyTarget}`);
    if (target && !target.classList.contains("empty")) copyText(target.textContent);
  });
});

el.messageList.addEventListener("click", (event) => {
  const citation = event.target.closest(".citation-link");
  if (!citation) return;
  event.stopPropagation();
  switchTab("evidence");
  const card = document.querySelector(`#evidence-${citation.dataset.marker}`);
  if (card) {
    card.classList.remove("highlight");
    void card.offsetWidth;
    card.classList.add("highlight");
    card.scrollIntoView({ behavior: "smooth", block: "center" });
  } else {
    showToast("이 답변에는 연결된 근거가 없습니다.");
  }
});

initializeTheme();
renderConversation();
renderInspector(activeMessageIndex >= 0 ? runForMessage(activeMessageIndex) : null);
loadConfig();
