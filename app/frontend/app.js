/* Inkference frontend logic. Talks to the FastAPI backend at <origin>/api
   (override with window.INKFERENCE_API for a separately-hosted backend). */
const API = (window.INKFERENCE_API || "") + "/api";

const state = {
  doc: null, page: 1, totalPages: 0, pageData: null, readerView: null,
  docs: [],                 // every document, for the library grid
  user: null,               // signed-in account, or null
  authRequired: false,      // does this deployment gate its API?
  gated: false,             // authRequired && signed out -> only #signin renders
  providers: [],            // federated sign-ins: [{key, label, enabled}]
  // Upload destination. true = the next upload starts a NEW book; false = it is
  // appended to state.doc. Set true by the "Add a book" tile, false by the Upload
  // tab. It exists because "whichever book is currently open" is not an intention
  // anyone expressed — that default silently filed new scans into the open book.
  newBookMode: false,
};

/* ---------- helpers ---------- */
const $ = (sel) => document.querySelector(sel);
const el = (tag, cls) => { const e = document.createElement(tag); if (cls) e.className = cls; return e; };

/* Page numbers are reused across uploads/reseeds, so /pages/{n}/image can map to
   different bytes over time. Append a content-derived version so a reused number gets
   a fresh URL the browser hasn't cached (seeded pages keep a stable URL → CDN cache). */
function imageUrl(docId, page) {
  const v = [page.status, page.avg_confidence, page.low_conf_words, page.width, page.height]
    .map((x) => (x == null ? "" : x)).join("_");
  return `${API}/documents/${docId}/pages/${page.page_number}/image?v=${encodeURIComponent(v)}`;
}

async function api(path, opts) {
  const r = await fetch(API + path, opts);
  // A session can expire mid-visit. Bounce to sign-in once rather than letting every
  // in-flight call fail silently and leave a half-empty page behind.
  if (r.status === 401 && !path.startsWith("/auth/")) {
    onSessionLost();
    throw new Error("session expired");
  }
  if (!r.ok) throw new Error((await r.text()) || r.status);
  return r.json();
}

function onSessionLost() {
  if (state.gated) return;          // already showing sign-in
  state.user = null;
  state.gated = true;
  renderAccount();
  setAuthMode("signin");
  authError("Your session expired. Please sign in again.");
  showView("signin");
  loadPublicStats();
}

/* confidence 1.0 -> dark ink, 0.0 -> faded; matches the legend gradient */
function confColor(c) {
  const dark = [43, 33, 26], light = [205, 191, 169];
  const t = Math.max(0, Math.min(1, c));
  const mix = dark.map((d, i) => Math.round(light[i] + (d - light[i]) * t));
  return `rgb(${mix[0]},${mix[1]},${mix[2]})`;
}

/* ---------- tabs / routing ---------- */
const VIEWS = ["library", "reader", "ask", "upload", "signin"];

document.querySelectorAll(".tab").forEach((t) => {
  t.addEventListener("click", () => {
    // Reaching Upload from the nav means "add to what I'm reading". Reaching it
    // from the Add-a-book tile means "start a new one" — see #add-book below.
    if (t.dataset.view === "upload") state.newBookMode = false;
    showView(t.dataset.view);
  });
});

$("#add-book").addEventListener("click", () => {
  state.newBookMode = true;
  showView("upload");
});
$("#brand-home").addEventListener("click", () => showView("library"));
$("#brand-home").addEventListener("keydown", (e) => {
  if (e.key === "Enter" || e.key === " ") { e.preventDefault(); showView("library"); }
});

function showView(name) {
  // While signed out of a gated backend every route collapses to sign-in, so a
  // hand-typed #reader can't render a shell whose data calls would all 401.
  if (state.gated) name = "signin";
  else if (name === "signin") name = "library";
  if (!VIEWS.includes(name)) name = "library";

  document.querySelectorAll(".tab").forEach((t) => t.classList.toggle("active", t.dataset.view === name));
  document.querySelectorAll(".view").forEach((v) => v.classList.add("hidden"));
  $("#view-" + name).classList.remove("hidden");
  // Sign-in is a full-bleed page: the app chrome belongs to a session that doesn't exist yet.
  $("#app-header").classList.toggle("hidden", name === "signin");
  if (name === "library") renderLibrary();
  if (name === "upload") renderUploadTarget();
  if (location.hash.slice(1) !== name) history.replaceState(null, "", "#" + name);
}
window.addEventListener("hashchange", () => showView(location.hash.slice(1)));

/* ---------- accounts ---------- */
/* The backend is the authority: /api/auth/me reports both the current user and
   whether this deployment gates its API at all (INKFERENCE_AUTH_REQUIRED). */
async function loadSession() {
  try {
    const info = await api("/auth/me");
    state.user = info.user;
    state.authRequired = info.auth_required;
    state.providers = info.providers || [];
  } catch (e) {
    state.user = null;
    state.authRequired = false;   // backend unreachable — don't trap behind a gate we can't verify
    state.providers = [];
  }
  state.gated = state.authRequired && !state.user;
  renderAccount();
  renderProviders();
}

/* Federated sign-in buttons.

   The markup ships them disabled and this turns on only the providers the backend
   reports as configured — a button that bounces the user to Google when this
   deployment holds no Google credentials is a worse experience than a greyed-out
   one with an explanation. */
function renderProviders() {
  const enabled = state.providers.filter((p) => p.enabled);
  state.providers.forEach((p) => {
    const btn = document.querySelector(`[data-provider="${p.key}"]`);
    if (!btn) return;                     // provider the frontend has no button for
    btn.disabled = !p.enabled;
    btn.title = p.enabled ? `Continue with ${p.label}` : `${p.label} sign-in is not configured here`;
  });
  const note = $("#provider-note");
  if (note) {
    note.textContent = enabled.length
      ? "You can also sign in with an email and password."
      : "Single sign-on is not configured on this deployment — use an email and password.";
  }
}

/* A provider sign-in is a full-page navigation, not fetch(): the browser has to
   follow the redirect to Google/Microsoft, and the state cookie the backend sets on
   the way out must be stored as a first-party cookie. XHR would do neither. */
document.querySelectorAll("[data-provider]").forEach((btn) => {
  btn.addEventListener("click", () => {
    if (btn.disabled) return;
    btn.classList.add("busy");
    window.location.href = `${API}/auth/oidc/${btn.dataset.provider}/start`;
  });
});

/* The OAuth callback reports failures by bouncing back to /?auth_error=…#signin,
   since a redirected browser has nowhere else to render one. Show it, then strip it
   from the URL so a refresh doesn't resurrect a stale message. */
function showOAuthErrorFromUrl() {
  const params = new URLSearchParams(location.search);
  const message = params.get("auth_error");
  if (!message) return;
  authError(message);
  params.delete("auth_error");
  const query = params.toString();
  history.replaceState(null, "", location.pathname + (query ? "?" + query : "") + location.hash);
}

function renderAccount() {
  $("#account").classList.toggle("hidden", !state.user);
}

/* Sign-in / create-account form */
let authMode = "signin";

function setAuthMode(mode) {
  authMode = mode;
  const signup = mode === "signup";
  $("#tab-signin").classList.toggle("active", !signup);
  $("#tab-signup").classList.toggle("active", signup);
  $("#field-name").classList.toggle("hidden", !signup);
  $("#auth-title").textContent = signup ? "Create your archive" : "Welcome back";
  $("#auth-sub").textContent = signup
    ? "Start reading and questioning your own scanned pages."
    : "Sign in to reach your library and transcriptions.";
  $("#google-label").textContent = signup ? "Sign up with Google" : "Continue with Google";
  $("#auth-password").placeholder = signup ? "At least 10 characters" : "••••••••••";
  $("#auth-password").autocomplete = signup ? "new-password" : "current-password";
  $("#auth-submit").textContent = signup ? "Create account" : "Sign in";
  $("#auth-note").textContent = signup
    ? "Your account and password are stored on this deployment only."
    : "New to Inkference? Create an account above.";
  authError("");
}

function authError(msg) {
  const box = $("#auth-error");
  box.textContent = msg || "";
  box.classList.toggle("hidden", !msg);
}

document.querySelectorAll(".auth-tab").forEach((t) => {
  t.addEventListener("click", () => setAuthMode(t.dataset.mode));
});

$("#auth-form").addEventListener("submit", async (e) => {
  e.preventDefault();
  const email = $("#auth-email").value.trim();
  const password = $("#auth-password").value;
  if (!email || !password) return authError("Enter your email and password.");

  const btn = $("#auth-submit");
  const label = btn.textContent;
  btn.disabled = true;
  btn.textContent = authMode === "signup" ? "Creating account…" : "Signing in…";
  authError("");
  try {
    const body = authMode === "signup"
      ? { email, password, name: $("#auth-name").value.trim() || null }
      : { email, password };
    await api("/auth/" + (authMode === "signup" ? "signup" : "login"), {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(body),
    });
    $("#auth-password").value = "";
    await loadSession();
    state.gated = false;
    await loadCorpus();
    showView("library");
  } catch (err) {
    authError(errorText(err) || "Something went wrong. Please try again.");
  } finally {
    btn.disabled = false;
    btn.textContent = label;
  }
});

/* FastAPI reports failures as {"detail": ...}; surface that rather than raw JSON. */
function errorText(err) {
  const raw = (err && err.message) || "";
  try {
    const parsed = JSON.parse(raw);
    const detail = parsed.detail;
    if (typeof detail === "string") return detail;
    if (Array.isArray(detail) && detail[0] && detail[0].msg) return detail[0].msg;
  } catch (e) { /* not JSON — fall through */ }
  return raw;
}

$("#sign-out").addEventListener("click", async () => {
  try { await api("/auth/logout", { method: "POST" }); } catch (e) { /* clear locally anyway */ }
  state.user = null;
  state.doc = null;
  state.gated = state.authRequired;
  renderAccount();
  setAuthMode("signin");
  showView(state.gated ? "signin" : "library");
});

/* Corpus totals for the signed-out brand panel. Public endpoint; if it fails the
   stats are simply left out rather than filled with invented numbers. */
async function loadPublicStats() {
  let stats;
  try { stats = await api("/stats"); } catch (e) { return; }
  const cells = [];
  if (stats.pages) {
    cells.push([stats.pages.toLocaleString(), "pages transcribed"]);
  }
  if (stats.avg_confidence != null) {
    cells.push([Math.round(stats.avg_confidence * 100) + "%", "average confidence"]);
  }
  $("#auth-stats").innerHTML = "";
  for (const [value, label] of cells) {
    const cell = el("div");
    const v = el("div", "auth-stat-value"); v.textContent = value;
    const l = el("div", "auth-stat-label"); l.textContent = label;
    cell.append(v, l);
    $("#auth-stats").append(cell);
  }
}

/* ---------- Library ---------- */
function renderLibrary() {
  const grid = $("#book-grid");
  const docs = state.docs || [];
  grid.querySelectorAll(".book-card").forEach((c) => c.remove());

  $("#library-count").textContent = docs.length
    ? `${docs.length} ${docs.length === 1 ? "book" : "books"} · ` +
      `${docs.reduce((n, d) => n + (d.page_count || 0), 0).toLocaleString()} pages`
    : "";

  for (const doc of docs) {
    const card = el("a", "book-card");
    card.href = "#reader";
    card.addEventListener("click", () => openDocument(doc));

    const cover = el("div", "book-cover");
    const pending = (doc.page_count || 0) - (doc.pages_done || 0);
    if (pending > 0) {
      const badge = el("div", "book-badge");
      badge.textContent = "Processing";
      cover.append(badge);
    }
    // The first page doubles as the cover — no separate cover art exists.
    if (doc.page_count) {
      const img = el("img");
      img.alt = "";
      img.loading = "lazy";
      img.src = `${API}/documents/${doc.id}/pages/1/image`;
      img.onerror = () => img.remove();
      cover.append(img);
    }

    const meta = el("div", "book-meta");
    const title = el("div", "book-title");
    title.textContent = doc.title;
    const sub = el("div", "book-sub");
    sub.textContent = [
      doc.subtitle,
      `${(doc.page_count || 0).toLocaleString()} pages`,
      doc.avg_confidence != null ? `${Math.round(doc.avg_confidence * 100)}% confidence` : null,
      pending > 0 ? `${pending} left` : null,
    ].filter(Boolean).join(" · ");
    meta.append(title, sub);

    card.append(cover, meta);
    grid.append(card);
  }

  renderContinueCard();
}

function renderContinueCard() {
  const card = $("#continue-card");
  const doc = state.doc;
  if (!doc) { card.classList.add("hidden"); return; }
  card.classList.remove("hidden");
  $("#continue-title").textContent = doc.title;
  const total = doc.page_count || 0;
  const page = Math.min(state.page || 1, total || 1);
  $("#continue-fill").style.width = total ? `${Math.max(2, (page / total) * 100)}%` : "0%";
  $("#continue-pos").textContent = total ? `page ${page} of ${total.toLocaleString()}` : "";
  $("#continue-cover-img").src = `${API}/documents/${doc.id}/pages/${page}/image`;
  $("#continue-cover-img").onerror = function () { this.style.display = "none"; };
}

async function openDocument(doc) {
  if (state.doc && state.doc.id === doc.id) return;
  state.doc = doc;
  state.totalPages = doc.page_count;
  $("#doc-subtitle").textContent = `${doc.title} · ${doc.subtitle || ""}`;
  $("#ask-banner").textContent =
    `Answers are drawn from all ${state.totalPages} transcribed pages of this document`;
  await loadPage(1);
}

/* ---------- init ---------- */
async function init() {
  await loadSession();
  if (state.gated) {
    setAuthMode("signin");
    showView("signin");
    // After setAuthMode, which clears the error box — otherwise the message a
    // failed provider sign-in just redirected us here to show gets wiped.
    showOAuthErrorFromUrl();
    loadPublicStats();
    return;
  }
  const hasDocs = await loadCorpus();
  // Empty backend: Upload is the only view with anything to do.
  if (!hasDocs) return showView("upload");
  showView(location.hash ? location.hash.slice(1) : "library");
}

async function loadCorpus() {
  let docs = [];
  try { docs = await api("/documents"); } catch (e) { /* backend down */ }
  state.docs = docs;
  if (!docs.length) {
    $("#doc-subtitle").textContent = "No documents — upload pages to begin";
    return false;
  }
  state.doc = docs[0];
  state.totalPages = state.doc.page_count;
  $("#doc-subtitle").textContent = `${state.doc.title} · ${state.doc.subtitle || ""}`;
  $("#ask-banner").textContent =
    `Answers are drawn from all ${state.totalPages} transcribed pages of this document`;
  // Hide the Deep research toggle when the backend has the agent switched off,
  // so the control is never offered when it would 503.
  try {
    const health = await api("/health");
    if (health && health.agent_enabled === false) $("#ask-modes").classList.add("hidden");
  } catch (e) { /* health is advisory — leave the toggle as-is */ }
  await loadPage(1);
  return true;
}

/* ---------- Reader ---------- */
async function loadPage(n) {
  if (!state.doc || n < 1 || n > state.totalPages) return;
  state.page = n;
  $("#page-cur").textContent = n;
  $("#page-total").textContent = "/ " + state.totalPages;

  let page;
  try { page = await api(`/documents/${state.doc.id}/pages/${n}`); }
  catch (e) { return; }

  // scan
  const img = $("#scan-img");
  img.src = imageUrl(state.doc.id, page);
  img.onerror = () => { img.style.display = "none"; $("#scan-empty").style.display = "block"; };
  img.onload = () => { img.style.display = "block"; $("#scan-empty").style.display = "none"; };
  $("#scan-name").textContent = `page ${n}`;
  $("#scan-dims").textContent = page.width ? `${page.width} × ${page.height}` : "";

  // readouts
  $("#trans-sub").textContent = `Machine reading · page ${n}`;
  $("#avg-conf").textContent = page.avg_confidence != null ? Math.round(page.avg_confidence * 100) : "–";
  const low = page.low_conf_words || 0;
  $("#lowconf-count").textContent = low ? `${low} word${low > 1 ? "s" : ""} below 60%` : "";

  // remember page; show the Raw/Corrected toggle only when correction exists
  state.pageData = page;
  // correction present if page-level corrected_lines OR any per-line corrected_words
  const hasCorrection = (page.corrected_lines && page.corrected_lines.length) ||
    page.lines.some((l) => l.corrected_words && l.corrected_words.length);
  $("#view-toggle").style.display = hasCorrection ? "inline-flex" : "none";
  if (hasCorrection && state.readerView == null) state.readerView = "corrected";
  renderTranscription();
}

/* build a line <div> of confidence-tinted words (green when Qwen-corrected) */
function renderWordLine(words, flagReview) {
  const div = el("div", "t-line" + (flagReview ? " review" : ""));
  words.forEach((w) => {
    const span = el("w");
    span.textContent = w.text;
    if (w.qwen_replaced) {
      span.className = "qwen";
      span.title = `Qwen correction (orig conf ${Math.round(w.confidence * 100)}%)`;
    } else {
      span.style.color = confColor(w.confidence);
      if (w.needs_review) span.title = `low confidence (${Math.round(w.confidence * 100)}%)`;
    }
    div.appendChild(span);
    div.appendChild(document.createTextNode(" "));
  });
  return div;
}

function renderTranscription() {
  const page = state.pageData;
  if (!page) return;
  const corrected = state.readerView === "corrected";
  document.querySelectorAll(".toggle-opt").forEach((o) =>
    o.classList.toggle("active", o.dataset.view === (corrected ? "corrected" : "raw")));

  const box = $("#transcription");
  box.innerHTML = "";

  // Corrected view: prefer page-level corrected_lines (preseed), else per-line corrected_words.
  if (corrected && page.corrected_lines && page.corrected_lines.length) {
    page.corrected_lines.forEach((words) => box.appendChild(renderWordLine(words, false)));
    return;
  }
  if (!page.lines.length) { box.innerHTML = '<div class="empty">No transcription for this page.</div>'; return; }
  for (const line of page.lines) {
    const words = corrected && line.corrected_words && line.corrected_words.length
      ? line.corrected_words : line.words;
    if (words && words.length) {
      box.appendChild(renderWordLine(words, line.needs_review && !corrected));
    } else {
      const div = el("div", "t-line");
      div.textContent = corrected && line.corrected_text != null ? line.corrected_text : line.text;
      box.appendChild(div);
    }
  }
}
// Cyclic navigation: next past the last page wraps to page 1, prev before page 1 wraps to the last.
$("#prev-page").addEventListener("click", () =>
  loadPage(state.page > 1 ? state.page - 1 : state.totalPages));
$("#next-page").addEventListener("click", () =>
  loadPage(state.page < state.totalPages ? state.page + 1 : 1));
$("#view-toggle").addEventListener("click", (e) => {
  if (!e.target.dataset.view) return;
  state.readerView = e.target.dataset.view;
  renderTranscription();
});

/* ---------- Ask the Archive ---------- */
/* Two modes. Default = POST /ask: one retrieval, one LLM call, ~2s. "Deep research"
   = POST /agent: a LangGraph loop that can search, then read pages in sequence, and
   remembers the conversation. Slower, so it only runs when explicitly asked for. */
const deepOn = () => $("#deep-research").checked;

/* Thread id is per-document and per-tab: a new tab starts a new conversation, and
   switching documents must not resurrect the wrong history. */
function threadId() {
  const key = "inkference.thread." + state.doc.id;
  let id = sessionStorage.getItem(key);
  if (!id) {
    id = (crypto.randomUUID ? crypto.randomUUID() : String(Date.now() + Math.random()));
    sessionStorage.setItem(key, id);
  }
  return id;
}

function renderSources(ans, pages) {
  if (!pages || !pages.length) return;
  const src = el("div", "sources");
  src.innerHTML = '<span class="sources-cap">Sources</span>';
  pages.forEach((p) => {
    const chip = el("span", "chip-page"); chip.textContent = "Page " + p;
    chip.addEventListener("click", () => { showView("reader"); loadPage(p); });
    src.appendChild(chip);
  });
  ans.appendChild(src);
}

/* The step trace is what makes a 20s wait legible: it shows the agent searching and
   reading rather than an idle spinner. */
function renderTrace(ans, trace) {
  if (!trace || !trace.length) return;
  const strip = el("div", "trace");
  trace.forEach((t) => {
    const step = el("span", "trace-step");
    step.textContent = t.label + (t.note ? ` (${t.note})` : "");
    strip.appendChild(step);
  });
  ans.insertBefore(strip, ans.querySelector(".answer-body"));
}

async function ask(question, persona) {
  if (!question.trim() || !state.doc) return;
  const cook = persona === "cook";
  const deep = deepOn();
  const thread = $("#thread");
  const q = el("div", "bubble-q"); q.textContent = question; thread.appendChild(q);

  const ans = el("div", "answer");
  const tag = cook ? '<span class="in-character">in character</span>' : "";
  const deepTag = deep ? '<span class="deep-tag">deep research</span>' : "";
  const loading = deep ? "Researching the journal…" : (cook ? "Consulting the journal…" : "…thinking…");
  ans.innerHTML = `<div class="answer-head"><div class="answer-mark">I</div>
    <span class="answer-who">${cook ? "Author" : "Inkference"}</span>${tag}${deepTag}</div>
    <div class="answer-body">${loading}</div>`;
  thread.appendChild(ans);
  thread.scrollTop = thread.scrollHeight;

  try {
    const body = deep
      ? { question, persona: persona || null, thread_id: threadId() }
      : { question, persona: persona || null };
    const res = await api(`/documents/${state.doc.id}/${deep ? "agent" : "ask"}`, {
      method: "POST", headers: { "content-type": "application/json" },
      body: JSON.stringify(body),
    });
    ans.querySelector(".answer-body").textContent = res.answer;
    renderTrace(ans, res.trace);
    renderSources(ans, res.source_pages);
    if (deep) $("#ask-reset").classList.remove("hidden");
  } catch (e) {
    ans.querySelector(".answer-body").textContent = "Error: " + e.message;
  }
  thread.scrollTop = thread.scrollHeight;
}
function submitAsk(persona) { const i = $("#ask-input"); ask(i.value, persona); i.value = ""; }
$("#ask-send").addEventListener("click", () => submitAsk());
$("#ask-cook").addEventListener("click", () => submitAsk("cook"));
$("#ask-input").addEventListener("keydown", (e) => { if (e.key === "Enter") submitAsk(); });
$("#suggestions").addEventListener("click", (e) => { if (e.target.dataset.q) ask(e.target.dataset.q); });

$("#ask-reset").addEventListener("click", async () => {
  if (!state.doc) return;
  const key = "inkference.thread." + state.doc.id;
  const id = sessionStorage.getItem(key);
  if (id) {
    try {
      await api(`/documents/${state.doc.id}/agent/threads/${id}`, { method: "DELETE" });
    } catch (e) { /* the thread may never have been persisted — clearing locally is enough */ }
    sessionStorage.removeItem(key);
  }
  const thread = $("#thread");
  [...thread.querySelectorAll(".bubble-q, .answer")].forEach((n) => n.remove());
  $("#ask-reset").classList.add("hidden");
});

/* ---------- Upload & Process ---------- */
const dz = $("#dropzone");
["dragover", "dragenter"].forEach((ev) => dz.addEventListener(ev, (e) => { e.preventDefault(); dz.classList.add("drag"); }));
["dragleave", "drop"].forEach((ev) => dz.addEventListener(ev, (e) => { e.preventDefault(); dz.classList.remove("drag"); }));
dz.addEventListener("drop", (e) => handleFiles(e.dataTransfer.files));
$("#file-input").addEventListener("change", (e) => handleFiles(e.target.files));

function setStep(stage) {
  const order = ["segmentation", "recognition", "confidence", "correction"];
  const idx = order.indexOf(stage);
  document.querySelectorAll(".step").forEach((s) => {
    const i = order.indexOf(s.dataset.stage);
    s.classList.toggle("active", i === idx);
    s.classList.toggle("done", i < idx);
  });
}

/* Show, and let the user change, where the next upload lands. */
function renderUploadTarget() {
  const box = $("#upload-target");
  if (!box) return;
  box.innerHTML = "";

  // With no books yet there is nothing to append to, so the only sensible mode is
  // "new book" — don't offer a choice that has one option.
  const mustBeNew = !state.doc;
  const creating = state.newBookMode || mustBeNew;

  if (creating) {
    const label = el("label", "target-label");
    label.textContent = "New book";
    const input = el("input", "target-input");
    input.id = "new-book-title";
    input.type = "text";
    input.placeholder = "Untitled manuscript";
    input.value = state._newBookTitle || "";
    // Survives a re-render (e.g. switching modes and back) so a typed title is
    // not silently discarded.
    input.addEventListener("input", () => { state._newBookTitle = input.value; });
    label.append(input);
    box.append(label);

    if (!mustBeNew) {
      const swap = el("button", "target-swap");
      swap.type = "button";
      swap.textContent = `Add to “${state.doc.title}” instead`;
      swap.addEventListener("click", () => {
        state.newBookMode = false;
        renderUploadTarget();
      });
      box.append(swap);
    }
  } else {
    const info = el("div", "target-label");
    info.innerHTML = `Adding pages to <strong></strong>`;
    info.querySelector("strong").textContent = state.doc.title;
    const swap = el("button", "target-swap");
    swap.type = "button";
    swap.textContent = "Start a new book instead";
    swap.addEventListener("click", () => {
      state.newBookMode = true;
      renderUploadTarget();
    });
    box.append(info, swap);
  }
}

/* The document this upload belongs to, creating it first if we are starting a new
   book. Returns null if creation failed, so the caller can abort rather than post
   pages into whatever was open before. */
async function resolveUploadTarget() {
  if (!state.newBookMode && state.doc) return state.doc;

  const title = (state._newBookTitle || "").trim() || "Untitled manuscript";
  let created;
  try {
    created = await api("/documents", {
      method: "POST", headers: { "content-type": "application/json" },
      body: JSON.stringify({ title }),
    });
  } catch (err) {
    $("#ingest-status").textContent = errorText(err) || "Could not create the book.";
    return null;
  }

  const doc = { id: created.id, slug: created.slug, title, subtitle: "", page_count: 0 };
  state.doc = doc;
  state.docs = [...(state.docs || []), doc];   // so the Library grid shows it
  state.totalPages = 0;
  state.page = 1;
  // The upload now belongs to this book: a second drop must extend it, not spawn
  // another empty book.
  state.newBookMode = false;
  state._newBookTitle = "";
  $("#doc-subtitle").textContent = title;
  renderUploadTarget();
  return doc;
}

async function handleFiles(fileList) {
  const files = Array.from(fileList).filter((f) => f.type.startsWith("image/"));
  if (!files.length) return;

  const doc = await resolveUploadTarget();
  if (!doc) return;

  // preview first file; reset any boxes/state from a previous upload
  state._segPreviewShown = false;
  state._segRedraw = null;
  $("#seg-overlay").innerHTML = "";
  const reader = new FileReader();
  reader.onload = () => { $("#ingest-img").src = reader.result; };
  reader.readAsDataURL(files[0]);

  // build queue rows
  const queue = $("#queue"); queue.innerHTML = "";
  const rows = files.map((f, i) => {
    const row = el("div", "q-row" + (i === 0 ? " active" : ""));
    row.innerHTML = `<div class="q-thumb"></div>
      <div class="q-main"><div class="q-name">${f.name}</div>
        <div class="q-bar"><div></div></div></div>
      <div class="q-status">Queued</div>`;
    queue.appendChild(row);
    return row;
  });

  const fd = new FormData();
  files.forEach((f) => fd.append("files", f));
  $("#ingest-status").textContent = `Uploading to “${doc.title}”…`;
  // doc, not state.doc: the destination was resolved before this await, and using
  // the live global here would follow the user if they navigated mid-upload.
  const resp = await api(`/documents/${doc.id}/pages`, { method: "POST", body: fd });
  // Use the authoritative page numbers the server assigned (don't recompute from a
  // client-side counter, which drifts across reloads/restarts and can point at a
  // previously-seeded page instead of the one just uploaded).
  pollJob(resp.job_id, rows, files.length, resp.pages || [], doc);
}

/* `doc` is the book this job is ingesting into — passed in rather than read from
   state, because ingestion outlives the view and the user may open another book
   while it runs. */
async function pollJob(jobId, rows, total, uploadedPages = [], doc = state.doc) {
  const timer = setInterval(async () => {
    let job;
    try { job = await api(`/jobs/${jobId}`); } catch (e) { return; }
    const done = job.done_pages || 0;
    const stage = job.stage || "segmentation";
    setStep(stage);
    $("#ingest-status").textContent = job.message || job.status;

    // Draw the segmentation boxes as soon as they're published (mid-pipeline),
    // once, for the first uploaded page (the one shown in the local preview).
    if (job.seg_preview && !state._segPreviewShown) {
      try {
        const sp = JSON.parse(job.seg_preview);
        const firstPage = uploadedPages[0];
        if (firstPage == null || sp.page_number === firstPage) {
          state._segPreviewShown = true;
          drawSegPreview(sp);
        }
      } catch (e) { /* ignore malformed preview */ }
    }

    rows.forEach((row, i) => {
      const bar = row.querySelector(".q-bar > div");
      const st = row.querySelector(".q-status");
      row.classList.toggle("active", i === done && job.status !== "complete");
      if (i < done || job.status === "complete") {
        bar.style.width = "100%"; st.textContent = "Complete"; st.className = "q-status done";
      } else if (i === done) {
        const frac = ((job.progress || 0) * total) - done;
        bar.style.width = Math.max(5, Math.min(100, frac * 100)) + "%";
        st.textContent = job.status === "recognizing" ? "Recognizing" :
                         job.status === "correcting" ? "Correcting" :
                         job.status === "scoring" ? "Scoring" : "Segmenting";
      }
    });

    if (job.status === "complete" || job.status === "failed") {
      clearInterval(timer);
      if (job.status === "complete") {
        doc.page_count = (doc.page_count || 0) + total;
        // Only move the reader's page count if the user is still on this book.
        if (state.doc && state.doc.id === doc.id) state.totalPages = doc.page_count;
        const firstPage = uploadedPages[0] ?? (doc.page_count - total + 1);
        await drawSegmentation(firstPage, doc);
      } else {
        $("#ingest-status").textContent = "Failed: " + (job.message || "error");
      }
    }
  }, 1000);
}

/* Draw line boxes over the #ingest-img preview. `boxes` = [{bbox:[x0,y0,x1,y1], review}].
   pageW/pageH are the (possibly downscaled) page dims the boxes are expressed in. */
function overlayBoxes(boxes, pageW, pageH) {
  const img = $("#ingest-img");
  const overlay = $("#seg-overlay");
  overlay.innerHTML = "";
  const r = img.getBoundingClientRect();
  const pr = overlay.getBoundingClientRect();
  const offX = r.left - pr.left, offY = r.top - pr.top;
  const sx = r.width / pageW, sy = r.height / pageH;
  boxes.forEach(({ bbox, review }) => {
    const [x0, y0, x1, y1] = bbox;
    const b = el("div", "seg-box" + (review ? " review" : ""));
    b.style.left = offX + x0 * sx + "px";
    b.style.top = offY + y0 * sy + "px";
    b.style.width = (x1 - x0) * sx + "px";
    b.style.height = (y1 - y0) * sy + "px";
    overlay.appendChild(b);
  });
}

/* Early preview: draw the raw segmentation boxes (no confidence yet) as soon as the
   segmentation stage publishes them, over the local file preview already on screen. */
function drawSegPreview(sp) {
  const img = $("#ingest-img");
  const boxes = (sp.boxes || []).map((bbox) => ({ bbox, review: false }));
  const draw = () => overlayBoxes(boxes, sp.width, sp.height);
  state._segRedraw = draw;
  if (img.complete && img.naturalWidth) draw();
  else img.onload = draw;
  $("#ingest-status").textContent = `Segmented · ${boxes.length} lines — recognizing…`;
}

/* Final overlay: fetch the finished page and draw boxes tinted by confidence. */
async function drawSegmentation(pageNumber, doc = state.doc) {
  try {
    const page = await api(`/documents/${doc.id}/pages/${pageNumber}`);
    const img = $("#ingest-img");
    const boxes = (page.lines || []).map((ln) => ({ bbox: ln.bbox, review: ln.needs_review }));
    const draw = () => {
      overlayBoxes(boxes, page.width, page.height);
      $("#ingest-status").textContent = `Done · ${page.lines.length} lines`;
    };
    // Attach handler BEFORE src (the scan is under /api/, which is browser-cached),
    // and draw immediately if it's already loaded — otherwise a cached image never
    // fires `load` and the overlay stays empty. Redraw on resize so boxes track the img.
    state._segRedraw = draw;
    img.onload = draw;
    img.src = imageUrl(doc.id, page);
    if (img.complete && img.naturalWidth) draw();
  } catch (e) { console.error("drawSegmentation failed", e); }
}
$("#open-reader").addEventListener("click", () => { showView("reader"); loadPage(state.totalPages); });
window.addEventListener("resize", () => { if (state._segRedraw) state._segRedraw(); });

init();
