/* Inkference frontend logic. Talks to the FastAPI backend at <origin>/api
   (override with window.INKFERENCE_API for a separately-hosted backend). */
const API = (window.INKFERENCE_API || "") + "/api";

const state = { doc: null, page: 1, totalPages: 0, pageData: null, readerView: null };

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
  if (!r.ok) throw new Error((await r.text()) || r.status);
  return r.json();
}

/* confidence 1.0 -> dark ink, 0.0 -> faded; matches the legend gradient */
function confColor(c) {
  const dark = [43, 33, 26], light = [205, 191, 169];
  const t = Math.max(0, Math.min(1, c));
  const mix = dark.map((d, i) => Math.round(light[i] + (d - light[i]) * t));
  return `rgb(${mix[0]},${mix[1]},${mix[2]})`;
}

/* ---------- tabs ---------- */
document.querySelectorAll(".tab").forEach((t) => {
  t.addEventListener("click", () => showView(t.dataset.view));
});
function showView(name) {
  if (!["reader", "ask", "upload"].includes(name)) name = "reader";
  document.querySelectorAll(".tab").forEach((t) => t.classList.toggle("active", t.dataset.view === name));
  document.querySelectorAll(".view").forEach((v) => v.classList.add("hidden"));
  $("#view-" + name).classList.remove("hidden");
  if (location.hash.slice(1) !== name) history.replaceState(null, "", "#" + name);
}
window.addEventListener("hashchange", () => showView(location.hash.slice(1)));

/* ---------- init ---------- */
async function init() {
  let docs = [];
  try { docs = await api("/documents"); } catch (e) { /* backend down */ }
  if (!docs.length) {
    $("#doc-subtitle").textContent = "No documents — upload pages to begin";
    showView("upload");
    return;
  }
  state.doc = docs[0];
  state.totalPages = state.doc.page_count;
  $("#doc-subtitle").textContent = `${state.doc.title} · ${state.doc.subtitle || ""}`;
  $("#ask-banner").textContent =
    `Answers are drawn from all ${state.totalPages} transcribed pages of this document`;
  await loadPage(1);
  if (location.hash) showView(location.hash.slice(1));
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
async function ask(question, persona) {
  if (!question.trim() || !state.doc) return;
  const cook = persona === "cook";
  const thread = $("#thread");
  const q = el("div", "bubble-q"); q.textContent = question; thread.appendChild(q);

  const ans = el("div", "answer");
  const tag = cook ? '<span class="in-character">in character</span>' : "";
  const loading = cook ? "Consulting the journal…" : "…thinking…";
  ans.innerHTML = `<div class="answer-head"><div class="answer-mark">I</div>
    <span class="answer-who">${cook ? "Author" : "Inkference"}</span>${tag}</div>
    <div class="answer-body">${loading}</div>`;
  thread.appendChild(ans);
  thread.scrollTop = thread.scrollHeight;

  try {
    const res = await api(`/documents/${state.doc.id}/ask`, {
      method: "POST", headers: { "content-type": "application/json" },
      body: JSON.stringify({ question, persona: persona || null }),
    });
    ans.querySelector(".answer-body").textContent = res.answer;
    if (res.source_pages && res.source_pages.length) {
      const src = el("div", "sources");
      src.innerHTML = '<span class="sources-cap">Sources</span>';
      res.source_pages.forEach((p) => {
        const chip = el("span", "chip-page"); chip.textContent = "Page " + p;
        chip.addEventListener("click", () => { showView("reader"); loadPage(p); });
        src.appendChild(chip);
      });
      ans.appendChild(src);
    }
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

async function handleFiles(fileList) {
  const files = Array.from(fileList).filter((f) => f.type.startsWith("image/"));
  if (!files.length) return;

  // ensure a document exists to ingest into
  if (!state.doc) {
    const d = await api("/documents", {
      method: "POST", headers: { "content-type": "application/json" },
      body: JSON.stringify({ title: "My Manuscript" }),
    });
    state.doc = { id: d.id, slug: d.slug, title: "My Manuscript", subtitle: "", page_count: 0 };
  }

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
  $("#ingest-status").textContent = "Uploading…";
  const resp = await api(`/documents/${state.doc.id}/pages`, { method: "POST", body: fd });
  // Use the authoritative page numbers the server assigned (don't recompute from a
  // client-side counter, which drifts across reloads/restarts and can point at a
  // previously-seeded page instead of the one just uploaded).
  pollJob(resp.job_id, rows, files.length, resp.pages || []);
}

async function pollJob(jobId, rows, total, uploadedPages = []) {
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
        state.doc.page_count = (state.doc.page_count || 0) + total;
        state.totalPages = state.doc.page_count;
        const firstPage = uploadedPages[0] ?? (state.totalPages - total + 1);
        await drawSegmentation(firstPage);
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
async function drawSegmentation(pageNumber) {
  try {
    const page = await api(`/documents/${state.doc.id}/pages/${pageNumber}`);
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
    img.src = imageUrl(state.doc.id, page);
    if (img.complete && img.naturalWidth) draw();
  } catch (e) { console.error("drawSegmentation failed", e); }
}
$("#open-reader").addEventListener("click", () => { showView("reader"); loadPage(state.totalPages); });
window.addEventListener("resize", () => { if (state._segRedraw) state._segRedraw(); });

init();
