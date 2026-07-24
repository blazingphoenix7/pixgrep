const $ = (id) => document.getElementById(id);
const grid = $("grid"), status = $("status");
let current = null; // row shown in lightbox
const activeFilters = {}; // {field: value}
let isSearching = false;
let tray = []; // ordered row integers for the export deck

// ── Feedback (thumbs-down) + user identity ─────────────────────────────────────
const USER_KEY = "pixgrep-user";
const FILTERS_OPEN_KEY = "pixgrep-filters-open";
let currentUser = localStorage.getItem(USER_KEY) || "";
let currentQuery = null; // text query the current result set was rendered for; null when not text-scoped (image/similar search)
let markedRows = new Set(); // rows marked bad for currentQuery
let pendingFeedback = null; // row awaiting a name before its toggle can fire

async function meta() {
  try {
    const r = await fetch("/api/meta");
    const j = await r.json();
    $("count").textContent = `${j.count} images indexed`;
  } catch { $("count").textContent = ""; }
}

async function loadFacets() {
  try {
    const r = await fetch("/api/facets");
    if (!r.ok) return;
    const facets = await r.json();
    const fields = Object.keys(facets);
    if (!fields.length) return;

    const container = $("filters");
    for (const field of fields) {
      const values = facets[field].slice(0, 8);
      if (!values.length) continue;

      const row = document.createElement("div");
      row.className = "filter-row";

      const label = document.createElement("span");
      label.className = "filter-label";
      label.textContent = field.replace(/_/g, " ");
      row.appendChild(label);

      for (const { value, count } of values) {
        const chip = document.createElement("button");
        chip.className = "chip";
        chip.dataset.field = field;
        chip.dataset.value = value;
        const nm = document.createElement("span");
        nm.textContent = value;
        const ct = document.createElement("span");
        ct.style.opacity = "0.6";
        ct.textContent = ` ${count}`;
        chip.appendChild(nm);
        chip.appendChild(ct);
        chip.addEventListener("click", () => toggleChip(chip, field, value));
        row.appendChild(chip);
      }

      container.appendChild(row);
    }
    $("filter-toolbar").classList.remove("hidden");
  } catch {}
}

function toggleChip(chip, field, value) {
  if (activeFilters[field] === value) {
    delete activeFilters[field];
    chip.classList.remove("active");
  } else {
    // Deactivate previous selection for this field
    const prev = $("filters").querySelector(`.chip.active[data-field]`);
    if (prev && prev.dataset.field === field) prev.classList.remove("active");
    activeFilters[field] = value;
    chip.classList.add("active");
  }
  renderActivePills();
}

function renderActivePills() {
  const container = $("active-pills");
  container.replaceChildren();
  for (const [field, value] of Object.entries(activeFilters)) {
    const pill = document.createElement("span");
    pill.className = "active-pill";
    const label = document.createElement("span");
    label.textContent = `${field.replace(/_/g, " ")}: ${value}`;
    const rm = document.createElement("button");
    rm.className = "active-pill-remove";
    rm.textContent = "×";
    rm.title = "Remove filter";
    rm.addEventListener("click", () => removeFilter(field));
    pill.appendChild(label);
    pill.appendChild(rm);
    container.appendChild(pill);
  }
}

function removeFilter(field) {
  delete activeFilters[field];
  const chip = $("filters").querySelector(`.chip.active[data-field="${field}"]`);
  if (chip) chip.classList.remove("active");
  renderActivePills();
  doSearch();
}

function setFiltersOpen(open) {
  $("filters").classList.toggle("open", open);
  $("filters-toggle").setAttribute("aria-expanded", String(open));
  $("filters-toggle").textContent = open ? "Filters ▴" : "Filters ▾";
  localStorage.setItem(FILTERS_OPEN_KEY, open ? "1" : "0");
}

$("filters-toggle").addEventListener("click", () => {
  setFiltersOpen(!$("filters").classList.contains("open"));
});

function appendFilters(url) {
  for (const [field, value] of Object.entries(activeFilters)) {
    url.searchParams.append("f", `${field}:${value}`);
  }
}

// Chunked rendering: results arrive uncapped (thousands for broad queries),
// so cards are appended in batches as the user scrolls instead of all at once.
const RENDER_CHUNK = 200;
let pendingResults = [];
let renderedCount = 0;
let gridSentinel = null;

const chunkObserver = new IntersectionObserver((entries) => {
  if (entries.some((e) => e.isIntersecting)) renderNextChunk();
}, { rootMargin: "1200px" });

function renderNextChunk() {
  const next = pendingResults.slice(renderedCount, renderedCount + RENDER_CHUNK);
  for (const r of next) grid.insertBefore(buildCard(r), gridSentinel);
  renderedCount += next.length;
  if (renderedCount >= pendingResults.length && gridSentinel) {
    chunkObserver.unobserve(gridSentinel);
    gridSentinel.remove();
    gridSentinel = null;
  }
}

function render(results) {
  grid.replaceChildren();
  if (gridSentinel) { chunkObserver.unobserve(gridSentinel); gridSentinel = null; }
  pendingResults = results;
  renderedCount = 0;
  // Both the status count and the card loop derive from the same array, so they are always in sync.
  if (!results.length) { status.textContent = "No matches."; return; }
  status.textContent = `${results.length} match${results.length === 1 ? "" : "es"}`;
  gridSentinel = document.createElement("div");
  gridSentinel.className = "grid-sentinel";
  grid.appendChild(gridSentinel);
  chunkObserver.observe(gridSentinel);
  renderNextChunk();
}

function buildCard(r) {
    const card = document.createElement("div");
    card.className = "card";
    const imgwrap = document.createElement("div");
    imgwrap.className = "imgwrap";
    const img = document.createElement("img");
    img.loading = "lazy";
    img.src = `/api/thumb/${r.row}`;
    img.alt = "";
    imgwrap.appendChild(img);
    const cap = document.createElement("div");
    cap.className = "cap";
    const nm = document.createElement("span");
    nm.className = "nm";
    nm.title = r.name;
    nm.textContent = r.name;
    const sc = document.createElement("span");
    sc.className = "sc";
    sc.textContent = r.score.toFixed(3);
    cap.appendChild(nm);
    cap.appendChild(sc);
    card.appendChild(imgwrap);
    card.appendChild(cap);
    card.addEventListener("click", () => openLightbox(r));

    const addBtn = document.createElement("button");
    addBtn.className = "add-btn";
    addBtn.title = "Add to deck";
    addBtn.textContent = "+";
    addBtn.addEventListener("click", (e) => {
      e.stopPropagation();
      if (!addToTray(r.row)) {
        addBtn.textContent = "✓";
        setTimeout(() => { addBtn.textContent = "+"; }, 700);
      } else {
        addBtn.classList.add("added");
        setTimeout(() => { addBtn.classList.remove("added"); }, 700);
      }
    });
    card.appendChild(addBtn);

    if (currentQuery) {
      const dnBtn = document.createElement("button");
      dnBtn.className = "down-btn";
      dnBtn.title = "Mark as bad result";
      dnBtn.textContent = "👎";
      dnBtn.dataset.row = String(r.row);
      if (markedRows.has(r.row)) {
        dnBtn.classList.add("marked");
        card.classList.add("marked-bad");
      }
      dnBtn.addEventListener("click", (e) => {
        e.stopPropagation();
        handleDownClick(r.row);
      });
      card.appendChild(dnBtn);
    }

    return card;
}

// ── Feedback (thumbs-down) ───────────────────────────────────────────────────

function paintMarked(row, marked) {
  if (marked) markedRows.add(row); else markedRows.delete(row);
  const btn = grid.querySelector(`.down-btn[data-row="${row}"]`);
  if (btn) {
    btn.classList.toggle("marked", marked);
    btn.closest(".card").classList.toggle("marked-bad", marked);
  }
  if (current && current.row === row) {
    $("lb-down").classList.toggle("marked", marked);
  }
}

function paintVisibleMarks() {
  for (const btn of grid.querySelectorAll(".down-btn")) {
    const row = parseInt(btn.dataset.row, 10);
    const marked = markedRows.has(row);
    btn.classList.toggle("marked", marked);
    btn.closest(".card").classList.toggle("marked-bad", marked);
  }
  if (current && currentQuery) {
    $("lb-down").classList.toggle("marked", markedRows.has(current.row));
  }
}

async function refreshMarks(query) {
  try {
    const url = new URL("/api/feedback/marks", location.origin);
    url.searchParams.set("query", query);
    const r = await fetch(url);
    if (!r.ok) return;
    const { rows } = await r.json();
    markedRows = new Set(rows);
    paintVisibleMarks();
  } catch {}
}

async function toggleFeedback(row) {
  if (!currentQuery) return;
  try {
    const r = await fetch("/api/feedback/toggle", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ user: currentUser, query: currentQuery, row }),
    });
    if (!r.ok) { status.textContent = `Could not update feedback (${r.status}).`; return; }
    const { marked } = await r.json();
    paintMarked(row, marked);
  } catch {
    status.textContent = "Network error — could not update feedback.";
  }
}

function handleDownClick(row) {
  if (!currentUser) {
    pendingFeedback = row;
    openUserPrompt();
    return;
  }
  toggleFeedback(row);
}

function setSearching(active) {
  isSearching = active;
  $("go").disabled = active;
}

async function doSearch() {
  const q = $("q").value.trim();
  if (!q || isSearching) return;
  setSearching(true);
  status.textContent = "Searching…";
  try {
    const url = new URL("/api/search", location.origin);
    url.searchParams.set("q", q);
    url.searchParams.set("k", "0");
    appendFilters(url);
    const r = await fetch(url);
    if (!r.ok) { status.textContent = `Search failed (${r.status}).`; return; }
    currentQuery = q;
    markedRows = new Set();
    render((await r.json()).results);
    refreshMarks(q);
  } catch {
    status.textContent = "Network error — is the server running?";
  } finally {
    setSearching(false);
  }
}

async function doImageSearch(file) {
  if (isSearching) return;
  setSearching(true);
  status.textContent = "Searching by image…";
  try {
    const url = new URL("/api/search/image", location.origin);
    url.searchParams.set("k", "0");
    appendFilters(url);
    const fd = new FormData();
    fd.append("file", file);
    const r = await fetch(url, { method: "POST", body: fd });
    if (!r.ok) { status.textContent = "Could not read that image."; return; }
    currentQuery = null;
    markedRows = new Set();
    render((await r.json()).results);
  } catch {
    status.textContent = "Network error — is the server running?";
  } finally {
    setSearching(false);
  }
}

async function doSimilar(row) {
  if (isSearching) return;
  setSearching(true);
  status.textContent = "Finding similar…";
  try {
    const url = new URL(`/api/similar/${row}`, location.origin);
    url.searchParams.set("k", "0");
    appendFilters(url);
    const r = await fetch(url);
    if (!r.ok) { status.textContent = "Search failed."; return; }
    currentQuery = null;
    markedRows = new Set();
    render((await r.json()).results);
  } catch {
    status.textContent = "Network error — is the server running?";
  } finally {
    setSearching(false);
  }
}

async function openLightbox(r) {
  current = r;
  $("lb-img").src = `/api/image/${r.row}`;
  $("lb-name").textContent = r.name;
  $("lb-path").textContent = r.path;
  $("lb-filmstrip").replaceChildren();
  updateLightboxDownBtn();
  $("lightbox").classList.remove("hidden");
  try {
    const resp = await fetch(`/api/group/${r.row}`);
    if (resp.ok) {
      const { results } = await resp.json();
      if (results.length > 1) renderFilmstrip(results, r.row);
    }
  } catch {}
}

function renderFilmstrip(members, activeRow) {
  const strip = $("lb-filmstrip");
  for (const m of members) {
    const img = document.createElement("img");
    img.src = `/api/thumb/${m.row}`;
    img.alt = "";
    img.dataset.row = String(m.row);
    img.className = "filmstrip-thumb" + (m.row === activeRow ? " active" : "");
    img.addEventListener("click", () => selectFilmstripMember(m));
    strip.appendChild(img);
  }
}

function selectFilmstripMember(m) {
  current = m;
  $("lb-img").src = `/api/image/${m.row}`;
  $("lb-name").textContent = m.name;
  $("lb-path").textContent = m.path;
  updateLightboxDownBtn();
  const strip = $("lb-filmstrip");
  for (const thumb of strip.querySelectorAll(".filmstrip-thumb")) {
    thumb.classList.toggle("active", parseInt(thumb.dataset.row) === m.row);
  }
}

function updateLightboxDownBtn() {
  const btn = $("lb-down");
  if (!currentQuery || !current) { btn.classList.add("hidden"); return; }
  btn.classList.remove("hidden");
  btn.classList.toggle("marked", markedRows.has(current.row));
}

function closeLightbox() {
  $("lightbox").classList.add("hidden");
}

$("go").addEventListener("click", doSearch);
$("q").addEventListener("keydown", (e) => { if (e.key === "Enter") doSearch(); });
$("lb-close").addEventListener("click", closeLightbox);
$("lightbox").addEventListener("click", (e) => {
  if (e.target === $("lightbox")) closeLightbox();
});
document.addEventListener("keydown", (e) => {
  if (e.key === "Escape") { closeLightbox(); closeUserPrompt(); }
});
$("lb-copy").addEventListener("click", () => {
  if (current) navigator.clipboard.writeText(current.path);
});
$("lb-similar").addEventListener("click", () => {
  if (current) { closeLightbox(); doSimilar(current.row); }
});
$("lb-down").addEventListener("click", (e) => {
  e.stopPropagation();
  if (current) handleDownClick(current.row);
});

const drop = $("drop");
["dragenter", "dragover"].forEach((ev) =>
  drop.addEventListener(ev, (e) => { e.preventDefault(); drop.classList.add("hover"); }));
["dragleave", "drop"].forEach((ev) =>
  drop.addEventListener(ev, (e) => { e.preventDefault(); drop.classList.remove("hover"); }));
drop.addEventListener("drop", (e) => {
  const f = e.dataTransfer.files[0];
  if (f) doImageSearch(f);
});

const browseInput = $("browse-input");
$("browse-btn").addEventListener("click", (e) => {
  e.preventDefault();
  browseInput.click();
});
browseInput.addEventListener("change", () => {
  const f = browseInput.files[0];
  if (f) doImageSearch(f);
  browseInput.value = ""; // allow re-selecting the same file
});

// ── Tray ──────────────────────────────────────────────────────────────────────

function addToTray(row) {
  if (tray.includes(row)) return false;
  tray.push(row);
  localStorage.setItem("pixgrep-tray", JSON.stringify(tray));
  renderTray();
  return true;
}

function removeFromTray(row) {
  tray = tray.filter((r) => r !== row);
  localStorage.setItem("pixgrep-tray", JSON.stringify(tray));
  renderTray();
}

function renderTray() {
  const trayEl = $("tray");
  if (!tray.length) { trayEl.classList.add("hidden"); return; }
  trayEl.classList.remove("hidden");

  const thumbsEl = $("tray-thumbs");
  thumbsEl.replaceChildren();
  for (const row of tray) {
    const wrap = document.createElement("div");
    wrap.className = "tray-thumb-wrap";

    const img = document.createElement("img");
    img.src = `/api/thumb/${row}`;
    img.alt = "";
    img.className = "tray-thumb";

    const rm = document.createElement("button");
    rm.className = "tray-thumb-remove";
    rm.textContent = "×";
    rm.title = "Remove";
    rm.addEventListener("click", () => removeFromTray(row));

    wrap.appendChild(img);
    wrap.appendChild(rm);
    thumbsEl.appendChild(wrap);
  }

  const n = tray.length;
  $("tray-count").textContent = `${n} slide${n === 1 ? "" : "s"}`;
}

async function downloadPptx() {
  const btn = $("tray-download");
  const msg = $("tray-msg");
  btn.disabled = true;
  btn.textContent = "Generating…";
  msg.textContent = "";
  try {
    const r = await fetch("/api/export/pptx", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({
        rows: tray,
        layout: $("tray-layout").value,
        captions: $("tray-captions").checked,
      }),
    });
    if (!r.ok) {
      const j = await r.json().catch(() => ({}));
      msg.textContent = `Export failed: ${j.detail || r.status}`;
      return;
    }
    const blob = await r.blob();
    const url = URL.createObjectURL(blob);
    const a = document.createElement("a");
    a.href = url;
    a.download = "pixgrep-export.pptx";
    a.click();
    URL.revokeObjectURL(url);
  } catch {
    msg.textContent = "Network error during export.";
  } finally {
    btn.disabled = false;
    btn.textContent = "Download PPTX";
  }
}

$("lb-add-deck").addEventListener("click", () => {
  if (!current) return;
  const btn = $("lb-add-deck");
  const added = addToTray(current.row);
  const orig = btn.textContent;
  btn.textContent = added ? "Added!" : "Already in deck";
  setTimeout(() => { btn.textContent = orig; }, 900);
});

$("tray-download").addEventListener("click", downloadPptx);
$("tray-clear").addEventListener("click", () => {
  tray = [];
  localStorage.setItem("pixgrep-tray", JSON.stringify(tray));
  renderTray();
});

// Restore tray from localStorage on load
try { tray = JSON.parse(localStorage.getItem("pixgrep-tray") || "[]"); } catch { tray = []; }
renderTray();

// ── User identity ────────────────────────────────────────────────────────────

function renderUserBadge() {
  const badge = $("user-badge");
  if (currentUser) {
    $("user-badge-btn").textContent = `▾ ${currentUser}`;
    badge.classList.remove("hidden");
  } else {
    badge.classList.add("hidden");
  }
}

function openUserPrompt() {
  $("user-name-input").value = currentUser;
  $("user-prompt").classList.remove("hidden");
  $("user-name-input").focus();
}

function closeUserPrompt() {
  $("user-prompt").classList.add("hidden");
}

function saveUserName() {
  const name = $("user-name-input").value.trim().slice(0, 40);
  if (!name) return;
  currentUser = name;
  localStorage.setItem(USER_KEY, currentUser);
  renderUserBadge();
  closeUserPrompt();
  if (pendingFeedback !== null) {
    const row = pendingFeedback;
    pendingFeedback = null;
    toggleFeedback(row);
  }
}

$("user-badge-btn").addEventListener("click", () => {
  pendingFeedback = null;
  openUserPrompt();
});
$("user-name-save").addEventListener("click", saveUserName);
$("user-name-input").addEventListener("keydown", (e) => { if (e.key === "Enter") saveUserName(); });
document.addEventListener("click", (e) => {
  const prompt = $("user-prompt");
  if (prompt.classList.contains("hidden")) return;
  if (prompt.contains(e.target) || e.target === $("user-badge-btn")) return;
  closeUserPrompt();
});

renderUserBadge();
setFiltersOpen(localStorage.getItem(FILTERS_OPEN_KEY) === "1");

meta();
loadFacets();
