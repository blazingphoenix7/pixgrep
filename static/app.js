const $ = (id) => document.getElementById(id);
const grid = $("grid"), status = $("status");
let current = null; // row shown in lightbox
const activeFilters = {}; // {field: value}
let isSearching = false;
let tray = []; // ordered row integers for the export deck

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
    container.classList.remove("hidden");
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
}

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

    return card;
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
    render((await r.json()).results);
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
  const strip = $("lb-filmstrip");
  for (const thumb of strip.querySelectorAll(".filmstrip-thumb")) {
    thumb.classList.toggle("active", parseInt(thumb.dataset.row) === m.row);
  }
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
  if (e.key === "Escape") closeLightbox();
});
$("lb-copy").addEventListener("click", () => {
  if (current) navigator.clipboard.writeText(current.path);
});
$("lb-similar").addEventListener("click", () => {
  if (current) { closeLightbox(); doSimilar(current.row); }
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

meta();
loadFacets();
