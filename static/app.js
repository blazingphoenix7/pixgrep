const $ = (id) => document.getElementById(id);
const grid = $("grid"), status = $("status");
let current = null; // row shown in lightbox
const activeFilters = {}; // {field: value}

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

function render(results) {
  grid.innerHTML = "";
  if (!results.length) { status.textContent = "No matches."; return; }
  status.textContent = `${results.length} match${results.length === 1 ? "" : "es"}`;
  for (const r of results) {
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
    grid.appendChild(card);
  }
}

async function doSearch() {
  const q = $("q").value.trim();
  if (!q) return;
  status.textContent = "Searching…";
  const url = new URL("/api/search", location.origin);
  url.searchParams.set("q", q);
  url.searchParams.set("k", "48");
  appendFilters(url);
  const r = await fetch(url);
  if (!r.ok) { status.textContent = "Search failed."; return; }
  render((await r.json()).results);
}

async function doImageSearch(file) {
  status.textContent = "Searching by image…";
  const url = new URL("/api/search/image", location.origin);
  url.searchParams.set("k", "48");
  appendFilters(url);
  const fd = new FormData();
  fd.append("file", file);
  const r = await fetch(url, { method: "POST", body: fd });
  if (!r.ok) { status.textContent = "Could not read that image."; return; }
  render((await r.json()).results);
}

async function doSimilar(row) {
  status.textContent = "Finding similar…";
  const url = new URL(`/api/similar/${row}`, location.origin);
  url.searchParams.set("k", "48");
  appendFilters(url);
  const r = await fetch(url);
  if (!r.ok) { status.textContent = "Failed."; return; }
  render((await r.json()).results);
}

function openLightbox(r) {
  current = r;
  $("lb-img").src = `/api/image/${r.row}`;
  $("lb-name").textContent = r.name;
  $("lb-path").textContent = r.path;
  $("lightbox").classList.remove("hidden");
}

$("go").addEventListener("click", doSearch);
$("q").addEventListener("keydown", (e) => { if (e.key === "Enter") doSearch(); });
$("lb-close").addEventListener("click", () => $("lightbox").classList.add("hidden"));
$("lightbox").addEventListener("click", (e) => {
  if (e.target === $("lightbox")) $("lightbox").classList.add("hidden");
});
$("lb-copy").addEventListener("click", () => {
  if (current) navigator.clipboard.writeText(current.path);
});
$("lb-similar").addEventListener("click", () => {
  if (current) { $("lightbox").classList.add("hidden"); doSimilar(current.row); }
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

meta();
loadFacets();
