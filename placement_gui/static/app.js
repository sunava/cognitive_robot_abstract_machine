"use strict";
// Frontend for the placement-pattern GUI.
//
// Tab "Placement": fetches the computed placements of a stored pattern from
// the server, draws a top-down view of the box + shapes, and lets the operator
// nudge the whole pattern via X/Y offset sliders.
//
// Tab "Pattern Editor": lists the shapes from the shape catalog, lets the
// operator pick one and define a new pattern (box size, rows/columns/gap) with
// a live preview, then saves it to the data source.

// ---------------------------------------------------------------------------
// Shared drawing helpers (used by both tabs and the shape previews)
// ---------------------------------------------------------------------------

function roundRect(ctx, x, y, w, h, r) {
  ctx.beginPath();
  ctx.moveTo(x + r, y);
  ctx.arcTo(x + w, y, x + w, y + h, r);
  ctx.arcTo(x + w, y + h, x, y + h, r);
  ctx.arcTo(x, y + h, x, y, r);
  ctx.arcTo(x, y, x + w, y, r);
  ctx.closePath();
}

// Rail centred at the current origin: length L along x, thickness T along y.
// A stylised top-down view of the black rail/bracket parts: long glossy dark
// body, wider head block with two holes at one end, groove down the length.
function drawRail(ctx, L, T) {
  const x0 = -L / 2, y0 = -T / 2;
  const bodyT = T * 0.60;            // slim shaft, thinner than the head
  const rBody = Math.min(bodyT / 2, 6);
  const headW = Math.min(L * 0.24, T * 1.8);  // wider end block

  // Glossy dark plastic/metal gradient (light sheen on top -> near-black).
  const g = ctx.createLinearGradient(0, y0, 0, y0 + T);
  g.addColorStop(0, "#42434a");
  g.addColorStop(0.45, "#17181c");
  g.addColorStop(1, "#050506");

  ctx.shadowColor = "rgba(0, 0, 0, 0.55)";
  ctx.shadowBlur = 10;
  ctx.fillStyle = g;

  // Shaft.
  roundRect(ctx, x0, -bodyT / 2, L, bodyT, rBody);
  ctx.fill();
  // Head block.
  ctx.shadowBlur = 8;
  roundRect(ctx, x0, y0, headW, T, Math.min(6, T / 4));
  ctx.fill();
  ctx.shadowBlur = 0;

  // Two square mounting holes in the head.
  const hole = Math.min(headW * 0.30, T * 0.24);
  const hx = x0 + headW * 0.5 - hole / 2;
  ctx.fillStyle = "rgba(0, 0, 0, 0.85)";
  [-T * 0.22, T * 0.22].forEach((cy) => {
    roundRect(ctx, hx, cy - hole / 2, hole, hole, 2);
    ctx.fill();
  });

  // Specular sheen streak along the shaft (glossy highlight).
  ctx.strokeStyle = "rgba(190, 200, 220, 0.32)";
  ctx.lineWidth = Math.max(1, bodyT * 0.10);
  ctx.beginPath();
  ctx.moveTo(x0 + headW, -bodyT * 0.18);
  ctx.lineTo(x0 + L - rBody, -bodyT * 0.18);
  ctx.stroke();

  // Groove line down the length.
  ctx.strokeStyle = "rgba(0, 0, 0, 0.5)";
  ctx.lineWidth = Math.max(1, bodyT * 0.06);
  ctx.beginPath();
  ctx.moveTo(x0 + headW, bodyT * 0.14);
  ctx.lineTo(x0 + L - rBody, bodyT * 0.14);
  ctx.stroke();

  // Faint edge outline to lift it off the box floor.
  ctx.strokeStyle = "rgba(255, 255, 255, 0.10)";
  ctx.lineWidth = 1;
  roundRect(ctx, x0, -bodyT / 2, L, bodyT, rBody);
  ctx.stroke();
}

// Draw one shape inside its bounding box (canvas px). The shape runs along its
// longer edge; vertical placements are drawn horizontally and rotated.
function drawShape(ctx, px, py, pw, ph) {
  const horizontal = pw >= ph;
  const L = horizontal ? pw : ph;   // length (along the long edge)
  const T = horizontal ? ph : pw;   // thickness (across)

  ctx.save();
  ctx.translate(px + pw / 2, py + ph / 2);
  if (!horizontal) ctx.rotate(Math.PI / 2);
  drawRail(ctx, L, T);
  ctx.restore();
}

// Draw a full placements result (box + shapes) onto a canvas.
function drawPlacements(canvas, result) {
  const ctx = canvas.getContext("2d");
  const dpr = window.devicePixelRatio || 1;
  const cw = canvas.clientWidth, ch = canvas.clientHeight;
  canvas.width = cw * dpr; canvas.height = ch * dpr;
  ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
  ctx.clearRect(0, 0, cw, ch);
  if (!result) return;

  const box = result.box;
  const pad = 30;
  const scale = Math.min((cw - 2 * pad) / box.width, (ch - 2 * pad) / box.height);
  const ox = (cw - box.width * scale) / 2;
  const oy = (ch - box.height * scale) / 2;

  // Box coords have origin lower-left; canvas y grows downward -> flip y.
  const X = x => ox + x * scale;
  const Y = y => oy + (box.height - y) * scale;

  // Box wall — frosted glass surface.
  ctx.save();
  ctx.fillStyle = "rgba(255, 255, 255, 0.05)";
  ctx.strokeStyle = "rgba(255, 255, 255, 0.22)";
  ctx.lineWidth = 2;
  roundRect(ctx, X(0), Y(box.height), box.width * scale, box.height * scale, 16);
  ctx.fill();
  ctx.stroke();
  ctx.restore();

  result.placements.forEach((p) => {
    drawShape(ctx, X(p.x), Y(p.y + p.height), p.width * scale, p.height * scale);
  });
}

// ---------------------------------------------------------------------------
// Header: status + tab switching
// ---------------------------------------------------------------------------

const demoBadge = document.getElementById("demo-badge");
const tabButtonPlacement = document.getElementById("tab-button-placement");
const tabButtonEditor = document.getElementById("tab-button-editor");
const tabPlacement = document.getElementById("tab-placement");
const tabEditor = document.getElementById("tab-editor");

async function loadStatus() {
  try {
    const s = await (await fetch("/api/status")).json();
    demoBadge.hidden = !s.demo;
    demoBadge.title = s.demo
      ? "Showing built-in sample data — no database connected."
      : "";
  } catch { /* status is optional; ignore if unavailable */ }
}

function showTab(editor) {
  tabPlacement.hidden = editor;
  tabEditor.hidden = !editor;
  tabButtonPlacement.classList.toggle("active", !editor);
  tabButtonEditor.classList.toggle("active", editor);
  // Canvases have zero size while hidden; redraw the one that just appeared.
  if (editor) refreshPreview(); else placements && drawPlacements(canvas, placements);
}
tabButtonPlacement.onclick = () => showTab(false);
tabButtonEditor.onclick = () => showTab(true);

// ---------------------------------------------------------------------------
// Tab 1: placement view
// ---------------------------------------------------------------------------

const canvas = document.getElementById("canvas");
const offsetXEl = document.getElementById("offset-x");
const offsetYEl = document.getElementById("offset-y");
const offsetXValue = document.getElementById("offset-x-value");
const offsetYValue = document.getElementById("offset-y-value");
const countEl = document.getElementById("count");
const metaEl = document.getElementById("meta");
const patternButtons = document.getElementById("pattern-buttons");

let patterns = [];
let currentPattern = null;
let placements = null;

async function loadPatterns(selectId) {
  patterns = await (await fetch("/api/patterns")).json();
  patternButtons.innerHTML = "";
  patterns.forEach(p => {
    const b = document.createElement("button");
    b.textContent = p.name;
    b.dataset.id = p.id;
    b.onclick = () => selectPattern(p.id);
    patternButtons.appendChild(b);
  });
  const initial = selectId || (patterns.length ? patterns[0].id : null);
  if (initial) await selectPattern(initial);
}

async function selectPattern(id) {
  currentPattern = id;
  [...patternButtons.children].forEach(b =>
    b.classList.toggle("active", b.dataset.id === id));
  // New pattern -> reset offset to the default (centred) placements.
  offsetXEl.value = 0;
  offsetYEl.value = 0;
  await refreshPlacements();
  // Bound the sliders to the pattern's allowed offset range.
  const m = placements.offset_range;
  offsetXEl.min = -m.max_x; offsetXEl.max = m.max_x;
  offsetYEl.min = -m.max_y; offsetYEl.max = m.max_y;
}

async function refreshPlacements() {
  const url = `/api/placements?pattern=${encodeURIComponent(currentPattern)}`
            + `&offset_x=${offsetXEl.value}&offset_y=${offsetYEl.value}`;
  placements = await (await fetch(url)).json();
  countEl.textContent = placements.count;
  offsetXValue.textContent = `${Math.round(placements.offset.x)} mm`;
  offsetYValue.textContent = `${Math.round(placements.offset.y)} mm`;
  metaEl.innerHTML =
    `Shape: ${placements.shape.name} `
    + `(${placements.shape.width} × ${placements.shape.height} mm)<br>`
    + `Box: ${placements.box.width} × ${placements.box.height} mm<br>`
    + `Grid: ${placements.columns} × ${placements.rows}`;
  drawPlacements(canvas, placements);
}

// Live drag without hammering the server: redraw locally, fetch on input.
let pendingPlacements = false;
function onSlide() {
  offsetXValue.textContent = `${Math.round(offsetXEl.value)} mm`;
  offsetYValue.textContent = `${Math.round(offsetYEl.value)} mm`;
  if (pendingPlacements) return;
  pendingPlacements = true;
  requestAnimationFrame(async () => {
    pendingPlacements = false;
    await refreshPlacements();
  });
}
offsetXEl.addEventListener("input", onSlide);
offsetYEl.addEventListener("input", onSlide);
document.getElementById("reset").onclick = () => {
  offsetXEl.value = 0; offsetYEl.value = 0; refreshPlacements();
};

// ---------------------------------------------------------------------------
// Tab 2: pattern editor (shape catalog + live preview + save)
// ---------------------------------------------------------------------------

const editorCanvas = document.getElementById("editor-canvas");
const shapeList = document.getElementById("shape-list");
const editorCount = document.getElementById("editor-count");
const patternNameEl = document.getElementById("pattern-name");
const boxWidthEl = document.getElementById("box-width");
const boxHeightEl = document.getElementById("box-height");
const rowsEl = document.getElementById("rows");
const columnsEl = document.getElementById("columns");
const gapEl = document.getElementById("gap");
const saveButton = document.getElementById("save-pattern");
const editorMessage = document.getElementById("editor-message");

let shapes = [];
let selectedShape = null;

async function loadShapes() {
  shapes = await (await fetch("/api/shapes")).json();
  shapeList.innerHTML = "";
  shapes.forEach(shape => {
    const card = document.createElement("button");
    card.className = "shape-card";
    card.dataset.id = shape.id;

    const preview = document.createElement("canvas");
    preview.className = "shape-preview";
    card.appendChild(preview);

    const label = document.createElement("div");
    label.className = "shape-label";
    label.innerHTML = `${shape.name}<span>${shape.width} × ${shape.height} mm</span>`;
    card.appendChild(label);

    card.onclick = () => selectShape(shape.id);
    shapeList.appendChild(card);
    drawShapePreview(preview, shape);
  });
  if (shapes.length) selectShape(shapes[0].id);
}

function drawShapePreview(preview, shape) {
  const ctx = preview.getContext("2d");
  const dpr = window.devicePixelRatio || 1;
  const cw = preview.clientWidth || 200, ch = preview.clientHeight || 56;
  preview.width = cw * dpr; preview.height = ch * dpr;
  ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
  const scale = Math.min((cw - 16) / shape.width, (ch - 12) / shape.height, 1);
  drawShape(ctx,
    (cw - shape.width * scale) / 2, (ch - shape.height * scale) / 2,
    shape.width * scale, shape.height * scale);
}

function selectShape(id) {
  selectedShape = id;
  [...shapeList.children].forEach(card =>
    card.classList.toggle("active", card.dataset.id === id));
  refreshPreview();
}

// Live preview: same server-side geometry as the placement view, computed for
// the (unsaved) pattern currently in the form.
let pendingPreview = false;
async function refreshPreview() {
  if (!selectedShape) return;
  if (pendingPreview) return;
  pendingPreview = true;
  requestAnimationFrame(async () => {
    pendingPreview = false;
    const url = `/api/preview?shape=${encodeURIComponent(selectedShape)}`
              + `&box_width=${boxWidthEl.value}&box_height=${boxHeightEl.value}`
              + `&rows=${rowsEl.value}&columns=${columnsEl.value}&gap=${gapEl.value}`;
    const result = await (await fetch(url)).json();
    if (result.error) { editorCount.textContent = "–"; return; }
    editorCount.textContent = result.count;
    drawPlacements(editorCanvas, result);
  });
}
[boxWidthEl, boxHeightEl, rowsEl, columnsEl, gapEl].forEach(el =>
  el.addEventListener("input", refreshPreview));

function showEditorMessage(text, isError) {
  editorMessage.textContent = text;
  editorMessage.classList.toggle("error", !!isError);
  editorMessage.classList.add("visible");
  setTimeout(() => editorMessage.classList.remove("visible"), 4000);
}

saveButton.onclick = async () => {
  if (!selectedShape) return;
  const name = patternNameEl.value.trim();
  if (!name) {
    showEditorMessage("Please enter a pattern name.", true);
    patternNameEl.focus();
    return;
  }
  const body = {
    name,
    shape_id: selectedShape,
    box: { width: +boxWidthEl.value, height: +boxHeightEl.value },
    rows: +rowsEl.value,
    columns: +columnsEl.value,
    gap: +gapEl.value,
  };
  const response = await fetch("/api/patterns", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(body),
  });
  const result = await response.json();
  if (!response.ok) {
    showEditorMessage(result.error || "Saving failed.", true);
    return;
  }
  showEditorMessage(`Saved "${result.name}" ✓`);
  await loadPatterns(result.id);
};

// ---------------------------------------------------------------------------

window.addEventListener("resize", () => {
  if (!tabPlacement.hidden && placements) drawPlacements(canvas, placements);
  if (!tabEditor.hidden) refreshPreview();
});

loadStatus();
loadPatterns();
loadShapes();
