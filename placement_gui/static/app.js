"use strict";
// Frontend for Anforderungsprofil 1. Fetches the precomputed pattern from the
// server, draws a top-down view of the box + parts, and lets the operator
// nudge the whole pattern via X/Y offset sliders and switch between recipes.

const canvas = document.getElementById("canvas");
const offsetXEl = document.getElementById("offset-x");
const offsetYEl = document.getElementById("offset-y");
const offsetXValue = document.getElementById("offset-x-value");
const offsetYValue = document.getElementById("offset-y-value");
const countEl = document.getElementById("count");
const metaEl = document.getElementById("meta");
const recipeButtons = document.getElementById("recipe-buttons");
const demoBadge = document.getElementById("demo-badge");

let recipes = [];
let currentRecipe = null;
let pattern = null;

async function fetchJson(url) {
  const response = await fetch(url);
  const payload = await response.json();
  if (!response.ok) throw new Error(payload.error || `HTTP ${response.status}`);
  return payload;
}

async function loadStatus() {
  try {
    const s = await fetchJson("/api/status");
    demoBadge.hidden = !s.demo;
    demoBadge.title = s.demo
      ? "Showing built-in sample recipes — no database connected."
      : "";
  } catch { /* status is optional; ignore if unavailable */ }
}

async function loadRecipes() {
  try {
    recipes = await fetchJson("/api/recipes");
  } catch (error) {
    metaEl.textContent = `Could not load recipes: ${error.message}`;
    return;
  }
  recipeButtons.innerHTML = "";
  recipes.forEach(r => {
    const b = document.createElement("button");
    b.textContent = r.name;
    b.dataset.id = r.id;
    b.onclick = () => selectRecipe(r.id);
    recipeButtons.appendChild(b);
  });
  if (recipes.length) selectRecipe(recipes[0].id);
}

async function selectRecipe(id) {
  currentRecipe = id;
  [...recipeButtons.children].forEach(b =>
    b.classList.toggle("active", b.dataset.id === id));
  // New recipe -> reset offset to the default (centred) pattern.
  offsetXEl.value = 0;
  offsetYEl.value = 0;
  await refreshPattern();
  if (!pattern) return;
  // Bound the sliders to the recipe's allowed offset range.
  const m = pattern.offset_range;
  offsetXEl.min = -m.max_x; offsetXEl.max = m.max_x;
  offsetYEl.min = -m.max_y; offsetYEl.max = m.max_y;
}

async function refreshPattern() {
  const url = `/api/pattern?recipe=${encodeURIComponent(currentRecipe)}`
            + `&offset_x=${offsetXEl.value}&offset_y=${offsetYEl.value}`;
  try {
    pattern = await fetchJson(url);
  } catch (error) {
    countEl.textContent = "–";
    metaEl.textContent =
      `Server unreachable (${error.message}) — is app.py still running?`;
    return;
  }
  countEl.textContent = pattern.count;
  offsetXValue.textContent = `${Math.round(pattern.offset.x)} mm`;
  offsetYValue.textContent = `${Math.round(pattern.offset.y)} mm`;
  metaEl.innerHTML =
    `Box: ${pattern.box.width} × ${pattern.box.height} mm<br>` +
    `Part: ${pattern.part.width} × ${pattern.part.height} mm<br>` +
    `Grid: ${pattern.columns} × ${pattern.rows}`;
  draw();
}

function draw() {
  // Fit the canvas to its CSS box at device resolution. Skip drawing while
  // the canvas has no usable size (e.g. during layout).
  const dpr = window.devicePixelRatio || 1;
  const cw = canvas.clientWidth, ch = canvas.clientHeight;
  const pad = 30;
  if (!pattern || cw <= 2 * pad || ch <= 2 * pad) return;
  canvas.width = cw * dpr; canvas.height = ch * dpr;
  const ctx = canvas.getContext("2d");
  ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
  ctx.clearRect(0, 0, cw, ch);

  const box = pattern.box;
  const scale = Math.min((cw - 2 * pad) / box.width,
                         (ch - 2 * pad) / box.height);
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

  // Parts — drawn in the shape of the real black rail parts from the
  // requirements doc. See drawPart().
  pattern.placements.forEach((p) => {
    drawPart(ctx, X(p.x), Y(p.y + p.height), p.width * scale, p.height * scale);
  });
}

// Draw one part inside its bounding box (canvas px). The part runs along its
// longer edge, with a wider "head" block (two square holes) at one end and a
// glossy sheen + groove along the body — a stylised top-down view of the black
// parts shown in the requirements doc.
function drawPart(ctx, px, py, pw, ph) {
  const horizontal = pw >= ph;
  const L = horizontal ? pw : ph;   // length (along the long edge)
  const T = horizontal ? ph : pw;   // thickness (across)

  ctx.save();
  ctx.translate(px + pw / 2, py + ph / 2);
  if (!horizontal) ctx.rotate(Math.PI / 2);  // draw horizontally, then rotate
  drawRail(ctx, L, T);
  ctx.restore();
}

// Rail centred at the current origin: length L along x, thickness T along y.
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

function roundRect(ctx, x, y, w, h, r) {
  // Clamp the radius so degenerate sizes can never produce a negative value,
  // which would make arcTo() throw and kill the calling code path.
  const radius = Math.max(0, Math.min(r, Math.abs(w) / 2, Math.abs(h) / 2));
  ctx.beginPath();
  ctx.moveTo(x + radius, y);
  ctx.arcTo(x + w, y, x + w, y + h, radius);
  ctx.arcTo(x + w, y + h, x, y + h, radius);
  ctx.arcTo(x, y + h, x, y, radius);
  ctx.arcTo(x, y, x + w, y, radius);
  ctx.closePath();
}

// Live drag without hammering the server: redraw locally, fetch on input.
let pending = false;
function onSlide() {
  offsetXValue.textContent = `${Math.round(offsetXEl.value)} mm`;
  offsetYValue.textContent = `${Math.round(offsetYEl.value)} mm`;
  if (pending) return;
  pending = true;
  requestAnimationFrame(async () => { pending = false; await refreshPattern(); });
}
offsetXEl.addEventListener("input", onSlide);
offsetYEl.addEventListener("input", onSlide);
document.getElementById("reset").onclick = () => {
  offsetXEl.value = 0; offsetYEl.value = 0; refreshPattern();
};
window.addEventListener("resize", draw);

loadStatus();
loadRecipes();
