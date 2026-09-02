/* ============================================================================
 * plan_builder.js — the Plan Builder page: compose a plan by drag-and-drop and
 * place objects in a top-down scene, then generate a runnable coraplex demo file.
 * A page script (owns the whole document), like models_page.js.
 * ==========================================================================*/
(function () {
  'use strict';

  // ---- available object meshes (coraplex/resources/objects) ----
  const MESHES = ['bowl.stl', 'spoon.stl', 'breakfast_cereal.stl', 'jeroen_cup.stl',
    'Static_CokeBottle.stl', 'big-knife.stl', 'whisk.stl', 'bread.stl', 'apartment_bowl.stl'];
  const OBJ_COLORS = ['#e6ecff', '#e6c07f', '#9aa1ad', '#8fd6c8', '#c9a0ff', '#ff9db1', '#9ecb6b'];

  // ---- action blocks ----
  const BLOCKS = {
    park_arms: { name: 'Park arms', color: '#b98cff', params: { arm: 'BOTH' } },
    move_torso: { name: 'Move torso', color: '#ff9db1', params: { torso: 'HIGH' } },
    navigate: { name: 'Navigate', color: '#8fd6c8', params: { x: 2.6, y: 1.8, z: 0.0, yaw: 0.0 } },
    transport: { name: 'Transport object', color: '#5b8cff', params: { object: '', x: 5.0, y: 3.3, z: 0.8, yaw: 1.57, arm: 'LEFT' } },
  };
  const ARMS = ['LEFT', 'RIGHT', 'BOTH'];
  const TORSO = ['HIGH', 'MID', 'LOW'];

  // ---- state ----
  let steps = [];       // [{type, params:{...}}]
  let objects = [];      // [{id, mesh, name, base, x, y, z, yaw, color}]
  let objSeq = 1, stepSeq = 1;

  // scene mapping: origin offset so the typical apartment area sits centred
  const SCALE = 40, ORIGIN_X = 2.5, ORIGIN_Y = 2.0;
  const $ = function (id) { return document.getElementById(id); };

  // ---------- palette ----------
  function renderBlocks() {
    const el = $('pb-blocks'); el.innerHTML = '';
    Object.keys(BLOCKS).forEach(function (k) {
      const b = BLOCKS[k];
      const d = document.createElement('div');
      d.className = 'pb-block'; d.draggable = true; d.dataset.block = k;
      d.innerHTML = '<span class="ic" style="background:' + b.color + '"></span>' + b.name;
      d.addEventListener('dragstart', function (e) { e.dataTransfer.setData('text/plain', 'block:' + k); });
      el.appendChild(d);
    });
    const meshSel = $('pb-mesh'); meshSel.innerHTML = MESHES.map(function (m) { return '<option>' + m + '</option>'; }).join('');
  }

  // ---------- objects ----------
  function addObject(mesh, opts) {
    opts = opts || {};
    const o = { id: 'o' + (objSeq++), mesh: mesh, name: mesh, base: !!opts.base,
      x: opts.x != null ? opts.x : 2.4, y: opts.y != null ? opts.y : 2.2,
      z: opts.z != null ? opts.z : 0.95, yaw: opts.yaw != null ? opts.yaw : 0.0,
      color: OBJ_COLORS[(objSeq) % OBJ_COLORS.length] };
    objects.push(o); renderObjects(); renderScene(); refreshObjectSelects();
    return o;
  }
  function renderObjects() {
    const el = $('pb-objects'); el.innerHTML = '';
    objects.forEach(function (o) {
      const d = document.createElement('div'); d.className = 'pb-obj';
      d.innerHTML =
        '<div class="row1"><span class="pb-swatch" style="background:' + o.color + '"></span>' +
        '<span class="oname" title="' + o.mesh + '">' + o.name + (o.base ? ' · base' : '') + '</span>' +
        (o.base ? '' : '<span class="odel" data-del="' + o.id + '">×</span>') + '</div>' +
        '<div class="fields">' +
        field(o, 'x') + field(o, 'y') + field(o, 'z') + field(o, 'yaw') + '</div>';
      el.appendChild(d);
    });
    el.querySelectorAll('.pb-num').forEach(function (inp) {
      inp.addEventListener('input', function () {
        const o = objects.find(function (x) { return x.id === inp.dataset.oid; });
        if (o) { o[inp.dataset.k] = parseFloat(inp.value) || 0; renderScene(); }
      });
    });
    el.querySelectorAll('.odel').forEach(function (x) {
      x.addEventListener('click', function () { objects = objects.filter(function (o) { return o.id !== x.dataset.del; }); renderObjects(); renderScene(); refreshObjectSelects(); });
    });
  }
  function field(o, k) {
    return '<label>' + k.toUpperCase() + '<input class="pb-num" data-oid="' + o.id + '" data-k="' + k + '" type="number" step="0.05" value="' + o[k] + '"' + (o.base && k !== 'yaw' ? '' : '') + '></label>';
  }

  // ---------- scene (top-down) ----------
  function worldToPx(x, y) {
    const sc = $('pb-scene'); const w = sc.clientWidth, h = sc.clientHeight;
    return { px: w / 2 + (x - ORIGIN_X) * SCALE, py: h / 2 - (y - ORIGIN_Y) * SCALE };
  }
  function pxToWorld(px, py) {
    const sc = $('pb-scene'); const w = sc.clientWidth, h = sc.clientHeight;
    return { x: ORIGIN_X + (px - w / 2) / SCALE, y: ORIGIN_Y - (py - h / 2) / SCALE };
  }
  function renderScene() {
    const sc = $('pb-scene');
    sc.querySelectorAll('.pb-marker').forEach(function (m) { m.remove(); });
    objects.forEach(function (o) {
      const p = worldToPx(o.x, o.y);
      const m = document.createElement('div'); m.className = 'pb-marker'; m.dataset.oid = o.id;
      m.style.left = p.px + 'px'; m.style.top = p.py + 'px'; m.style.background = o.color;
      m.innerHTML = '<span class="lbl">' + o.name.replace(/\.stl$/i, '') + '</span>' + o.name.charAt(0).toUpperCase();
      m.addEventListener('mousedown', function (e) { startDragMarker(e, o, m); });
      sc.appendChild(m);
    });
  }
  function startDragMarker(e, o, m) {
    e.preventDefault();
    const sc = $('pb-scene');
    function move(ev) {
      const r = sc.getBoundingClientRect();
      const px = Math.max(0, Math.min(r.width, ev.clientX - r.left));
      const py = Math.max(0, Math.min(r.height, ev.clientY - r.top));
      m.style.left = px + 'px'; m.style.top = py + 'px';
      const w = pxToWorld(px, py); o.x = Math.round(w.x * 100) / 100; o.y = Math.round(w.y * 100) / 100;
      syncObjectFields(o);
    }
    function up() { document.removeEventListener('mousemove', move); document.removeEventListener('mouseup', up); renderObjects(); }
    document.addEventListener('mousemove', move); document.addEventListener('mouseup', up);
  }
  function syncObjectFields(o) {
    document.querySelectorAll('.pb-num[data-oid="' + o.id + '"]').forEach(function (inp) {
      if (inp.dataset.k === 'x') inp.value = o.x; if (inp.dataset.k === 'y') inp.value = o.y;
    });
  }

  // ---------- plan steps ----------
  function addStep(type) {
    const b = BLOCKS[type]; if (!b) return;
    steps.push({ id: 's' + (stepSeq++), type: type, params: Object.assign({}, b.params) });
    renderSteps();
  }
  function renderSteps() {
    const el = $('pb-steps');
    $('pb-step-count').textContent = steps.length ? '(' + steps.length + ')' : '';
    if (!steps.length) { el.innerHTML = '<div class="pb-drop-hint">Drop action blocks here to build the sequence</div>'; return; }
    el.innerHTML = '';
    steps.forEach(function (s, i) {
      const b = BLOCKS[s.type];
      const d = document.createElement('div'); d.className = 'pb-step'; d.style.borderLeftColor = b.color;
      d.innerHTML =
        '<div class="sh"><span class="snum">' + (i + 1) + '</span><span class="sname">' + b.name + '</span>' +
        '<span class="sctl"><button data-up="' + s.id + '" title="Move up">↑</button>' +
        '<button data-down="' + s.id + '" title="Move down">↓</button>' +
        '<button data-del="' + s.id + '" title="Remove">×</button></span></div>' +
        '<div class="sparams">' + stepParams(s) + '</div>';
      el.appendChild(d);
    });
    wireStepEvents();
  }
  function stepParams(s) {
    const p = s.params;
    if (s.type === 'park_arms') return sel(s, 'arm', ARMS);
    if (s.type === 'move_torso') return sel(s, 'torso', TORSO);
    if (s.type === 'navigate') return num(s, 'x') + num(s, 'y') + num(s, 'z') + num(s, 'yaw');
    if (s.type === 'transport') return objSel(s) + num(s, 'x') + num(s, 'y') + num(s, 'z') + num(s, 'yaw') + sel(s, 'arm', ARMS);
    return '';
  }
  function num(s, k) { return '<label>' + k.toUpperCase() + '<input class="pb-num xyz" data-sid="' + s.id + '" data-k="' + k + '" type="number" step="0.05" value="' + s.params[k] + '"></label>'; }
  function sel(s, k, opts) { return '<label>' + k + '<select class="pb-sel" data-sid="' + s.id + '" data-k="' + k + '">' + opts.map(function (o) { return '<option' + (s.params[k] === o ? ' selected' : '') + '>' + o + '</option>'; }).join('') + '</select></label>'; }
  function objSel(s) {
    const opts = objects.map(function (o) { return '<option value="' + o.mesh + '"' + (s.params.object === o.mesh ? ' selected' : '') + '>' + o.name + '</option>'; }).join('');
    return '<label>object<select class="pb-sel" data-sid="' + s.id + '" data-k="object">' + (opts || '<option value="">— add an object —</option>') + '</select></label>';
  }
  function wireStepEvents() {
    const el = $('pb-steps');
    el.querySelectorAll('.pb-num,.pb-sel').forEach(function (inp) {
      inp.addEventListener('input', function () {
        const s = steps.find(function (x) { return x.id === inp.dataset.sid; }); if (!s) return;
        const v = inp.classList.contains('pb-num') ? (parseFloat(inp.value) || 0) : inp.value;
        s.params[inp.dataset.k] = v;
      });
    });
    el.querySelectorAll('[data-del]').forEach(function (b) { b.addEventListener('click', function () { steps = steps.filter(function (s) { return s.id !== b.dataset.del; }); renderSteps(); }); });
    el.querySelectorAll('[data-up]').forEach(function (b) { b.addEventListener('click', function () { moveStep(b.dataset.up, -1); }); });
    el.querySelectorAll('[data-down]').forEach(function (b) { b.addEventListener('click', function () { moveStep(b.dataset.down, 1); }); });
  }
  function moveStep(id, dir) {
    const i = steps.findIndex(function (s) { return s.id === id; }); const j = i + dir;
    if (i < 0 || j < 0 || j >= steps.length) return;
    const t = steps[i]; steps[i] = steps[j]; steps[j] = t; renderSteps();
  }
  function refreshObjectSelects() { renderSteps(); }

  // drop zone
  const stepsEl = $('pb-steps');
  stepsEl.addEventListener('dragover', function (e) { e.preventDefault(); stepsEl.classList.add('drop-ok'); });
  stepsEl.addEventListener('dragleave', function () { stepsEl.classList.remove('drop-ok'); });
  stepsEl.addEventListener('drop', function (e) {
    e.preventDefault(); stepsEl.classList.remove('drop-ok');
    const d = e.dataTransfer.getData('text/plain') || '';
    if (d.indexOf('block:') === 0) addStep(d.slice(6));
  });

  // ---------- code generation ----------
  function py(v) { return (Math.round(v * 1000) / 1000).toString(); }
  function pose(p) { return 'Pose.from_xyz_rpy(' + py(p.x) + ', ' + py(p.y) + ', ' + py(p.z) + ', yaw=' + py(p.yaw) + ', reference_frame=world.root)'; }
  function body(mesh) { return 'world.get_body_by_name("' + mesh + '")'; }
  function generate() {
    const added = objects.filter(function (o) { return !o.base; });
    const L = [];
    L.push('"""Generated by the cramera Plan Builder."""');
    L.push('import os');
    L.push('from coraplex.datastructures.dataclasses import Context');
    L.push('from coraplex.datastructures.enums import Arms, VisualizationBackend');
    L.push('from coraplex.execution_environment import simulated_robot');
    L.push('from coraplex.plans.factories import sequential');
    L.push('from coraplex.visualization import WorldVisualization');
    L.push('from coraplex.robot_plans.actions.composite.transporting import TransportAction');
    L.push('from coraplex.robot_plans.actions.core.navigation import NavigateAction');
    L.push('from coraplex.robot_plans.actions.core.robot_body import ParkArmsAction, MoveTorsoAction');
    L.push('from coraplex.testing import setup_world');
    L.push('from semantic_digital_twin.adapters.mesh import STLParser');
    L.push('from semantic_digital_twin.datastructures.definitions import TorsoState');
    L.push('from semantic_digital_twin.reasoning.world_reasoner import WorldReasoner');
    L.push('from semantic_digital_twin.robots.pr2 import PR2');
    L.push('from semantic_digital_twin.spatial_types import HomogeneousTransformationMatrix');
    L.push('from semantic_digital_twin.spatial_types.spatial_types import Pose');
    L.push('from semantic_digital_twin.world_description.geometry import Color');
    L.push('');
    L.push('world = setup_world()');
    L.push('visualization = WorldVisualization.from_environment(');
    L.push('    world, default_backend=VisualizationBackend.CRAMERA).start()');
    L.push('');
    if (added.length) {
      L.push('# --- objects placed in the Plan Builder ---');
      added.forEach(function (o, i) {
        L.push('_obj' + i + ' = STLParser(os.path.join(os.path.dirname(__file__),');
        L.push('    "..", "..", "resources", "objects", "' + o.mesh + '")).parse()');
      });
      L.push('with world.modify_world():');
      added.forEach(function (o, i) {
        L.push('    world.merge_world_at_pose(_obj' + i + ', HomogeneousTransformationMatrix.from_xyz_rpy(');
        L.push('        ' + py(o.x) + ', ' + py(o.y) + ', ' + py(o.z) + ', yaw=' + py(o.yaw) + ', reference_frame=world.root))');
      });
      added.forEach(function (o) {
        const c = hexToRgb(o.color);
        L.push(body(o.mesh) + '.visual.shapes[0].color = Color(' + c[0] + ', ' + c[1] + ', ' + c[2] + ')');
      });
      L.push('');
    }
    L.push('pr2 = PR2.from_world(world)');
    L.push('context = Context(world=world, robot=pr2, _debug=False, ros_node=visualization.ros_node)');
    L.push('with world.modify_world():');
    L.push('    WorldReasoner(world).reason()');
    L.push('context.evaluate_conditions = False');
    L.push('');
    L.push('plan = sequential([');
    steps.forEach(function (s) { L.push('    ' + stepCode(s) + ','); });
    L.push('], context=context).plan');
    L.push('visualization.attach_plan(plan)');
    L.push('');
    L.push('with simulated_robot:');
    L.push('    plan.perform()');
    L.push('');
    return L.join('\n');
  }
  function stepCode(s) {
    const p = s.params;
    if (s.type === 'park_arms') return 'ParkArmsAction(Arms.' + p.arm + ')';
    if (s.type === 'move_torso') return 'MoveTorsoAction(TorsoState.' + p.torso + ')';
    if (s.type === 'navigate') return 'NavigateAction(' + pose(p) + ')';
    if (s.type === 'transport') return 'TransportAction(' + body(p.object || 'object') + ', ' + pose(p) + ', Arms.' + p.arm + ')';
    return 'None';
  }
  function hexToRgb(h) { const n = parseInt(h.slice(1), 16); return [(n >> 16 & 255) / 255, (n >> 8 & 255) / 255, (n & 255) / 255].map(function (v) { return Math.round(v * 100) / 100; }); }

  function showCode() { $('pb-code').textContent = generate(); status('', ''); }
  function status(msg, cls) { const el = $('pb-status'); el.textContent = msg; el.className = 'pb-status ' + (cls || ''); }
  function fileName() { return (($('pb-name').value || 'my_demo').replace(/[^a-z0-9_\-]/gi, '_')) + '.py'; }

  function download() {
    const code = generate();
    const blob = new Blob([code], { type: 'text/x-python' });
    const a = document.createElement('a'); a.href = URL.createObjectURL(blob); a.download = fileName(); a.click();
    URL.revokeObjectURL(a.href); status('downloaded ' + fileName(), 'ok');
  }
  function save() {
    const code = generate();
    fetch('/api/plan/save', { method: 'POST', headers: { 'content-type': 'application/json' }, body: JSON.stringify({ name: fileName(), code: code }) })
      .then(function (r) { return r.json(); })
      .then(function (j) { if (j.ok) status('saved → ' + j.path + '  (run: cramera-live ' + j.path + ')', 'ok'); else status('save failed: ' + (j.error || '?'), 'err'); })
      .catch(function (e) { status('save failed: ' + e, 'err'); });
  }

  // ---------- boot ----------
  renderBlocks();
  addObject('milk.stl', { base: true, x: 2.5, y: 2.3, z: 0.9 });   // milk is already in the apartment
  addObject('bowl.stl', { x: 2.4, y: 2.0, z: 0.95 });
  renderSteps();
  // a friendly starter plan
  addStep('park_arms'); addStep('move_torso');
  $('pb-generate').addEventListener('click', showCode);
  $('pb-download').addEventListener('click', download);
  $('pb-save').addEventListener('click', save);
  window.addEventListener('resize', renderScene);
})();
