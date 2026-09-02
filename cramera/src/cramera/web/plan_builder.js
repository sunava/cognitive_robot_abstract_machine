/* ============================================================================
 * plan_builder.js — the Plan Builder page: compose a plan by drag-and-drop and
 * place objects in a top-down scene, then generate a runnable coraplex demo file.
 * A page script (owns the whole document), like models_page.js.
 * ==========================================================================*/
(function () {
  'use strict';

  // ---- available object meshes (coraplex/resources/objects) ----
  const MESHES = ['milk.stl', 'bowl.stl', 'spoon.stl', 'breakfast_cereal.stl', 'jeroen_cup.stl',
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
  let objects = [];      // [{id, mesh, name, x, y, z, yaw, color}]
  let objSeq = 1, stepSeq = 1;
  let robotXY = { x: 1.5, y: 2.5 };   // robot spawn (draggable in the scene)

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
    const o = { id: 'o' + (objSeq++), mesh: mesh, name: mesh,
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
        '<span class="oname" title="' + o.mesh + '">' + o.name + '</span>' +
        '<span class="ocap" data-cap="' + o.id + '" title="capture its pose from the live 3D scene">⟳ capture</span>' +
        '<span class="odel" data-del="' + o.id + '">×</span></div>' +
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
    el.querySelectorAll('.ocap').forEach(function (x) {
      x.addEventListener('click', function () { captureObject(x.dataset.cap); });
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
    if (!sc) return;   // the 2D scene was replaced by the live 3D capture flow
    sc.querySelectorAll('.pb-marker,.pb-tmarker').forEach(function (m) { m.remove(); });
    // robot spawn (draggable)
    const rp = worldToPx(robotXY.x, robotXY.y);
    const rm = document.createElement('div'); rm.className = 'pb-marker pb-rm';
    rm.style.left = rp.px + 'px'; rm.style.top = rp.py + 'px'; rm.style.background = '#38405c'; rm.style.fontSize = '13px';
    rm.innerHTML = '<span class="lbl">robot</span>🤖';
    rm.addEventListener('mousedown', function (e) {
      dragMarker(e, rm, function (x, y) { robotXY.x = x; robotXY.y = y; }, function () {});
    });
    sc.appendChild(rm);
    // objects (draggable)
    objects.forEach(function (o) {
      const p = worldToPx(o.x, o.y);
      const m = document.createElement('div'); m.className = 'pb-marker';
      m.style.left = p.px + 'px'; m.style.top = p.py + 'px'; m.style.background = o.color;
      m.innerHTML = '<span class="lbl">' + o.name.replace(/\.stl$/i, '') + '</span>' + o.name.charAt(0).toUpperCase();
      m.addEventListener('mousedown', function (e) {
        dragMarker(e, m, function (x, y) { o.x = x; o.y = y; syncNum(o.id); }, function () { renderObjects(); });
      });
      sc.appendChild(m);
    });
    // transport destinations (ghost target, draggable) — "where the object should go"
    steps.forEach(function (s, i) {
      if (s.type !== 'transport') return;
      const p = worldToPx(s.params.x, s.params.y);
      const m = document.createElement('div'); m.className = 'pb-marker pb-tmarker';
      m.style.left = p.px + 'px'; m.style.top = p.py + 'px';
      m.innerHTML = '<span class="lbl">→ ' + (s.params.object ? s.params.object.replace(/\.stl$/i, '') : 'step ' + (i + 1)) + '</span>◎';
      m.addEventListener('mousedown', function (e) {
        dragMarker(e, m, function (x, y) { s.params.x = x; s.params.y = y; syncStepNum(s.id); }, function () { renderSteps(); });
      });
      sc.appendChild(m);
    });
  }
  function dragMarker(e, m, apply, onEnd) {
    e.preventDefault();
    const sc = $('pb-scene');
    function move(ev) {
      const r = sc.getBoundingClientRect();
      const px = Math.max(0, Math.min(r.width, ev.clientX - r.left));
      const py = Math.max(0, Math.min(r.height, ev.clientY - r.top));
      m.style.left = px + 'px'; m.style.top = py + 'px';
      const w = pxToWorld(px, py); apply(Math.round(w.x * 100) / 100, Math.round(w.y * 100) / 100);
    }
    function up() { document.removeEventListener('mousemove', move); document.removeEventListener('mouseup', up); onEnd(); }
    document.addEventListener('mousemove', move); document.addEventListener('mouseup', up);
  }
  function syncNum(oid) {
    document.querySelectorAll('.pb-num[data-oid="' + oid + '"]').forEach(function (inp) {
      const o = objects.find(function (x) { return x.id === oid; }); if (!o) return;
      if (inp.dataset.k === 'x') inp.value = o.x; if (inp.dataset.k === 'y') inp.value = o.y;
    });
  }
  function syncStepNum(sid) {
    document.querySelectorAll('.pb-num[data-sid="' + sid + '"]').forEach(function (inp) {
      const s = steps.find(function (x) { return x.id === sid; }); if (!s) return;
      if (inp.dataset.k === 'x') inp.value = s.params.x; if (inp.dataset.k === 'y') inp.value = s.params.y;
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
    if (!steps.length) { el.innerHTML = '<div class="pb-drop-hint">Drop action blocks here to build the sequence</div>'; renderScene(); return; }
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
    renderScene();
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
    return '<label>object<select class="pb-sel" data-sid="' + s.id + '" data-k="object">' + (opts || '<option value="">— add an object —</option>') + '</select></label>' +
      '<label>target<button class="pb-capbtn" data-capstep="' + s.id + '" title="use the object\'s current pose in the live 3D scene as this step\'s target">◎ capture from 3D</button></label>';
  }
  function wireStepEvents() {
    const el = $('pb-steps');
    el.querySelectorAll('.pb-num,.pb-sel').forEach(function (inp) {
      inp.addEventListener('input', function () {
        const s = steps.find(function (x) { return x.id === inp.dataset.sid; }); if (!s) return;
        const v = inp.classList.contains('pb-num') ? (parseFloat(inp.value) || 0) : inp.value;
        s.params[inp.dataset.k] = v;
        renderScene();
      });
    });
    el.querySelectorAll('[data-del]').forEach(function (b) { b.addEventListener('click', function () { steps = steps.filter(function (s) { return s.id !== b.dataset.del; }); renderSteps(); }); });
    el.querySelectorAll('[data-up]').forEach(function (b) { b.addEventListener('click', function () { moveStep(b.dataset.up, -1); }); });
    el.querySelectorAll('[data-down]').forEach(function (b) { b.addEventListener('click', function () { moveStep(b.dataset.down, 1); }); });
    el.querySelectorAll('.pb-capbtn').forEach(function (b) { b.addEventListener('click', function (e) { e.preventDefault(); captureStepTarget(b.dataset.capstep); }); });
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
  function generate(stepsOverride) {
    const added = objects;
    const useSteps = stepsOverride || steps;
    const env = ($('pb-env') && $('pb-env').value) || 'apartment.urdf';
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
    L.push('from semantic_digital_twin.adapters.mesh import STLParser');
    L.push('from semantic_digital_twin.adapters.urdf import URDFParser');
    L.push('from semantic_digital_twin.datastructures.definitions import TorsoState');
    L.push('from semantic_digital_twin.reasoning.world_reasoner import WorldReasoner');
    L.push('from semantic_digital_twin.robots.pr2 import PR2');
    L.push('from semantic_digital_twin.spatial_types import HomogeneousTransformationMatrix');
    L.push('from semantic_digital_twin.spatial_types.spatial_types import Pose');
    L.push('from semantic_digital_twin.world_description.geometry import Color');
    L.push('');
    L.push('_HERE = os.path.dirname(__file__)');
    L.push('_WORLDS = os.path.join(_HERE, "..", "..", "resources", "worlds")');
    L.push('_OBJECTS = os.path.join(_HERE, "..", "..", "resources", "objects")');
    L.push('');
    L.push('');
    L.push('def build_world(env_file, robot_xy):');
    L.push('    """Parse the chosen environment + PR2 and spawn the robot at robot_xy."""');
    L.push('    robot_world = URDFParser.from_file(PR2.get_ros_file_path()).parse()');
    L.push('    world = URDFParser.from_file(os.path.join(_WORLDS, env_file)).parse()');
    L.push('    with world.modify_world():');
    L.push('        robot_root = robot_world.get_body_by_name(PR2._get_root_body_name())');
    L.push('        drive = PR2.get_drive_connection_type().create_with_dofs(');
    L.push('            parent=world.root, child=robot_root, world=world)');
    L.push('        world.merge_world(robot_world, drive)');
    L.push('        drive.origin = HomogeneousTransformationMatrix.from_xyz_rpy(robot_xy[0], robot_xy[1], 0)');
    L.push('    standing = max(0.0, -world.height_of_lowest_collision_point_of_branch(robot_root))');
    L.push('    with world.modify_world():');
    L.push('        drive.parent_T_connection_expression = HomogeneousTransformationMatrix.from_xyz_rpy(');
    L.push('            z=standing, reference_frame=world.root)');
    L.push('    return world');
    L.push('');
    L.push('');
    L.push('world = build_world("' + env + '", (' + py(robotXY.x) + ', ' + py(robotXY.y) + '))');
    L.push('visualization = WorldVisualization.from_environment(');
    L.push('    world, default_backend=VisualizationBackend.CRAMERA).start()');
    L.push('');
    if (added.length) {
      L.push('# --- objects placed in the Plan Builder ---');
      added.forEach(function (o, i) {
        L.push('_obj' + i + ' = STLParser(os.path.join(_OBJECTS, "' + o.mesh + '")).parse()');
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
    useSteps.forEach(function (s) { L.push('    ' + stepCode(s) + ','); });
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

  // ---------- live 3D capture ----------
  function bridgeUrl() { return 'http://' + window.location.hostname + ':8765'; }
  function quatToYaw(q) { // q = [qx,qy,qz,qw] -> yaw
    return Math.atan2(2 * (q[3] * q[2] + q[0] * q[1]), 1 - 2 * (q[1] * q[1] + q[2] * q[2]));
  }
  function liveStatus(msg, cls) { const el = $('pb-live-status'); el.textContent = msg; el.className = 'pb-live-status ' + (cls || ''); }
  function fetchCaptured() {
    return fetch(bridgeUrl() + '/captured_objects').then(function (r) { return r.json(); }).then(function (d) { return (d && d.objects) || {}; });
  }
  function poseFromCaptured(objs, mesh) {
    const p = objs[mesh]; if (!p || p.length < 7) return null;
    return { x: Math.round(p[0] * 100) / 100, y: Math.round(p[1] * 100) / 100, z: Math.round(p[2] * 100) / 100, yaw: Math.round(quatToYaw(p.slice(3)) * 1000) / 1000 };
  }
  function captureObject(oid) {
    const o = objects.find(function (x) { return x.id === oid; }); if (!o) return;
    fetchCaptured().then(function (objs) {
      const pz = poseFromCaptured(objs, o.mesh);
      if (!pz) { status('no live pose for ' + o.mesh + ' — is the scene running?', 'err'); return; }
      o.x = pz.x; o.y = pz.y; o.z = pz.z; o.yaw = pz.yaw; renderObjects();
      status('captured ' + o.name + ' → (' + pz.x + ', ' + pz.y + ', ' + pz.z + ')', 'ok');
    }).catch(function () { status('capture failed — start the live scene first', 'err'); });
  }
  function captureStepTarget(sid) {
    const s = steps.find(function (x) { return x.id === sid; }); if (!s) return;
    if (!s.params.object) { status('pick an object for this Transport step first', 'err'); return; }
    fetchCaptured().then(function (objs) {
      const pz = poseFromCaptured(objs, s.params.object);
      if (!pz) { status('no live pose for ' + s.params.object, 'err'); return; }
      s.params.x = pz.x; s.params.y = pz.y; s.params.z = pz.z; s.params.yaw = pz.yaw; renderSteps();
      status('captured target for ' + s.params.object + ' → (' + pz.x + ', ' + pz.y + ', ' + pz.z + ')', 'ok');
    }).catch(function () { status('capture failed — start the live scene first', 'err'); });
  }
  function startLive() {
    const code = generate([{ type: 'park_arms', params: { arm: 'BOTH' } }]);   // scaffold: world + objects, idle
    liveStatus('starting… (first run parses meshes, ~1 min)', '');
    fetch('/api/plan/scaffold', { method: 'POST', headers: { 'content-type': 'application/json' }, body: JSON.stringify({ code: code }) })
      .then(function (r) { return r.json(); })
      .then(function (j) { if (!j.ok) { liveStatus('failed: ' + (j.error || '?'), 'err'); return; } pollLive(0); })
      .catch(function (e) { liveStatus('failed: ' + e, 'err'); });
  }
  function pollLive(n) {
    fetch(bridgeUrl() + '/captured_objects').then(function (r) { return r.ok ? r.json() : null; })
      .then(function (d) {
        if (d) { liveStatus('● live — open the 3D view, click ◉ Live, drag objects, then capture', 'ok'); }
        else if (n < 40) { setTimeout(function () { pollLive(n + 1); }, 3000); }
        else liveStatus('scene did not come up — check the terminal', 'err');
      })
      .catch(function () { if (n < 40) setTimeout(function () { pollLive(n + 1); }, 3000); else liveStatus('scene did not come up', 'err'); });
  }
  function stopLive() {
    fetch('/api/plan/scaffold/stop', { method: 'POST' }).then(function () { liveStatus('stopped', ''); }).catch(function () {});
  }

  // ---------- boot ----------
  renderBlocks();
  $('pb-add-obj').addEventListener('click', function () { addObject($('pb-mesh').value); });
  $('pb-env').addEventListener('change', renderScene);
  $('pb-live-start').addEventListener('click', startLive);
  $('pb-live-stop').addEventListener('click', stopLive);
  $('pb-rx').addEventListener('input', function () { robotXY.x = parseFloat(this.value) || 0; });
  $('pb-ry').addEventListener('input', function () { robotXY.y = parseFloat(this.value) || 0; });
  addObject('milk.stl', { x: 2.5, y: 2.3, z: 0.9 });
  addObject('bowl.stl', { x: 2.4, y: 2.0, z: 0.95 });
  renderSteps();
  // a friendly starter plan
  addStep('park_arms'); addStep('move_torso');
  $('pb-generate').addEventListener('click', showCode);
  $('pb-download').addEventListener('click', download);
  $('pb-save').addEventListener('click', save);
  window.addEventListener('resize', renderScene);
})();
