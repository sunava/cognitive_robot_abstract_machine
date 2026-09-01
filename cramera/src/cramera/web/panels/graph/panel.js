/* ============================================================================
 * panels/graph/panel.js — the graph view with five tabs.
 *
 *   Knowledge   the entity graph + CRAM architecture (double-click drills in)
 *   Kinematics  the robot's URDF tree (links as nodes, joints as edges)
 *   Plan        the executed plan tree, node border = execution status
 *   Statechart  the giskardpy motion statechart of the running motion group
 *   Transforms  the executing world's connection graph, node border = freshness
 *
 * The tabs in LIVE_ENDPOINT take their node status from the cramera-live bridge
 * while it is attached: structure changes rebuild the graph, pure status
 * changes only re-colour the rings (no layout jumps). Without a bridge, the
 * Statechart tab follows the playhead through the statecharts the played
 * recording captured.
 *
 * Bus events:
 *   emits    entity:select {id, detail, relations}   node clicked
 *   listens  entity:highlight {ids, focus?}          spotlight matching nodes
 *   listens  scene:step {step}                       highlight the running episode
 *   listens  scene:frame {index}                    follow the replay's statecharts
 *   listens  live:changed {on, url}                  start/stop the status poll
 *
 * Rendering is delegated to graph.js (window.Graph, vis-network wrapper).
 * ==========================================================================*/
Panels.define('graph', function (root, bus) {
  root.innerHTML =
    '<div class="graph-wrap">' +
    '  <div class="graph-tabs" id="graph-tabs">' +
    '    <button data-view="knowledge" class="active" title="the entity graph (EQL / knowledge base)">Knowledge</button>' +
    '    <button data-view="kinematics" title="the robot\'s kinematic structure — URDF links &amp; joints">Kinematics</button>' +
    '    <button data-view="plan" title="the plan tree, with the execution status of every node">Plan</button>' +
    '    <button data-view="chart" title="the giskardpy motion statechart of the running motion group">Statechart</button>' +
    '    <button data-view="transforms" title="the world\'s connection graph — which frame hangs from which, and how recently it moved">Transforms</button>' +
    '    <span class="gt-live" id="gt-live" title="node status is streaming from the running demo">◉ live status</span>' +
    '  </div>' +
    '  <div class="graph-canvas"></div>' +
    '  <div class="graph-steps" id="graph-steps" style="display:none"></div>' +
    '  <button class="graph-steps-toggle" id="graph-steps-toggle" style="display:none" '
    +   'title="Switch between the plan graph and a readable step list">\u2630 Steps</button>' +
    '  <div class="graph-zoom">' +
    '    <button id="graph-zoom-in" title="Zoom in — or pinch on a touchpad">+</button>' +
    '    <button id="graph-zoom-out" title="Zoom out — or pinch on a touchpad">−</button>' +
    '    <button id="graph-zoom-fit" title="Fit the whole graph">⤡</button>' +
    '  </div>' +
    '  <div id="graph-empty" class="graph-empty" style="display:none"></div>' +
    '  <div id="graph-nav" class="graph-nav" style="display:none">' +
    '    <button id="gnav-home" title="back to the overview">⌂</button>' +
    '    <button id="gnav-up" title="one level up">↑ back</button>' +
    '    <span id="gnav-path"></span>' +
    '  </div>' +
    '  <div class="legend" id="legend"></div>' +
    '</div>';

  const emptyEl = root.querySelector('#graph-empty');
  const navEl = root.querySelector('#graph-nav');
  const navUp = root.querySelector('#gnav-up');
  const navHome = root.querySelector('#gnav-home');
  const navPath = root.querySelector('#gnav-path');
  const tabsEl = root.querySelector('#graph-tabs');
  const liveBadge = root.querySelector('#gt-live');
  Graph.attach(root.querySelector('.graph-canvas'), root.querySelector('#legend'));

  // vis-network draws onto a canvas of the size it had when it was built, so a window
  // the reader drags to another size leaves the graph in the old one. Re-fitted once the
  // dragging settles rather than per resize event, of which a drag fires many.
  const REFIT_DELAY_MILLISECONDS = 120;
  let refit = null;
  window.addEventListener('resize', function () {
    if (refit) window.clearTimeout(refit);
    refit = window.setTimeout(function () { refit = null; Graph.resize(); }, REFIT_DELAY_MILLISECONDS);
  });

  // %% zoom controls
  // one step in and its exact inverse out, so clicking + then − lands where it started
  const ZOOM_STEP = 1.3;
  root.querySelector('#graph-zoom-in').addEventListener('click', function () { Graph.zoomBy(ZOOM_STEP); });
  root.querySelector('#graph-zoom-out').addEventListener('click', function () { Graph.zoomBy(1 / ZOOM_STEP); });
  root.querySelector('#graph-zoom-fit').addEventListener('click', function () { Graph.fit(); });

  // %% tabs
  const TABS = {
    knowledge:  { url: '/api/knowledge' },
    kinematics: { url: '/api/knowledge/view?name=kinematics' },
    plan:       { url: '/api/knowledge/view?name=plan' },
    chart:      { url: '/api/knowledge/view?name=chart' },
    transforms: { url: '/api/knowledge/view?name=transforms' },
  };
  // the bridge endpoint each live view polls
  const LIVE_ENDPOINT = { plan: '/plan', chart: '/chart', transforms: '/transforms' };
  let tab = 'knowledge';
  let view = null;            // the currently rendered payload
  const base = {};            // tab -> payload as loaded from the server
  const shown = {};           // tab -> payload currently rendered (drill-downs)
  const stacks = {};          // tab -> parent payloads for the back button
  Object.keys(TABS).forEach(function (t) { stacks[t] = []; });
  let inGraphSet = {};

  // %% Plan tab: readable step-list rendering (an alternative to the vis graph)
  let stepsMode = false;
  const stepsEl = root.querySelector('#graph-steps');
  const stepsToggle = root.querySelector('#graph-steps-toggle');
  const STEP_KINDS = { ActionNode: 'action', AttachNode: 'attach', DetachNode: 'attach' };
  const DETAIL_KINDS = { ConditionNode: 'condition', MotionNode: 'motion', MonitorNode: 'monitor' };
  const STRUCT_KINDS = { SequentialNode: 1, ParallelNode: 1, UnderspecifiedNode: 1 };
  const STEP_STATUS = { SUCCEEDED: 'done', DONE: 'done', RUNNING: 'running', FAILED: 'failed', CREATED: 'not started', NOT_STARTED: 'not started' };
  function stepWords(x) { return String(x || '').replace(/([a-z0-9])([A-Z])/g, '$1 $2').trim().toLowerCase().replace(/^./, function (c) { return c.toUpperCase(); }); }
  function stepLabel(n) {
    if (n.kind === 'ConditionNode') return 'condition check';
    if (n.kind === 'AttachNode') return 'grasp' + (n.target ? ' ' + String(n.target).replace(/\.(stl|obj|dae)$/i, '') : '');
    if (n.kind === 'DetachNode') return 'release' + (n.target ? ' ' + String(n.target).replace(/\.(stl|obj|dae)$/i, '') : '');
    if (n.kind === 'MotionNode') return stepWords((n.label || 'motion').replace(/Motion$/, '')) || 'motion';
    var l = stepWords((n.label || n.kind).replace(/(Action|Node)$/, ''));
    if (n.target) l += ' ' + String(n.target).replace(/\.(stl|obj|dae)$/i, '');
    return l || 'step';
  }
  function stepTreeFrom(nodes) {
    const by = {}, roots = [];
    nodes.forEach(function (n) { by[n.id] = { n: n, kids: [] }; });
    nodes.forEach(function (n) { const o = by[n.id]; if (n.parent && by[n.parent]) by[n.parent].kids.push(o); else roots.push(o); });
    return roots;
  }
  function stepPill(status) {
    const key = status === 'NOT_STARTED' ? 'CREATED' : (status || 'CREATED');
    return '<span class="sp sp-' + key + '">' + (STEP_STATUS[status] || String(status || '').toLowerCase()) + '</span>';
  }
  // flatten structural containers; keep action/attach as numbered steps, details collapsed
  function stepItems(node) {
    const out = [];
    (node.kids || []).forEach(function (c) {
      const k = c.n.kind;
      if (STEP_KINDS[k]) out.push(c);
      else if (STRUCT_KINDS[k]) Array.prototype.push.apply(out, stepItems(c));
    });
    return out;
  }
  // %% constraints palette (in-tab): natural-language -> real giskard goal -> POST /constraint
  let CONSTRAINTS = [
    { id: 'c1', text: 'Milk must always stay upright' },
    { id: 'c2', text: 'Robot must look where it operates' },
    { id: 'c3', text: 'Keep the gripper closed while carrying' },
  ];
  let cSeq = 4;
  function objIn(text, node) {
    const m = String(text).toLowerCase().match(/\b(milk|bowl|spoon|plate|cup|tray|bottle|flask|vial|sample)\b/);
    if (m) return m[1];
    if (node && node.target) return String(node.target).replace(/\.(stl|obj|dae)$/i, '');
    return 'object';
  }
  function compileConstraint(text, node) {
    const t = String(text).toLowerCase();
    if (/upright|stay up|vertical|tip over|spill/.test(t)) {
      const o = objIn(text, node);
      return { goal: 'VectorsAligned', params: { root_link: 'map', tip_link: o, tip_normal: [0, 0, 1], goal_normal: [0, 0, 1], threshold: 0.1 } };
    }
    if (/look|watch|gaze|point|face|operat/.test(t))
      return { goal: 'PointingAt', params: { tip_link: 'head_camera', root_link: 'map', pointing_axis: [0, 0, 1], goal_point: '@operation_target', threshold: 0.05 } };
    if (/gripper.*clos|clos.*gripper|hold.*tight|carry/.test(t))
      return { goal: 'JointPositionReached', params: { joint: 'gripper_joint', goal_position: 0.0, threshold: 0.005 } };
    return { goal: null, params: {} };
  }
  const stepNodeById = {};   // id -> raw plan node, filled during renderSteps
  function postConstraint(nodeId, cid) {
    const node = stepNodeById[nodeId];
    const c = CONSTRAINTS.find(function (x) { return x.id === cid; });
    if (!node || !c) return;
    const comp = compileConstraint(c.text, node);
    if (!comp.goal) { flashStep(nodeId, 'no match', false); return; }
    if (!liveState.url) { flashStep(nodeId, 'not live', false); return; }
    const body = { op: 'attach_monitor', text: c.text, apply: 'next_activation',
      target_plan_node: { id: node.id, kind: node.kind, label: node.label },
      giskard_node: { type: comp.goal, params: comp.params } };
    fetch(liveState.url + '/constraint', { method: 'POST', headers: { 'content-type': 'application/json' }, body: JSON.stringify(body) })
      .then(function (r) { return r.json(); })
      .then(function (j) { flashStep(nodeId, j.ok ? (comp.goal + ' ✓') : ('✗ ' + (j.error || 'error')), !!j.ok); })
      .catch(function (e) { flashStep(nodeId, '✗ ' + e, false); });
  }
  function flashStep(nodeId, msg, ok) {
    const row = stepsEl.querySelector('.st-row[data-id="' + nodeId + '"]');
    if (!row) return;
    let tag = row.querySelector('.st-cflash');
    if (!tag) { tag = document.createElement('span'); tag.className = 'st-cflash'; row.querySelector('.st-meta').prepend(tag); }
    tag.textContent = '⛓ ' + msg; tag.classList.toggle('bad', !ok);
  }
  function renderPalette() {
    const items = CONSTRAINTS.map(function (c) {
      return '<div class="cpal-card" draggable="true" data-cid="' + c.id + '"><span class="cpal-grip">⠿</span>' + c.text + '<span class="cpal-del" data-del="' + c.id + '">×</span></div>';
    }).join('');
    return '<div class="cpal"><div class="cpal-h">Constraints — drag onto a step (live)</div>' +
      '<div class="cpal-list">' + items + '</div>' +
      '<div class="cpal-add"><input class="cpal-in" placeholder="e.g. milk must stay upright"><button class="cpal-btn">Add</button></div></div>';
  }
  function wirePalette() {
    stepsEl.querySelectorAll('.cpal-card').forEach(function (card) {
      card.addEventListener('dragstart', function (e) { e.dataTransfer.setData('text/plain', 'c:' + card.dataset.cid); e.dataTransfer.effectAllowed = 'copy'; });
    });
    stepsEl.querySelectorAll('.cpal-del').forEach(function (x) {
      x.addEventListener('click', function (e) { e.stopPropagation(); CONSTRAINTS = CONSTRAINTS.filter(function (c) { return c.id !== x.dataset.del; }); renderSteps(shown[tab] || base[tab]); });
    });
    const inp = stepsEl.querySelector('.cpal-in'), btn = stepsEl.querySelector('.cpal-btn');
    function add() { const v = inp.value.trim(); if (!v) return; CONSTRAINTS.push({ id: 'c' + (cSeq++), text: v }); renderSteps(shown[tab] || base[tab]); }
    if (btn) btn.addEventListener('click', add);
    if (inp) inp.addEventListener('keydown', function (e) { if (e.key === 'Enter') { e.preventDefault(); add(); } });
    stepsEl.querySelectorAll('.st-row[data-id]').forEach(function (row) {
      row.addEventListener('dragover', function (e) { e.preventDefault(); row.classList.add('st-drop'); });
      row.addEventListener('dragleave', function () { row.classList.remove('st-drop'); });
      row.addEventListener('drop', function (e) {
        e.preventDefault(); e.stopPropagation(); row.classList.remove('st-drop');
        const d = e.dataTransfer.getData('text/plain') || '';
        if (d.indexOf('c:') === 0) postConstraint(row.dataset.id, d.slice(2));
      });
    });
  }

  function renderSteps(payload) {
    const roots = stepTreeFrom(payload.nodes || []);
    const top = roots.length === 1 && STRUCT_KINDS[roots[0].n.kind] ? stepItems(roots[0]) : roots;
    for (const k in stepNodeById) delete stepNodeById[k];
    const html = [renderPalette(), '<div class="steps-tree">'];
    function walk(item, number) {
      const n = item.n;
      stepNodeById[n.id] = n;
      const sub = stepItems(item);
      const details = (item.kids || []).filter(function (c) { return DETAIL_KINDS[c.n.kind]; });
      const hk = sub.length || details.length;
      html.push('<div class="st-row' + (hk ? ' hk' : '') + '" data-id="' + n.id + '">' +
        '<span class="st-tw">' + (hk ? '▸' : '') + '</span>' +
        '<span class="st-num">' + number + '</span>' +
        '<span class="st-name">' + stepLabel(n) + '</span>' +
        '<span class="st-meta">' + (details.length ? '<span class="st-dc">' + details.length + ' detail' + (details.length > 1 ? 's' : '') + '</span>' : '') + stepPill(n.status) + '</span>' +
        '</div>');
      if (hk) {
        html.push('<div class="st-kids">');
        details.forEach(function (d) { html.push('<div class="st-leaf"><span class="st-name detail">' + stepLabel(d.n) + '</span>' + stepPill(d.n.status) + '</div>'); });
        var i = 1; sub.forEach(function (c) { walk(c, number + '.' + (i++)); });
        html.push('</div>');
      }
    }
    var i = 1; top.forEach(function (c) { walk(c, String(i++)); });
    html.push('</div>');
    stepsEl.innerHTML = html.join('');
    wirePalette();
    // collapse/expand
    stepsEl.querySelectorAll('.st-row.hk').forEach(function (r) {
      r.addEventListener('click', function () {
        const kids = r.nextElementSibling;
        if (kids && kids.classList.contains('st-kids')) {
          const open = kids.style.display !== 'none';
          kids.style.display = open ? 'none' : '';
          r.querySelector('.st-tw').textContent = open ? '▸' : '▾';
        }
      });
    });
  }
  function maybeRenderSteps(payload) {
    const on = stepsMode && tab === 'plan';
    const canvas = root.querySelector('.graph-canvas');
    const legend = root.querySelector('#legend');
    if (on && payload && (payload.nodes || []).length) {
      canvas.style.display = 'none'; if (legend) legend.style.display = 'none';
      stepsEl.style.display = ''; renderSteps(payload);
    } else {
      stepsEl.style.display = 'none'; canvas.style.display = ''; if (legend) legend.style.display = '';
    }
  }
  if (stepsToggle) stepsToggle.addEventListener('click', function () {
    stepsMode = !stepsMode;
    stepsToggle.classList.toggle('on', stepsMode);
    maybeRenderSteps(shown[tab] || base[tab]);
  });

  function setView(payload) {
    view = payload;
    shown[tab] = payload;
    inGraphSet = {};
    payload.nodes.forEach(function (n) { inGraphSet[n.id] = 1; });
    if (emptyEl) {
      const empty = !payload.nodes.length;
      emptyEl.style.display = empty ? '' : 'none';
      emptyEl.textContent = empty ? (payload.empty || 'Nothing to show in this view.') : '';
    }
    Graph.build({
      nodes: payload.nodes, edges: payload.edges, legend: payload.legend,
      layout: payload.layout, arrows: !!payload.arrows,
      // a view may name the statuses its legend lists, instead of taking the default
      statusLegend: payload.statusLegend || false,
      key: (payload.key || tab) + '#' + stacks[tab].length,
    });
    updateNav();
    maybeRenderSteps(payload);
  }
  function updateNav() {
    const inside = stacks[tab].length > 0;
    navEl.style.display = inside ? '' : 'none';
    if (inside) {
      const path = stacks[tab].slice(1).map(function (v) { return v.breadcrumb; }).concat([view.breadcrumb]);
      navPath.textContent = path.join(' / ');
    }
  }
  async function drill(id) {
    if (!view.details[id]) return;
    try {
      const r = await fetch(SceneContext.withScene('/api/knowledge/expand?node=' + encodeURIComponent(id)));
      const p = await ResponseUtil.parseJson(r);
      if (!p.ok) return;                       // node has no inside view
      stacks[tab].push(view);
      setView(p);
      select(id);
    } catch (err) { /* server unreachable — stay where we are */ }
  }
  function goBack() { if (stacks[tab].length) setView(stacks[tab].pop()); }
  function goHome() {
    if (!stacks[tab].length) return;
    stacks[tab] = [];
    setView(base[tab]);
  }
  navUp.addEventListener('click', goBack);
  navHome.addEventListener('click', goHome);

  async function showTab(name) {
    if (!TABS[name]) return;
    tab = name;
    tabsEl.querySelectorAll('button').forEach(function (b) {
      b.classList.toggle('active', b.dataset.view === name);
    });
    if (!base[name]) {
      emptyEl.style.display = '';
      emptyEl.textContent = 'loading…';
      try {
        const r = await fetch(SceneContext.withScene(TABS[name].url));
        if (r.status === 404) throw new Error('this build needs the /api/knowledge/view route — restart the server');
        const p = await ResponseUtil.parseJson(r);
        if (!p.ok) {
          emptyEl.textContent = p.error || 'view unavailable';
          return;
        }
        p.key = name;
        base[name] = p;
      } catch (err) {
        emptyEl.textContent = 'Could not load this view: ' + ((err && err.message) || err);
        return;
      }
    }
    if (stepsToggle) stepsToggle.style.display = (name === 'plan') ? '' : 'none';
    setView(shown[name] || base[name]);
    liveRefresh(true);            // a live tab picks the bridge status up at once
    showRecordedStatechart();     // and a replayed one, the moment it is playing
  }
  tabsEl.querySelectorAll('button').forEach(function (b) {
    b.addEventListener('click', function () { showTab(b.dataset.view); });
  });

  // %% node click → describe in whatever panel listens
  function select(id) {
    const d = view && view.details && view.details[id];
    if (!d) return;
    const relations = (view.edges || [])
      .filter(function (e) { return e.from === id || e.to === id; })
      .map(function (e) {
        return { s: labelOf(e.from), p: e.label || e.kind, o: labelOf(e.to) };
      });
    bus.emit('entity:select', { id: id, detail: d, relations: relations });
    spotlight({ ids: [id], focus: id });
  }
  function labelOf(id) { return (view.details[id] && view.details[id].label) || id; }
  Graph.onSelect(select);
  Graph.onDoubleSelect(drill);

  // %% highlights (from EQL results or our own selection)
  function spotlight(p) {
    const ids = (p && p.ids) || [];
    let hi = ids.filter(function (id) { return inGraphSet[id]; });
    if (p && p.focus && inGraphSet[p.focus]) {
      const neighbours = (view.edges || [])
        .filter(function (e) { return e.from === p.focus || e.to === p.focus; })
        .map(function (e) { return e.from === p.focus ? e.to : e.from; });
      hi = hi.concat(neighbours.filter(function (id) { return inGraphSet[id]; }));
    }
    if (hi.length) Graph.highlight(hi); else Graph.reset();
  }
  bus.on('entity:highlight', spotlight);
  bus.on('scene:step', function (p) {
    if (p.step === '__done__') { Graph.reset(); return; }
    if (tab === 'knowledge' && !stacks[tab].length && inGraphSet[p.step]) select(p.step);
  });

  // %% live status overlay (Plan / Statechart tabs)
  // The bridge publishes the plan tree and the executing motion statechart with
  // per-node status. Structure changes (the plan grows as actions expand, a new
  // statechart is compiled per motion group) rebuild the graph; a pure status
  // change only re-colours the rings, so the layout never jumps.
  const CHART_LEGEND = [
    { group: 'task', label: 'Task (motion constraint)' },
    { group: 'monitor', label: 'Monitor / observation' },
    { group: 'motion_goal', label: 'Goal (contains nodes)' },
    { group: 'motion_end', label: 'End / cancel motion' },
  ];
  const TRANSFORM_LEGEND = [
    { group: 'world_frame', label: 'World root' },
    { group: 'actuated_frame', label: 'Actuated joint' },
    { group: 'free_frame', label: 'Free-floating' },
    { group: 'fixed_frame', label: 'Fixed to its parent' },
  ];
  const FRESHNESS_LEGEND = ['MOVING', 'SETTLED', 'STALE', 'STATIC'];
  const FRAME_GROUP = { actuated: 'actuated_frame', free: 'free_frame', fixed: 'fixed_frame' };
  const liveSig = { plan: '', chart: '', transforms: '' };
  let liveTimer = null;
  let liveState = { on: false, url: '' };

  function liveSource() {
    const p = shown[tab] || base[tab];
    return (p && p.live) || null;               // 'plan' | 'chart' | null
  }

  // drop the redundant 'Action' suffix only — a label that merely contains the word,
  // such as 'ActionNode', must survive intact. Mirrors
  // PlanViewPayload._shorten_action_label: the bridge sends the raw designator name,
  // so the live path shortens it here.
  function shortenActionLabel(label) {
    const shortened = label.replace(/Action$/, '');
    return shortened || label;
  }

  function planPayload(live) {
    const nodes = [], edges = [], details = {};
    (live.nodes || []).forEach(function (n) {
      const label = shortenActionLabel(n.label || '?');
      const lines = ['a ' + n.kind,
                     'status: ' + n.status + (n.derived ? ' (derived from the motion statechart)' : '')];
      if (n.arm) lines.push('arm: ' + n.arm);
      if (n.target) lines.push('target: ' + n.target);
      nodes.push({ id: n.id, label: label, group: n.group,
                   title: [label].concat(lines).join('\n'), status: n.status });
      details[n.id] = { label: label, group: n.group, lines: lines };
      if (n.parent) edges.push({ from: n.parent, to: n.id, kind: 'property', label: 'has step' });
    });
    return { ok: true, breadcrumb: 'live plan', nodes: nodes, edges: edges, details: details,
             legend: live.legend || [], layout: 'hier', arrows: true, statusLegend: true,
             live: 'plan', key: 'plan-live',
             empty: 'The bridge is attached but the demo has not started its plan yet.' };
  }

  function chartPayload(live) {
    const nodes = [], edges = [], details = {}, isParent = {};
    (live.nodes || []).forEach(function (n) { if (n.parent) isParent[n.parent] = 1; });
    (live.nodes || []).forEach(function (n) {
      const group = isParent[n.id] ? 'motion_goal'
        : /EndMotion|CancelMotion/.test(n.class_name) ? 'motion_end'
        : /Monitor|Reached|Observation|Condition/.test(n.class_name + n.name) ? 'monitor' : 'task';
      const lines = ['a ' + n.class_name, 'life cycle: ' + n.life_cycle, 'observation: ' + n.observation];
      nodes.push({ id: n.id, label: n.name, group: group,
                   title: [n.name].concat(lines).join('\n'), status: n.life_cycle });
      details[n.id] = { label: n.name, group: group, lines: lines };
      if (n.parent) edges.push({ from: n.parent, to: n.id, kind: 'type', label: 'contains' });
    });
    (live.edges || []).forEach(function (e) {
      edges.push({ from: e.from, to: e.to, kind: e.kind, label: (e.kind || '').toLowerCase() + ' transition' });
    });
    return { ok: true, breadcrumb: 'statechart' + (live.title ? ' · ' + live.title : ''),
             nodes: nodes, edges: edges, details: details, legend: CHART_LEGEND,
             layout: 'hier', arrows: true, statusLegend: true, live: 'chart', key: 'chart-live',
             empty: 'Attached, but no motion statechart is executing right now.' };
  }

  // The world's connection graph: one node per frame, one edge per connection. A
  // frame's ring is the freshness of the connection that carries it — the root frame,
  // which no connection carries, cannot go stale.
  function transformsPayload(live) {
    const nodes = [], edges = [], details = {}, carried = {};
    const connections = live.connections || [];
    connections.forEach(function (c) { carried[c.child] = c; });
    const frames = {};
    connections.forEach(function (c) { frames[c.parent] = 1; frames[c.child] = 1; });
    Object.keys(frames).forEach(function (frame) {
      const connection = carried[frame];
      const group = connection ? (FRAME_GROUP[connection.kind] || 'fixed_frame') : 'world_frame';
      const status = connection ? connection.freshness : 'STATIC';
      const lines = connection
        ? ['a frame carried by ' + connection.name,
           'connection: ' + connection.kind,
           'last written by: ' + connection.writer,
           'last changed: ' + (connection.ageSeconds === null || connection.ageSeconds === undefined
             ? 'never, since the bridge attached'
             : connection.ageSeconds + ' s ago')]
        : ['the world root frame'];
      nodes.push({ id: frame, label: frame, group: group,
                   title: [frame].concat(lines).join('\n'), status: status });
      details[frame] = { label: frame, group: group, lines: lines };
    });
    connections.forEach(function (c) {
      edges.push({ from: c.parent, to: c.child, kind: c.kind, label: c.name });
    });
    return { ok: true, breadcrumb: 'transform graph', nodes: nodes, edges: edges,
             details: details, legend: TRANSFORM_LEGEND, layout: 'hier', arrows: true,
             statusLegend: FRESHNESS_LEGEND, live: 'transforms', key: 'transforms-live',
             empty: 'Attached, but the demo has not published a world yet.' };
  }

  function livePayload(source, live) {
    if (source === 'plan') return planPayload(live);
    if (source === 'chart') return chartPayload(live);
    return transformsPayload(live);
  }

  async function liveRefresh(force) {
    const src = liveSource();
    const active = !!src && liveState.on;
    liveBadge.classList.toggle('on', active);
    if (!active) return;
    if (stacks[tab].length) return;              // inside a drill-down: leave it alone
    let live;
    try {
      live = await fetch(liveState.url + LIVE_ENDPOINT[src]).then(ResponseUtil.parseJson);
    } catch (err) { return; }                    // bridge gone — the 3D side handles it
    if (!live || !(live.nodes || live.connections)) return;
    const payload = livePayload(src, live);
    if (force || live.signature !== liveSig[src]) {    // structure changed → rebuild
      liveSig[src] = live.signature;
      base[tab] = payload;
      setView(payload);
      return;
    }
    // same structure: only re-colour, and keep the detail lines in sync
    const map = {};
    payload.nodes.forEach(function (n) { map[n.id] = n.status; });
    if (!Graph.setStatuses(map)) { base[tab] = payload; setView(payload); return; }
    base[tab] = payload;
    if (view && view.details) view.details = payload.details;
    maybeRenderSteps(payload);
  }

  bus.on('live:changed', function (p) {
    liveState = { on: !!p.on, url: p.url || '' };
    if (liveTimer) { clearInterval(liveTimer); liveTimer = null; }
    if (liveState.on) {
      liveTimer = setInterval(function () { liveRefresh(false); }, 700);
      liveRefresh(true);
    } else {
      liveBadge.classList.remove('on');
      liveSig.plan = liveSig.chart = liveSig.transforms = '';
      // drop the live payloads so the live tabs fall back to the recorded bundle
      const liveTabs = Object.keys(LIVE_ENDPOINT);
      liveTabs.forEach(function (t) { delete base[t]; delete shown[t]; stacks[t] = []; });
      if (liveTabs.indexOf(tab) >= 0) showTab(tab);
    }
  });

  // %% recorded statecharts (Statechart tab of a replay)
  // A recording keeps every statechart its run ticked (see
  // cramera.knowledge.recorded_statecharts); the tab follows the playhead through
  // them, re-colouring rather than rebuilding while the played moments stay inside
  // the same chart.
  const NO_STATECHART = -1;           // mirrors recorded_statecharts.NO_STATECHART
  let recordedFrame = 0;              // the played frame the tab is showing
  let drawnChart = NO_STATECHART;     // index of the recorded chart currently drawn

  function recordedStatecharts() {
    return (base.chart && base.chart.recorded) || null;
  }

  function momentAt(index) {
    const recorded = recordedStatecharts();
    const at = recorded.frames[index];
    return at === undefined || at === NO_STATECHART ? null : recorded.moments[at];
  }

  // one recorded moment in the shape the bridge publishes, so a statechart is drawn
  // by the same renderer whether it is streaming or replayed
  function recordedSnapshot(moment) {
    const chart = recordedStatecharts().charts[moment.chart];
    return {
      signature: chart.signature,
      title: chart.title,
      nodes: chart.nodes.map(function (node, index) {
        return { id: node.id, name: node.name, class_name: node.class_name,
                 parent: node.parent, life_cycle: moment.lifeCycles[index],
                 observation: moment.observations[index] };
      }),
      edges: chart.edges,
    };
  }

  function showRecordedStatechart() {
    if (tab !== 'chart' || liveState.on || stacks.chart.length) return;
    if (!recordedStatecharts()) return;
    const moment = momentAt(recordedFrame);
    if (!moment) { drawnChart = NO_STATECHART; setView(base.chart); return; }
    const payload = chartPayload(recordedSnapshot(moment));
    payload.key = 'chart-recorded-' + moment.chart;
    payload.empty = base.chart.empty;
    if (moment.chart === drawnChart) {
      const statuses = {};
      payload.nodes.forEach(function (node) { statuses[node.id] = node.status; });
      if (Graph.setStatuses(statuses)) {
        shown.chart = payload;
        if (view && view.details) view.details = payload.details;
        return;
      }
    }
    drawnChart = moment.chart;
    setView(payload);
  }

  bus.on('scene:frame', function (p) {
    recordedFrame = p.index;
    showRecordedStatechart();
  });

  // %% boot
  showTab('knowledge');

  return {
    destroy: function () {
      if (liveTimer) { clearInterval(liveTimer); liveTimer = null; }
    },
  };
});
