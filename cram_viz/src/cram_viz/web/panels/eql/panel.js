/* ============================================================================
 * panels/eql/panel.js — the EQL (Entity Query Language) console.
 *
 * Query box + presets + answer panel. Queries are executed server-side
 * (POST /api/eql) against the episode knowledge base; the KB overview
 * (GET /api/kb) provides presets and per-entity details.
 *
 * Bus events:
 *   emits    kb:ready {payload}          the /api/kb overview, once loaded
 *   emits    entity:highlight {ids, focus?}   results / described entity
 *   listens  scene:part-clicked {id}     describe the clicked part
 *   listens  scene:step {step}           describe the running episode
 *   listens  entity:select {id, detail, relations}   node clicked in a graph
 * ==========================================================================*/
Panels.define('eql', function (root, bus) {
  root.innerHTML =
    '<div class="panel-head">' +
    '  <h2>EQL · entity query language</h2>' +
    '  <span id="kb-status" class="kb-status">loading knowledge base…</span>' +
    '</div>' +
    '<div class="query-box">' +
    '  <div class="query-row">' +
    '    <textarea id="query-input" rows="2" spellcheck="false" placeholder="the(entity(object).where(object.name == \'milk\'))   —  vars: object, episode, arm, joint, robot"></textarea>' +
    '    <button id="query-run">Run</button>' +
    '  </div>' +
    '  <div id="presets" class="presets"></div>' +
    '</div>' +
    '<div id="answer" class="answer"><div class="answer-empty">Loading the episode knowledge base…</div></div>';

  const kbStatus = root.querySelector('#kb-status');
  const answerEl = root.querySelector('#answer');
  const input = root.querySelector('#query-input');
  const runBtn = root.querySelector('#query-run');
  const presetsEl = root.querySelector('#presets');

  let knowledgeBase = null;   // /api/kb overview (presets + entity details)

  // ---- boot -----------------------------------------------------------------
  fetch('/api/kb').then(function (response) { return response.json(); }).then(boot).catch(function (err) {
    kbStatus.textContent = 'KB error';
    answerEl.innerHTML = '<div class="qerr">Failed to reach the EQL server:\n' + esc(String(err)) + '</div>';
  });

  function boot(payload) {
    if (!payload.ok) {
      kbStatus.textContent = 'EQL unavailable';
      answerEl.innerHTML = '<div class="qerr">' + esc(payload.error || 'unknown error') + '</div>';
      return;
    }
    knowledgeBase = payload;
    kbStatus.textContent = payload.status;
    kbStatus.classList.add('ready');
    buildPresets(payload.presets || []);
    welcome();
    bus.emit('kb:ready', { payload: payload });
  }

  function welcome() {
    answerEl.innerHTML =
      '<div class="goal">EQL · knowledge &amp; reasoning</div>' +
      '<p class="headline"><b>Correctness, concepts and specifications</b> are captured as ' +
      '<b>rules</b> and <b>description-logic axioms / predicates</b>, and made explorable as a ' +
      '<b>graph</b> — queried with <b>EQL</b>, krrood’s pythonic entity query language from the ' +
      'CRAM architecture.</p>' +
      '<p class="hint-txt">Ready-made variables: <code>object</code> (bench objects), <code>episode</code> ' +
      '(action episodes), <code>arm</code>, <code>joint</code> (joint motion), <code>robot</code>, ' +
      '<code>package</code> / <code>subpackage</code> / <code>python_class</code> (CRAM packages, subpackages, classes). ' +
      'Build queries like <code>the(entity(object).where(object.name == \'milk\'))</code> — ' +
      'or click a preset below, or a node in the graph.</p>';
  }

  // ---- describe an entity in the answer panel --------------------------------
  // Two sources: our own knowledgeBase.details (scene clicks) and graph panels,
  // which send the full detail payload of the node the user clicked (entity:select).
  function describe(id, detail, relations) {
    const entityDetail = detail || (knowledgeBase && knowledgeBase.details && knowledgeBase.details[id]);
    if (!entityDetail) return;
    let html = '<div class="goal">entity · ' + esc(id) + '</div>';
    html += '<div class="ansrow"><span class="tag" style="background:' + groupColor(entityDetail.group) + '">' +
      esc(entityDetail.group) + '</span><div class="body"><span class="name">' + esc(entityDetail.label) + '</span></div></div>';
    (entityDetail.lines || []).forEach(function (line) {
      html += '<div class="ansrow"><div class="body"><span class="name">' + esc(line) + '</span></div></div>';
    });
    if (relations && relations.length) {
      html += '<div class="ansub">Relations</div>';
      relations.slice(0, 40).forEach(function (relation) {
        html += '<div class="ansrow"><div class="body"><span class="name">' + esc(relation.s) +
          ' <span class="rel">' + esc(relation.p) + '</span> ' + esc(relation.o) + '</span></div></div>';
      });
      if (relations.length > 40) {
        html += '<div class="ansrow"><div class="body"><span class="sub">… ' + (relations.length - 40) + ' more</span></div></div>';
      }
    }
    answerEl.innerHTML = html;
    bus.emit('entity:highlight', { ids: [id], focus: id });
  }

  bus.on('scene:part-clicked', function (payload) { describe(payload.id); });
  bus.on('entity:select', function (payload) { describe(payload.id, payload.detail, payload.relations); });
  bus.on('scene:step', function (payload) {
    if (payload.step !== '__done__' && knowledgeBase && knowledgeBase.details && knowledgeBase.details[payload.step]) describe(payload.step);
  });

  // ---- presets ----------------------------------------------------------------
  function buildPresets(presets) {
    presetsEl.innerHTML = '';
    presets.forEach(function (preset) {
      const presetButton = document.createElement('div');
      presetButton.className = 'preset'; presetButton.textContent = preset.text;
      presetButton.title = preset.code;
      presetButton.addEventListener('click', function () {
        input.value = preset.code;
        runQuery(preset.code);
      });
      presetsEl.appendChild(presetButton);
    });
  }

  // ---- run an EQL query --------------------------------------------------------
  let running = false;
  async function runQuery(code) {
    code = (code || '').trim(); if (!code || running) return;
    running = true;
    runBtn.textContent = '…';
    try {
      const response = await fetch('/api/eql', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ code: code }),
      });
      render(code, await response.json());
    } catch (err) {
      render(code, { ok: false, error: String(err) });
    }
    running = false;
    runBtn.textContent = 'Run';
  }

  function render(code, res) {
    let html = '<div class="goal">&gt;&gt;&gt; ' + esc(code) + '</div>';
    if (!res.ok) {
      answerEl.innerHTML = html + '<div class="qerr">' + esc(res.error || 'query failed') + '</div>';
      bus.emit('entity:highlight', { ids: [] });
      return;
    }
    if (!res.count) {
      answerEl.innerHTML = html + '<div class="nores">No solutions — the query returned nothing.</div>';
      bus.emit('entity:highlight', { ids: [] });
      return;
    }
    html += '<p class="headline"><b>' + res.count + '</b> result' + (res.count === 1 ? '' : 's') +
      (res.more ? ' (truncated)' : '') + '.</p>';
    res.rows.forEach(function (row) {
      if (row.__entity__ !== undefined) html += entityRow(row);
      else html += valueRow(row);
    });
    answerEl.innerHTML = html;
    bus.emit('entity:highlight', { ids: res.highlight || [] });
  }

  function entityRow(row) {
    const group = groupOfType(row.__type__);
    const subDetails = [];
    for (const key in row) {
      if (key.indexOf('__') === 0 || row[key] === null || row[key] === undefined) continue;
      subDetails.push(key + ': ' + row[key]);
    }
    return '<div class="ansrow"><span class="tag" style="background:' + groupColor(group) + '">' +
      esc(row.__type__) + '</span><div class="body"><span class="name">' + esc(row.__entity__) +
      '</span><span class="sub">' + esc(subDetails.join('  ·  ')) + '</span></div></div>';
  }
  function valueRow(row) {
    const parts = Object.keys(row).map(function (key) {
      return '<code>' + esc(key) + ' = ' + esc(String(row[key])) + '</code>';
    }).join(' ');
    return '<div class="ansrow"><div class="body">' + parts + '</div></div>';
  }

  runBtn.addEventListener('click', function () { runQuery(input.value); });
  input.addEventListener('keydown', function (e) {
    if (e.key === 'Enter' && !e.shiftKey) { e.preventDefault(); runQuery(input.value); }
  });

  // ---- helpers ----------------------------------------------------------------
  // group keys/colours come from the shared palette (core/palette.js) so this
  // panel's tags and the graph panel's nodes can never drift apart again
  const TYPE_GROUP = {
    BenchObject: 'object', ActionEpisode: 'event', Arm: 'robot', Gripper: 'robot',
    Robot: 'robot', JointMotion: 'robot', Position: 'concept',
    Package: 'concept', SubPackage: 'klass', PythonClass: 'pyclass',
  };
  function groupOfType(type) { return TYPE_GROUP[type] || 'ind'; }
  function groupColor(group) { return (window.EntityPalette[group] || window.EntityPalette.ind).color; }
  function esc(s) { return String(s).replace(/[&<>]/g, function (c) { return { '&': '&amp;', '<': '&lt;', '>': '&gt;' }[c]; }); }
});
