/* ============================================================================
 * deck_player.js — recorded cramera episodes inside the defense deck.
 *
 * The rendering / URDF-loading / material core is reused from the cramera
 * viewer (cramera/src/cramera/web/panels/robot_scene/panel.js on the
 * cram-viz-integration branch); this file adapts it to the deck: three scene
 * bundles switchable from the HUD, a scrubber + plan-step chips, and one
 * shared viewer that is re-parented between the title slide (ambient
 * autoplay) and the recorded-episodes slide (interactive).
 *
 * Scene bundles are loaded from ./scenes/<name>/ exactly like cramera does —
 * clone or symlink cram2/cram-scenes there (see defense/README.md).
 * ==========================================================================*/
(function () {
  'use strict';
  const SCENES = 'scenes/';
  const EPISODE_NAMES = ['PR2_Apartment', 'HSR_Apartment', 'TIAGO_Apartment'];
  const ROBOT_LABEL = { PR2_Apartment: 'pr2', HSR_Apartment: 'hsrb', TIAGO_Apartment: 'tiago' };
  //: how long after a bundle load imported materials keep being re-tamed, in seconds
  const MATERIAL_SETTLE_SECONDS = 20;

  const titleStage = document.getElementById('titleStage');
  const epStage = document.getElementById('epStage');
  if (!epStage) return;

  // %% host container, re-parented between slides
  const host = document.createElement('div');
  host.style.cssText = 'position:absolute;inset:0;';
  epStage.appendChild(host);
  const statusEl = document.createElement('div');
  statusEl.style.cssText =
    'position:absolute;left:50%;top:50%;transform:translate(-50%,-50%);z-index:5;' +
    'font-family:var(--mono);font-size:12px;color:var(--mute);pointer-events:none;';
  statusEl.textContent = '';
  host.appendChild(statusEl);

  // %% three.js setup (from cramera panel.js)
  const scene3 = new THREE.Scene();
  const camera = new THREE.PerspectiveCamera(45, 1, 0.01, 200);
  camera.position.set(3, 2.4, 4);
  const renderer = new THREE.WebGLRenderer({ antialias: true, alpha: true });
  renderer.setPixelRatio(Math.min(window.devicePixelRatio, 2));
  renderer.shadowMap.enabled = true;
  renderer.shadowMap.type = THREE.PCFSoftShadowMap;
  renderer.outputEncoding = THREE.sRGBEncoding;
  renderer.toneMapping = THREE.ACESFilmicToneMapping;
  renderer.toneMappingExposure = 0.95;
  host.insertBefore(renderer.domElement, statusEl);

  // theme-aware studio backdrop (deck has a light beamer mode and a dark mode)
  function backdrop(dark) {
    const cv = document.createElement('canvas');
    cv.width = 2; cv.height = 256;
    const ctx = cv.getContext('2d');
    const g = ctx.createLinearGradient(0, 0, 0, 256);
    if (dark) { g.addColorStop(0, '#232833'); g.addColorStop(0.55, '#151922'); g.addColorStop(1, '#0c0e13'); }
    else { g.addColorStop(0, '#f4f6f9'); g.addColorStop(0.55, '#e9edf1'); g.addColorStop(1, '#dde3e9'); }
    ctx.fillStyle = g; ctx.fillRect(0, 0, 2, 256);
    return new THREE.CanvasTexture(cv);
  }
  const hemi = new THREE.HemisphereLight(0xf4efe6, 0x2a2d33, 0.32);
  function applyPlayerTheme() {
    const dark = document.documentElement.dataset.theme === 'dark';
    scene3.background = backdrop(dark);
    hemi.intensity = dark ? 0.32 : 0.45;
    renderer.toneMappingExposure = dark ? 0.95 : 1.05;
    needsRender = true;
  }
  new MutationObserver(applyPlayerTheme)
    .observe(document.documentElement, { attributes: true, attributeFilter: ['data-theme'] });

  // image-based lighting (imported .dae lights are stripped on load)
  if (THREE.RoomEnvironment && THREE.PMREMGenerator) {
    try {
      const pmrem = new THREE.PMREMGenerator(renderer);
      scene3.environment = pmrem.fromScene(new THREE.RoomEnvironment(), 0.04).texture;
    } catch (e) { /* optional */ }
  }
  scene3.add(hemi);
  scene3.add(new THREE.AmbientLight(0xfff4e6, 0.05));
  const key = new THREE.DirectionalLight(0xfff2df, 1.05);
  key.position.set(4, 7, 3);
  key.castShadow = true;
  key.shadow.mapSize.set(4096, 4096);
  key.shadow.camera.near = 0.5; key.shadow.camera.far = 30;
  key.shadow.camera.left = -6; key.shadow.camera.right = 6;
  key.shadow.camera.top = 6; key.shadow.camera.bottom = -6;
  key.shadow.bias = -0.0003;
  key.shadow.normalBias = 0.02;
  key.shadow.radius = 3;
  scene3.add(key);
  const back = new THREE.DirectionalLight(0xdfe6f2, 0.3);
  back.position.set(-4, 4, -3);
  scene3.add(back);

  // %% procedural textures (floor planks, counter/table wood) — from cramera
  function floorTexture() {
    const cv = document.createElement('canvas'); cv.width = 512; cv.height = 512;
    const ctx = cv.getContext('2d');
    ctx.fillStyle = '#b9a184'; ctx.fillRect(0, 0, 512, 512);
    const plank = 512 / 6;
    for (let r = 0; r < 6; r++) {
      const shade = 168 + ((r * 37) % 40) - 20;
      ctx.fillStyle = 'rgb(' + (shade + 20) + ',' + (shade - 4) + ',' + (shade - 30) + ')';
      ctx.fillRect(0, r * plank, 512, plank - 2);
      ctx.strokeStyle = 'rgba(60,40,25,0.35)'; ctx.lineWidth = 2;
      ctx.strokeRect(0, r * plank, 512, plank - 2);
      for (let i = 0; i < 40; i++) {
        ctx.strokeStyle = 'rgba(90,60,40,0.06)';
        const y = r * plank + Math.random() * plank;
        ctx.beginPath(); ctx.moveTo(0, y); ctx.lineTo(512, y + (Math.random() - 0.5) * 4); ctx.stroke();
      }
    }
    const t = new THREE.CanvasTexture(cv);
    t.wrapS = t.wrapT = THREE.RepeatWrapping; t.anisotropy = 8;
    return t;
  }
  function woodTexture(base, streak) {
    const cv = document.createElement('canvas'); cv.width = 256; cv.height = 256;
    const ctx = cv.getContext('2d');
    ctx.fillStyle = base; ctx.fillRect(0, 0, 256, 256);
    for (let i = 0; i < 220; i++) {
      const y = Math.floor((i / 220) * 256) + (i % 3);
      ctx.strokeStyle = 'rgba(' + streak + ',' + (0.03 + (i % 5) * 0.015) + ')';
      ctx.lineWidth = 1 + (i % 3);
      ctx.beginPath(); ctx.moveTo(0, y);
      for (let x = 0; x <= 256; x += 16) ctx.lineTo(x, y + Math.sin((x + i) * 0.05) * 1.5);
      ctx.stroke();
    }
    const t = new THREE.CanvasTexture(cv);
    t.wrapS = t.wrapT = THREE.RepeatWrapping; t.anisotropy = 8;
    return t;
  }
  const WOOD_COUNTER = woodTexture('#9c6b3f', '70,45,25');
  const WOOD_TABLE = woodTexture('#b98a52', '90,60,35');
  const floorTex = floorTexture();
  floorTex.repeat.set(12, 12);
  const ground = new THREE.Mesh(
    new THREE.PlaneGeometry(60, 60),
    new THREE.MeshStandardMaterial({ map: floorTex, roughness: 0.85, metalness: 0.0 })
  );
  ground.rotation.x = -Math.PI / 2;
  ground.receiveShadow = true;
  ground.visible = false;            // shown once the first bundle sizes it
  scene3.add(ground);

  const controls = new THREE.OrbitControls(camera, renderer.domElement);
  controls.enableDamping = true;
  controls.dampingFactor = 0.08;
  controls.minDistance = 1;
  controls.maxDistance = 25;
  controls.maxPolarAngle = Math.PI * 0.5;

  // z-up (URDF/map) -> three y-up
  const worldRoot = new THREE.Group();
  worldRoot.rotation.x = -Math.PI / 2;
  scene3.add(worldRoot);

  let needsRender = true;
  controls.addEventListener('change', function () { needsRender = true; });

  // %% materials (identical treatment to cramera)
  function tameMat(mat) {
    if (!mat) return;
    if (mat.emissive) mat.emissive.setRGB(0, 0, 0);
    mat.emissiveIntensity = 0;
    if (mat.color) {
      const lum = (mat.color.r + mat.color.g + mat.color.b) / 3;
      if (lum > 0.92) mat.color.setRGB(0.82, 0.82, 0.83);
    }
    if (mat.isMeshPhongMaterial) { mat.shininess = 25; if (mat.specular) mat.specular.setRGB(0.05, 0.05, 0.05); }
    if ('roughness' in mat) mat.roughness = Math.min(Math.max(mat.roughness || 0.7, 0.4), 0.92);
    if ('metalness' in mat) mat.metalness = Math.min(mat.metalness || 0, 0.2);
    if ('envMapIntensity' in mat) mat.envMapIntensity = 0.45;
    mat.needsUpdate = true;
  }
  function linkNameOf(c) {
    let o = c;
    while (o) { if (o.isURDFLink && o.name) return String(o.name); o = o.parent; }
    return '';
  }
  function themeEnvironment(mat, link) {
    tameMat(mat);
    const n = link.toLowerCase();
    if (/cooktop|hotplate|ceran|stove/.test(n)) {
      mat.color.setHex(0x0a0b0d); mat.map = null; mat.roughness = 0.18; mat.metalness = 0.15;
    } else if (/island_countertop|countertop|worktop/.test(n)) {
      mat.color.setHex(0xffffff); mat.map = WOOD_COUNTER; mat.roughness = 0.55; mat.metalness = 0.02;
    } else if (/coffee_table|table_area|bedside_table|table_top|dining/.test(n)) {
      mat.color.setHex(0xffffff); mat.map = WOOD_TABLE; mat.roughness = 0.5; mat.metalness = 0.02;
    } else if (/handle|tap_body|tap_handle|sink|faucet/.test(n)) {
      mat.color.setHex(0xc6ccd4); mat.map = null; mat.roughness = 0.28; mat.metalness = 0.85;
    } else if (/cabinet|drawer|door|wardrobe|dishwasher|oven|coffe_machine|island_back|island_waterfall|side_[ab]|fridge/.test(n)) {
      mat.color.setHex(0x1b1d21); mat.map = null; mat.roughness = 0.42; mat.metalness = 0.12;
    } else if (/wall/.test(n)) {
      mat.color.setHex(0xd9d4cb); mat.map = null; mat.roughness = 0.95; mat.metalness = 0.0;
    }
    mat.needsUpdate = true;
  }
  const OWN_LIGHTS = new Set();
  scene3.traverse(function (c) { if (c.isLight) OWN_LIGHTS.add(c); });
  function stripImportedLights(root) {
    const kill = [];
    root.traverse(function (c) { if (c.isLight && !OWN_LIGHTS.has(c)) kill.push(c); });
    kill.forEach(function (l) { if (l.parent) l.parent.remove(l); });
  }
  function tameModel(entry) {
    stripImportedLights(entry.obj);
    entry.obj.traverse(function (c) {
      if (!c.isMesh || c.userData._tamed) return;
      c.castShadow = true; c.receiveShadow = true;
      const link = entry.robot ? '' : linkNameOf(c);
      const mats = Array.isArray(c.material) ? c.material : [c.material];
      mats.forEach(function (m) { entry.robot ? tameMat(m) : themeEnvironment(m, link); });
      c.userData._tamed = true;
    });
  }

  // %% scene bundles: one cached slot per episode, swapped under worldRoot
  // slot: {group, models[], robotModel, objectMeshes{}, traj, sc, ready}
  const slots = {};
  let active = null;             // active slot
  let activeName = null;

  function makeUrdfLoader(manager) {
    const loader = new URDFLoader(manager);
    loader.packages = {};
    loader.parseCollision = false;
    const def = loader.defaultMeshLoader.bind(loader);
    loader.loadMeshCb = function (path, mgr, done) {
      if (/\.obj$/i.test(path)) {
        new THREE.OBJLoader(mgr).load(path, function (o) { done(o); },
          undefined, function () { done(new THREE.Object3D()); });
      } else {
        def(path, mgr, done);
      }
    };
    return loader;
  }

  function loadSlot(name, onReady) {
    if (slots[name]) { onReady(slots[name]); return; }
    const base = SCENES + name + '/';
    const slot = {
      group: new THREE.Group(), models: [], robotModel: null,
      objectMeshes: {}, traj: null, sc: null, ready: false, settleT0: null,
    };
    slots[name] = slot;
    slot.group.visible = false;
    worldRoot.add(slot.group);
    statusEl.textContent = 'Loading ' + name + '…';
    fetch(base + 'scene.json').then(function (r) { return r.json(); }).then(function (sc) {
      slot.sc = sc;
      const manager = new THREE.LoadingManager();
      sc.models.forEach(function (m) {
        makeUrdfLoader(manager).load(base + m.urdf, function (obj) {
          const entry = { name: m.name, prefix: m.prefix || '', robot: !!m.robot, obj: obj };
          slot.models.push(entry);
          if (m.robot) slot.robotModel = entry;
          slot.group.add(obj);
          needsRender = true;
        });
      });
      (sc.objects || []).forEach(function (o) {
        const mat = new THREE.MeshStandardMaterial({
          color: new THREE.Color(o.color || '#cccccc'),
          roughness: 0.5, metalness: 0.05, envMapIntensity: 0.7,
        });
        new THREE.STLLoader(manager).load(base + o.mesh, function (geom) {
          const mesh = new THREE.Mesh(geom, mat);
          mesh.castShadow = true; mesh.receiveShadow = true;
          const g = new THREE.Group();
          g.add(mesh);
          slot.objectMeshes[o.key] = g;
          slot.group.add(g);
          needsRender = true;
        });
      });
      manager.onLoad = function () {
        fetch(base + (sc.trajectory || 'trajectory.json'))
          .then(function (r) { return r.ok ? r.json() : null; })
          .then(function (traj) {
            slot.traj = traj;
            slot.ready = true;
            slot.settleT0 = clock.getElapsedTime();
            slot.models.forEach(tameModel);
            statusEl.textContent = '';
            onReady(slot);
            needsRender = true;
          });
      };
    }).catch(function (e) {
      statusEl.innerHTML = 'Scene bundle not found.<br>Clone cram2/cram-scenes to defense/scenes — see defense/README.md';
    });
  }

  function dropGroundToSlot(slot) {
    const envs = slot.models.filter(function (m) { return !m.robot; });
    if (!envs.length) return;
    const box = new THREE.Box3();
    envs.forEach(function (m) { box.expandByObject(m.obj); });
    if (!isFinite(box.min.y) || !isFinite(box.min.x)) return;
    const w = (box.max.x - box.min.x) + 0.4;
    const d = (box.max.z - box.min.z) + 0.4;
    ground.geometry.dispose();
    ground.geometry = new THREE.PlaneGeometry(w, d);
    ground.position.set((box.min.x + box.max.x) / 2, box.min.y + 0.002, (box.min.z + box.max.z) / 2);
    ground.visible = true;
    floorTex.repeat.set(w / 0.6, d / 0.6);
    floorTex.needsUpdate = true;
    needsRender = true;
  }

  // %% playback (from cramera, dt-based so speed is adjustable)
  let playing = true, playhead = 0, speed = 2, scrubbing = false, lastSegIdx = -1;
  let sweep = null;              // {t0, prev, amount} while the title sweep runs
  let onTitle = false;
  const _p0 = new THREE.Vector3(), _p1 = new THREE.Vector3();
  const _q0 = new THREE.Quaternion(), _q1 = new THREE.Quaternion();
  function setPose(obj, a, b, t) {
    _p0.set(a[0], a[1], a[2]); _p1.set(b[0], b[1], b[2]);
    obj.position.copy(_p0).lerp(_p1, t);
    _q0.set(a[3], a[4], a[5], a[6]); _q1.set(b[3], b[4], b[5], b[6]);
    obj.quaternion.copy(_q0).slerp(_q1, t);
  }
  function modelByPrefix(slot, prefix) {
    for (let i = 0; i < slot.models.length; i++)
      if (slot.models[i].prefix === prefix) return slot.models[i];
    return null;
  }
  function applyFrame(slot, f) {
    const traj = slot.traj;
    if (!traj) return;
    const F = traj.frames, i0 = Math.floor(f), i1 = Math.min(i0 + 1, F.length - 1), t = f - i0;
    const f0 = F[i0], f1 = F[i1];
    for (const k in f0) {
      const cut = k.indexOf('/');
      const m = modelByPrefix(slot, cut < 0 ? '' : k.slice(0, cut));
      if (!m) continue;
      const j = m.obj.joints[cut < 0 ? k : k.slice(cut + 1)];
      if (j) j.setJointValue(f0[k] + ((f1[k] !== undefined ? f1[k] : f0[k]) - f0[k]) * t);
    }
    if (slot.robotModel && traj.base && traj.base[i0]) {
      setPose(slot.robotModel.obj, traj.base[i0], traj.base[i1] || traj.base[i0], t);
    }
    if (traj.objects) {
      const o0 = traj.objects[i0], o1 = traj.objects[i1] || o0;
      for (const name in slot.objectMeshes) {
        if (o0[name]) setPose(slot.objectMeshes[name], o0[name], o1[name] || o0[name], t);
      }
    }
  }

  // %% HUD wiring (deck elements)
  const chipsBox = document.getElementById('epChips');
  const desigEl = document.getElementById('epDesig');
  const metaEl = document.getElementById('epMeta');
  const legendEl = document.getElementById('epLegend');
  const scrubEl = document.getElementById('epScrub');
  const timeEl = document.getElementById('epTime');
  const playBtn = document.getElementById('epPlay');
  const speedBtn = document.getElementById('epSpeed');

  function segIndexAt(sc, frame) {
    const segs = sc.segments || [];
    for (let i = segs.length - 1; i >= 0; i--) if (frame >= segs[i].start) return i;
    return 0;
  }
  function fillHud(slot) {
    const sc = slot.sc;
    chipsBox.innerHTML = (sc.segments || []).map(function (s) {
      return '<span class="pc">' + s.step.replace(/_/g, ' ') + '</span>';
    }).join('');
    Array.prototype.forEach.call(chipsBox.querySelectorAll('.pc'), function (c, i) {
      c.addEventListener('click', function () {
        playhead = sc.segments[i].start; lastSegIdx = -1; needsRender = true;
      });
    });
    legendEl.innerHTML = (sc.objects || []).map(function (o) {
      return '<span style="display:inline-flex;align-items:center;gap:5px;margin-right:12px;">' +
        '<span style="width:8px;height:8px;border-radius:50%;background:' + o.color +
        ';border:1px solid var(--line2);display:inline-block;"></span>' +
        o.id.replace(/_/g, ' ') + '</span>';
    }).join('');
    const fps = (slot.traj && (slot.traj.fps || slot.traj.framesPerSecond)) || 30;
    metaEl.innerHTML = 'robot: ' + ROBOT_LABEL[activeName] + ' · world: apartment · ' +
      (slot.traj ? slot.traj.frames.length : 0) + ' frames @ ' + fps + ' fps<br>' +
      'source: cram2/cram-scenes · full meshes &amp; textures · cramera stack';
    lastSegIdx = -1;
  }
  function updateHud(slot) {
    const sc = slot.sc, traj = slot.traj;
    if (!sc || !traj) return;
    const fi = Math.floor(playhead);
    const si = segIndexAt(sc, fi);
    if (si !== lastSegIdx) {
      lastSegIdx = si;
      Array.prototype.forEach.call(chipsBox.querySelectorAll('.pc'), function (c, i) {
        c.classList.toggle('now', i === si);
        c.classList.toggle('done', i < si);
      });
      const s = (sc.segments || [])[si];
      if (s) {
        desigEl.innerHTML = s.picks
          ? '<span style="color:var(--acc)">Transport</span>(object=<span style="color:var(--ok)">' + s.picks +
            '</span>,<br>&nbsp;&nbsp;&nbsp;&nbsp;target=<span style="color:var(--ok)">(' +
            (s.place ? s.place.slice(0, 3).map(function (v) { return v.toFixed(2); }).join(', ') : '…') +
            ')</span>, robot=<span style="color:var(--ok)">' + ROBOT_LABEL[activeName] + '</span>)'
          : '<span style="color:var(--acc)">ParkArms</span>(arm=<span style="color:var(--ok)">BOTH</span>)';
      }
    }
    const n = traj.frames.length;
    if (!scrubbing) scrubEl.value = String(Math.round(fi / (n - 1) * 1000));
    const fps = traj.fps || traj.framesPerSecond || 30;
    timeEl.textContent = (fi / fps).toFixed(1) + 's / ' + (n / fps).toFixed(0) + 's · f' + fi;
  }

  function setScene(name) {
    activeName = name;
    document.querySelectorAll('#epHud .actbtns button').forEach(function (b) {
      b.classList.toggle('sel', b.dataset.ep === name);
    });
    playhead = 0; lastSegIdx = -1;
    if (active) active.group.visible = false;
    loadSlot(name, function (slot) {
      if (activeName !== name) return;      // user switched again mid-load
      active = slot;
      slot.group.visible = true;
      dropGroundToSlot(slot);
      applyFrame(slot, 0);
      fillHud(slot);
      frameCamera(slot);
      if (onTitle) startSweep();
      needsRender = true;
    });
  }

  // %% camera
  let follow = true;
  const _target = new THREE.Vector3(), _base = new THREE.Vector3();
  function frameCamera(slot) {
    if (!slot.robotModel) return;
    const box = new THREE.Box3().setFromObject(slot.robotModel.obj);
    const c = box.getCenter(new THREE.Vector3());
    controls.target.copy(c);
    camera.position.set(c.x + 3.2, c.y + 1.6, c.z + 3.4);
    controls.update();
    needsRender = true;
  }
  function robotCenter(out) {
    if (!active || !active.robotModel) return false;
    active.robotModel.obj.getWorldPosition(_base);
    out.copy(_base); out.y += 0.6;
    return true;
  }

  // %% SSAO (optional, from cramera)
  let composer = null, ssaoPass = null;
  (function setupSSAO() {
    if (!(THREE.EffectComposer && THREE.RenderPass && THREE.SSAOPass && THREE.ShaderPass && THREE.CopyShader)) return;
    try {
      const w = host.clientWidth || 800, h = host.clientHeight || 600;
      composer = new THREE.EffectComposer(renderer);
      composer.addPass(new THREE.RenderPass(scene3, camera));
      ssaoPass = new THREE.SSAOPass(scene3, camera, w, h);
      ssaoPass.kernelRadius = 0.12;
      ssaoPass.minDistance = 0.001;
      ssaoPass.maxDistance = 0.04;
      composer.addPass(ssaoPass);
      const copy = new THREE.ShaderPass(THREE.CopyShader);
      copy.renderToScreen = true;
      composer.addPass(copy);
    } catch (e) { composer = null; }
  })();
  function renderFrame() {
    if (composer) composer.render();
    else renderer.render(scene3, camera);
  }

  // %% render loop
  const clock = new THREE.Clock();
  function tick() {
    requestAnimationFrame(tick);
    const dt = Math.min(0.1, clock.getDelta());
    if (active && active.settleT0 !== null &&
        clock.getElapsedTime() - active.settleT0 < MATERIAL_SETTLE_SECONDS) {
      active.models.forEach(tameModel);
      needsRender = true;
    }
    const moved = controls.update();
    if (sweep) {
      const u = Math.min(1, (clock.getElapsedTime() - sweep.t0) / TITLE_SWEEP_SECONDS);
      const eased = u * u * (3 - 2 * u);
      rotateAroundTarget((eased - sweep.prev) * sweep.amount);
      sweep.prev = eased;
      if (u >= 1) sweep = null;
      needsRender = true;
    }
    if (active && active.traj && playing && !scrubbing) {
      const fps = active.traj.fps || active.traj.framesPerSecond || 30;
      playhead += dt * fps * speed;
      if (playhead >= active.traj.frames.length - 1) playhead = 0;
      needsRender = true;
    }
    if (active && active.traj && needsRender) {
      applyFrame(active, playhead);
      updateHud(active);
      if (follow && robotCenter(_target)) controls.target.lerp(_target, 0.06);
    }
    if (!needsRender && !moved && !controls.autoRotate) return;
    renderFrame();
    needsRender = false;
  }
  tick();

  function resize() {
    const w = host.clientWidth, h = host.clientHeight;
    if (!w || !h) return;
    renderer.setSize(w, h, false);
    if (composer) composer.setSize(w, h);
    if (ssaoPass) ssaoPass.setSize(w, h);
    camera.aspect = w / h; camera.updateProjectionMatrix();
    needsRender = true;
  }
  window.addEventListener('resize', resize);
  new ResizeObserver(resize).observe(host);

  // %% controls wiring
  playBtn.addEventListener('click', function () {
    playing = !playing;
    playBtn.textContent = playing ? '⏸' : '▶';
  });
  speedBtn.addEventListener('click', function () {
    speed = speed >= 4 ? 1 : speed * 2;
    speedBtn.textContent = speed + '×';
  });
  scrubEl.addEventListener('pointerdown', function () { scrubbing = true; });
  window.addEventListener('pointerup', function () { scrubbing = false; });
  scrubEl.addEventListener('input', function () {
    if (!active || !active.traj) return;
    playhead = (+scrubEl.value) / 1000 * (active.traj.frames.length - 1);
    lastSegIdx = -1;
    needsRender = true;
  });
  document.querySelectorAll('#epHud .actbtns button').forEach(function (b) {
    b.addEventListener('click', function () { setScene(b.dataset.ep); });
  });

  // %% slide routing: title slide shows the ambient view, episode slide the HUD one
  //: title-slide camera sweep: total angle in degrees and duration in seconds —
  //: the camera pans once from left to right, eases out and then holds still
  const TITLE_SWEEP_DEGREES = 24;
  const TITLE_SWEEP_SECONDS = 22;
  const _sweepUp = new THREE.Vector3(0, 1, 0);
  const _sweepOff = new THREE.Vector3();
  function rotateAroundTarget(angle) {
    _sweepOff.copy(camera.position).sub(controls.target);
    _sweepOff.applyAxisAngle(_sweepUp, angle);
    camera.position.copy(controls.target).add(_sweepOff);
  }
  function startSweep() {
    const amount = THREE.MathUtils.degToRad(TITLE_SWEEP_DEGREES);
    rotateAroundTarget(amount / 2);          // start panned to the left edge
    sweep = { t0: clock.getElapsedTime(), prev: 0, amount: -amount };
    needsRender = true;
  }
  function moveTo(stage, ambient) {
    if (host.parentElement !== stage) stage.appendChild(host);
    onTitle = ambient;
    controls.enabled = !ambient;
    sweep = null;
    if (ambient) startSweep();
    resize();
    needsRender = true;
  }
  window.DeckPlayer = {
    onSlide: function (id) {
      if (id === 's1' && titleStage) moveTo(titleStage, true);
      else if (id === 'sEp') moveTo(epStage, false);
    },
  };

  applyPlayerTheme();
  setScene(EPISODE_NAMES[0]);
  // route the initial slide (the deck's go() may have run before this script)
  const on = document.querySelector('.slide.on');
  if (on) window.DeckPlayer.onSlide(on.id);
})();
