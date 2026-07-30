/* ============================================================================
 * panels/robot_scene/model-loader.js — URDF loading & material theming.
 *
 * Owns the procedural floor/wood textures, the URDFLoader factory (routes
 * .obj mesh refs through THREE.OBJLoader since URDFLoader's default mesh
 * loader only understands STL/DAE), and taming imported materials so meshes
 * authored in unrelated tools read as one consistent scene (robot vs
 * furniture palettes, stripped imported lights).
 * ==========================================================================*/
window.RobotSceneModelLoader = (function () {
  function floorTexture() {
    const canvas = document.createElement('canvas'); canvas.width = 512; canvas.height = 512;
    const context = canvas.getContext('2d');
    context.fillStyle = '#b9a184'; context.fillRect(0, 0, 512, 512);
    const plank = 512 / 6;
    for (let row = 0; row < 6; row++) {
      const shade = 168 + ((row * 37) % 40) - 20;
      context.fillStyle = 'rgb(' + (shade + 20) + ',' + (shade - 4) + ',' + (shade - 30) + ')';
      context.fillRect(0, row * plank, 512, plank - 2);
      context.strokeStyle = 'rgba(60,40,25,0.35)'; context.lineWidth = 2;
      context.strokeRect(0, row * plank, 512, plank - 2);
      for (let i = 0; i < 40; i++) {
        context.strokeStyle = 'rgba(90,60,40,0.06)';
        const y = row * plank + Math.random() * plank;
        context.beginPath(); context.moveTo(0, y); context.lineTo(512, y + (Math.random() - 0.5) * 4); context.stroke();
      }
    }
    const texture = new THREE.CanvasTexture(canvas);
    texture.wrapS = texture.wrapT = THREE.RepeatWrapping; texture.anisotropy = 8;
    return texture;
  }

  function woodTexture(base, streak) {
    const canvas = document.createElement('canvas'); canvas.width = 256; canvas.height = 256;
    const context = canvas.getContext('2d');
    context.fillStyle = base; context.fillRect(0, 0, 256, 256);
    for (let i = 0; i < 220; i++) {
      const y = Math.floor((i / 220) * 256) + (i % 3);
      context.strokeStyle = 'rgba(' + streak + ',' + (0.03 + (i % 5) * 0.015) + ')';
      context.lineWidth = 1 + (i % 3);
      context.beginPath(); context.moveTo(0, y);
      for (let x = 0; x <= 256; x += 16) context.lineTo(x, y + Math.sin((x + i) * 0.05) * 1.5);
      context.stroke();
    }
    const texture = new THREE.CanvasTexture(canvas);
    texture.wrapS = texture.wrapT = THREE.RepeatWrapping; texture.anisotropy = 8;
    return texture;
  }

  function makeUrdfLoader(manager) {
    const loader = new URDFLoader(manager);
    loader.packages = {};
    loader.parseCollision = false;
    const defaultMeshLoader = loader.defaultMeshLoader.bind(loader);
    loader.loadMeshCb = function (path, mgr, done) {
      if (/\.obj$/i.test(path)) {
        new THREE.OBJLoader(mgr).load(path, function (obj) { done(obj); },
          undefined, function () { done(new THREE.Object3D()); });
      } else {
        defaultMeshLoader(path, mgr, done);
      }
    };
    return loader;
  }

  // one instance per scene: captures which lights are the scene's own so
  // imported models' baked-in lights (harsh, uncontrolled) can be stripped
  // on load without touching the studio lighting rig.
  function ModelTamer(scene3) {
    const ownLights = new Set();
    scene3.traverse(function (c) { if (c.isLight) ownLights.add(c); });
    const woodCounter = woodTexture('#9c6b3f', '70,45,25');
    const woodTable = woodTexture('#b98a52', '90,60,35');

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

    function linkNameOf(object3d) {
      let node = object3d;
      while (node) { if (node.isURDFLink && node.name) return String(node.name); node = node.parent; }
      return '';
    }

    // furniture palette, keyed on common link-name vocabulary — applies to
    // any environment model; unmatched links keep their tamed authored look
    function themeEnvironment(mat, link) {
      tameMat(mat);
      const name = link.toLowerCase();
      if (/cooktop|hotplate|ceran|stove/.test(name)) {
        mat.color.setHex(0x0a0b0d); mat.map = null; mat.roughness = 0.18; mat.metalness = 0.15;
      } else if (/island_countertop|countertop|worktop/.test(name)) {
        mat.color.setHex(0xffffff); mat.map = woodCounter; mat.roughness = 0.55; mat.metalness = 0.02;
      } else if (/coffee_table|table_area|bedside_table|table_top|dining/.test(name)) {
        mat.color.setHex(0xffffff); mat.map = woodTable; mat.roughness = 0.5; mat.metalness = 0.02;
      } else if (/handle|tap_body|tap_handle|sink|faucet/.test(name)) {
        mat.color.setHex(0xc6ccd4); mat.map = null; mat.roughness = 0.28; mat.metalness = 0.85;
      } else if (/cabinet|drawer|door|wardrobe|dishwasher|oven|coffe_machine|island_back|island_waterfall|side_[ab]|fridge/.test(name)) {
        mat.color.setHex(0x1b1d21); mat.map = null; mat.roughness = 0.42; mat.metalness = 0.12;
      } else if (/wall/.test(name)) {
        mat.color.setHex(0xd9d4cb); mat.map = null; mat.roughness = 0.95; mat.metalness = 0.0;
      }
      mat.needsUpdate = true;
    }

    function stripImportedLights(root) {
      const stale = [];
      root.traverse(function (c) { if (c.isLight && !ownLights.has(c)) stale.push(c); });
      stale.forEach(function (light) { if (light.parent) light.parent.remove(light); });
    }

    function tameModel(entry) {
      stripImportedLights(entry.obj);
      entry.obj.traverse(function (c) {
        if (!c.isMesh || c.userData._tamed) return;
        c.castShadow = true; c.receiveShadow = true;
        const link = entry.robot ? '' : linkNameOf(c);
        const materials = Array.isArray(c.material) ? c.material : [c.material];
        materials.forEach(function (m) { entry.robot ? tameMat(m) : themeEnvironment(m, link); });
        c.userData._tamed = true;
      });
    }

    this.upgrade = function (models) { models.forEach(tameModel); };
  }

  return {
    floorTexture: floorTexture,
    woodTexture: woodTexture,
    makeUrdfLoader: makeUrdfLoader,
    ModelTamer: ModelTamer,
  };
})();
