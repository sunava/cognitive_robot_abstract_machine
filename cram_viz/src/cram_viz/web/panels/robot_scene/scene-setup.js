/* ============================================================================
 * panels/robot_scene/scene-setup.js — three.js bootstrap for the robot-scene
 * panel: renderer, camera, studio lighting/backdrop, ground plane, orbit
 * controls and the optional SSAO post-processing chain.
 *
 * `create(container)` returns everything the panel composes on top of, plus
 * `resize()`/`dispose()` so the panel can tear the whole environment down.
 * ==========================================================================*/
window.RobotSceneEnvironment = (function () {
  function create(container) {
    const scene3 = new THREE.Scene();
    scene3.background = null;

    const camera = new THREE.PerspectiveCamera(45, 1, 0.01, 200);
    camera.position.set(3, 2.4, 4);

    const renderer = new THREE.WebGLRenderer({ antialias: true, alpha: true });
    renderer.setPixelRatio(Math.min(window.devicePixelRatio, 2));
    renderer.shadowMap.enabled = true;
    renderer.shadowMap.type = THREE.PCFSoftShadowMap;
    renderer.outputEncoding = THREE.sRGBEncoding;
    renderer.toneMapping = THREE.ACESFilmicToneMapping;
    renderer.toneMappingExposure = 0.95;
    container.appendChild(renderer.domElement);

    // soft vertical-gradient studio backdrop
    (function () {
      const canvas = document.createElement('canvas');
      canvas.width = 2; canvas.height = 256;
      const context = canvas.getContext('2d');
      const gradient = context.createLinearGradient(0, 0, 0, 256);
      gradient.addColorStop(0, '#232833'); gradient.addColorStop(0.55, '#151922'); gradient.addColorStop(1, '#0c0e13');
      context.fillStyle = gradient; context.fillRect(0, 0, 2, 256);
      scene3.background = new THREE.CanvasTexture(canvas);
    })();

    // image-based lighting (imported .dae lights are stripped on load)
    let pmrem = null;
    if (THREE.RoomEnvironment && THREE.PMREMGenerator) {
      try {
        pmrem = new THREE.PMREMGenerator(renderer);
        scene3.environment = pmrem.fromScene(new THREE.RoomEnvironment(), 0.04).texture;
        pmrem.dispose();
      } catch (e) { /* optional */ }
    }

    scene3.add(new THREE.HemisphereLight(0xf4efe6, 0x2a2d33, 0.32));
    scene3.add(new THREE.AmbientLight(0xfff4e6, 0.05));
    const keyLight = new THREE.DirectionalLight(0xfff2df, 1.05);
    keyLight.position.set(4, 7, 3);
    keyLight.castShadow = true;
    keyLight.shadow.mapSize.set(4096, 4096);
    keyLight.shadow.camera.near = 0.5; keyLight.shadow.camera.far = 30;
    keyLight.shadow.camera.left = -6; keyLight.shadow.camera.right = 6;
    keyLight.shadow.camera.top = 6; keyLight.shadow.camera.bottom = -6;
    keyLight.shadow.bias = -0.0003;
    keyLight.shadow.normalBias = 0.02;
    keyLight.shadow.radius = 3;
    scene3.add(keyLight);
    const backLight = new THREE.DirectionalLight(0xdfe6f2, 0.3);
    backLight.position.set(-4, 4, -3);
    scene3.add(backLight);

    const floorTex = window.RobotSceneModelLoader.floorTexture();
    floorTex.repeat.set(12, 12);
    const ground = new THREE.Mesh(
      new THREE.PlaneGeometry(60, 60),
      new THREE.MeshStandardMaterial({ map: floorTex, roughness: 0.85, metalness: 0.0 })
    );
    ground.rotation.x = -Math.PI / 2;
    ground.receiveShadow = true;
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

    // ---- SSAO --------------------------------------------------------------
    let composer = null, ssaoPass = null;
    if (THREE.EffectComposer && THREE.RenderPass && THREE.SSAOPass && THREE.ShaderPass && THREE.CopyShader) {
      try {
        const w = container.clientWidth || 800, h = container.clientHeight || 600;
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
    }

    function renderFrame() {
      if (composer) composer.render();
      else renderer.render(scene3, camera);
    }

    function resize() {
      const w = container.clientWidth, h = container.clientHeight;
      if (!w || !h) return;
      renderer.setSize(w, h, false);
      if (composer) composer.setSize(w, h);
      if (ssaoPass) ssaoPass.setSize(w, h);
      camera.aspect = w / h; camera.updateProjectionMatrix();
    }

    const resizeObserver = new ResizeObserver(resize);
    resizeObserver.observe(container);
    window.addEventListener('resize', resize);

    function dispose() {
      window.removeEventListener('resize', resize);
      resizeObserver.disconnect();
      controls.dispose();
      renderer.dispose();
      if (composer) composer.dispose();
    }

    return {
      scene3: scene3, camera: camera, renderer: renderer, controls: controls, worldRoot: worldRoot,
      ground: ground, floorTex: floorTex,
      renderFrame: renderFrame, resize: resize, dispose: dispose,
    };
  }

  return { create: create };
})();
