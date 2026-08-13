/* ============================================================================
 * core/obj-mesh-material.js — finds the one mesh inside an OBJLoader result that
 * a URDF <material> should be applied to.
 *
 * URDFLoader assigns a URDF link's <material> to the object its mesh loader hands
 * back only when that object `instanceof THREE.Mesh`. THREE.OBJLoader never returns
 * a bare Mesh though: it always wraps the parsed geometry in a Group, even for an
 * OBJ file (such as one written by trimesh) that contains exactly one mesh and no
 * per-face materials of its own. Left as the Group, the URDF's colour is silently
 * dropped and the mesh renders in OBJLoader's own default material instead.
 *
 * Handing URDFLoader the single mesh inside that Group -- instead of the Group
 * itself -- lets the `instanceof THREE.Mesh` check succeed. An OBJ that already
 * carries its own per-face materials (a companion .mtl was loaded) is left as its
 * Group, since a single URDF colour would flatten the object's variation.
 * ==========================================================================*/
(function () {
  'use strict';

  function singleMeshChild(object) {
    if (object.isMesh) {
      return object;
    }
    const meshes = [];
    (function walk(node) {
      (node.children || []).forEach(function (child) {
        if (child.isMesh) {
          meshes.push(child);
        }
        walk(child);
      });
    })(object);
    return meshes.length === 1 ? meshes[0] : null;
  }

  window.ObjMeshMaterial = { singleMeshChild: singleMeshChild };
})();
