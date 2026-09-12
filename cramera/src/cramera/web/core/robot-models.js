/* ============================================================================
 * core/robot-models.js — the robots among the loaded models, and what describes each.
 *
 * A scene bundle lists its environment and then one model per robot, each under its own
 * world prefix ("walker_s2_description", "uMe"). The viewer loads those models
 * asynchronously, so the robots are ordered by their place in the bundle rather than by
 * whichever URDF finished first: the first of them is the *primary* robot — the one a
 * live `/state` `base` and a recording's `trajectory.base` describe on their own, and
 * the one everything robot-shaped falls back to. Every other robot is posed from the
 * per-prefix `modelBases` the bridge streams and the recording stores beside them.
 *
 * A robot's parts (link name -> part name, e.g. PR2LeftArm) come from its own payload:
 * `scene.robots` carries one per robot, while a bundle written before robots became a
 * list carries only the first, as `scene.robot`. Two robots can declare the very same
 * link names, so a part table belongs to one model rather than to the scene.
 *
 * Pure lookups, no DOM and no three.js, so they are testable under node.
 * ==========================================================================*/
(function (global) {
  'use strict';

  const MOVED_EPSILON = 1e-4;
  /* How far a base pose must differ from where the model already stands to count as a
     move, in metres and in quaternion units. A pose re-sent unchanged — which is every
     snapshot of a robot that is standing still — must not make that robot the acting
     one and pull the camera off the robot that is actually driving. */

  /* The robot entries among `models`, in bundle order: `order` is the index the bundle
     listed the model at, which is not the order the URDFs finish loading in. */
  function robots(models) {
    return (models || [])
      .filter(function (model) { return !!model.robot; })
      .sort(function (a, b) { return (a.order || 0) - (b.order || 0); });
  }

  /* The primary robot among `models`, or null while no robot has loaded yet. */
  function primary(models) {
    return robots(models)[0] || null;
  }

  /* One payload per robot of the bundle. `scene.robots` is the list; `scene.robot` is
     the first robot alone, all a bundle written before the list carries. */
  function payloads(scene) {
    if (scene && scene.robots && scene.robots.length) return scene.robots;
    return scene && scene.robot ? [scene.robot] : [];
  }

  /* The payload describing `model`: by world prefix, then by name. A bundle with a
     single robot and a single payload always belongs together, whatever the two sides
     called it — that is every scene recorded before robots became a list.

     `robotCount` is how many robot models the scene has, so that fallback cannot pair
     the one payload of an old bundle with the wrong robot of a new one. */
  function payloadFor(payloadList, model, robotCount) {
    const list = payloadList || [];
    let index;
    for (index = 0; index < list.length; index += 1) {
      if (list[index].prefix && list[index].prefix === model.prefix) return list[index];
    }
    for (index = 0; index < list.length; index += 1) {
      if (list[index].name && list[index].name === model.name) return list[index];
    }
    return (list.length === 1 && robotCount === 1) ? list[0] : null;
  }

  /* link name -> part name for one robot payload. */
  function linkToPart(payload) {
    const table = {};
    const parts = (payload && payload.parts) || {};
    Object.keys(parts).forEach(function (part) {
      (parts[part] || []).forEach(function (link) { table[link] = part; });
    });
    return table;
  }

  /* The `{<robot prefix>: pose}` a recording stores for frame `index`, or null for a
     recording made before trajectories carried one base pose per robot. */
  function basesAt(trajectory, index) {
    const recorded = trajectory && trajectory.modelBases;
    return (recorded && recorded[index]) || null;
  }

  /* The pose `bases` holds for `model`, or null when it names no such model. A model
     without a prefix cannot be addressed this way — it is whatever `base` describes. */
  function baseOf(bases, model) {
    if (!bases || !model || !model.prefix) return null;
    const pose = bases[model.prefix];
    return Array.isArray(pose) ? pose : null;
  }

  /* Whether putting `object` (anything with a `position` and a `quaternion`) at `pose`
     would actually move it. */
  function moved(object, pose) {
    if (!object || !pose) return false;
    const position = object.position, quaternion = object.quaternion;
    return Math.abs(position.x - pose[0]) > MOVED_EPSILON
      || Math.abs(position.y - pose[1]) > MOVED_EPSILON
      || Math.abs(position.z - pose[2]) > MOVED_EPSILON
      || Math.abs(quaternion.x - pose[3]) > MOVED_EPSILON
      || Math.abs(quaternion.y - pose[4]) > MOVED_EPSILON
      || Math.abs(quaternion.z - pose[5]) > MOVED_EPSILON
      || Math.abs(quaternion.w - pose[6]) > MOVED_EPSILON;
  }

  global.RobotModels = {
    MOVED_EPSILON: MOVED_EPSILON,
    robots: robots,
    primary: primary,
    payloads: payloads,
    payloadFor: payloadFor,
    linkToPart: linkToPart,
    basesAt: basesAt,
    baseOf: baseOf,
    moved: moved,
  };
})(window);
