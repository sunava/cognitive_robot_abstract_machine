/* ============================================================================
 * core/scene_picker.js — resolves the header's robot/environment/task/version
 * dropdowns to one onboarded scene bundle.
 *
 * There is no independent mixing across the four axes: each onboarded scene is a
 * fixed (robot, environment, task, version) recording. This module only looks up,
 * among the scenes actually onboarded, which one matches a chosen combination — so the
 * four dropdowns in robot_scene/panel.js look independent while only ever landing on
 * one that was actually recorded. Narrowing any earlier axis (robot, say) narrows what
 * the later ones (environment, task, version) can offer; version is always enough to
 * disambiguate scenes that would otherwise share every other axis (e.g. two saved live
 * recordings of the same robot/environment/task), so there is no separate flat "scene
 * name" picker.
 *
 * `scenes` is `scenes/index.json`'s `scenes` array: entries shaped like
 * {name, robot, environment, task, version}, where `environment` is null for a
 * bench-only scene (no environment model, e.g. a robot recorded alone on a bench).
 * ==========================================================================*/
(function () {
  'use strict';

  function uniq(values) {
    const seen = {}, out = [];
    values.forEach(function (v) {
      const key = String(v);
      if (!seen[key]) { seen[key] = true; out.push(v); }
    });
    return out;
  }

  // scenes still matching every axis narrowed so far
  function matching(scenes, robot, environment, task) {
    return (scenes || []).filter(function (s) {
      if (robot !== undefined && s.robot !== robot) return false;
      if (environment !== undefined && (s.environment || null) !== (environment || null)) return false;
      if (task !== undefined && s.task !== task) return false;
      return true;
    });
  }

  // every robot with at least one onboarded scene
  function robots(scenes) {
    return uniq((scenes || []).map(function (s) { return s.robot; }));
  }

  // every environment onboarded together with the given robot (null included
  // for a bench-only recording), in the order scenes first offers them
  function environments(scenes, robot) {
    return uniq(matching(scenes, robot).map(function (s) { return s.environment; }));
  }

  // every task onboarded together with the given (robot, environment) pair
  function tasks(scenes, robot, environment) {
    return uniq(matching(scenes, robot, environment).map(function (s) { return s.task; }));
  }

  // every version onboarded for the given (robot, environment, task) triple
  function versions(scenes, robot, environment, task) {
    return uniq(matching(scenes, robot, environment, task).map(function (s) { return s.version; }));
  }

  // the scene bundle recorded for this exact (robot, environment, task, version)
  // combination, or null if it was never onboarded
  function sceneFor(scenes, robot, environment, task, version) {
    const env = environment || null;
    const match = (scenes || []).find(function (s) {
      return s.robot === robot && (s.environment || null) === env
        && s.task === task && s.version === version;
    });
    return match ? match.name : null;
  }

  // the (robot, environment, task, version) a named scene was recorded with, or null
  // if the name isn't in the index
  function describe(scenes, name) {
    const match = (scenes || []).find(function (s) { return s.name === name; });
    return match
      ? { robot: match.robot, environment: match.environment || null, task: match.task, version: match.version }
      : null;
  }

  window.ScenePicker = {
    robots: robots,
    environments: environments,
    tasks: tasks,
    versions: versions,
    sceneFor: sceneFor,
    describe: describe,
  };
})();
