// The coraplex execution environments a generated demo can run its plan in.
// Collision avoidance is the only thing that separates them, and the two output styles
// spell it differently -- a flat script enters the environment by name, a
// RobotDemonstration takes the flag -- so both spellings live together here.
(function () {
  'use strict';

  const ENVIRONMENTS = [
    {
      name: 'simulated_robot',
      label: 'off — faster, may clip through obstacles',
      collisionAvoidance: false,
    },
    {
      name: 'simulated_robot_advanced',
      label: 'on — plan around obstacles',
      collisionAvoidance: true,
    },
  ];

  window.ExecutionEnvironments = {
    // every environment to offer, in the order they are offered
    all: function () { return ENVIRONMENTS.slice(); },
    // the environment of that name, falling back to the first one offered, so an
    // unknown name never turns collision avoidance on behind the user's back
    byName: function (name) {
      const found = ENVIRONMENTS.filter(function (e) { return e.name === name; });
      return found.length ? found[0] : ENVIRONMENTS[0];
    },
  };
})();
