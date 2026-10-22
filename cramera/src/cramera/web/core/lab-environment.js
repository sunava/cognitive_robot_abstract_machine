/* ============================================================================
 * core/lab-environment.js — image-based lighting from a bundle's lighting description.
 *
 * An imported scene describes its lights in `rendering.environment` (lighting.json):
 *   {sun:    {direction: [x,y,z], intensity, colorTemperature, color},
 *    dome:   {color: [r,g,b], intensity},
 *    panels: [{center: [x,y,z], normal: [x,y,z], width, height, intensity, colorTemperature, color}]}
 * in the bundle frame (Z up, metres); `direction` is the way the sunlight travels and
 * `normal` the way a panel emits. `color` (linear RGB) and `colorTemperature` (kelvin)
 * are both optional; a light shows its colour times its black body, as USD lights do. three r128 has no shadowed area lights, so the
 * panels light the scene through its environment map instead: a room the size of the
 * scene with a glowing quad per panel, rendered once through PMREM.
 *
 * One shadow-casting key light carries the strongest direct light: the sun, unless
 * the panels light the floor more strongly than the sun would. Then the key shines
 * from the panels, because a room under a ceiling never sees the sun, and a key
 * along the sun would leave the whole room in the ceiling's shadow. The environment
 * map already holds all of the panels' light, unshadowed, so a key from the panels
 * adds only a share of it: enough for objects to shade what lies below them. Its
 * shadows cover only the space below the panels, so the ceiling they hang in never
 * shades the room.
 *
 * A colour temperature becomes a linear sRGB colour of unit luminance (white is D65);
 * nothing is white-balanced, so warm panels light the room warm.
 *
 * Intensities are the source renderer's (luminance for panels and dome, irradiance
 * for the sun). All are divided by one reference luminance — the panels' light spread
 * over the room's floor, else the dome's — so the lit ceiling glows at about 1 and
 * `rendering.exposure` alone sets how bright the scene is, as an auto-exposed render
 * of the source scene does.
 *
 * three.js is handed in, so it is testable under node.
 * ==========================================================================*/
(function (global) {
  'use strict';

  //: colour of a light that states neither a colour nor a colour temperature
  const WHITE = Object.freeze([1, 1, 1]);
  //: coolest and warmest colour temperature the Planckian-locus fit covers, in kelvin
  const LOCUS_RANGE = Object.freeze([1667, 25000]);
  //: the room of a description that has neither bounds nor panels to size it, bundle frame
  const FALLBACK_ROOM = [[-5, -5, 0], [5, 5, 3]];
  //: height of the ground a room without bounds stands on, bundle frame
  const GROUND = 0;
  //: index of every face of three's BoxGeometry in the bundle frame
  const Face = Object.freeze({ CEILING: 4, FLOOR: 5, WALLS: Object.freeze([0, 1, 2, 3]) });
  //: number of faces of three's BoxGeometry
  const BOX_FACES = 6;

  // %% colour temperature
  /** CIE 1931 chromaticity [x, y] of a black body (Kim et al.'s cubic fit of the Planckian locus). */
  function planckianChromaticity(kelvin) {
    const t = Math.min(LOCUS_RANGE[1], Math.max(LOCUS_RANGE[0], kelvin));
    const x = t <= 4000
      ? -0.2661239e9 / (t * t * t) - 0.2343589e6 / (t * t) + 0.8776956e3 / t + 0.179910
      : -3.0258469e9 / (t * t * t) + 2.1070379e6 / (t * t) + 0.2226347e3 / t + 0.240390;
    let y;
    if (t <= 2222) y = -1.1063814 * x * x * x - 1.34811020 * x * x + 2.18555832 * x - 0.20219683;
    else if (t <= 4000) y = -0.9549476 * x * x * x - 1.37418593 * x * x + 2.09137015 * x - 0.16748867;
    else y = 3.0817580 * x * x * x - 5.87338670 * x * x + 3.75112997 * x - 0.37001483;
    return [x, y];
  }

  /** Linear sRGB (D65) of a black body at a temperature in kelvin, at a luminance of 1. */
  function blackBody(kelvin) {
    const [x, y] = planckianChromaticity(kelvin);
    const X = x / y, Y = 1, Z = (1 - x - y) / y;
    return [
      3.2404542 * X - 1.5371385 * Y - 0.4985314 * Z,
      -0.9692660 * X + 1.8760108 * Y + 0.0415560 * Z,
      0.0556434 * X - 0.2040259 * Y + 1.0572252 * Z,
    ].map(function (channel) { return Math.max(0, channel); });
  }

  // %% the description, with unusable entries left out
  function isDirection(value) {
    return global.SceneBounds.isPoint(value) && value.some(function (component) { return component !== 0; });
  }

  function isAmount(value) {
    return Number.isFinite(value) && value >= 0;
  }

  /** Linear colour of a light: its colour times the black body of its colour temperature. */
  function colorOf(light) {
    const tint = global.SceneBounds.isPoint(light.color) ? light.color : WHITE;
    const temperature = light.colorTemperature;
    const body = Number.isFinite(temperature) && temperature > 0 ? blackBody(temperature) : WHITE;
    return tint.map(function (channel, index) { return Math.max(0, channel) * body[index]; });
  }

  class LightingDescription {
    /**
     * @param {object|null} sun {direction, intensity, color}
     * @param {object|null} dome {color, intensity}
     * @param {object[]} panels [{center, normal, width, height, intensity, color}]
     * with every `color` the light's linear colour
     */
    constructor(sun, dome, panels) {
      this.sun = sun;
      this.dome = dome;
      this.panels = panels;
    }

    /** The usable lights of a lighting.json payload. */
    static of(payload) {
      const source = payload || {};
      const sun = source.sun && isDirection(source.sun.direction) && isAmount(source.sun.intensity)
        ? { direction: source.sun.direction.slice(), intensity: source.sun.intensity, color: colorOf(source.sun) }
        : null;
      const dome = source.dome && isAmount(source.dome.intensity)
        ? { color: colorOf({ color: source.dome.color }), intensity: source.dome.intensity }
        : null;
      const panels = (Array.isArray(source.panels) ? source.panels : []).filter(function (panel) {
        return panel && global.SceneBounds.isPoint(panel.center) && isDirection(panel.normal) && isAmount(panel.intensity) &&
          Number.isFinite(panel.width) && panel.width > 0 && Number.isFinite(panel.height) && panel.height > 0;
      }).map(function (panel) {
        return {
          center: panel.center.slice(), normal: panel.normal.slice(), width: panel.width, height: panel.height,
          intensity: panel.intensity, color: colorOf(panel),
        };
      });
      return new LightingDescription(sun, dome, panels);
    }

    /** Whether no usable light is described, so a viewer keeps its own lighting. */
    get empty() {
      return !this.sun && !this.dome && !this.panels.length;
    }

    /**
     * Whether light arrives from all around — from panels or a dome — so it replaces a
     * viewer's own environment and fill lights; a sun alone lights one side only.
     */
    get surrounding() {
      return !!this.dome || this.panels.length > 0;
    }
  }

  // %% the environment
  class LabEnvironment {
    /**
     * @param {LightingDescription} lighting
     * @param {SceneBounds|null} bounds the scene's shadow bounds, which the room spans
     */
    constructor(lighting, bounds) {
      this.lighting = lighting;
      this.bounds = bounds;
      this.room = LabEnvironment.roomOf(lighting, bounds);
      const size = this.room.size();
      const floorArea = size[0] * size[1];
      //: the panels' luminous intensity spread over the room's floor: the lit ceiling's mean luminance
      this.ceilingLuminance = floorArea > 0 ? LabEnvironment.intensityOf(lighting.panels) / floorArea : 0;
      this.referenceLuminance = LabEnvironment.referenceOf(lighting, this.ceilingLuminance);
    }

    /**
     * The room box: the bounds' floor and walls, closed just above the highest panel;
     * without bounds the panels' footprint standing on the ground.
     */
    static roomOf(lighting, bounds) {
      const panels = lighting.panels;
      if (!panels.length) return bounds || global.SceneBounds.of(FALLBACK_ROOM);
      const ceiling = Math.max.apply(null, panels.map(function (panel) { return panel.center[2]; })) + LabEnvironment.PANEL_GAP;
      if (bounds) return new global.SceneBounds(bounds.min.slice(), [bounds.max[0], bounds.max[1], ceiling]);
      const reach = function (panel) { return Math.max(panel.width, panel.height) / 2; };
      const lowest = function (axis) {
        return Math.min.apply(null, panels.map(function (panel) { return panel.center[axis] - reach(panel); }));
      };
      const highest = function (axis) {
        return Math.max.apply(null, panels.map(function (panel) { return panel.center[axis] + reach(panel); }));
      };
      return new global.SceneBounds([lowest(0), lowest(1), GROUND], [highest(0), highest(1), ceiling]);
    }

    /** Luminous intensity along the normal (luminance times area), summed over the panels. */
    static intensityOf(panels) {
      return panels.reduce(function (sum, panel) { return sum + panel.intensity * panel.width * panel.height; }, 0);
    }

    /** The luminance shown at 1: the lit ceiling's, else the dome's, else the sun's reflected by white. */
    static referenceOf(lighting, ceilingLuminance) {
      if (ceilingLuminance > 0) return ceilingLuminance;
      if (lighting.dome && lighting.dome.intensity > 0) return lighting.dome.intensity;
      if (lighting.sun && lighting.sun.intensity > 0) return lighting.sun.intensity / Math.PI;
      return 1;
    }

    /** Luminance from above, relative to the reference: the lit ceiling, else the open sky. */
    overheadLuminance() {
      if (this.lighting.panels.length) return this.ceilingLuminance / this.referenceLuminance;
      return this.lighting.dome ? this.lighting.dome.intensity / this.referenceLuminance : 0;
    }

    /** Linear colour of the panels' light mixed by their output, or null when they emit none. */
    panelColor() {
      const panels = this.lighting.panels;
      const total = LabEnvironment.intensityOf(panels);
      if (!(total > 0)) return null;
      return [0, 1, 2].map(function (channel) {
        return panels.reduce(function (sum, panel) {
          return sum + panel.color[channel] * LabEnvironment.intensityOf([panel]) / total;
        }, 0);
      });
    }

    /** Linear colour of the light filling the room: the panels', else the dome's. */
    lightColor() {
      const panels = this.panelColor();
      if (panels) return panels;
      return this.lighting.dome ? this.lighting.dome.color.slice() : WHITE.slice();
    }

    /** The way the panels' light travels: their normals averaged by output, or null when they cancel out. */
    panelDirection() {
      const sum = [0, 0, 0];
      this.lighting.panels.forEach(function (panel) {
        const length = Math.hypot(panel.normal[0], panel.normal[1], panel.normal[2]);
        const output = LabEnvironment.intensityOf([panel]);
        panel.normal.forEach(function (component, axis) { sum[axis] += component / length * output; });
      });
      const length = Math.hypot(sum[0], sum[1], sum[2]);
      return length > 0 ? sum.map(function (component) { return component / length; }) : null;
    }

    /** The point the environment is seen from, bundle frame: the centre of the room. */
    probe() {
      return this.room.center();
    }

    /** Distance from the probe to the farthest point of the room, in metres. */
    reach() {
      const size = this.room.size();
      return Math.hypot(size[0], size[1], size[2]) / 2 + LabEnvironment.PANEL_GAP;
    }

    /**
     * The shadow-casting key light, {direction, color, intensity, shadowBounds}: from
     * the panels when their light on the floor outshines the sun's, else along the sun,
     * else null. A directional light of intensity 1 lights white as a luminance of 1
     * shows; a key from the panels carries KEY_SHARE of their light. `shadowBounds` is
     * the region its shadows cover, or null for the viewer's default region.
     */
    keyLight() {
      const sun = this.lighting.sun;
      const direction = this.panelDirection();
      // a uniformly lit ceiling of luminance L gives the floor below it an irradiance of pi L
      const ceilingIrradiance = Math.PI * this.ceilingLuminance;
      if (direction && ceilingIrradiance > 0 && !(sun && sun.intensity > ceilingIrradiance)) {
        return {
          direction: direction, color: this.lightColor(),
          intensity: LabEnvironment.KEY_SHARE * this.ceilingLuminance / this.referenceLuminance,
          shadowBounds: this.belowPanels(),
        };
      }
      if (!sun) return null;
      return {
        direction: sun.direction.slice(),
        color: sun.color.slice(),
        intensity: sun.intensity / Math.PI / this.referenceLuminance,
        shadowBounds: this.bounds,
      };
    }

    /**
     * The scene's bounds, else the room, cut off below the fixture of the ceiling's
     * lowest panel: the space whose objects shade what the panels light. A panel more
     * than CEILING_DEPTH below the highest one hangs in the room, as a lamp over a
     * bench does, and is no part of the ceiling.
     */
    belowPanels() {
      const region = this.bounds || this.room;
      const heights = this.lighting.panels.map(function (panel) { return panel.center[2]; });
      const highest = Math.max.apply(null, heights);
      const lowest = Math.min.apply(null, heights.filter(function (height) {
        return height >= highest - LabEnvironment.CEILING_DEPTH;
      }));
      const top = Math.min(region.max[2], lowest - LabEnvironment.FIXTURE_DEPTH);
      if (top <= region.min[2]) return region;
      return new global.SceneBounds(region.min.slice(), [region.max[0], region.max[1], top]);
    }

    /**
     * Linear colour of every room face, lit once by the ceiling: the floor reflects
     * the light from above, the ceiling between the panels reflects the floor, and the
     * walls see half of each. Without panels the ceiling is the dome's open sky.
     */
    faceColors() {
      const light = this.lightColor();
      const overhead = this.overheadLuminance();
      const floor = LabEnvironment.FLOOR_REFLECTANCE * overhead;
      const wall = LabEnvironment.WALL_REFLECTANCE * (overhead + floor) / 2;
      const dome = this.lighting.dome;
      const tint = function (luminance) { return light.map(function (channel) { return channel * luminance; }); };
      const colors = new Array(BOX_FACES);
      Face.WALLS.forEach(function (face) { colors[face] = tint(wall); });
      colors[Face.FLOOR] = tint(floor);
      if (this.lighting.panels.length) colors[Face.CEILING] = tint(LabEnvironment.CEILING_REFLECTANCE * floor);
      else if (dome) colors[Face.CEILING] = dome.color.map(function (channel) { return channel * overhead; });
      else colors[Face.CEILING] = [0, 0, 0];
      return colors;
    }

    /** The room as a three scene for PMREM, seen from the probe at the origin. */
    scene(THREE) {
      const scene = new THREE.Scene();
      const frame = new THREE.Group();
      frame.rotation.x = -Math.PI / 2;   // the bundle's Z up becomes three's Y up
      scene.add(frame);
      const probe = this.probe();
      const offset = function (point) { return point.map(function (value, axis) { return value - probe[axis]; }); };
      const size = this.room.size();
      const room = new THREE.Mesh(
        new THREE.BoxGeometry(size[0], size[1], size[2]),
        this.faceColors().map(function (color) {
          return new THREE.MeshBasicMaterial({ color: new THREE.Color().fromArray(color), side: THREE.BackSide });
        }));
      room.name = LabEnvironment.ROOM;
      room.position.fromArray(offset(this.room.center()));
      frame.add(room);
      const facing = new THREE.Vector3(0, 0, 1);
      const lab = this;
      this.lighting.panels.forEach(function (panel) {
        const radiance = panel.intensity / lab.referenceLuminance;
        const quad = new THREE.Mesh(
          new THREE.PlaneGeometry(panel.width, panel.height),
          new THREE.MeshBasicMaterial({
            color: new THREE.Color().fromArray(panel.color.map(function (channel) { return channel * radiance; })),
          }));
        quad.name = LabEnvironment.PANEL;
        quad.position.fromArray(offset(panel.center));
        quad.quaternion.setFromUnitVectors(facing, new THREE.Vector3().fromArray(panel.normal).normalize());
        frame.add(quad);
      });
      return scene;
    }
  }

  LabEnvironment.blackBody = blackBody;
  LabEnvironment.Face = Face;
  //: name of the room box in the environment scene
  LabEnvironment.ROOM = 'room';
  //: name of every panel quad in the environment scene
  LabEnvironment.PANEL = 'panel';
  //: metres between the highest panel and the room's ceiling, so the quads never sink into it
  LabEnvironment.PANEL_GAP = 0.01;
  //: metres below a panel's centre its fixture and the ceiling around it reach
  LabEnvironment.FIXTURE_DEPTH = 0.1;
  //: metres below the highest panel a panel still counts as part of the ceiling
  LabEnvironment.CEILING_DEPTH = 0.3;
  //: assumed share of light a light laboratory floor reflects
  LabEnvironment.FLOOR_REFLECTANCE = 0.6;
  //: assumed share of light the painted walls reflect
  LabEnvironment.WALL_REFLECTANCE = 0.7;
  //: assumed share of light the ceiling tiles between the panels reflect
  LabEnvironment.CEILING_REFLECTANCE = 0.8;
  //: share of the panels' light a key from the panels carries, matched to the floor-to-wall
  //: brightness of path-traced renders of a panel-lit laboratory
  LabEnvironment.KEY_SHARE = 0.25;

  global.LabEnvironment = LabEnvironment;
  global.LightingDescription = LightingDescription;
})(window);
