/* Cameras attached to articulated robot links in the rendered scene. */
(function (global) {
  'use strict';

  // %% camera frames
  const CAMERA_CONFIGURATION = Object.freeze({ fieldOfView: 60, aspectRatio: 4 / 3, near: 0.02, far: 200 });
  const CAMERA_LABEL = 'Robot';
  const PREFERRED_FRAMES = Object.freeze([
    'wide_stereo_optical_frame', 'head_center_camera_frame', 'head_rgbd_sensor_link',
  ]);
  // HSRB camera frames have an optical basis without an optical suffix.
  const OPTICAL_FRAMES = new Set([
    'head_center_camera_frame', 'head_l_stereo_camera_link',
    'head_r_stereo_camera_link', 'head_rgbd_sensor_link', 'hand_camera_frame',
  ]);
  const HEAD_FRAMES = Object.freeze([
    'head_tilt_link', 'head_link', 'head', 'head_plate_frame', 'head_pan_link',
  ]);
  const FRAME_PRIORITY = Object.freeze({ head: 10, camera: 20, hand: 30 });

  /** One articulated camera frame belonging to a distinct robot instance. */
  class Source {
    constructor(model, name, optical) {
      this.id = encodeURIComponent(Source.identity(model)) + '/' + encodeURIComponent(name);
      /** Stable selection key across asynchronous model loading. */
      this.label = Source.identity(model) + ' · ' + Source.localName(name).replace(/_/g, ' ');
      /** Robot instance and camera frame displayed in the source selector. */
      this.link = model.obj.links[name];
      /** Articulated scene node whose world pose places the camera. */
      this.optical = optical;
      /** Whether local positive Z looks forward and negative Y points upward. */
    }

    /** Identify a robot independently of its loading order. */
    static identity(model) {
      return String(model.identifier || model.prefix || model.name || CAMERA_LABEL);
    }

    /** Strip a composed world's robot namespace from a link name. */
    static localName(name) {
      return name.slice(name.lastIndexOf('/') + 1);
    }

    /** Recognize optical frames, including the HSRB's named camera frames. */
    static optical(name) {
      const local = Source.localName(name).toLowerCase();
      return /(?:^|_)optical(?:_|$)/.test(local) || OPTICAL_FRAMES.has(local);
    }

    /** Prefer the robot's head camera over cameras attached to its hands. */
    static priority(name) {
      const local = Source.localName(name).toLowerCase();
      const preferred = PREFERRED_FRAMES.indexOf(local);
      if (preferred >= 0) return preferred;
      if (/(?:hand|forearm|wrist|gripper)/.test(local)) return FRAME_PRIORITY.hand;
      if (/(?:head|wide_stereo)/.test(local)) return FRAME_PRIORITY.head;
      return FRAME_PRIORITY.camera;
    }

    /** List camera links, falling back to a head frame when no camera is available. */
    static forModel(model) {
      const names = Object.keys(model.obj.links).filter(function (name) {
        return !/gazebo/i.test(name);
      });
      let candidates = names.filter(Source.optical);
      const optical = candidates.length > 0;
      if (!optical) {
        candidates = names.filter(function (name) {
          return /(?:camera|stereo|rgbd)/i.test(Source.localName(name));
        });
      }
      if (!candidates.length) {
        const head = HEAD_FRAMES.map(function (frame) {
          return names.find(function (name) { return Source.localName(name).toLowerCase() === frame; });
        }).find(Boolean);
        candidates = head ? [head] : [];
      }
      candidates.sort(function (first, second) {
        return Source.priority(first) - Source.priority(second) || first.localeCompare(second);
      });
      return candidates.map(function (name) { return new Source(model, name, optical); });
    }

    /** Collect sources in a stable order across independently loaded robots. */
    static forModels(models) {
      return models.filter(function (model) {
        return model.robot && model.obj && model.obj.links;
      }).sort(function (first, second) {
        return Source.identity(first).localeCompare(Source.identity(second));
      }).flatMap(Source.forModel);
    }
  }

  // %% articulated camera pose
  /** Keep a perspective camera aligned with a selected robot link. */
  class Pose {
    constructor(three) {
      this.camera = new three.PerspectiveCamera(
        CAMERA_CONFIGURATION.fieldOfView, CAMERA_CONFIGURATION.aspectRatio,
        CAMERA_CONFIGURATION.near, CAMERA_CONFIGURATION.far
      );
      /** Camera used to render the robot's view into its viewport. */
      this.orientation = new three.Quaternion();
      /** Current world orientation of the selected link. */
      this.forward = new three.Vector3();
      /** Selected frame's viewing direction in world coordinates. */
      this.target = new three.Vector3();
      /** World point one unit ahead of the camera. */
    }

    /** Follow the link's current world pose while preserving the camera's field of view. */
    update(source, width, height) {
      if (!source || !source.link || !Number.isFinite(width) || !Number.isFinite(height) ||
          width <= 0 || height <= 0) return false;
      source.link.updateWorldMatrix(true, false);
      source.link.getWorldPosition(this.camera.position);
      source.link.getWorldQuaternion(this.orientation);
      this.forward.set(source.optical ? 0 : 1, 0, source.optical ? 1 : 0).applyQuaternion(this.orientation);
      this.camera.up.set(0, source.optical ? -1 : 0, source.optical ? 0 : 1).applyQuaternion(this.orientation);
      this.target.copy(this.camera.position).add(this.forward);
      this.camera.lookAt(this.target);
      this.camera.updateMatrixWorld(true);
      return true;
    }
  }

  global.RobotCamera = { sources: Source.forModels, Pose: Pose };
})(window);
