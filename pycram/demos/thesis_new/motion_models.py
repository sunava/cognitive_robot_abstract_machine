import numpy as np


def _as_float_array(value):
    """Return a float numpy array without forcing dtype on unsupported types."""
    # Avoid passing dtype to __array__ implementations that don't accept it.
    arr = np.asarray(value)
    if arr.dtype != float:
        try:
            arr = arr.astype(float)
        except (TypeError, ValueError):
            pass
    return arr


class Pose:
    def __init__(self, R=None, p=None):
        """Store a rotation matrix R and translation p."""
        self.R = (
            np.eye(3, dtype=float) if R is None else _as_float_array(R).reshape(3, 3)
        )
        self.p = (
            np.zeros(3, dtype=float) if p is None else _as_float_array(p).reshape(3)
        )

    def transform_point(self, q_local):
        """Transform a local point into the pose frame."""
        q_local = _as_float_array(q_local).reshape(3)
        return self.p + self.R @ q_local


class FrameProvider:
    def get_pose(self) -> Pose:
        """Return the current pose for this provider."""
        raise NotImplementedError


class FixedFrameProvider(FrameProvider):
    def __init__(self, pose: Pose):
        """Always return the same pose."""
        self._pose = pose

    def get_pose(self) -> Pose:
        """Return the fixed pose."""
        return self._pose


class MotionSegment:
    def __init__(self, name, duration_s, local_curve):
        """Define a local motion curve over a fixed duration."""
        self.name = str(name)
        self.duration_s = float(duration_s)
        self.local_curve = local_curve

    def sample(self, frame_provider: FrameProvider, dt: float, t0: float = 0.0):
        """Sample the local curve in the given frame."""
        F = frame_provider.get_pose()

        n = max(2, int(np.ceil(self.duration_s / float(dt))) + 1)
        times = np.linspace(t0, t0 + self.duration_s, n)

        tau = (times - t0) / self.duration_s
        pts = np.empty((n, 3), dtype=float)
        for i, u in enumerate(tau):
            pts[i] = F.transform_point(self.local_curve(float(u)))

        return times, pts


class MotionSequence:
    def __init__(self, phases):
        """Store an ordered list of phases."""
        self.phases = list(phases)

    @property
    def duration_s(self):
        """Total duration across all phases."""
        return float(sum(p.duration_s for p in self.phases))

    def sample(self, frame_provider: FrameProvider, dt: float, t0: float = 0.0):
        """Sample all phases into one concatenated sequence."""
        all_t, all_p, all_id = [], [], []
        t = float(t0)

        for k, ph in enumerate(self.phases):
            tt, pp = ph.sample(frame_provider, dt=dt, t0=t)
            if all_t:
                tt = tt[1:]
                pp = pp[1:]

            all_t.append(tt)
            all_p.append(pp)
            all_id.append(np.full(len(tt), k, dtype=int))
            t += ph.duration_s

        return np.concatenate(all_t), np.vstack(all_p), np.concatenate(all_id)
