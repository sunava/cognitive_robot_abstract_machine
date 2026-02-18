import numpy as np

from semantic_digital_twin.spatial_types import HomogeneousTransformationMatrix

class MotionSegment:
    def __init__(self, name, duration_s, local_curve):
        """Define a local motion curve over a fixed duration."""
        self.name = str(name)
        self.duration_s = float(duration_s)
        self.local_curve = local_curve

    def sample(self, frame_np: np.ndarray, dt: float, t0: float = 0.0):
        n = max(2, int(np.ceil(self.duration_s / float(dt))) + 1)
        times = np.linspace(t0, t0 + self.duration_s, n)
        tau = (times - t0) / self.duration_s

        F = np.asarray(frame_np, dtype=float)

        R = F[:3, :3]
        t = F[:3, 3]

        q = np.array([self.local_curve(float(u)) for u in tau], dtype=float).reshape(-1, 3)
        pts = q @ R.T + t

        return times, pts


class MotionSequence:
    def __init__(self, phases):
        """Store an ordered list of phases."""
        self.phases = list(phases)

    @property
    def duration_s(self):
        """Total duration across all phases."""
        return float(sum(p.duration_s for p in self.phases))

    def sample(self,  frame: HomogeneousTransformationMatrix, dt: float, t0: float = 0.0):
        """Sample all phases into one concatenated sequence."""
        all_t, all_p, all_id = [], [], []
        t = float(t0)

        for k, ph in enumerate(self.phases):
            tt, pp = ph.sample(frame, dt=dt, t0=t)
            if all_t:
                tt = tt[1:]
                pp = pp[1:]

            all_t.append(tt)
            all_p.append(pp)
            all_id.append(np.full(len(tt), k, dtype=int))
            t += ph.duration_s

        return np.concatenate(all_t), np.vstack(all_p), np.concatenate(all_id)
