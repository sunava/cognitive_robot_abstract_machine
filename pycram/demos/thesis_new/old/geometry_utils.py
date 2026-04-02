import numpy as np


def unit(v, eps=1e-12):
    """Return a unit-length 3D vector."""
    v = np.asarray(v, dtype=float).reshape(3)
    n = np.linalg.norm(v)
    if n < eps:
        raise ValueError("zero-length vector")
    return v / n


def aligned_plane_frame(origin, normal, tangent_hint):
    """Build a right-handed frame aligned to a plane normal."""
    p = np.asarray(origin, dtype=float).reshape(3)
    z = unit(normal)

    th = np.asarray(tangent_hint, dtype=float).reshape(3)
    x = th - (th @ z) * z
    x = unit(x)

    y = unit(np.cross(z, x))
    R = np.column_stack([x, y, z])
    return R, p
