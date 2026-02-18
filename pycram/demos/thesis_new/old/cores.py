import numpy as np
import matplotlib.pyplot as plt


def unit(v, eps=1e-12):
    """
    Normalize a 3D vector.

    Parameters
    ----------
    v : array-like, shape (3,)
        Input vector.
    eps : float
        Threshold below which the vector is considered zero-length.

    Returns
    -------
    np.ndarray, shape (3,)
        Unit-length vector pointing in the same direction as v.

    Raises
    ------
    ValueError
        If the vector norm is below the given threshold.
    """
    v = np.asarray(v, dtype=float).reshape(3)
    n = np.linalg.norm(v)
    if n < eps:
        raise ValueError("zero-length vector")
    return v / n


def ramp_depth(t_end: float, d_max: float):
    """
    Create a monotone displacement function.

    The returned function increases linearly from 0 to d_max
    over the interval [0, t_end] and remains constant afterwards.

    Parameters
    ----------
    t_end : float
        Time at which the maximum displacement is reached.
    d_max : float
        Maximum displacement magnitude.

    Returns
    -------
    Callable[[float], float]
        A scalar function d(t) describing displacement over time.
    """
    t_end = float(t_end)
    d_max = float(d_max)

    def d(t: float) -> float:
        t = float(t)
        if t <= 0.0:
            return 0.0
        if t >= t_end:
            return d_max
        return d_max * (t / t_end)

    return d


def linear_displacement(p0, n, d, t):
    """
    Compute a monotone displacement along a fixed direction.

    This implements the geometric form
        p(t) = p0 - d(t) * n

    Parameters
    ----------
    p0 : array-like, shape (3,)
        Initial position.
    n : array-like, shape (3,)
        Direction vector defining the displacement axis.
    d : Callable[[float], float]
        Scalar displacement function.
    t : float
        Time parameter.

    Returns
    -------
    np.ndarray, shape (3,)
        Position at time t.
    """
    n = unit(n)
    return p0 - d(t) * n


def displacement_plus_tangential_oscillation(p0, n, tau, d, A, f, t):
    """
    Compute a displacement composed of a monotone term and
    a bounded tangential oscillation.

    This implements the geometric form
        p(t) = p0 - d(t) * n + A * sin(2*pi*f*t) * tau

    Parameters
    ----------
    p0 : array-like, shape (3,)
        Initial position.
    n : array-like, shape (3,)
        Primary displacement direction.
    tau : array-like, shape (3,)
        Tangential direction orthogonal to n.
    d : Callable[[float], float]
        Monotone displacement function.
    A : float
        Oscillation amplitude.
    f : float
        Oscillation frequency.
    t : float
        Time parameter.

    Returns
    -------
    np.ndarray, shape (3,)
        Position at time t.
    """
    n = unit(n)
    tau = unit(tau)
    s = A * np.sin(2.0 * np.pi * f * t)
    return p0 - d(t) * n + s * tau


def rotation_angle(omega, t):
    """
    Compute a pure rotational angle as a function of time.

    This corresponds to the scalar part of a rotation
        R(t) = exp(omega * t)

    Parameters
    ----------
    omega : float
        Constant angular velocity.
    t : float or np.ndarray
        Time parameter.

    Returns
    -------
    float or np.ndarray
        Rotation angle at time t.
    """
    return omega * t


# ---------------------------------------------------------------------
# Demonstration
# ---------------------------------------------------------------------

# Total duration and time discretization
T = 4.0
ts = np.linspace(0.0, T, 800)

# Initial position
p0 = np.array([0.0, 0.0, 0.0])

# Fixed directions defining a local coordinate frame
n = np.array([0.0, 0.0, 1.0])  # primary displacement direction
tau = np.array([1.0, 0.0, 0.0])  # tangential direction

# Displacement and oscillation parameters
d = ramp_depth(t_end=3.0, d_max=0.06)
A = 0.01
f = 3.0
omega = 2.5

# Evaluate trajectories over time
P_lin = np.vstack([linear_displacement(p0, n, d, t) for t in ts])
P_osc = np.vstack(
    [displacement_plus_tangential_oscillation(p0, n, tau, d, A, f, t) for t in ts]
)
ang = rotation_angle(omega, ts)

# ---------------------------------------------------------------------
# Visualization
# ---------------------------------------------------------------------

# 3D trajectory showing combined displacement and oscillation
fig = plt.figure()
ax = fig.add_subplot(111, projection="3d")
ax.plot(P_osc[:, 0], P_osc[:, 1], P_osc[:, 2])
ax.set_xlabel("x")
ax.set_ylabel("y")
ax.set_zlabel("z")
ax.set_title("Monotone displacement with superposed tangential oscillation")
plt.show()

# Normal displacement over time
fig = plt.figure()
ax = fig.add_subplot(111)
ax.plot(ts, P_lin[:, 2], label="monotone displacement only")
ax.plot(ts, P_osc[:, 2], label="with tangential oscillation")
ax.set_xlabel("t [s]")
ax.set_ylabel("displacement along n [m]")
ax.set_title("Displacement component over time")
ax.legend()
plt.show()

# Pure rotation angle over time
fig = plt.figure()
ax = fig.add_subplot(111)
ax.plot(ts, ang)
ax.set_xlabel("t [s]")
ax.set_ylabel("angle [rad]")
ax.set_title("Pure rotational component")
plt.show()
