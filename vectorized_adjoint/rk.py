"""Adaptive and fixed-step explicit Runge-Kutta integrators."""
import numpy as np
from .butcher import DORMAND_PRINCE_45, RK4_CLASSIC


def adaptive_rk45(f, x0, t_span, tol=1e-8, h_init=None, h_min=1e-12, h_max=1.0,
                  safety=0.9, max_steps=1000000, butcher=None):
    """
    Adaptive RK45 integrator (Dormand-Prince by default).

    Parameters
    ----------
    f : callable
        Right-hand side f(x, t) -> dx/dt. x is np.array of shape (n,).
    x0 : array_like
        Initial state, shape (n,).
    t_span : (t0, tf)
        Integration interval.
    tol : float
        Local error tolerance (combined absolute + relative).
    h_init : float or None
        Initial step size. If None, uses (tf-t0)/100.
    butcher : ButcherTableau or None
        Butcher tableau. Default: Dormand-Prince 4(5).

    Returns
    -------
    t_list : list of float
        Time points (variable spacing).
    x_list : list of np.array
        States at each time point.
    h_list : list of float
        Step sizes used (len = len(t_list) - 1).
    k_stages : list of list of np.array
        RK stages at each step (for adjoint). k_stages[step][stage] = k_i.
    """
    if butcher is None:
        butcher = DORMAND_PRINCE_45

    t0, tf = t_span
    x = np.array(x0, dtype=float)
    n = len(x)
    s = butcher.s

    if h_init is None:
        h_init = (tf - t0) / 100.0

    t = t0
    h = h_init

    t_list = [t]
    x_list = [x.copy()]
    h_list = []
    k_stages = []

    steps = 0
    while t < tf - 1e-14 * abs(tf):
        if t + h > tf:
            h = tf - t

        # Compute stages
        k = [None] * s
        for i in range(s):
            xi = x.copy()
            for j in range(i):
                xi += h * butcher.a[i, j] * k[j]
            k[i] = f(xi, t + butcher.c[i] * h)

        # Higher-order solution
        x_new = x.copy()
        for i in range(s):
            x_new += h * butcher.b[i] * k[i]

        # Error estimate
        err = np.zeros(n)
        for i in range(s):
            err += h * (butcher.b[i] - butcher.b_hat[i]) * k[i]

        err_norm = np.linalg.norm(err / (1.0 + np.abs(x_new))) / np.sqrt(n)

        if err_norm <= tol or h <= h_min:
            # Accept step
            t += h
            x = x_new
            t_list.append(t)
            x_list.append(x.copy())
            h_list.append(h)
            k_stages.append([ki.copy() for ki in k])
            steps += 1

        # Adjust step size
        if err_norm > 0:
            h_new = safety * h * (tol / err_norm) ** 0.2
        else:
            h_new = h * 2.0
        h = max(h_min, min(h_max, h_new))

        if steps >= max_steps:
            break

    return t_list, x_list, h_list, k_stages


def fixed_rk4(f, x0, t_span, dt, butcher=None):
    """
    Fixed-step RK4 integrator.

    Returns same format as adaptive_rk45 for compatibility.
    """
    if butcher is None:
        butcher = RK4_CLASSIC

    t0, tf = t_span
    x = np.array(x0, dtype=float)
    n = len(x)
    s = butcher.s

    t = t0
    t_list = [t]
    x_list = [x.copy()]
    h_list = []
    k_stages = []

    while t < tf - 1e-14 * abs(tf):
        h = min(dt, tf - t)

        k = [None] * s
        for i in range(s):
            xi = x.copy()
            for j in range(i):
                xi += h * butcher.a[i, j] * k[j]
            k[i] = f(xi, t + butcher.c[i] * h)

        x_new = x.copy()
        for i in range(s):
            x_new += h * butcher.b[i] * k[i]

        t += h
        x = x_new
        t_list.append(t)
        x_list.append(x.copy())
        h_list.append(h)
        k_stages.append([ki.copy() for ki in k])

    return t_list, x_list, h_list, k_stages
