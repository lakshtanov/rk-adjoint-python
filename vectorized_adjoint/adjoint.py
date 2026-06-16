"""Discrete adjoint sensitivity analysis for explicit Runge-Kutta methods.

Python port of the C++ algorithm from:
  R. Martins, E. Lakshtanov, "A C++ implementation of the discrete adjoint
  sensitivity analysis method for explicit adaptive Runge-Kutta methods...",
  Applied Mathematics and Computation, 2025. arXiv:2410.01911

  C++ source: github.com/RuiMartins1996/VectorizedAdjoint
  See lib/include/detail/backpropagation.hpp, back_prop_step()
"""
import numpy as np


def _back_prop_step(aad_rhs, x_n, p, t_n, dt, k_stages_n, butcher, wbarend):
    """
    One backward step of the discrete adjoint.

    Translated from C++ back_prop_step() in detail/backpropagation.hpp lines 161-229.

    Parameters
    ----------
    aad_rhs : AadRhs
    x_n : np.array(n) — state at beginning of step
    p : np.array(m) — parameters
    t_n : float — time at beginning of step
    dt : float — step size
    k_stages_n : list of np.array — RK stages k[0]..k[s-1]
    butcher : ButcherTableau
    wbarend : np.array(n) — adjoint at end of step (modified in-place)

    Returns
    -------
    mu_step : np.array(m) — parameter sensitivity contribution from this step
    """
    n = len(x_n)
    s = butcher.s
    m = len(p)

    # Initialize w_bar_n: columns 0..s+1
    # Column indices: 0 = state adjoint, 1..s = stage adjoints, s+1 = incoming adjoint
    w_bar = np.zeros((n, s + 2))
    w_bar[:, s + 1] = wbarend

    # Step s+1: distribute incoming adjoint to stage adjoints (C++ lines 179-184)
    for i in range(n):
        w_bar[i, 0] += w_bar[i, s + 1]
        for mm in range(1, s + 1):
            w_bar[i, mm] += butcher.b[mm - 1] * dt * w_bar[i, s + 1]

    # Recompute intermediate states (K vectors)
    # k_stages_n[m] = f(X_m, t + c[m]*dt) already available from forward pass

    mu_step = np.zeros(m)

    # Steps s down to 1 (C++ lines 196-224)
    for mm in range(s, 0, -1):
        t_mn = t_n + butcher.c[mm - 1] * dt  # Note: C++ uses c(m) with 1-based m

        # Get intermediate state X^{m,n} (C++ get_intermediate_state, lines 67-81)
        # X^{m,n} = x_n + dt * sum_{k=1}^{m-1} a[m-1, k-1] * K[k-1]
        x_mn = x_n.copy()
        for k in range(1, mm):
            x_mn += dt * butcher.a[mm - 1, k - 1] * k_stages_n[k - 1]

        # Get adjoint vector for this stage
        w_bar_m = w_bar[:, mm].copy()

        # Vector-Jacobian product: w_bar_m^T * df/dx, w_bar_m^T * df/dp
        vjp_x, vjp_p = aad_rhs.vjp(x_mn, p, t_mn, w_bar_m)

        # Update stage 0 adjoint (C++ line 211-212)
        for i in range(n):
            w_bar[i, 0] += vjp_x[i]

        # Update stage k < m adjoints (C++ lines 214-216)
        for k in range(1, mm):
            for i in range(n):
                w_bar[i, k] += vjp_x[i] * butcher.a[mm - 1, k - 1] * dt

        # Accumulate parameter sensitivity (C++ lines 220-222)
        mu_step += vjp_p

    # Update wbarend for next iteration (C++ lines 227-228)
    wbarend[:] = w_bar[:, 0]

    return mu_step


def discrete_adjoint_erk(aad_rhs, x_traj, h_list, k_stages, butcher, p,
                         dJdx_T=None, cost_func=None):
    """
    Discrete adjoint for explicit Runge-Kutta methods.

    Translated from C++ back_prop() in detail/backpropagation.hpp lines 257-278.

    Parameters
    ----------
    aad_rhs : AadRhs
    x_traj : list of np.array — forward trajectory, len = N+1
    h_list : list of float — step sizes, len = N
    k_stages : list of list of np.array — RK stages per step
    butcher : ButcherTableau
    p : array_like — parameter values
    dJdx_T : np.array or None — gradient of cost w.r.t. final state
    cost_func : callable or None — J(x_T) -> scalar (for FD of terminal condition)

    Returns
    -------
    dJdp : np.array(m) — gradient of J w.r.t. parameters
    """
    N = len(h_list)
    n = len(x_traj[0])
    p = np.array(p, dtype=float)

    # Terminal condition
    if dJdx_T is not None:
        wbarend = np.array(dJdx_T, dtype=float)
    elif cost_func is not None:
        x_T = x_traj[-1]
        J0 = cost_func(x_T)
        wbarend = np.zeros(n)
        eps = 1e-7
        for i in range(n):
            x_up = x_T.copy(); x_up[i] += eps
            wbarend[i] = (cost_func(x_up) - J0) / eps
    else:
        raise ValueError("Either dJdx_T or cost_func must be provided")

    # Accumulate dJ/dp
    alphabar = np.zeros(len(p))

    # Backward sweep: n = N-1 down to 0 (C++ lines 270-276)
    # Reconstruct times from h_list
    times = [0.0]
    for i in range(N):
        times.append(times[-1] + h_list[i])

    for step in range(N - 1, -1, -1):
        mu_step = _back_prop_step(
            aad_rhs, x_traj[step], p, times[step], h_list[step],
            k_stages[step], butcher, wbarend)
        alphabar += mu_step

    return alphabar
