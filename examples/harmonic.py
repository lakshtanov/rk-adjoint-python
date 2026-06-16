#!/usr/bin/env python3
"""
Harmonic oscillator: discrete adjoint with analytical validation.

dx/dt = v, dv/dt = -k*x. Analytical solution exists,
so we can validate both forward and adjoint.

Run: python examples/harmonic.py
"""
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import time
import numpy as np
import aadc
from vectorized_adjoint import adaptive_rk45, discrete_adjoint_erk, AadRhs, DORMAND_PRINCE_45


def harmonic_rhs(x, p, t):
    """Harmonic oscillator: dx/dt = v, dv/dt = -k*x."""
    k = p[0]
    return [x[1], -k * x[0]]


def main():
    print("Harmonic Oscillator: Discrete Adjoint (arXiv:2410.01911)")
    print("=" * 50)

    k = 4.0
    x0 = np.array([1.0, 0.0])  # x=1, v=0
    p0 = np.array([k])
    T = 3.0

    # Analytical: x(T) = cos(sqrt(k)*T), v(T) = -sqrt(k)*sin(sqrt(k)*T)
    w = np.sqrt(k)
    x_exact = np.array([np.cos(w*T), -w*np.sin(w*T)])
    print(f"Analytical x(T) = {x_exact}")

    def cost(x_T):
        return x_T[0]**2 + x_T[1]**2

    aad = AadRhs(harmonic_rhs, n_states=2, n_params=1, x0=x0, p0=p0)
    aad.set_params(p0)

    t_list, x_list, h_list, k_stages = adaptive_rk45(aad.f_numpy, x0, (0, T), tol=1e-10)
    print(f"Numerical x(T)  = {x_list[-1]}")
    print(f"Forward error: {np.max(np.abs(x_list[-1] - x_exact)):.2e}")

    dJdp = discrete_adjoint_erk(aad, x_list, h_list, k_stages, DORMAND_PRINCE_45, p0,
                                 dJdx_T=2*x_list[-1])
    print(f"\ndJ/dk (adjoint) = {dJdp[0]:.6f}")

    # FD
    eps = 1e-7
    aad.set_params([k + eps])
    _, x_up, _, _ = adaptive_rk45(aad.f_numpy, x0, (0, T), tol=1e-10)
    aad.set_params([k - eps])
    _, x_dn, _, _ = adaptive_rk45(aad.f_numpy, x0, (0, T), tol=1e-10)
    fd = (cost(x_up[-1]) - cost(x_dn[-1])) / (2*eps)
    print(f"dJ/dk (FD)      = {fd:.6f}")
    print(f"Ratio: {dJdp[0]/fd:.6f}")


if __name__ == "__main__":
    main()
