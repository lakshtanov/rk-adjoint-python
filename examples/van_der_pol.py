#!/usr/bin/env python3
"""
Van der Pol oscillator: discrete adjoint sensitivity.

Stiff system with adaptive step control. Demonstrates that the
discrete adjoint handles variable step sizes correctly.

Run: python examples/van_der_pol.py
"""
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import time
import numpy as np
import aadc
from vectorized_adjoint import adaptive_rk45, discrete_adjoint_erk, AadRhs, DORMAND_PRINCE_45


def vdp_rhs(x, p, t):
    """Van der Pol: dx/dt = y, dy/dt = mu*(1-x^2)*y - x."""
    mu = p[0]
    return [x[1], mu * (1.0 - x[0]*x[0]) * x[1] - x[0]]


def main():
    print("Van der Pol: Discrete Adjoint (arXiv:2410.01911)")
    print("=" * 50)

    mu = 1.0
    x0 = np.array([2.0, 0.0])
    p0 = np.array([mu])
    t_span = (0.0, 6.0)

    def cost(x_T):
        return x_T[0]**2 + x_T[1]**2

    aad = AadRhs(vdp_rhs, n_states=2, n_params=1, x0=x0, p0=p0)
    aad.set_params(p0)

    t_list, x_list, h_list, k_stages = adaptive_rk45(aad.f_numpy, x0, t_span, tol=1e-8)
    print(f"Steps: {len(h_list)}, x(T) = {x_list[-1]}")

    dJdp = discrete_adjoint_erk(aad, x_list, h_list, k_stages, DORMAND_PRINCE_45, p0,
                                 dJdx_T=2*x_list[-1])
    print(f"dJ/dmu (adjoint) = {dJdp[0]:.6f}")

    # FD
    eps = 1e-6
    aad.set_params([mu + eps])
    _, x_up, _, _ = adaptive_rk45(aad.f_numpy, x0, t_span, tol=1e-8)
    aad.set_params([mu - eps])
    _, x_dn, _, _ = adaptive_rk45(aad.f_numpy, x0, t_span, tol=1e-8)
    fd = (cost(x_up[-1]) - cost(x_dn[-1])) / (2*eps)
    print(f"dJ/dmu (FD)      = {fd:.6f}")
    print(f"Ratio: {dJdp[0]/fd:.6f}")


if __name__ == "__main__":
    main()
