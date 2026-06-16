#!/usr/bin/env python3
"""
Lotka-Volterra: discrete adjoint sensitivity via VectorizedAdjoint Python.

Demonstrates the full pipeline from the paper (arXiv:2410.01911):
1. Record ODE RHS with AAD
2. Adaptive RK45 forward integration
3. Discrete adjoint backward pass
4. Compare gradient with finite differences

Run: python examples/lotka_volterra.py
"""
import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import time
import numpy as np
import aadc
from vectorized_adjoint import adaptive_rk45, discrete_adjoint_erk, AadRhs, DORMAND_PRINCE_45


def lotka_rhs(x, p, t):
    """Lotka-Volterra RHS: dx/dt = f(x, p, t).
    x = [prey, predator], p = [alpha, beta, delta, gamma].
    """
    alpha, beta, delta, gamma = p[0], p[1], p[2], p[3]
    prey, pred = x[0], x[1]
    dprey = alpha * prey - beta * prey * pred
    dpred = delta * prey * pred - gamma * pred
    return [dprey, dpred]


def main():
    print("=" * 60)
    print("Lotka-Volterra: Discrete Adjoint Sensitivity Analysis")
    print("Method: arXiv:2410.01911 (Martins & Lakshtanov)")
    print("=" * 60)

    # Parameters
    p0 = np.array([1.5, 1.0, 3.0, 1.0])  # alpha, beta, delta, gamma
    x0 = np.array([1.0, 1.0])              # initial prey, predator
    t_span = (0.0, 5.0)
    tol = 1e-8

    # Cost: J = x(T)^2 + y(T)^2
    def cost(x_T):
        return x_T[0]**2 + x_T[1]**2
    def dJdx(x_T):
        return np.array([2*x_T[0], 2*x_T[1]])

    # Step 1: Record RHS with AAD
    print("\nStep 1: Recording RHS with AAD...")
    t0 = time.time()
    aad = AadRhs(lotka_rhs, n_states=2, n_params=4, x0=x0, p0=p0)
    print(f"  Done in {time.time()-t0:.3f}s")

    # Step 2: Adaptive RK45 forward integration
    print("\nStep 2: Adaptive RK45 forward integration...")
    aad.set_params(p0)
    t0 = time.time()
    t_list, x_list, h_list, k_stages = adaptive_rk45(
        aad.f_numpy, x0, t_span, tol=tol)
    fwd_time = time.time() - t0
    print(f"  Steps: {len(h_list)}")
    print(f"  x(T) = [{x_list[-1][0]:.6f}, {x_list[-1][1]:.6f}]")
    print(f"  J(x(T)) = {cost(x_list[-1]):.6f}")
    print(f"  Time: {fwd_time*1000:.1f} ms")

    # Step 3: Discrete adjoint
    print("\nStep 3: Discrete adjoint backward pass...")
    t0 = time.time()
    dJdp = discrete_adjoint_erk(
        aad, x_list, h_list, k_stages, DORMAND_PRINCE_45, p0,
        dJdx_T=dJdx(x_list[-1]))
    adj_time = time.time() - t0
    print(f"  dJ/dp = {dJdp}")
    print(f"  Time: {adj_time*1000:.1f} ms")

    # Step 4: Finite difference verification
    print("\nStep 4: Finite difference verification...")
    eps = 1e-6
    dJdp_fd = np.zeros(4)
    for i in range(4):
        p_up = p0.copy(); p_up[i] += eps
        p_dn = p0.copy(); p_dn[i] -= eps

        aad.set_params(p_up)
        _, x_up, _, _ = adaptive_rk45(aad.f_numpy, x0, t_span, tol=tol)
        J_up = cost(x_up[-1])

        aad.set_params(p_dn)
        _, x_dn, _, _ = adaptive_rk45(aad.f_numpy, x0, t_span, tol=tol)
        J_dn = cost(x_dn[-1])

        dJdp_fd[i] = (J_up - J_dn) / (2 * eps)

    print(f"  dJ/dp (FD) = {dJdp_fd}")

    # Comparison
    print(f"\n{'='*60}")
    print("Comparison: Adjoint vs Finite Differences")
    print(f"{'='*60}")
    names = ['alpha', 'beta', 'delta', 'gamma']
    all_ok = True
    for i in range(4):
        ratio = dJdp[i] / dJdp_fd[i] if dJdp_fd[i] != 0 else float('inf')
        ok = abs(ratio - 1.0) < 0.01
        all_ok = all_ok and ok
        print(f"  dJ/d{names[i]:6s}: adj={dJdp[i]:12.6f}  fd={dJdp_fd[i]:12.6f}  ratio={ratio:.6f}  {'OK' if ok else 'FAIL'}")

    print(f"\nAll match: {all_ok}")


if __name__ == "__main__":
    main()
