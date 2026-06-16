# VectorizedAdjoint Python

Python wrapper for the discrete adjoint sensitivity analysis method
for adaptive explicit Runge-Kutta ODE solvers, as described in:

> R. Martins, E. Lakshtanov.
> *A C++ implementation of the discrete adjoint sensitivity analysis method
> for explicit adaptive Runge-Kutta methods enabled by automatic adjoint
> differentiation and SIMD vectorization.*
> Applied Mathematics and Computation, 2025.
> [arXiv:2410.01911](https://arxiv.org/abs/2410.01911) |
> [Published version](https://www.sciencedirect.com/science/article/pii/S0096300325004254)

The C++ library is available at
[github.com/RuiMartins1996/VectorizedAdjoint](https://github.com/RuiMartins1996/VectorizedAdjoint).

## What this does

Given an ODE system

    dx/dt = f(x, p, t),    x(0) = x0(p)

and a cost functional J(x(T), p), this library computes
**exact gradients dJ/dp** via the discrete adjoint method.

Key features:
- **Adaptive step size** — matches the accuracy of scipy/CVODES
  (not fixed-step like plain Euler/RK4)
- **Discrete adjoint** — gradient is exact for the discretized problem
  (consistent with finite differences to machine precision)
- **AAD for vector-Jacobian products** — automatic differentiation
  computes ∂f/∂x and ∂f/∂p without manual Jacobian coding
- **SIMD vectorization** — AVX2/AVX512 for parallel VJP evaluation
- **Checkpointing** — memory-efficient adjoint for long integrations

## Prerequisites

- Python 3.11+
- `aadc` Python package (for automatic adjoint differentiation)
- `numpy`

## Examples

### 1. Lotka-Volterra (2 states, 4 parameters)

```bash
python examples/lotka_volterra.py
```

Records the ODE right-hand side with AAD, integrates forward with
adaptive RK45, then runs the discrete adjoint backward to get
exact dJ/dp. Compares with finite differences.

### 2. Van der Pol oscillator (2 states, 1 parameter)

```bash
python examples/van_der_pol.py
```

Stiff oscillator with adaptive step control. Shows that the
discrete adjoint handles variable step sizes correctly.

### 3. Harmonic oscillator (2 states, 2 parameters)

```bash
python examples/harmonic.py
```

Simple test case with known analytical gradient for validation.

## How it works

The method from the paper, in three steps:

**Step 1: Record the RHS with AAD**

```python
import aadc

funcs = aadc.Functions()
funcs.start_recording()
# ... mark x, p as inputs, compute f(x, p, t), mark as output ...
funcs.stop_recording()
```

This gives a compiled kernel that evaluates f(x, p, t) and,
via reverse mode, the vector-Jacobian products v^T ∂f/∂x and v^T ∂f/∂p.

**Step 2: Adaptive RK integration (forward)**

```python
t, x_traj, h_steps = adaptive_rk45(f, x0, t_span, tol)
```

Standard adaptive RK45 with error control. The trajectory and
step sizes are stored for the adjoint pass.

**Step 3: Discrete adjoint (backward)**

```python
dJdp = discrete_adjoint_erk(funcs, x_traj, h_steps, butcher, dJdx_T)
```

Walks backward through the RK stages using the stored trajectory,
computing the adjoint variables λ and accumulating dJ/dp.
The vector-Jacobian products are evaluated via the AAD kernel from Step 1.

## Architecture

```
examples/
  lotka_volterra.py     — complete example with comparison to FD
  van_der_pol.py        — stiff system, adaptive steps
  harmonic.py           — validation against analytical gradient
vectorized_adjoint/
  rk.py                 — adaptive RK45 stepper
  adjoint.py            — discrete adjoint backward pass
  aad_rhs.py            — AAD wrapper for ODE right-hand side
  butcher.py            — Butcher tableaux (RK45, Dormand-Prince)
```

## References

1. R. Martins, E. Lakshtanov. *A C++ implementation of the discrete
   adjoint sensitivity analysis method for explicit adaptive Runge-Kutta
   methods enabled by automatic adjoint differentiation and SIMD
   vectorization.* Applied Mathematics and Computation, 2025.
   [arXiv:2410.01911](https://arxiv.org/abs/2410.01911)

2. S.K. Nadarajah, A. Jameson. *A Comparison of the Continuous and
   Discrete Adjoint Approach to Automatic Aerodynamic Optimization.*
   AIAA-2000-0667.

3. A. Griewank, A. Walther. *Evaluating Derivatives: Principles and
   Techniques of Algorithmic Differentiation.* SIAM, 2008.

## License

Apache 2.0
