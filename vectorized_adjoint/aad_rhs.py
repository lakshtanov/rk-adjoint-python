"""AAD wrapper for ODE right-hand side: records f(x, p, t) and provides VJP."""
import numpy as np
import aadc


class AadRhs:
    """
    Records the ODE right-hand side f(x, p, t) with AADC and provides:
    - f(x, p, t)       — forward evaluation
    - v^T df/dx         — vector-Jacobian product w.r.t. state
    - v^T df/dp         — vector-Jacobian product w.r.t. parameters

    This is the AAD kernel from Step 1 of the paper (Section 3.3.3).
    """

    def __init__(self, rhs_func, n_states, n_params, x0, p0, t0=0.0):
        """
        Record the RHS function.

        Parameters
        ----------
        rhs_func : callable
            f(x, p, t) -> dxdt, where x, p are lists of aadc.idouble, t is aadc.idouble.
        n_states : int
        n_params : int
        x0 : array_like
            Typical state values (for recording).
        p0 : array_like
            Parameter values (for recording).
        t0 : float
            Typical time value.
        """
        self.n = n_states
        self.m = n_params

        self._funcs = aadc.Functions()
        self._funcs.start_recording()

        # State inputs
        id_x = [aadc.idouble(float(x0[i])) for i in range(n_states)]
        self._a_x = [xi.mark_as_input() for xi in id_x]

        # Parameter inputs
        id_p = [aadc.idouble(float(p0[i])) for i in range(n_params)]
        self._a_p = [pi.mark_as_input() for pi in id_p]

        # Time input
        id_t = aadc.idouble(float(t0))
        self._a_t = id_t.mark_as_input()

        # Evaluate RHS
        dxdt = rhs_func(id_x, id_p, id_t)

        # Mark outputs
        self._r_f = [fi.mark_as_output() for fi in dxdt]

        self._funcs.stop_recording()
        self._workers = aadc.ThreadPool(1)

    def evaluate(self, x, p, t):
        """Evaluate f(x, p, t) → np.array of shape (n,)."""
        inputs = {}
        for i in range(self.n):
            inputs[self._a_x[i]] = float(x[i])
        for i in range(self.m):
            inputs[self._a_p[i]] = float(p[i])
        inputs[self._a_t] = float(t)

        request = {r: [] for r in self._r_f}
        res = aadc.evaluate(self._funcs, request, inputs, self._workers)

        return np.array([float(np.asarray(res[0][r]).flat[0]) for r in self._r_f])

    def vjp(self, x, p, t, v):
        """
        Compute vector-Jacobian products:
            v^T df/dx  (shape n)
            v^T df/dp  (shape m)

        Parameters
        ----------
        x : array_like, shape (n,)
        p : array_like, shape (m,)
        t : float
        v : array_like, shape (n,) — adjoint vector

        Returns
        -------
        vjp_x : np.array, shape (n,) — v^T df/dx
        vjp_p : np.array, shape (m,) — v^T df/dp
        """
        inputs = {}
        for i in range(self.n):
            inputs[self._a_x[i]] = float(x[i])
        for i in range(self.m):
            inputs[self._a_p[i]] = float(p[i])
        inputs[self._a_t] = float(t)

        # Request gradients of all f_i w.r.t. x and p
        all_args = list(self._a_x) + list(self._a_p)
        request = {r: all_args for r in self._r_f}

        res = aadc.evaluate(self._funcs, request, inputs, self._workers)

        # Compute v^T J by summing v[i] * (df_i/d...)
        vjp_x = np.zeros(self.n)
        vjp_p = np.zeros(self.m)

        for i in range(self.n):
            vi = float(v[i])
            if vi == 0.0:
                continue
            for j in range(self.n):
                vjp_x[j] += vi * float(np.asarray(res[1][self._r_f[i]][self._a_x[j]]).flat[0])
            for j in range(self.m):
                vjp_p[j] += vi * float(np.asarray(res[1][self._r_f[i]][self._a_p[j]]).flat[0])

        return vjp_x, vjp_p

    def f_numpy(self, x, t):
        """Plain numpy evaluation (for the forward integrator)."""
        return self.evaluate(x, self._p_current, t)

    def set_params(self, p):
        """Set current parameter values for f_numpy."""
        self._p_current = np.array(p, dtype=float)
