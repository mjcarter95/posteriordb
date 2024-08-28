from stannumpyro.distributions import *
from stannumpyro.dppllib import ops_index, ops_index_update, lax_cond, numpyro_cond, lax_while_loop, lax_fori_loop, fori_loop, foreach_loop, lax_foreach_loop, jit, sample, param, observe, factor, array, zeros, ones, empty, matmul, true_divide, floor_divide, transpose, dtype_long, dtype_float, vmap, get_element
from stannumpyro.stanlib import integrate_ode_rk45_function_array_int_array_array_array_array_real_real_real, log_array, log_int, log_real, rep_array_int_int, rep_array_real_int

def dz_dt(t, z, theta, x_r, x_i):
    u = z[1 - 1]
    v = z[2 - 1]
    alpha = theta[1 - 1]
    beta__ = theta[2 - 1]
    gamma__ = theta[3 - 1]
    delta = theta[4 - 1]
    du_dt = (alpha - beta__ * v) * u
    dv_dt = (- gamma__ + delta * u) * v
    return array([du_dt, dv_dt], dtype=dtype_float)

def convert_inputs(inputs):
    N = inputs['N']
    ts = array(inputs['ts'], dtype=dtype_float)
    y_init = array(inputs['y_init'], dtype=dtype_float)
    y = array(inputs['y'], dtype=dtype_float)
    return { 'N': N, 'ts': ts, 'y_init': y_init, 'y': y }

def model(*, N, ts, y_init, y):
    # Parameters
    theta = sample('theta', lower_constrained_improper_uniform(0, shape=[
    4]))
    z_init = sample('z_init', lower_constrained_improper_uniform(0, shape=[
    2]))
    sigma = sample('sigma', lower_constrained_improper_uniform(0, shape=[
    2]))
    # Transformed parameters
    z = integrate_ode_rk45_function_array_int_array_array_array_array_real_real_real(
    dz_dt, z_init, 0, ts, theta,
    rep_array_real_int(array(0.0, dtype=dtype_float), 0),
    rep_array_int_int(0, 0), array(1e-5, dtype=dtype_float),
    array(1e-3, dtype=dtype_float), array(5e2, dtype=dtype_float))
    # Model
    observe('_theta__1', normal(1, array(0.5, dtype=dtype_float)), theta[
    array([1, 3], dtype=dtype_long) - 1])
    observe('_theta__2', normal(array(0.05, dtype=dtype_float),
                                array(0.05, dtype=dtype_float)), theta[
    array([2, 4], dtype=dtype_long) - 1])
    observe('_sigma__3', lognormal(- 1, 1), sigma)
    observe('_z_init__4', lognormal(log_int(10), 1), z_init)
    def _fori__5(k, _acc__6):
        observe(f'_y_init__{k}__7', lognormal(log_real(z_init[k - 1]),
                                              sigma[k - 1]), y_init[k - 1])
        observe(f'_y__{k}__8', lognormal(log_array(z[:, k - 1]), sigma[k - 1]), y[
        :, k - 1])
        return None
    _ = fori_loop(1, 2 + 1, _fori__5, None)


def generated_quantities(*, N, ts, y_init, y, theta, z_init, sigma):
    # Transformed parameters
    z = integrate_ode_rk45_function_array_int_array_array_array_array_real_real_real(
    dz_dt, z_init, 0, ts, theta,
    rep_array_real_int(array(0.0, dtype=dtype_float), 0),
    rep_array_int_int(0, 0), array(1e-5, dtype=dtype_float),
    array(1e-3, dtype=dtype_float), array(5e2, dtype=dtype_float))
    # Generated quantities
    y_init_rep = empty([2], dtype=dtype_float)
    y_rep = empty([N, 2], dtype=dtype_float)
    @jit
    def _fori__9(k, _acc__10):
        (y_init_rep, y_rep) = _acc__10
        y_init_rep = ops_index_update(y_init_rep, ops_index[k - 1], lognormal_rng(
        log_real(z_init[k - 1]), sigma[k - 1]))
        @jit
        def _fori__11(n, _acc__12):
            y_rep = _acc__12
            y_rep = ops_index_update(y_rep, ops_index[n - 1, k - 1], lognormal_rng(
            log_real(z[n - 1, k - 1]), sigma[k - 1]))
            return y_rep
        y_rep = lax_fori_loop(1, N + 1, _fori__11, y_rep)
        return (y_init_rep, y_rep)
    (y_init_rep, y_rep) = lax_fori_loop(1, 2 + 1, _fori__9,
                                        (y_init_rep, y_rep))
    return { 'z': z, 'y_init_rep': y_init_rep, 'y_rep': y_rep }

def map_generated_quantities(_samples, *, N, ts, y_init, y):
    def _generated_quantities(theta, z_init, sigma):
        return generated_quantities(N=N, ts=ts, y_init=y_init, y=y,
                                    theta=theta, z_init=z_init, sigma=sigma)
    _f = jit(vmap(_generated_quantities))
    return _f(_samples['theta'], _samples['z_init'], _samples['sigma'])

def parameters_info(*, N, ts, y_init, y):
    return { 'theta': { 'shape': [4] },'z_init': { 'shape': [2] },
             'sigma': { 'shape': [2] }, }

