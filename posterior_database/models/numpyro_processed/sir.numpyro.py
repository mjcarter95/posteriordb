from stannumpyro.distributions import *
from stannumpyro.dppllib import lax_cond, numpyro_cond, lax_while_loop, lax_fori_loop, fori_loop, foreach_loop, lax_foreach_loop, jit, sample, param, observe, factor, array, zeros, ones, empty, matmul, true_divide, floor_divide, transpose, dtype_long, dtype_float, vmap, get_element
from stannumpyro.stanlib import col_matrix_int, integrate_ode_rk45_function_array_real_array_array_array_array, log_vector, to_matrix_array


def simple_SIR(t, y, theta, x_r, x_i):
    dydt = empty([4], dtype=dtype_float)
    dydt = dydt.at[1 - 1].set(true_divide(-theta[1 - 1] * y[4 - 1], y[4 - 1
        ] + theta[2 - 1]) * y[1 - 1])
    dydt = dydt.at[2 - 1].set(true_divide(theta[1 - 1] * y[4 - 1], y[4 - 1] +
        theta[2 - 1]) * y[1 - 1] - theta[3 - 1] * y[2 - 1])
    dydt = dydt.at[3 - 1].set(theta[3 - 1] * y[2 - 1])
    dydt = dydt.at[4 - 1].set(theta[4 - 1] * y[2 - 1] - theta[5 - 1] * y[4 - 1]
        )
    return dydt


def convert_inputs(inputs):
    N_t = inputs['N_t']
    t = array(inputs['t'], dtype=dtype_float)
    y0 = array(inputs['y0'], dtype=dtype_float)
    stoi_hat = array(inputs['stoi_hat'], dtype=dtype_long)
    B_hat = array(inputs['B_hat'], dtype=dtype_float)
    return {'N_t': N_t, 't': t, 'y0': y0, 'stoi_hat': stoi_hat, 'B_hat': B_hat}


def transformed_data(*, N_t, t, y0, stoi_hat, B_hat):
    t0 = 0
    kappa = 1000000
    x_r = empty([0], dtype=dtype_float)
    x_i = empty([0], dtype=dtype_long)
    return {'t0': t0, 'kappa': kappa, 'x_r': x_r, 'x_i': x_i}


def model(*, N_t, t, y0, stoi_hat, B_hat, t0, kappa, x_r, x_i):
    beta__ = sample('beta', lower_constrained_improper_uniform(0, shape=[]))
    gamma__ = sample('gamma', lower_constrained_improper_uniform(0, shape=[]))
    xi = sample('xi', lower_constrained_improper_uniform(0, shape=[]))
    delta = sample('delta', lower_constrained_improper_uniform(0, shape=[]))
    theta = array([beta__, kappa, gamma__, xi, delta], dtype=dtype_float)
    y = integrate_ode_rk45_function_array_real_array_array_array_array(
        simple_SIR, y0, t0, t, theta, x_r, x_i)
    observe('_beta__1', cauchy(0, array(2.5, dtype=dtype_float)), beta__)
    observe('_gamma__2', cauchy(0, 1), gamma__)
    observe('_xi__3', cauchy(0, 25), xi)
    observe('_delta__4', cauchy(0, 1), delta)
    observe('_stoi_hat__5', poisson(y0[1 - 1] - y[1 - 1, 1 - 1]), stoi_hat[
        1 - 1])

    def _fori__6(n, _acc__7):
        observe(f'_stoi_hat__{n}__8', poisson(y[n - 1 - 1, 1 - 1] - y[n - 1,
            1 - 1]), stoi_hat[n - 1])
        return None
    _ = fori_loop(2, N_t + 1, _fori__6, None)
    observe('_B_hat__9', lognormal(log_vector(col_matrix_int(
        to_matrix_array(y), 4)), array(0.15, dtype=dtype_float)), B_hat)


def generated_quantities(*, N_t, t, y0, stoi_hat, B_hat, t0, kappa, x_r,
    x_i, beta__, gamma__, xi, delta):
    theta = array([beta__, kappa, gamma__, xi, delta], dtype=dtype_float)
    y = integrate_ode_rk45_function_array_real_array_array_array_array(
        simple_SIR, y0, t0, t, theta, x_r, x_i)
    return {'theta': theta, 'y': y}


def map_generated_quantities(_samples, *, N_t, t, y0, stoi_hat, B_hat, t0,
    kappa, x_r, x_i):

    def _generated_quantities(beta__, gamma__, xi, delta):
        return generated_quantities(N_t=N_t, t=t, y0=y0, stoi_hat=stoi_hat,
            B_hat=B_hat, t0=t0, kappa=kappa, x_r=x_r, x_i=x_i, beta__=
            beta__, gamma__=gamma__, xi=xi, delta=delta)
    _f = jit(vmap(_generated_quantities))
    return _f(_samples['beta'], _samples['gamma'], _samples['xi'], _samples
        ['delta'])


def parameters_info(*, N_t, t, y0, stoi_hat, B_hat, t0, kappa, x_r, x_i):
    return {'beta': {'shape': []}, 'gamma': {'shape': []}, 'xi': {'shape':
        []}, 'delta': {'shape': []}}
