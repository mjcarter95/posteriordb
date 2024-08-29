from stannumpyro.distributions import *
from stannumpyro.dppllib import lax_cond, numpyro_cond, lax_while_loop, lax_fori_loop, fori_loop, foreach_loop, lax_foreach_loop, jit, sample, param, observe, factor, array, zeros, ones, empty, matmul, true_divide, floor_divide, transpose, dtype_long, dtype_float, vmap, get_element
from stannumpyro.stanlib import integrate_ode_rk45_function_array_real_array_array_array_array, sum_array


def two_pool_feedback(t, C, theta, x_r, x_i):
    k1 = theta[1 - 1]
    k2 = theta[2 - 1]
    alpha21 = theta[3 - 1]
    alpha12 = theta[4 - 1]
    dC_dt = empty([2], dtype=dtype_float)
    dC_dt = dC_dt.at[1 - 1].set(-k1 * C[1 - 1] + alpha12 * k2 * C[2 - 1])
    dC_dt = dC_dt.at[2 - 1].set(-k2 * C[2 - 1] + alpha21 * k1 * C[1 - 1])
    return dC_dt


def evolved_CO2(N_t, t0, ts, gamma__, totalC_t0, k1, k2, alpha21, alpha12,
    x_r, x_i):
    C_t0 = empty([2], dtype=dtype_float)
    C_t0 = C_t0.at[1 - 1].set(gamma__ * totalC_t0)
    C_t0 = C_t0.at[2 - 1].set((1 - gamma__) * totalC_t0)
    theta = empty([4], dtype=dtype_float)
    theta = theta.at[1 - 1].set(k1)
    theta = theta.at[2 - 1].set(k2)
    theta = theta.at[3 - 1].set(alpha21)
    theta = theta.at[4 - 1].set(alpha12)
    C_hat = integrate_ode_rk45_function_array_real_array_array_array_array(
        two_pool_feedback, C_t0, t0, ts, theta, x_r, x_i)
    eCO2_hat = empty([N_t], dtype=dtype_float)

    @jit
    def _fori__1(t, _acc__2):
        eCO2_hat = _acc__2
        eCO2_hat = eCO2_hat.at[t - 1].set(totalC_t0 - sum_array(C_hat[t - 1]))
        return eCO2_hat
    eCO2_hat = lax_fori_loop(1, N_t + 1, _fori__1, eCO2_hat)
    return eCO2_hat


def convert_inputs(inputs):
    totalC_t0 = array(inputs['totalC_t0'], dtype=dtype_float)
    t0 = array(inputs['t0'], dtype=dtype_float)
    N_t = inputs['N_t']
    ts = array(inputs['ts'], dtype=dtype_float)
    eCO2mean = array(inputs['eCO2mean'], dtype=dtype_float)
    return {'totalC_t0': totalC_t0, 't0': t0, 'N_t': N_t, 'ts': ts,
        'eCO2mean': eCO2mean}


def transformed_data(*, totalC_t0, t0, N_t, ts, eCO2mean):
    x_r = empty([0], dtype=dtype_float)
    x_i = empty([0], dtype=dtype_long)
    return {'x_r': x_r, 'x_i': x_i}


def model(*, totalC_t0, t0, N_t, ts, eCO2mean, x_r, x_i):
    k1 = sample('k1', lower_constrained_improper_uniform(0, shape=[]))
    k2 = sample('k2', lower_constrained_improper_uniform(0, shape=[]))
    alpha21 = sample('alpha21', lower_constrained_improper_uniform(0, shape=[])
        )
    alpha12 = sample('alpha12', lower_constrained_improper_uniform(0, shape=[])
        )
    gamma__ = sample('gamma', uniform(0, 1))
    sigma = sample('sigma', lower_constrained_improper_uniform(0, shape=[]))
    eCO2_hat = evolved_CO2(N_t, t0, ts, gamma__, totalC_t0, k1, k2, alpha21,
        alpha12, x_r, x_i)
    observe('_gamma__3', beta(10, 1), gamma__)
    observe('_k1__4', normal(0, 1), k1)
    observe('_k2__5', normal(0, 1), k2)
    observe('_alpha21__6', normal(0, 1), alpha21)
    observe('_alpha12__7', normal(0, 1), alpha12)
    observe('_sigma__8', cauchy(0, 1), sigma)

    def _fori__9(t, _acc__10):
        observe(f'_eCO2mean__{t}__11', normal(eCO2_hat[t - 1], sigma),
            eCO2mean[t - 1])
        return None
    _ = fori_loop(1, N_t + 1, _fori__9, None)


def generated_quantities(*, totalC_t0, t0, N_t, ts, eCO2mean, x_r, x_i, k1,
    k2, alpha21, alpha12, gamma__, sigma):
    eCO2_hat = evolved_CO2(N_t, t0, ts, gamma__, totalC_t0, k1, k2, alpha21,
        alpha12, x_r, x_i)
    return {'eCO2_hat': eCO2_hat}


def map_generated_quantities(_samples, *, totalC_t0, t0, N_t, ts, eCO2mean,
    x_r, x_i):

    def _generated_quantities(k1, k2, alpha21, alpha12, gamma__, sigma):
        return generated_quantities(totalC_t0=totalC_t0, t0=t0, N_t=N_t, ts
            =ts, eCO2mean=eCO2mean, x_r=x_r, x_i=x_i, k1=k1, k2=k2, alpha21
            =alpha21, alpha12=alpha12, gamma__=gamma__, sigma=sigma)
    _f = jit(vmap(_generated_quantities))
    return _f(_samples['k1'], _samples['k2'], _samples['alpha21'], _samples
        ['alpha12'], _samples['gamma'], _samples['sigma'])


def parameters_info(*, totalC_t0, t0, N_t, ts, eCO2mean, x_r, x_i):
    return {'k1': {'shape': []}, 'k2': {'shape': []}, 'alpha21': {'shape':
        []}, 'alpha12': {'shape': []}, 'gamma': {'shape': []}, 'sigma': {
        'shape': []}}
