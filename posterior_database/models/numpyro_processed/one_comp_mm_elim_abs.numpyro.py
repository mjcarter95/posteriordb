from stannumpyro.distributions import *
from stannumpyro.dppllib import lax_cond, numpyro_cond, lax_while_loop, lax_fori_loop, fori_loop, foreach_loop, lax_foreach_loop, jit, sample, param, observe, factor, array, zeros, ones, empty, matmul, true_divide, floor_divide, transpose, dtype_long, dtype_float, vmap, get_element
from stannumpyro.stanlib import exp_real, integrate_ode_bdf_function_array_real_array_array_array_array, log_real


def one_comp_mm_elim_abs(t, y, theta, x_r, x_i):
    k_a = theta[1 - 1]
    K_m = theta[2 - 1]
    V_m = theta[3 - 1]
    D = x_r[1 - 1]
    V = x_r[2 - 1]
    dose = 0
    elim = true_divide(true_divide(V_m, V) * y[1 - 1], K_m + y[1 - 1])

    @jit
    def _then__1(_acc__2):
        dose = _acc__2
        dose = true_divide(exp_real(-k_a * t) * D * k_a, V)
        return dose

    @jit
    def _else__3(acc):
        return acc
    dose = lax_cond(t > 0, _then__1, _else__3, dose)
    dydt = empty([1], dtype=dtype_float)
    dydt = dydt.at[1 - 1].set(dose - elim)
    return dydt


def convert_inputs(inputs):
    t0 = array(inputs['t0'], dtype=dtype_float)
    D = array(inputs['D'], dtype=dtype_float)
    V = array(inputs['V'], dtype=dtype_float)
    N_t = inputs['N_t']
    times = array(inputs['times'], dtype=dtype_float)
    C_hat = array(inputs['C_hat'], dtype=dtype_float)
    return {'t0': t0, 'D': D, 'V': V, 'N_t': N_t, 'times': times, 'C_hat':
        C_hat}


def transformed_data(*, t0, D, V, N_t, times, C_hat):
    C0 = array([array(0.0, dtype=dtype_float)], dtype=dtype_float)
    x_r = array([D, V], dtype=dtype_float)
    x_i = empty([0], dtype=dtype_long)
    return {'C0': C0, 'x_r': x_r, 'x_i': x_i}


def model(*, t0, D, V, N_t, times, C_hat, C0, x_r, x_i):
    k_a = sample('k_a', lower_constrained_improper_uniform(0, shape=[]))
    K_m = sample('K_m', lower_constrained_improper_uniform(0, shape=[]))
    V_m = sample('V_m', lower_constrained_improper_uniform(0, shape=[]))
    sigma = sample('sigma', lower_constrained_improper_uniform(0, shape=[]))
    theta = array([k_a, K_m, V_m], dtype=dtype_float)
    C = integrate_ode_bdf_function_array_real_array_array_array_array(
        one_comp_mm_elim_abs, C0, t0, times, theta, x_r, x_i)
    observe('_k_a__4', cauchy(0, 1), k_a)
    observe('_K_m__5', cauchy(0, 1), K_m)
    observe('_V_m__6', cauchy(0, 1), V_m)
    observe('_sigma__7', cauchy(0, 1), sigma)

    def _fori__8(n, _acc__9):
        observe(f'_C_hat__{n}__10', lognormal(log_real(C[n - 1, 1 - 1]),
            sigma), C_hat[n - 1])
        return None
    _ = fori_loop(1, N_t + 1, _fori__8, None)


def generated_quantities(*, t0, D, V, N_t, times, C_hat, C0, x_r, x_i, k_a,
    K_m, V_m, sigma):
    theta = array([k_a, K_m, V_m], dtype=dtype_float)
    C = integrate_ode_bdf_function_array_real_array_array_array_array(
        one_comp_mm_elim_abs, C0, t0, times, theta, x_r, x_i)
    C_ppc = empty([N_t], dtype=dtype_float)

    @jit
    def _fori__11(n, _acc__12):
        C_ppc = _acc__12
        C_ppc = C_ppc.at[n - 1].set(lognormal_rng(log_real(C[n - 1, 1 - 1]),
            sigma))
        return C_ppc
    C_ppc = lax_fori_loop(1, N_t + 1, _fori__11, C_ppc)
    return {'theta': theta, 'C': C, 'C_ppc': C_ppc}


def map_generated_quantities(_samples, *, t0, D, V, N_t, times, C_hat, C0,
    x_r, x_i):

    def _generated_quantities(k_a, K_m, V_m, sigma):
        return generated_quantities(t0=t0, D=D, V=V, N_t=N_t, times=times,
            C_hat=C_hat, C0=C0, x_r=x_r, x_i=x_i, k_a=k_a, K_m=K_m, V_m=V_m,
            sigma=sigma)
    _f = jit(vmap(_generated_quantities))
    return _f(_samples['k_a'], _samples['K_m'], _samples['V_m'], _samples[
        'sigma'])


def parameters_info(*, t0, D, V, N_t, times, C_hat, C0, x_r, x_i):
    return {'k_a': {'shape': []}, 'K_m': {'shape': []}, 'V_m': {'shape': []
        }, 'sigma': {'shape': []}}
