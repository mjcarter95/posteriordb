from stannumpyro.distributions import *
from stannumpyro.dppllib import lax_cond, numpyro_cond, lax_while_loop, lax_fori_loop, fori_loop, foreach_loop, lax_foreach_loop, jit, sample, param, observe, factor, array, zeros, ones, empty, matmul, true_divide, floor_divide, transpose, dtype_long, dtype_float, vmap, get_element
from stannumpyro.stanlib import mean_array


def convert_inputs(inputs):
    N = inputs['N']
    R = inputs['R']
    culm = array(inputs['culm'], dtype=dtype_long)
    T = inputs['T']
    response = array(inputs['response'], dtype=dtype_long)
    return {'N': N, 'R': R, 'culm': culm, 'T': T, 'response': response}


def transformed_data(*, N, R, culm, T, response):
    r = empty([T, N], dtype=dtype_long)

    @jit
    def _fori__1(j, _acc__2):
        r = _acc__2

        @jit
        def _fori__3(k, _acc__4):
            r = _acc__4
            r = r.at[k - 1, j - 1].set(response[1 - 1, k - 1])
            return r
        r = lax_fori_loop(1, T + 1, _fori__3, r)
        return r
    r = lax_fori_loop(1, culm[1 - 1] + 1, _fori__1, r)

    @jit
    def _fori__5(i, _acc__6):
        r = _acc__6

        @jit
        def _fori__7(j, _acc__8):
            r = _acc__8

            @jit
            def _fori__9(k, _acc__10):
                r = _acc__10
                r = r.at[k - 1, j - 1].set(response[i - 1, k - 1])
                return r
            r = lax_fori_loop(1, T + 1, _fori__9, r)
            return r
        r = lax_fori_loop(culm[i - 1 - 1] + 1, culm[i - 1] + 1, _fori__7, r)
        return r
    r = lax_fori_loop(2, R + 1, _fori__5, r)
    ones__ = empty([N], dtype=dtype_float)

    @jit
    def _fori__11(i, _acc__12):
        ones__ = _acc__12
        ones__ = ones__.at[i - 1].set(array(1.0, dtype=dtype_float))
        return ones__
    ones__ = lax_fori_loop(1, N + 1, _fori__11, ones__)
    return {'r': r, 'ones__': ones__}


def model(*, N, R, culm, T, response, r, ones__):
    alpha = sample('alpha', improper_uniform(shape=[T]))
    theta = sample('theta', improper_uniform(shape=[N]))
    beta__ = sample('beta', lower_constrained_improper_uniform(0, shape=[]))
    observe('_alpha__13', normal(0, array(100.0, dtype=dtype_float)), alpha)
    observe('_theta__14', normal(0, 1), theta)
    observe('_beta__15', normal(array(0.0, dtype=dtype_float), array(100.0,
        dtype=dtype_float)), beta__)

    def _fori__16(k, _acc__17):
        observe(f'_r__{k}__18', bernoulli_logit(beta__ * theta - alpha[k - 
            1] * ones__), r[k - 1])
        return None
    _ = fori_loop(1, T + 1, _fori__16, None)


def generated_quantities(*, N, R, culm, T, response, r, ones__, alpha,
    theta, beta__):
    mean_alpha = mean_array(alpha)
    a = empty([T], dtype=dtype_float)

    @jit
    def _fori__19(t, _acc__20):
        a = _acc__20
        a = a.at[t - 1].set(alpha[t - 1] - mean_alpha)
        return a
    a = lax_fori_loop(1, T + 1, _fori__19, a)
    return {'mean_alpha': mean_alpha, 'a': a}


def map_generated_quantities(_samples, *, N, R, culm, T, response, r, ones__):

    def _generated_quantities(alpha, theta, beta__):
        return generated_quantities(N=N, R=R, culm=culm, T=T, response=
            response, r=r, ones__=ones__, alpha=alpha, theta=theta, beta__=
            beta__)
    _f = jit(vmap(_generated_quantities))
    return _f(_samples['alpha'], _samples['theta'], _samples['beta'])


def parameters_info(*, N, R, culm, T, response, r, ones__):
    return {'alpha': {'shape': [T]}, 'theta': {'shape': [N]}, 'beta': {
        'shape': []}}
