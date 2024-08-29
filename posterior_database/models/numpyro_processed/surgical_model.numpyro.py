from stannumpyro.distributions import *
from stannumpyro.dppllib import lax_cond, numpyro_cond, lax_while_loop, lax_fori_loop, fori_loop, foreach_loop, lax_foreach_loop, jit, sample, param, observe, factor, array, zeros, ones, empty, matmul, true_divide, floor_divide, transpose, dtype_long, dtype_float, vmap, get_element
from stannumpyro.stanlib import inv_logit_real, sqrt_real


def convert_inputs(inputs):
    N = inputs['N']
    r = array(inputs['r'], dtype=dtype_long)
    n = array(inputs['n'], dtype=dtype_long)
    return {'N': N, 'r': r, 'n': n}


def model(*, N, r, n):
    mu = sample('mu', improper_uniform(shape=[]))
    sigmasq = sample('sigmasq', lower_constrained_improper_uniform(0, shape=[])
        )
    b = sample('b', improper_uniform(shape=[N]))
    sigma = sqrt_real(sigmasq)
    p = empty([N], dtype=dtype_float)

    @jit
    def _fori__1(i, _acc__2):
        p = _acc__2
        p = p.at[i - 1].set(inv_logit_real(b[i - 1]))
        return p
    p = lax_fori_loop(1, N + 1, _fori__1, p)
    observe('_mu__3', normal(array(0.0, dtype=dtype_float), array(1000.0,
        dtype=dtype_float)), mu)
    observe('_sigmasq__4', inv_gamma(array(0.001, dtype=dtype_float), array
        (0.001, dtype=dtype_float)), sigmasq)
    observe('_b__5', normal(mu, sigma), b)
    observe('_r__6', binomial_logit(n, b), r)


def generated_quantities(*, N, r, n, mu, sigmasq, b):
    sigma = sqrt_real(sigmasq)
    p = empty([N], dtype=dtype_float)

    @jit
    def _fori__7(i, _acc__8):
        p = _acc__8
        p = p.at[i - 1].set(inv_logit_real(b[i - 1]))
        return p
    p = lax_fori_loop(1, N + 1, _fori__7, p)
    pop_mean = inv_logit_real(mu)
    return {'sigma': sigma, 'p': p, 'pop_mean': pop_mean}


def map_generated_quantities(_samples, *, N, r, n):

    def _generated_quantities(mu, sigmasq, b):
        return generated_quantities(N=N, r=r, n=n, mu=mu, sigmasq=sigmasq, b=b)
    _f = jit(vmap(_generated_quantities))
    return _f(_samples['mu'], _samples['sigmasq'], _samples['b'])


def parameters_info(*, N, r, n):
    return {'mu': {'shape': []}, 'sigmasq': {'shape': []}, 'b': {'shape': [N]}}
