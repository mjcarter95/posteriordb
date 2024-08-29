from stannumpyro.distributions import *
from stannumpyro.dppllib import lax_cond, numpyro_cond, lax_while_loop, lax_fori_loop, fori_loop, foreach_loop, lax_foreach_loop, jit, sample, param, observe, factor, array, zeros, ones, empty, matmul, true_divide, floor_divide, transpose, dtype_long, dtype_float, vmap, get_element


def convert_inputs(inputs):
    n = inputs['n']
    k = inputs['k']
    return {'n': n, 'k': k}


def model(*, n, k):
    theta = sample('theta', uniform(0, 1))
    thetaprior = sample('thetaprior', uniform(0, 1))
    observe('_theta__1', beta(1, 1), theta)
    observe('_thetaprior__2', beta(1, 1), thetaprior)
    observe('_k__3', binomial(n, theta), k)


def generated_quantities(*, n, k, theta, thetaprior):
    postpredk = binomial_rng(n, theta)
    priorpredk = binomial_rng(n, thetaprior)
    return {'postpredk': postpredk, 'priorpredk': priorpredk}


def map_generated_quantities(_samples, *, n, k):

    def _generated_quantities(theta, thetaprior):
        return generated_quantities(n=n, k=k, theta=theta, thetaprior=
            thetaprior)
    _f = jit(vmap(_generated_quantities))
    return _f(_samples['theta'], _samples['thetaprior'])


def parameters_info(*, n, k):
    return {'theta': {'shape': []}, 'thetaprior': {'shape': []}}
