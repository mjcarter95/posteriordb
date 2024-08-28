from stannumpyro.distributions import *
from stannumpyro.dppllib import ops_index, ops_index_update, lax_cond, numpyro_cond, lax_while_loop, lax_fori_loop, fori_loop, foreach_loop, lax_foreach_loop, jit, sample, param, observe, factor, array, zeros, ones, empty, matmul, true_divide, floor_divide, transpose, dtype_long, dtype_float, vmap, get_element
from stannumpyro.stanlib import mean_vector, sd_vector, sum_vector

def convert_inputs(inputs):
    n = inputs['n']
    y = array(inputs['y'], dtype=dtype_float)
    x = array(inputs['x'], dtype=dtype_float)
    w = array(inputs['w'], dtype=dtype_float)
    return { 'n': n, 'y': y, 'x': x, 'w': w }

def model(*, n, y, x, w):
    # Parameters
    mu = sample('mu', uniform(mean_vector(y) - 3 * sd_vector(y) * ones([
    n]), mean_vector(y) + 3 * sd_vector(y)))
    seasonal = sample('seasonal', improper_uniform(shape=[n]))
    beta__ = sample('beta', improper_uniform(shape=[]))
    lambda__ = sample('lambda', improper_uniform(shape=[]))
    sigma = sample('sigma', positive_ordered_constrained_improper_uniform(shape=[
    3]))
    # Transformed parameters
    yhat = mu + beta__ * x + lambda__ * w
    # Model
    def _fori__1(t, _acc__2):
        observe(f'_seasonal__{t}__3', normal(- sum_vector(seasonal[t - 11 - 1:t - 1]),
                                             sigma[1 - 1]), seasonal[
        t - 1])
        return None
    _ = fori_loop(12, n + 1, _fori__1, None)
    def _fori__4(t, _acc__5):
        observe(f'_mu__{t}__6', normal(mu[t - 1 - 1], sigma[2 - 1]), mu[
        t - 1])
        return None
    _ = fori_loop(2, n + 1, _fori__4, None)
    observe('_y__7', normal(yhat + seasonal, sigma[3 - 1]), y)
    observe('_sigma__8', student_t(4, 0, 1), sigma)


def generated_quantities(*, n, y, x, w, mu, seasonal, beta__, lambda__, sigma):
    # Transformed parameters
    yhat = mu + beta__ * x + lambda__ * w
    return { 'yhat': yhat }

def map_generated_quantities(_samples, *, n, y, x, w):
    def _generated_quantities(mu, seasonal, beta__, lambda__, sigma):
        return generated_quantities(n=n, y=y, x=x, w=w, mu=mu,
                                    seasonal=seasonal, beta__=beta__,
                                    lambda__=lambda__, sigma=sigma)
    _f = jit(vmap(_generated_quantities))
    return _f(_samples['mu'], _samples['seasonal'], _samples['beta'],
              _samples['lambda'], _samples['sigma'])

def parameters_info(*, n, y, x, w):
    return { 'mu': { 'shape': [n] },'seasonal': { 'shape': [n] },
             'beta': { 'shape': [] },'lambda': { 'shape': [] },
             'sigma': { 'shape': [3] }, }

