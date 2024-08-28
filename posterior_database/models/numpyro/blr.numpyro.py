from stannumpyro.distributions import *
from stannumpyro.dppllib import ops_index, ops_index_update, lax_cond, numpyro_cond, lax_while_loop, lax_fori_loop, fori_loop, foreach_loop, lax_foreach_loop, jit, sample, param, observe, factor, array, zeros, ones, empty, matmul, true_divide, floor_divide, transpose, dtype_long, dtype_float, vmap, get_element

def convert_inputs(inputs):
    D = inputs['D']
    N = inputs['N']
    X = array(inputs['X'], dtype=dtype_float)
    y = array(inputs['y'], dtype=dtype_float)
    return { 'D': D, 'N': N, 'X': X, 'y': y }

def model(*, D, N, X, y):
    # Parameters
    beta__ = sample('beta', improper_uniform(shape=[D]))
    sigma = sample('sigma', lower_constrained_improper_uniform(0, shape=[]))
    # Model
    factor('_expr__1', normal_lpdf(beta__, 0, 10))
    factor('_expr__2', normal_lpdf(sigma, 0, 10))
    factor('_expr__3', normal_lpdf(y, matmul(X, beta__), sigma))

def parameters_info(*, D, N, X, y):
    return { 'beta': { 'shape': [D] },'sigma': { 'shape': [] }, }

