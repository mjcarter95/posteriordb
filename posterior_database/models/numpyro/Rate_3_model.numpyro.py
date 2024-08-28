from stannumpyro.distributions import *
from stannumpyro.dppllib import ops_index, ops_index_update, lax_cond, numpyro_cond, lax_while_loop, lax_fori_loop, fori_loop, foreach_loop, lax_foreach_loop, jit, sample, param, observe, factor, array, zeros, ones, empty, matmul, true_divide, floor_divide, transpose, dtype_long, dtype_float, vmap, get_element

def convert_inputs(inputs):
    n1 = inputs['n1']
    n2 = inputs['n2']
    k1 = inputs['k1']
    k2 = inputs['k2']
    return { 'n1': n1, 'n2': n2, 'k1': k1, 'k2': k2 }

def model(*, n1, n2, k1, k2):
    # Parameters
    theta = sample('theta', uniform(0, 1))
    # Model
    observe('_theta__1', beta(1, 1), theta)
    observe('_k1__2', binomial(n1, theta), k1)
    observe('_k2__3', binomial(n2, theta), k2)

def parameters_info(*, n1, n2, k1, k2):
    return { 'theta': { 'shape': [] }, }

