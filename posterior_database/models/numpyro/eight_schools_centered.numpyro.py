from stannumpyro.distributions import *
from stannumpyro.dppllib import ops_index, ops_index_update, lax_cond, numpyro_cond, lax_while_loop, lax_fori_loop, fori_loop, foreach_loop, lax_foreach_loop, jit, sample, param, observe, factor, array, zeros, ones, empty, matmul, true_divide, floor_divide, transpose, dtype_long, dtype_float, vmap, get_element

def convert_inputs(inputs):
    J = inputs['J']
    y = array(inputs['y'], dtype=dtype_float)
    sigma = array(inputs['sigma'], dtype=dtype_float)
    return { 'J': J, 'y': y, 'sigma': sigma }

def model(*, J, y, sigma):
    # Parameters
    theta = sample('theta', improper_uniform(shape=[J]))
    mu = sample('mu', improper_uniform(shape=[]))
    tau = sample('tau', lower_constrained_improper_uniform(0, shape=[]))
    # Model
    observe('_tau__1', cauchy(0, 5), tau)
    observe('_theta__2', normal(mu, tau), theta)
    observe('_y__3', normal(theta, sigma), y)
    observe('_mu__4', normal(0, 5), mu)

def parameters_info(*, J, y, sigma):
    return { 'theta': { 'shape': [J] },'mu': { 'shape': [] },
             'tau': { 'shape': [] }, }

