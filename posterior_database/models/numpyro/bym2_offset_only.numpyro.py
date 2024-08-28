from stannumpyro.distributions import *
from stannumpyro.dppllib import ops_index, ops_index_update, lax_cond, numpyro_cond, lax_while_loop, lax_fori_loop, fori_loop, foreach_loop, lax_foreach_loop, jit, sample, param, observe, factor, array, zeros, ones, empty, matmul, true_divide, floor_divide, transpose, dtype_long, dtype_float, vmap, get_element
from stannumpyro.stanlib import dot_self_vector, exp_vector, log_real, log_vector, sqrt_real, sum_vector

def convert_inputs(inputs):
    N_edges = inputs['N_edges']
    N = inputs['N']
    node1 = array(inputs['node1'], dtype=dtype_long)
    node2 = array(inputs['node2'], dtype=dtype_long)
    y = array(inputs['y'], dtype=dtype_long)
    E = array(inputs['E'], dtype=dtype_float)
    scaling_factor = array(inputs['scaling_factor'], dtype=dtype_float)
    return { 'N_edges': N_edges, 'N': N, 'node1': node1, 'node2': node2,
             'y': y, 'E': E, 'scaling_factor': scaling_factor }

def transformed_data(*, N_edges, N, node1, node2, y, E, scaling_factor):
    # Transformed data
    log_E = log_vector(E)
    return { 'log_E': log_E }

def model(*, N_edges, N, node1, node2, y, E, scaling_factor, log_E):
    # Parameters
    beta0 = sample('beta0', improper_uniform(shape=[]))
    sigma = sample('sigma', lower_constrained_improper_uniform(0, shape=[]))
    rho = sample('rho', uniform(0, 1))
    theta = sample('theta', improper_uniform(shape=[N]))
    phi = sample('phi', improper_uniform(shape=[N]))
    # Transformed parameters
    convolved_re = sqrt_real(1 - rho) * theta + sqrt_real(true_divide(rho, scaling_factor)) * phi
    # Model
    observe('_y__1', poisson_log(log_E + beta0 + convolved_re * sigma), y)
    factor('_expr__2', - array(0.5, dtype=dtype_float) * dot_self_vector(
    phi[node1 - 1] - phi[node2 - 1]))
    observe('_beta0__3', normal(0, 1), beta0)
    observe('_theta__4', normal(0, 1), theta)
    observe('_sigma__5', normal(0, 1), sigma)
    observe('_rho__6', beta(array(0.5, dtype=dtype_float),
                            array(0.5, dtype=dtype_float)), rho)
    observe('_expr__7', normal(0, array(0.001, dtype=dtype_float) * N), sum_vector(
    phi))


def generated_quantities(*, N_edges, N, node1, node2, y, E, scaling_factor,
                            log_E, beta0, sigma, rho, theta, phi):
    # Transformed parameters
    convolved_re = sqrt_real(1 - rho) * theta + sqrt_real(true_divide(rho, scaling_factor)) * phi
    # Generated quantities
    log_precision = - array(2.0, dtype=dtype_float) * log_real(sigma)
    logit_rho = log_real(true_divide(rho, (array(1.0, dtype=dtype_float) - rho)))
    eta = log_E + beta0 + convolved_re * sigma
    mu = exp_vector(eta)
    return { 'convolved_re': convolved_re, 'log_precision': log_precision,
             'logit_rho': logit_rho, 'eta': eta, 'mu': mu }

def map_generated_quantities(_samples, *, N_edges, N, node1, node2, y, E,
                                          scaling_factor, log_E):
    def _generated_quantities(beta0, sigma, rho, theta, phi):
        return generated_quantities(N_edges=N_edges, N=N, node1=node1,
                                    node2=node2, y=y, E=E,
                                    scaling_factor=scaling_factor,
                                    log_E=log_E, beta0=beta0, sigma=sigma,
                                    rho=rho, theta=theta, phi=phi)
    _f = jit(vmap(_generated_quantities))
    return _f(_samples['beta0'], _samples['sigma'], _samples['rho'],
              _samples['theta'], _samples['phi'])

def parameters_info(*, N_edges, N, node1, node2, y, E, scaling_factor, log_E):
    return { 'beta0': { 'shape': [] },'sigma': { 'shape': [] },
             'rho': { 'shape': [] },'theta': { 'shape': [N] },
             'phi': { 'shape': [N] }, }

