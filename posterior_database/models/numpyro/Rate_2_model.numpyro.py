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
    theta1 = sample('theta1', uniform(0, 1))
    theta2 = sample('theta2', uniform(0, 1))
    # Transformed parameters
    delta = theta1 - theta2
    # Model
    observe('_theta1__1', beta(1, 1), theta1)
    observe('_theta2__2', beta(1, 1), theta2)
    observe('_k1__3', binomial(n1, theta1), k1)
    observe('_k2__4', binomial(n2, theta2), k2)


def generated_quantities(*, n1, n2, k1, k2, theta1, theta2):
    # Transformed parameters
    delta = theta1 - theta2
    return { 'delta': delta }

def map_generated_quantities(_samples, *, n1, n2, k1, k2):
    def _generated_quantities(theta1, theta2):
        return generated_quantities(n1=n1, n2=n2, k1=k1, k2=k2,
                                    theta1=theta1, theta2=theta2)
    _f = jit(vmap(_generated_quantities))
    return _f(_samples['theta1'], _samples['theta2'])

def parameters_info(*, n1, n2, k1, k2):
    return { 'theta1': { 'shape': [] },'theta2': { 'shape': [] }, }

