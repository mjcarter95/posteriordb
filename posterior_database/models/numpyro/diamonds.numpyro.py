from stannumpyro.distributions import *
from stannumpyro.dppllib import ops_index, ops_index_update, lax_cond, numpyro_cond, lax_while_loop, lax_fori_loop, fori_loop, foreach_loop, lax_foreach_loop, jit, sample, param, observe, factor, array, zeros, ones, empty, matmul, true_divide, floor_divide, transpose, dtype_long, dtype_float, vmap, get_element
from stannumpyro.stanlib import dot_product_vector_vector, mean_vector


def convert_inputs(inputs):
    N = inputs['N']
    Y = array(inputs['Y'], dtype=dtype_float)
    K = inputs['K']
    X = array(inputs['X'], dtype=dtype_float)
    prior_only = inputs['prior_only']
    return { 'N': N, 'Y': Y, 'K': K, 'X': X, 'prior_only': prior_only }

def transformed_data(*, N, Y, K, X, prior_only):
    # Transformed data
    Kc = K - 1
    Xc = empty([N, Kc], dtype=dtype_float)
    means_X = empty([Kc], dtype=dtype_float)
    @jit
    def _fori__1(i, _acc__2):
        (Xc, means_X) = _acc__2
        means_X = ops_index_update(means_X, ops_index[i - 1 - 1], mean_vector(
        X[:, i - 1]))
        Xc = ops_index_update(Xc, ops_index[:, i - 1 - 1], X[:, i - 1] - means_X[
        i - 1 - 1])
        return (Xc, means_X)
    (Xc, means_X) = lax_fori_loop(2, K + 1, _fori__1, (Xc, means_X))
    return { 'Kc': Kc, 'Xc': Xc, 'means_X': means_X }

def model(*, N, Y, K, X, prior_only, Kc, Xc, means_X):
    # Parameters
    b = sample('b', improper_uniform(shape=[Kc]))
    Intercept = sample('Intercept', improper_uniform(shape=[]))
    sigma = sample('sigma', lower_constrained_improper_uniform(0, shape=[]))
    # Transformed parameters
    
    # Model
    factor('_expr__3', normal_lpdf(b, 0, 1))
    factor('_expr__4', student_t_lpdf(Intercept, 3, 8, 10))
    factor('_expr__5', student_t_lpdf(sigma, 3, 0, 10) - 1 * student_t_lccdf(
    0, 3, 0, 10))
    def _then__6(_acc__7):
        factor('_expr__9', normal_id_glm_lpdf(Y, Xc, Intercept, b, sigma))
        return None
    def _else__8(acc):
        return acc
    _ = numpyro_cond(not prior_only, _then__6, _else__8, None)


def generated_quantities(*, N, Y, K, X, prior_only, Kc, Xc, means_X, b,
                            Intercept, sigma):
    # Transformed parameters
    
    # Generated quantities
    b_Intercept = Intercept - dot_product_vector_vector(means_X, b)
    return { 'b_Intercept': b_Intercept }

def map_generated_quantities(_samples, *, N, Y, K, X, prior_only, Kc, Xc,
                                          means_X):
    def _generated_quantities(b, Intercept, sigma):
        return generated_quantities(N=N, Y=Y, K=K, X=X,
                                    prior_only=prior_only, Kc=Kc, Xc=Xc,
                                    means_X=means_X, b=b,
                                    Intercept=Intercept, sigma=sigma)
    _f = jit(vmap(_generated_quantities))
    return _f(_samples['b'], _samples['Intercept'], _samples['sigma'])

def parameters_info(*, N, Y, K, X, prior_only, Kc, Xc, means_X):
    return { 'b': { 'shape': [Kc] },'Intercept': { 'shape': [] },
             'sigma': { 'shape': [] }, }

