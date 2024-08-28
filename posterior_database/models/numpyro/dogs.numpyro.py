from stannumpyro.distributions import *
from stannumpyro.dppllib import ops_index, ops_index_update, lax_cond, numpyro_cond, lax_while_loop, lax_fori_loop, fori_loop, foreach_loop, lax_foreach_loop, jit, sample, param, observe, factor, array, zeros, ones, empty, matmul, true_divide, floor_divide, transpose, dtype_long, dtype_float, vmap, get_element

def convert_inputs(inputs):
    n_dogs = inputs['n_dogs']
    n_trials = inputs['n_trials']
    y = array(inputs['y'], dtype=dtype_long)
    return { 'n_dogs': n_dogs, 'n_trials': n_trials, 'y': y }

def model(*, n_dogs, n_trials, y):
    # Parameters
    beta__ = sample('beta', improper_uniform(shape=[3]))
    # Transformed parameters
    n_avoid = empty([n_dogs, n_trials], dtype=dtype_float)
    n_shock = empty([n_dogs, n_trials], dtype=dtype_float)
    p = empty([n_dogs, n_trials], dtype=dtype_float)
    @jit
    def _fori__1(j, _acc__2):
        (n_avoid, n_shock, p) = _acc__2
        n_avoid = ops_index_update(n_avoid, ops_index[j - 1, 1 - 1], 0)
        n_shock = ops_index_update(n_shock, ops_index[j - 1, 1 - 1], 0)
        @jit
        def _fori__3(t, _acc__4):
            (n_avoid, n_shock) = _acc__4
            n_avoid = ops_index_update(n_avoid, ops_index[j - 1, t - 1], n_avoid[
            j - 1, t - 1 - 1] + 1 - y[j - 1, t - 1 - 1])
            n_shock = ops_index_update(n_shock, ops_index[j - 1, t - 1], n_shock[
            j - 1, t - 1 - 1] + y[j - 1, t - 1 - 1])
            return (n_avoid, n_shock)
        (n_avoid, n_shock) = lax_fori_loop(2, n_trials + 1, _fori__3,
                                           (n_avoid, n_shock))
        @jit
        def _fori__5(t, _acc__6):
            p = _acc__6
            p = ops_index_update(p, ops_index[j - 1, t - 1], beta__[1 - 1] + beta__[
            2 - 1] * n_avoid[j - 1, t - 1] + beta__[3 - 1] * n_shock[
            j - 1, t - 1])
            return p
        p = lax_fori_loop(1, n_trials + 1, _fori__5, p)
        return (n_avoid, n_shock, p)
    (n_avoid, n_shock, p) = lax_fori_loop(1, n_dogs + 1, _fori__1,
                                          (n_avoid, n_shock, p))
    # Model
    observe('_beta__7', normal(0, 100), beta__)
    def _fori__8(i, _acc__9):
        def _fori__10(j, _acc__11):
            observe(f'_y__{j}__{i}__12', bernoulli_logit(p[i - 1, j - 1]), y[
            i - 1, j - 1])
            return None
        _ = fori_loop(1, n_trials + 1, _fori__10, None)
        return None
    _ = fori_loop(1, n_dogs + 1, _fori__8, None)


def generated_quantities(*, n_dogs, n_trials, y, beta__):
    # Transformed parameters
    n_avoid = empty([n_dogs, n_trials], dtype=dtype_float)
    n_shock = empty([n_dogs, n_trials], dtype=dtype_float)
    p = empty([n_dogs, n_trials], dtype=dtype_float)
    @jit
    def _fori__13(j, _acc__14):
        (n_avoid, n_shock, p) = _acc__14
        n_avoid = ops_index_update(n_avoid, ops_index[j - 1, 1 - 1], 0)
        n_shock = ops_index_update(n_shock, ops_index[j - 1, 1 - 1], 0)
        @jit
        def _fori__15(t, _acc__16):
            (n_avoid, n_shock) = _acc__16
            n_avoid = ops_index_update(n_avoid, ops_index[j - 1, t - 1], n_avoid[
            j - 1, t - 1 - 1] + 1 - y[j - 1, t - 1 - 1])
            n_shock = ops_index_update(n_shock, ops_index[j - 1, t - 1], n_shock[
            j - 1, t - 1 - 1] + y[j - 1, t - 1 - 1])
            return (n_avoid, n_shock)
        (n_avoid, n_shock) = lax_fori_loop(2, n_trials + 1, _fori__15,
                                           (n_avoid, n_shock))
        @jit
        def _fori__17(t, _acc__18):
            p = _acc__18
            p = ops_index_update(p, ops_index[j - 1, t - 1], beta__[1 - 1] + beta__[
            2 - 1] * n_avoid[j - 1, t - 1] + beta__[3 - 1] * n_shock[
            j - 1, t - 1])
            return p
        p = lax_fori_loop(1, n_trials + 1, _fori__17, p)
        return (n_avoid, n_shock, p)
    (n_avoid, n_shock, p) = lax_fori_loop(1, n_dogs + 1, _fori__13,
                                          (n_avoid, n_shock, p))
    return { 'n_avoid': n_avoid, 'n_shock': n_shock, 'p': p }

def map_generated_quantities(_samples, *, n_dogs, n_trials, y):
    def _generated_quantities(beta__):
        return generated_quantities(n_dogs=n_dogs, n_trials=n_trials, y=y,
                                    beta__=beta__)
    _f = jit(vmap(_generated_quantities))
    return _f(_samples['beta'])

def parameters_info(*, n_dogs, n_trials, y):
    return { 'beta': { 'shape': [3] }, }

