from stannumpyro.distributions import *
from stannumpyro.dppllib import ops_index, ops_index_update, lax_cond, numpyro_cond, lax_while_loop, lax_fori_loop, fori_loop, foreach_loop, lax_foreach_loop, jit, sample, param, observe, factor, array, zeros, ones, empty, matmul, true_divide, floor_divide, transpose, dtype_long, dtype_float, vmap, get_element

def convert_inputs(inputs):
    n_age = inputs['n_age']
    N = inputs['N']
    age = array(inputs['age'], dtype=dtype_long)
    n_age_edu = inputs['n_age_edu']
    age_edu = array(inputs['age_edu'], dtype=dtype_long)
    black = array(inputs['black'], dtype=dtype_float)
    n_edu = inputs['n_edu']
    edu = array(inputs['edu'], dtype=dtype_long)
    female = array(inputs['female'], dtype=dtype_float)
    n_region_full = inputs['n_region_full']
    region_full = array(inputs['region_full'], dtype=dtype_long)
    n_state = inputs['n_state']
    state = array(inputs['state'], dtype=dtype_long)
    v_prev_full = array(inputs['v_prev_full'], dtype=dtype_float)
    y = array(inputs['y'], dtype=dtype_long)
    return { 'n_age': n_age, 'N': N, 'age': age, 'n_age_edu': n_age_edu,
             'age_edu': age_edu, 'black': black, 'n_edu': n_edu, 'edu': edu,
             'female': female, 'n_region_full': n_region_full,
             'region_full': region_full, 'n_state': n_state, 'state': state,
             'v_prev_full': v_prev_full, 'y': y }

def model(*, n_age, N, age, n_age_edu, age_edu, black, n_edu, edu, female,
             n_region_full, region_full, n_state, state, v_prev_full, y):
    # Parameters
    a = sample('a', improper_uniform(shape=[n_age]))
    b = sample('b', improper_uniform(shape=[n_edu]))
    c = sample('c', improper_uniform(shape=[n_age_edu]))
    d = sample('d', improper_uniform(shape=[n_state]))
    e__ = sample('e', improper_uniform(shape=[n_region_full]))
    beta__ = sample('beta', improper_uniform(shape=[5]))
    sigma_a = sample('sigma_a', uniform(0, 100))
    sigma_b = sample('sigma_b', uniform(0, 100))
    sigma_c = sample('sigma_c', uniform(0, 100))
    sigma_d = sample('sigma_d', uniform(0, 100))
    sigma_e = sample('sigma_e', uniform(0, 100))
    # Transformed parameters
    y_hat = empty([N], dtype=dtype_float)
    @jit
    def _fori__1(i, _acc__2):
        y_hat = _acc__2
        y_hat = ops_index_update(y_hat, ops_index[i - 1], beta__[1 - 1] + beta__[
        2 - 1] * black[i - 1] + beta__[3 - 1] * female[i - 1] + beta__[
        5 - 1] * female[i - 1] * black[i - 1] + beta__[4 - 1] * v_prev_full[
        i - 1] + a[age[i - 1] - 1] + b[edu[i - 1] - 1] + c[age_edu[i - 1] - 1] + d[
        state[i - 1] - 1] + e__[region_full[i - 1] - 1])
        return y_hat
    y_hat = lax_fori_loop(1, N + 1, _fori__1, y_hat)
    # Model
    observe('_a__3', normal(0, sigma_a), a)
    observe('_b__4', normal(0, sigma_b), b)
    observe('_c__5', normal(0, sigma_c), c)
    observe('_d__6', normal(0, sigma_d), d)
    observe('_e__7', normal(0, sigma_e), e__)
    observe('_beta__8', normal(0, 100), beta__)
    observe('_y__9', bernoulli_logit(y_hat), y)


def generated_quantities(*, n_age, N, age, n_age_edu, age_edu, black, n_edu,
                            edu, female, n_region_full, region_full, n_state,
                            state, v_prev_full, y, a, b, c, d, e__, beta__,
                            sigma_a, sigma_b, sigma_c, sigma_d, sigma_e):
    # Transformed parameters
    y_hat = empty([N], dtype=dtype_float)
    @jit
    def _fori__10(i, _acc__11):
        y_hat = _acc__11
        y_hat = ops_index_update(y_hat, ops_index[i - 1], beta__[1 - 1] + beta__[
        2 - 1] * black[i - 1] + beta__[3 - 1] * female[i - 1] + beta__[
        5 - 1] * female[i - 1] * black[i - 1] + beta__[4 - 1] * v_prev_full[
        i - 1] + a[age[i - 1] - 1] + b[edu[i - 1] - 1] + c[age_edu[i - 1] - 1] + d[
        state[i - 1] - 1] + e__[region_full[i - 1] - 1])
        return y_hat
    y_hat = lax_fori_loop(1, N + 1, _fori__10, y_hat)
    return { 'y_hat': y_hat }

def map_generated_quantities(_samples, *, n_age, N, age, n_age_edu, age_edu,
                                          black, n_edu, edu, female,
                                          n_region_full, region_full,
                                          n_state, state, v_prev_full, y):
    def _generated_quantities(a, b, c, d, e__, beta__, sigma_a, sigma_b,
                              sigma_c, sigma_d, sigma_e):
        return generated_quantities(n_age=n_age, N=N, age=age,
                                    n_age_edu=n_age_edu, age_edu=age_edu,
                                    black=black, n_edu=n_edu, edu=edu,
                                    female=female,
                                    n_region_full=n_region_full,
                                    region_full=region_full, n_state=n_state,
                                    state=state, v_prev_full=v_prev_full,
                                    y=y, a=a, b=b, c=c, d=d, e__=e__,
                                    beta__=beta__, sigma_a=sigma_a,
                                    sigma_b=sigma_b, sigma_c=sigma_c,
                                    sigma_d=sigma_d, sigma_e=sigma_e)
    _f = jit(vmap(_generated_quantities))
    return _f(_samples['a'], _samples['b'], _samples['c'], _samples['d'],
              _samples['e'], _samples['beta'], _samples['sigma_a'],
              _samples['sigma_b'], _samples['sigma_c'], _samples['sigma_d'],
              _samples['sigma_e'])

def parameters_info(*, n_age, N, age, n_age_edu, age_edu, black, n_edu, edu,
                       female, n_region_full, region_full, n_state, state,
                       v_prev_full, y):
    return { 'a': { 'shape': [n_age] },'b': { 'shape': [n_edu] },
             'c': { 'shape': [n_age_edu] },'d': { 'shape': [n_state] },
             'e': { 'shape': [n_region_full] },'beta': { 'shape': [5] },
             'sigma_a': { 'shape': [] },'sigma_b': { 'shape': [] },
             'sigma_c': { 'shape': [] },'sigma_d': { 'shape': [] },
             'sigma_e': { 'shape': [] }, }

