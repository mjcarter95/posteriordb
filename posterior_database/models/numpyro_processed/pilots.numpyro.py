from stannumpyro.distributions import *
from stannumpyro.dppllib import lax_cond, numpyro_cond, lax_while_loop, lax_fori_loop, fori_loop, foreach_loop, lax_foreach_loop, jit, sample, param, observe, factor, array, zeros, ones, empty, matmul, true_divide, floor_divide, transpose, dtype_long, dtype_float, vmap, get_element


def convert_inputs(inputs):
    n_groups = inputs['n_groups']
    N = inputs['N']
    group_id = array(inputs['group_id'], dtype=dtype_long)
    n_scenarios = inputs['n_scenarios']
    scenario_id = array(inputs['scenario_id'], dtype=dtype_long)
    y = array(inputs['y'], dtype=dtype_float)
    return {'n_groups': n_groups, 'N': N, 'group_id': group_id,
        'n_scenarios': n_scenarios, 'scenario_id': scenario_id, 'y': y}


def model(*, n_groups, N, group_id, n_scenarios, scenario_id, y):
    a = sample('a', improper_uniform(shape=[n_groups]))
    b = sample('b', improper_uniform(shape=[n_scenarios]))
    mu_a = sample('mu_a', improper_uniform(shape=[]))
    mu_b = sample('mu_b', improper_uniform(shape=[]))
    sigma_a = sample('sigma_a', uniform(0, 100))
    sigma_b = sample('sigma_b', uniform(0, 100))
    sigma_y = sample('sigma_y', uniform(0, 100))
    y_hat = empty([N], dtype=dtype_float)

    @jit
    def _fori__1(i, _acc__2):
        y_hat = _acc__2
        y_hat = y_hat.at[i - 1].set(a[group_id[i - 1] - 1] + b[scenario_id[
            i - 1] - 1])
        return y_hat
    y_hat = lax_fori_loop(1, N + 1, _fori__1, y_hat)
    observe('_mu_a__3', normal(0, 1), mu_a)
    observe('_a__4', normal(10 * mu_a, sigma_a), a)
    observe('_mu_b__5', normal(0, 1), mu_b)
    observe('_b__6', normal(10 * mu_b, sigma_b), b)
    observe('_y__7', normal(y_hat, sigma_y), y)


def generated_quantities(*, n_groups, N, group_id, n_scenarios, scenario_id,
    y, a, b, mu_a, mu_b, sigma_a, sigma_b, sigma_y):
    y_hat = empty([N], dtype=dtype_float)

    @jit
    def _fori__8(i, _acc__9):
        y_hat = _acc__9
        y_hat = y_hat.at[i - 1].set(a[group_id[i - 1] - 1] + b[scenario_id[
            i - 1] - 1])
        return y_hat
    y_hat = lax_fori_loop(1, N + 1, _fori__8, y_hat)
    return {'y_hat': y_hat}


def map_generated_quantities(_samples, *, n_groups, N, group_id,
    n_scenarios, scenario_id, y):

    def _generated_quantities(a, b, mu_a, mu_b, sigma_a, sigma_b, sigma_y):
        return generated_quantities(n_groups=n_groups, N=N, group_id=
            group_id, n_scenarios=n_scenarios, scenario_id=scenario_id, y=y,
            a=a, b=b, mu_a=mu_a, mu_b=mu_b, sigma_a=sigma_a, sigma_b=
            sigma_b, sigma_y=sigma_y)
    _f = jit(vmap(_generated_quantities))
    return _f(_samples['a'], _samples['b'], _samples['mu_a'], _samples[
        'mu_b'], _samples['sigma_a'], _samples['sigma_b'], _samples['sigma_y'])


def parameters_info(*, n_groups, N, group_id, n_scenarios, scenario_id, y):
    return {'a': {'shape': [n_groups]}, 'b': {'shape': [n_scenarios]},
        'mu_a': {'shape': []}, 'mu_b': {'shape': []}, 'sigma_a': {'shape':
        []}, 'sigma_b': {'shape': []}, 'sigma_y': {'shape': []}}
