from stannumpyro.distributions import *
from stannumpyro.dppllib import lax_cond, numpyro_cond, lax_while_loop, lax_fori_loop, fori_loop, foreach_loop, lax_foreach_loop, jit, sample, param, observe, factor, array, zeros, ones, empty, matmul, true_divide, floor_divide, transpose, dtype_long, dtype_float, vmap, get_element


def convert_inputs(inputs):
    n_dogs = inputs['n_dogs']
    n_trials = inputs['n_trials']
    y = array(inputs['y'], dtype=dtype_long)
    return {'n_dogs': n_dogs, 'n_trials': n_trials, 'y': y}


def transformed_data(*, n_dogs, n_trials, y):
    J = n_dogs
    T = n_trials
    prev_shock = empty([J, T], dtype=dtype_float)
    prev_avoid = empty([J, T], dtype=dtype_float)

    @jit
    def _fori__1(j, _acc__2):
        prev_avoid, prev_shock = _acc__2
        prev_shock = prev_shock.at[j - 1, 1 - 1].set(0)
        prev_avoid = prev_avoid.at[j - 1, 1 - 1].set(0)

        @jit
        def _fori__3(t, _acc__4):
            prev_avoid, prev_shock = _acc__4
            prev_shock = prev_shock.at[j - 1, t - 1].set(prev_shock[j - 1, 
                t - 1 - 1] + y[j - 1, t - 1 - 1])
            prev_avoid = prev_avoid.at[j - 1, t - 1].set(prev_avoid[j - 1, 
                t - 1 - 1] + 1 - y[j - 1, t - 1 - 1])
            return prev_avoid, prev_shock
        prev_avoid, prev_shock = lax_fori_loop(2, T + 1, _fori__3, (
            prev_avoid, prev_shock))
        return prev_avoid, prev_shock
    prev_avoid, prev_shock = lax_fori_loop(1, J + 1, _fori__1, (prev_avoid,
        prev_shock))
    return {'J': J, 'T': T, 'prev_shock': prev_shock, 'prev_avoid': prev_avoid}


def model(*, n_dogs, n_trials, y, J, T, prev_shock, prev_avoid):
    a = sample('a', uniform(0, 1))
    b = sample('b', uniform(0, 1))

    def _fori__5(j, _acc__6):

        def _fori__7(t, _acc__8):
            p = a ** prev_shock[j - 1, t - 1] * b ** prev_avoid[j - 1, t - 1]
            observe(f'_y__{t}__{j}__9', bernoulli(p), y[j - 1, t - 1])
            return None
        _ = fori_loop(1, T + 1, _fori__7, None)
        return None
    _ = fori_loop(1, J + 1, _fori__5, None)


def generated_quantities(*, n_dogs, n_trials, y, J, T, prev_shock,
    prev_avoid, a, b):
    y_rep = empty([n_dogs, n_trials], dtype=dtype_long)
    prev_shock_rep = None
    prev_avoid_rep = None
    p_rep = None

    @jit
    def _fori__10(j, _acc__11):
        p_rep, prev_avoid_rep, prev_shock_rep, y_rep = _acc__11
        prev_shock_rep = 0
        prev_avoid_rep = 0
        y_rep = y_rep.at[j - 1, 1 - 1].set(1)

        @jit
        def _fori__12(t, _acc__13):
            p_rep, prev_avoid_rep, prev_shock_rep, y_rep = _acc__13
            prev_shock_rep = prev_shock_rep + y_rep[j - 1, t - 1 - 1]
            prev_avoid_rep = prev_avoid_rep + 1 - y_rep[j - 1, t - 1 - 1]
            p_rep = a ** prev_shock_rep * b ** prev_avoid_rep
            y_rep = y_rep.at[j - 1, t - 1].set(bernoulli_rng(p_rep))
            return p_rep, prev_avoid_rep, prev_shock_rep, y_rep
        p_rep, prev_avoid_rep, prev_shock_rep, y_rep = lax_fori_loop(2, T +
            1, _fori__12, (p_rep, prev_avoid_rep, prev_shock_rep, y_rep))
        return p_rep, prev_avoid_rep, prev_shock_rep, y_rep
    p_rep, prev_avoid_rep, prev_shock_rep, y_rep = lax_fori_loop(1, J + 1,
        _fori__10, (p_rep, prev_avoid_rep, prev_shock_rep, y_rep))
    return {'y_rep': y_rep, 'prev_shock_rep': prev_shock_rep,
        'prev_avoid_rep': prev_avoid_rep, 'p_rep': p_rep}


def map_generated_quantities(_samples, *, n_dogs, n_trials, y, J, T,
    prev_shock, prev_avoid):

    def _generated_quantities(a, b):
        return generated_quantities(n_dogs=n_dogs, n_trials=n_trials, y=y,
            J=J, T=T, prev_shock=prev_shock, prev_avoid=prev_avoid, a=a, b=b)
    _f = jit(vmap(_generated_quantities))
    return _f(_samples['a'], _samples['b'])


def parameters_info(*, n_dogs, n_trials, y, J, T, prev_shock, prev_avoid):
    return {'a': {'shape': []}, 'b': {'shape': []}}
