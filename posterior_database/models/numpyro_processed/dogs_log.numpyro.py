from stannumpyro.distributions import *
from stannumpyro.dppllib import lax_cond, numpyro_cond, lax_while_loop, lax_fori_loop, fori_loop, foreach_loop, lax_foreach_loop, jit, sample, param, observe, factor, array, zeros, ones, empty, matmul, true_divide, floor_divide, transpose, dtype_long, dtype_float, vmap, get_element
from stannumpyro.stanlib import inv_logit_real


def convert_inputs(inputs):
    n_trials = inputs['n_trials']
    n_dogs = inputs['n_dogs']
    y = array(inputs['y'], dtype=dtype_long)
    return {'n_trials': n_trials, 'n_dogs': n_dogs, 'y': y}


def model(*, n_trials, n_dogs, y):
    beta__ = sample('beta', improper_uniform(shape=[2]))
    n_avoid = empty([n_dogs, n_trials], dtype=dtype_float)
    n_shock = empty([n_dogs, n_trials], dtype=dtype_float)
    p = empty([n_dogs, n_trials], dtype=dtype_float)

    @jit
    def _fori__1(j, _acc__2):
        n_avoid, n_shock, p = _acc__2
        n_avoid = n_avoid.at[j - 1, 1 - 1].set(0)
        n_shock = n_shock.at[j - 1, 1 - 1].set(0)

        @jit
        def _fori__3(t, _acc__4):
            n_avoid, n_shock = _acc__4
            n_avoid = n_avoid.at[j - 1, t - 1].set(n_avoid[j - 1, t - 1 - 1
                ] + 1 - y[j - 1, t - 1 - 1])
            n_shock = n_shock.at[j - 1, t - 1].set(n_shock[j - 1, t - 1 - 1
                ] + y[j - 1, t - 1 - 1])
            return n_avoid, n_shock
        n_avoid, n_shock = lax_fori_loop(2, n_trials + 1, _fori__3, (
            n_avoid, n_shock))

        @jit
        def _fori__5(t, _acc__6):
            p = _acc__6
            p = p.at[j - 1, t - 1].set(inv_logit_real(beta__[1 - 1] *
                n_avoid[j - 1, t - 1] + beta__[2 - 1] * n_shock[j - 1, t - 1]))
            return p
        p = lax_fori_loop(1, n_trials + 1, _fori__5, p)
        return n_avoid, n_shock, p
    n_avoid, n_shock, p = lax_fori_loop(1, n_dogs + 1, _fori__1, (n_avoid,
        n_shock, p))
    observe('_beta__7', uniform(-100, 0), beta__[1 - 1])
    observe('_beta__8', uniform(0, 100), beta__[2 - 1])

    def _fori__9(i, _acc__10):

        def _fori__11(j, _acc__12):
            observe(f'_y__{j}__{i}__13', bernoulli(p[i - 1, j - 1]), y[i - 
                1, j - 1])
            return None
        _ = fori_loop(1, n_trials + 1, _fori__11, None)
        return None
    _ = fori_loop(1, n_dogs + 1, _fori__9, None)


def generated_quantities(*, n_trials, n_dogs, y, beta__):
    n_avoid = empty([n_dogs, n_trials], dtype=dtype_float)
    n_shock = empty([n_dogs, n_trials], dtype=dtype_float)
    p = empty([n_dogs, n_trials], dtype=dtype_float)

    @jit
    def _fori__14(j, _acc__15):
        n_avoid, n_shock, p = _acc__15
        n_avoid = n_avoid.at[j - 1, 1 - 1].set(0)
        n_shock = n_shock.at[j - 1, 1 - 1].set(0)

        @jit
        def _fori__16(t, _acc__17):
            n_avoid, n_shock = _acc__17
            n_avoid = n_avoid.at[j - 1, t - 1].set(n_avoid[j - 1, t - 1 - 1
                ] + 1 - y[j - 1, t - 1 - 1])
            n_shock = n_shock.at[j - 1, t - 1].set(n_shock[j - 1, t - 1 - 1
                ] + y[j - 1, t - 1 - 1])
            return n_avoid, n_shock
        n_avoid, n_shock = lax_fori_loop(2, n_trials + 1, _fori__16, (
            n_avoid, n_shock))

        @jit
        def _fori__18(t, _acc__19):
            p = _acc__19
            p = p.at[j - 1, t - 1].set(inv_logit_real(beta__[1 - 1] *
                n_avoid[j - 1, t - 1] + beta__[2 - 1] * n_shock[j - 1, t - 1]))
            return p
        p = lax_fori_loop(1, n_trials + 1, _fori__18, p)
        return n_avoid, n_shock, p
    n_avoid, n_shock, p = lax_fori_loop(1, n_dogs + 1, _fori__14, (n_avoid,
        n_shock, p))
    return {'n_avoid': n_avoid, 'n_shock': n_shock, 'p': p}


def map_generated_quantities(_samples, *, n_trials, n_dogs, y):

    def _generated_quantities(beta__):
        return generated_quantities(n_trials=n_trials, n_dogs=n_dogs, y=y,
            beta__=beta__)
    _f = jit(vmap(_generated_quantities))
    return _f(_samples['beta'])


def parameters_info(*, n_trials, n_dogs, y):
    return {'beta': {'shape': [2]}}
