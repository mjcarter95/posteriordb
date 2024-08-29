from stannumpyro.distributions import *
from stannumpyro.dppllib import lax_cond, numpyro_cond, lax_while_loop, lax_fori_loop, fori_loop, foreach_loop, lax_foreach_loop, jit, sample, param, observe, factor, array, zeros, ones, empty, matmul, true_divide, floor_divide, transpose, dtype_long, dtype_float, vmap, get_element
from stannumpyro.stanlib import diag_pre_multiply_vector_matrix, inv_logit_vector, quad_form_diag_matrix_vector, rep_vector_int_int, to_vector_matrix


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
    mu_logit_ab = sample('mu_logit_ab', improper_uniform(shape=[2]))
    sigma_logit_ab = sample('sigma_logit_ab',
        lower_constrained_improper_uniform(0, shape=[2]))
    L_logit_ab = sample('L_logit_ab',
        cholesky_factor_corr_constrained_improper_uniform(shape=[2, 2]))
    z = sample('z', improper_uniform(shape=[J, 2]))
    logit_ab = matmul(rep_vector_int_int(1, J), mu_logit_ab) + matmul(z,
        diag_pre_multiply_vector_matrix(sigma_logit_ab, L_logit_ab))
    Omega_logit_ab = matmul(L_logit_ab, transpose(L_logit_ab, 0, 1))
    Sigma_logit_ab = quad_form_diag_matrix_vector(Omega_logit_ab,
        sigma_logit_ab)
    a = inv_logit_vector(logit_ab[:, 1 - 1])
    b = inv_logit_vector(logit_ab[:, 2 - 1])

    def _fori__5(j, _acc__6):

        def _fori__7(t, _acc__8):
            p = a[j - 1] ** prev_shock[j - 1, t - 1] * b[j - 1] ** prev_avoid[
                j - 1, t - 1]
            observe(f'_y__{t}__{j}__9', bernoulli(p), y[j - 1, t - 1])
            return None
        _ = fori_loop(1, T + 1, _fori__7, None)
        return None
    _ = fori_loop(1, J + 1, _fori__5, None)
    observe('_mu_logit_ab__10', logistic(0, 1), mu_logit_ab)
    observe('_sigma_logit_ab__11', normal(0, 1), sigma_logit_ab)
    observe('_L_logit_ab__12', lkj_corr_cholesky(2), L_logit_ab)
    observe('_expr__13', normal(0, 1), to_vector_matrix(z))


def generated_quantities(*, n_dogs, n_trials, y, J, T, prev_shock,
    prev_avoid, mu_logit_ab, sigma_logit_ab, L_logit_ab, z):
    logit_ab = matmul(rep_vector_int_int(1, J), mu_logit_ab) + matmul(z,
        diag_pre_multiply_vector_matrix(sigma_logit_ab, L_logit_ab))
    Omega_logit_ab = matmul(L_logit_ab, transpose(L_logit_ab, 0, 1))
    Sigma_logit_ab = quad_form_diag_matrix_vector(Omega_logit_ab,
        sigma_logit_ab)
    a = inv_logit_vector(logit_ab[:, 1 - 1])
    b = inv_logit_vector(logit_ab[:, 2 - 1])
    y_rep = empty([J, T], dtype=dtype_long)
    prev_shock_rep = None
    prev_avoid_rep = None
    p_rep = None

    @jit
    def _fori__14(j, _acc__15):
        p_rep, prev_avoid_rep, prev_shock_rep, y_rep = _acc__15
        prev_shock_rep = 0
        prev_avoid_rep = 0
        y_rep = y_rep.at[j - 1, 1 - 1].set(1)

        @jit
        def _fori__16(t, _acc__17):
            p_rep, prev_avoid_rep, prev_shock_rep, y_rep = _acc__17
            prev_shock_rep = prev_shock_rep + y_rep[j - 1, t - 1 - 1]
            prev_avoid_rep = prev_avoid_rep + 1 - y_rep[j - 1, t - 1 - 1]
            p_rep = a[j - 1] ** prev_shock_rep * b[j - 1] ** prev_avoid_rep
            y_rep = y_rep.at[j - 1, t - 1].set(bernoulli_rng(p_rep))
            return p_rep, prev_avoid_rep, prev_shock_rep, y_rep
        p_rep, prev_avoid_rep, prev_shock_rep, y_rep = lax_fori_loop(2, T +
            1, _fori__16, (p_rep, prev_avoid_rep, prev_shock_rep, y_rep))
        return p_rep, prev_avoid_rep, prev_shock_rep, y_rep
    p_rep, prev_avoid_rep, prev_shock_rep, y_rep = lax_fori_loop(1, J + 1,
        _fori__14, (p_rep, prev_avoid_rep, prev_shock_rep, y_rep))
    return {'logit_ab': logit_ab, 'Omega_logit_ab': Omega_logit_ab,
        'Sigma_logit_ab': Sigma_logit_ab, 'a': a, 'b': b, 'y_rep': y_rep,
        'prev_shock_rep': prev_shock_rep, 'prev_avoid_rep': prev_avoid_rep,
        'p_rep': p_rep}


def map_generated_quantities(_samples, *, n_dogs, n_trials, y, J, T,
    prev_shock, prev_avoid):

    def _generated_quantities(mu_logit_ab, sigma_logit_ab, L_logit_ab, z):
        return generated_quantities(n_dogs=n_dogs, n_trials=n_trials, y=y,
            J=J, T=T, prev_shock=prev_shock, prev_avoid=prev_avoid,
            mu_logit_ab=mu_logit_ab, sigma_logit_ab=sigma_logit_ab,
            L_logit_ab=L_logit_ab, z=z)
    _f = jit(vmap(_generated_quantities))
    return _f(_samples['mu_logit_ab'], _samples['sigma_logit_ab'], _samples
        ['L_logit_ab'], _samples['z'])


def parameters_info(*, n_dogs, n_trials, y, J, T, prev_shock, prev_avoid):
    return {'mu_logit_ab': {'shape': [2]}, 'sigma_logit_ab': {'shape': [2]},
        'L_logit_ab': {'shape': [2, 2]}, 'z': {'shape': [J, 2]}}
