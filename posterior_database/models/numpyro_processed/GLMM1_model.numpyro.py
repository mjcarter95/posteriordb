from stannumpyro.distributions import *
from stannumpyro.dppllib import lax_cond, numpyro_cond, lax_while_loop, lax_fori_loop, fori_loop, foreach_loop, lax_foreach_loop, jit, sample, param, observe, factor, array, zeros, ones, empty, matmul, true_divide, floor_divide, transpose, dtype_long, dtype_float, vmap, get_element
from stannumpyro.stanlib import rep_matrix_rowvector_int


def convert_inputs(inputs):
    nyear = inputs['nyear']
    nsite = inputs['nsite']
    nobs = inputs['nobs']
    obs = array(inputs['obs'], dtype=dtype_long)
    obsyear = array(inputs['obsyear'], dtype=dtype_long)
    obssite = array(inputs['obssite'], dtype=dtype_long)
    nmis = inputs['nmis']
    misyear = array(inputs['misyear'], dtype=dtype_long)
    missite = array(inputs['missite'], dtype=dtype_long)
    return {'nyear': nyear, 'nsite': nsite, 'nobs': nobs, 'obs': obs,
        'obsyear': obsyear, 'obssite': obssite, 'nmis': nmis, 'misyear':
        misyear, 'missite': missite}


def model(*, nyear, nsite, nobs, obs, obsyear, obssite, nmis, misyear, missite
    ):
    alpha = sample('alpha', improper_uniform(shape=[nsite]))
    mu_alpha = sample('mu_alpha', improper_uniform(shape=[]))
    sd_alpha = sample('sd_alpha', uniform(0, 5))
    log_lambda = rep_matrix_rowvector_int(alpha, nyear)
    observe('_alpha__1', normal(mu_alpha, sd_alpha), alpha)
    observe('_mu_alpha__2', normal(0, 10), mu_alpha)

    def _fori__3(i, _acc__4):
        observe(f'_obs__{i}__5', poisson_log(log_lambda[obsyear[i - 1] - 1,
            obssite[i - 1] - 1]), obs[i - 1])
        return None
    _ = fori_loop(1, nobs + 1, _fori__3, None)


def generated_quantities(*, nyear, nsite, nobs, obs, obsyear, obssite, nmis,
    misyear, missite, alpha, mu_alpha, sd_alpha):
    log_lambda = rep_matrix_rowvector_int(alpha, nyear)
    mis = empty([nmis], dtype=dtype_long)

    @jit
    def _fori__6(i, _acc__7):
        mis = _acc__7
        mis = mis.at[i - 1].set(poisson_log_rng(log_lambda[misyear[i - 1] -
            1, missite[i - 1] - 1]))
        return mis
    mis = lax_fori_loop(1, nmis + 1, _fori__6, mis)
    return {'log_lambda': log_lambda, 'mis': mis}


def map_generated_quantities(_samples, *, nyear, nsite, nobs, obs, obsyear,
    obssite, nmis, misyear, missite):

    def _generated_quantities(alpha, mu_alpha, sd_alpha):
        return generated_quantities(nyear=nyear, nsite=nsite, nobs=nobs,
            obs=obs, obsyear=obsyear, obssite=obssite, nmis=nmis, misyear=
            misyear, missite=missite, alpha=alpha, mu_alpha=mu_alpha,
            sd_alpha=sd_alpha)
    _f = jit(vmap(_generated_quantities))
    return _f(_samples['alpha'], _samples['mu_alpha'], _samples['sd_alpha'])


def parameters_info(*, nyear, nsite, nobs, obs, obsyear, obssite, nmis,
    misyear, missite):
    return {'alpha': {'shape': [nsite]}, 'mu_alpha': {'shape': []},
        'sd_alpha': {'shape': []}}
