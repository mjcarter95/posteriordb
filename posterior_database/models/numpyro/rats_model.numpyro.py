from stannumpyro.distributions import *
from stannumpyro.dppllib import ops_index, ops_index_update, lax_cond, numpyro_cond, lax_while_loop, lax_fori_loop, fori_loop, foreach_loop, lax_foreach_loop, jit, sample, param, observe, factor, array, zeros, ones, empty, matmul, true_divide, floor_divide, transpose, dtype_long, dtype_float, vmap, get_element

def convert_inputs(inputs):
    N = inputs['N']
    Npts = inputs['Npts']
    rat = array(inputs['rat'], dtype=dtype_long)
    x = array(inputs['x'], dtype=dtype_float)
    y = array(inputs['y'], dtype=dtype_float)
    xbar = array(inputs['xbar'], dtype=dtype_float)
    return { 'N': N, 'Npts': Npts, 'rat': rat, 'x': x, 'y': y, 'xbar': xbar }

def model(*, N, Npts, rat, x, y, xbar):
    # Parameters
    alpha = sample('alpha', improper_uniform(shape=[N]))
    beta__ = sample('beta', improper_uniform(shape=[N]))
    mu_alpha = sample('mu_alpha', improper_uniform(shape=[]))
    mu_beta = sample('mu_beta', improper_uniform(shape=[]))
    sigma_y = sample('sigma_y', lower_constrained_improper_uniform(0, shape=[]))
    sigma_alpha = sample('sigma_alpha', lower_constrained_improper_uniform(0, shape=[]))
    sigma_beta = sample('sigma_beta', lower_constrained_improper_uniform(0, shape=[]))
    # Model
    observe('_mu_alpha__1', normal(0, 100), mu_alpha)
    observe('_mu_beta__2', normal(0, 100), mu_beta)
    observe('_alpha__3', normal(mu_alpha, sigma_alpha), alpha)
    observe('_beta__4', normal(mu_beta, sigma_beta), beta__)
    def _fori__5(n, _acc__6):
        irat = rat[n - 1]
        observe(f'_y__{n}__7', normal(alpha[irat - 1] + beta__[irat - 1] * (x[
                                      n - 1] - xbar), sigma_y), y[n - 1])
        return None
    _ = fori_loop(1, Npts + 1, _fori__5, None)


def generated_quantities(*, N, Npts, rat, x, y, xbar, alpha, beta__,
                            mu_alpha, mu_beta, sigma_y, sigma_alpha,
                            sigma_beta):
    # Generated quantities
    alpha0 = mu_alpha - xbar * mu_beta
    return { 'alpha0': alpha0 }

def map_generated_quantities(_samples, *, N, Npts, rat, x, y, xbar):
    def _generated_quantities(alpha, beta__, mu_alpha, mu_beta, sigma_y,
                              sigma_alpha, sigma_beta):
        return generated_quantities(N=N, Npts=Npts, rat=rat, x=x, y=y,
                                    xbar=xbar, alpha=alpha, beta__=beta__,
                                    mu_alpha=mu_alpha, mu_beta=mu_beta,
                                    sigma_y=sigma_y, sigma_alpha=sigma_alpha,
                                    sigma_beta=sigma_beta)
    _f = jit(vmap(_generated_quantities))
    return _f(_samples['alpha'], _samples['beta'], _samples['mu_alpha'],
              _samples['mu_beta'], _samples['sigma_y'],
              _samples['sigma_alpha'], _samples['sigma_beta'])

def parameters_info(*, N, Npts, rat, x, y, xbar):
    return { 'alpha': { 'shape': [N] },'beta': { 'shape': [N] },
             'mu_alpha': { 'shape': [] },'mu_beta': { 'shape': [] },
             'sigma_y': { 'shape': [] },'sigma_alpha': { 'shape': [] },
             'sigma_beta': { 'shape': [] }, }

