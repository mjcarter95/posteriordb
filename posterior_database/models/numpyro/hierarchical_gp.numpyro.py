from stannumpyro.distributions import *
from stannumpyro.dppllib import ops_index, ops_index_update, lax_cond, numpyro_cond, lax_while_loop, lax_fori_loop, fori_loop, foreach_loop, lax_foreach_loop, jit, sample, param, observe, factor, array, zeros, ones, empty, matmul, true_divide, floor_divide, transpose, dtype_long, dtype_float, vmap, get_element
from stannumpyro.stanlib import cholesky_decompose_matrix, sqrt_real, to_vector_matrix

def convert_inputs(inputs):
    N_regions = inputs['N_regions']
    N_years_obs = inputs['N_years_obs']
    N_years = inputs['N_years']
    N_states = inputs['N_states']
    state_region_ind = array(inputs['state_region_ind'], dtype=dtype_long)
    N = inputs['N']
    state_ind = array(inputs['state_ind'], dtype=dtype_long)
    region_ind = array(inputs['region_ind'], dtype=dtype_long)
    year_ind = array(inputs['year_ind'], dtype=dtype_long)
    y = array(inputs['y'], dtype=dtype_float)
    return { 'N_regions': N_regions, 'N_years_obs': N_years_obs,
             'N_years': N_years, 'N_states': N_states,
             'state_region_ind': state_region_ind, 'N': N,
             'state_ind': state_ind, 'region_ind': region_ind,
             'year_ind': year_ind, 'y': y }

def transformed_data(*, N_regions, N_years_obs, N_years, N_states,
                        state_region_ind, N, state_ind, region_ind, year_ind,
                        y):
    # Transformed data
    years = empty([N_years], dtype=dtype_float)
    @jit
    def _fori__1(t, _acc__2):
        years = _acc__2
        years = ops_index_update(years, ops_index[t - 1], t)
        return years
    years = lax_fori_loop(1, N_years + 1, _fori__1, years)
    counts = empty([17], dtype=dtype_float)
    @jit
    def _fori__3(i, _acc__4):
        counts = _acc__4
        counts = ops_index_update(counts, ops_index[i - 1], 2)
        return counts
    counts = lax_fori_loop(1, 17 + 1, _fori__3, counts)
    return { 'years': years, 'counts': counts }

def model(*, N_regions, N_years_obs, N_years, N_states, state_region_ind, N,
             state_ind, region_ind, year_ind, y, years, counts):
    # Parameters
    GP_region_std = sample('GP_region_std', improper_uniform(shape=[N_years,
                                                                    N_regions]))
    GP_state_std = sample('GP_state_std', improper_uniform(shape=[N_years,
                                                                  N_states]))
    year_std = sample('year_std', improper_uniform(shape=[N_years_obs]))
    state_std = sample('state_std', improper_uniform(shape=[N_states]))
    region_std = sample('region_std', improper_uniform(shape=[N_regions]))
    tot_var = sample('tot_var', lower_constrained_improper_uniform(0, shape=[]))
    prop_var = sample('prop_var', simplex_constrained_improper_uniform(shape=[
    17]))
    mu = sample('mu', improper_uniform(shape=[]))
    length_GP_region_long = sample('length_GP_region_long', lower_constrained_improper_uniform(0, shape=[]))
    length_GP_state_long = sample('length_GP_state_long', lower_constrained_improper_uniform(0, shape=[]))
    length_GP_region_short = sample('length_GP_region_short', lower_constrained_improper_uniform(0, shape=[]))
    length_GP_state_short = sample('length_GP_state_short', lower_constrained_improper_uniform(0, shape=[]))
    # Transformed parameters
    vars = 17 * prop_var * tot_var
    sigma_year = sqrt_real(vars[1 - 1])
    sigma_region = sqrt_real(vars[2 - 1])
    sigma_state = empty([10], dtype=dtype_float)
    @jit
    def _fori__5(i, _acc__6):
        sigma_state = _acc__6
        sigma_state = ops_index_update(sigma_state, ops_index[i - 1], sqrt_real(
        vars[i + 2 - 1]))
        return sigma_state
    sigma_state = lax_fori_loop(1, 10 + 1, _fori__5, sigma_state)
    sigma_GP_region_long = sqrt_real(vars[13 - 1])
    sigma_GP_state_long = sqrt_real(vars[14 - 1])
    sigma_GP_region_short = sqrt_real(vars[15 - 1])
    sigma_GP_state_short = sqrt_real(vars[16 - 1])
    sigma_error_state_2 = sqrt_real(vars[17 - 1])
    region_re = sigma_region * region_std
    year_re = sigma_year * year_std
    state_re = sigma_state[state_region_ind - 1] * state_std
    cov_region = gp_exp_quad_cov_array_real_real(years, sigma_GP_region_long,
                                                 length_GP_region_long) + gp_exp_quad_cov_array_real_real(
    years, sigma_GP_region_short, length_GP_region_short)
    cov_state = gp_exp_quad_cov_array_real_real(years, sigma_GP_state_long,
                                                length_GP_state_long) + gp_exp_quad_cov_array_real_real(
    years, sigma_GP_state_short, length_GP_state_short)
    @jit
    def _fori__7(year, _acc__8):
        (cov_region, cov_state) = _acc__8
        cov_region = ops_index_update(cov_region, ops_index[year - 1,
                                                            year - 1], cov_region[
        year - 1, year - 1] + array(1e-6, dtype=dtype_float))
        cov_state = ops_index_update(cov_state, ops_index[year - 1, year - 1], cov_state[
        year - 1, year - 1] + array(1e-6, dtype=dtype_float))
        return (cov_region, cov_state)
    (cov_region, cov_state) = lax_fori_loop(1, N_years + 1, _fori__7,
                                            (cov_region, cov_state))
    L_cov_region = cholesky_decompose_matrix(cov_region)
    L_cov_state = cholesky_decompose_matrix(cov_state)
    GP_region = matmul(L_cov_region, GP_region_std)
    GP_state = matmul(L_cov_state, GP_state_std)
    # Model
    obs_mu = empty([N], dtype=dtype_float)
    @jit
    def _fori__9(n, _acc__10):
        obs_mu = _acc__10
        obs_mu = ops_index_update(obs_mu, ops_index[n - 1], mu + year_re[
        year_ind[n - 1] - 1] + state_re[state_ind[n - 1] - 1] + region_re[
        region_ind[n - 1] - 1] + GP_region[year_ind[n - 1] - 1,
                                           region_ind[n - 1] - 1] + GP_state[
        year_ind[n - 1] - 1, state_ind[n - 1] - 1])
        return obs_mu
    obs_mu = lax_fori_loop(1, N + 1, _fori__9, obs_mu)
    observe('_y__11', normal(obs_mu, sigma_error_state_2), y)
    observe('_expr__12', normal(0, 1), to_vector_matrix(GP_region_std))
    observe('_expr__13', normal(0, 1), to_vector_matrix(GP_state_std))
    observe('_year_std__14', normal(0, 1), year_std)
    observe('_state_std__15', normal(0, 1), state_std)
    observe('_region_std__16', normal(0, 1), region_std)
    observe('_mu__17', normal(array(.5, dtype=dtype_float),
                              array(.5, dtype=dtype_float)), mu)
    observe('_tot_var__18', gamma(3, 3), tot_var)
    observe('_prop_var__19', dirichlet(counts), prop_var)
    observe('_length_GP_region_long__20', weibull(30, 8), length_GP_region_long)
    observe('_length_GP_state_long__21', weibull(30, 8), length_GP_state_long)
    observe('_length_GP_region_short__22', weibull(30, 3), length_GP_region_short)
    observe('_length_GP_state_short__23', weibull(30, 3), length_GP_state_short)



def generated_quantities(*, N_regions, N_years_obs, N_years, N_states,
                            state_region_ind, N, state_ind, region_ind,
                            year_ind, y, years, counts, GP_region_std,
                            GP_state_std, year_std, state_std, region_std,
                            tot_var, prop_var, mu, length_GP_region_long,
                            length_GP_state_long, length_GP_region_short,
                            length_GP_state_short):
    # Transformed parameters
    vars = 17 * prop_var * tot_var
    sigma_year = sqrt_real(vars[1 - 1])
    sigma_region = sqrt_real(vars[2 - 1])
    sigma_state = empty([10], dtype=dtype_float)
    @jit
    def _fori__24(i, _acc__25):
        sigma_state = _acc__25
        sigma_state = ops_index_update(sigma_state, ops_index[i - 1], sqrt_real(
        vars[i + 2 - 1]))
        return sigma_state
    sigma_state = lax_fori_loop(1, 10 + 1, _fori__24, sigma_state)
    sigma_GP_region_long = sqrt_real(vars[13 - 1])
    sigma_GP_state_long = sqrt_real(vars[14 - 1])
    sigma_GP_region_short = sqrt_real(vars[15 - 1])
    sigma_GP_state_short = sqrt_real(vars[16 - 1])
    sigma_error_state_2 = sqrt_real(vars[17 - 1])
    region_re = sigma_region * region_std
    year_re = sigma_year * year_std
    state_re = sigma_state[state_region_ind - 1] * state_std
    cov_region = gp_exp_quad_cov_array_real_real(years, sigma_GP_region_long,
                                                 length_GP_region_long) + gp_exp_quad_cov_array_real_real(
    years, sigma_GP_region_short, length_GP_region_short)
    cov_state = gp_exp_quad_cov_array_real_real(years, sigma_GP_state_long,
                                                length_GP_state_long) + gp_exp_quad_cov_array_real_real(
    years, sigma_GP_state_short, length_GP_state_short)
    @jit
    def _fori__26(year, _acc__27):
        (cov_region, cov_state) = _acc__27
        cov_region = ops_index_update(cov_region, ops_index[year - 1,
                                                            year - 1], cov_region[
        year - 1, year - 1] + array(1e-6, dtype=dtype_float))
        cov_state = ops_index_update(cov_state, ops_index[year - 1, year - 1], cov_state[
        year - 1, year - 1] + array(1e-6, dtype=dtype_float))
        return (cov_region, cov_state)
    (cov_region, cov_state) = lax_fori_loop(1, N_years + 1, _fori__26,
                                            (cov_region, cov_state))
    L_cov_region = cholesky_decompose_matrix(cov_region)
    L_cov_state = cholesky_decompose_matrix(cov_state)
    GP_region = matmul(L_cov_region, GP_region_std)
    GP_state = matmul(L_cov_state, GP_state_std)
    # Generated quantities
    level = normal_rng(array(0.5, dtype=dtype_float), sigma_year)
    y_new = empty([N_years, N_states], dtype=dtype_float)
    y_new_pred = empty([N_years, N_states], dtype=dtype_float)
    @jit
    def _fori__28(state, _acc__29):
        (y_new, y_new_pred) = _acc__29
        @jit
        def _fori__30(t, _acc__31):
            (y_new, y_new_pred) = _acc__31
            @jit
            def _then__32(_acc__33):
                y_new = _acc__33
                y_new = ops_index_update(y_new, ops_index[t - 1, state - 1], state_re[
                state - 1] + region_re[state_region_ind[state - 1] - 1] + GP_state[
                t - 1, state - 1] + GP_region[t - 1,
                                              state_region_ind[state - 1] - 1] + (mu + year_re[
                t - 1]))
                return y_new
            @jit
            def _else__34(_acc__35):
                y_new = _acc__35
                y_new = ops_index_update(y_new, ops_index[t - 1, state - 1], state_re[
                state - 1] + region_re[state_region_ind[state - 1] - 1] + GP_state[
                t - 1, state - 1] + GP_region[t - 1,
                                              state_region_ind[state - 1] - 1] + level)
                return y_new
            y_new = lax_cond(t < 12, _then__32, _else__34, y_new)
            y_new_pred = ops_index_update(y_new_pred, ops_index[t - 1,
                                                                state - 1], normal_rng(
            y_new[t - 1, state - 1], sigma_error_state_2))
            return (y_new, y_new_pred)
        (y_new, y_new_pred) = lax_fori_loop(1, N_years + 1, _fori__30,
                                            (y_new, y_new_pred))
        return (y_new, y_new_pred)
    (y_new, y_new_pred) = lax_fori_loop(1, N_states + 1, _fori__28,
                                        (y_new, y_new_pred))
    return { 'vars': vars, 'sigma_year': sigma_year,
             'sigma_region': sigma_region, 'sigma_state': sigma_state,
             'sigma_GP_region_long': sigma_GP_region_long,
             'sigma_GP_state_long': sigma_GP_state_long,
             'sigma_GP_region_short': sigma_GP_region_short,
             'sigma_GP_state_short': sigma_GP_state_short,
             'sigma_error_state_2': sigma_error_state_2,
             'region_re': region_re, 'year_re': year_re,
             'state_re': state_re, 'cov_region': cov_region,
             'cov_state': cov_state, 'L_cov_region': L_cov_region,
             'L_cov_state': L_cov_state, 'GP_region': GP_region,
             'GP_state': GP_state, 'level': level, 'y_new': y_new,
             'y_new_pred': y_new_pred }

def map_generated_quantities(_samples, *, N_regions, N_years_obs, N_years,
                                          N_states, state_region_ind, N,
                                          state_ind, region_ind, year_ind, y,
                                          years, counts):
    def _generated_quantities(GP_region_std, GP_state_std, year_std,
                              state_std, region_std, tot_var, prop_var, mu,
                              length_GP_region_long, length_GP_state_long,
                              length_GP_region_short, length_GP_state_short):
        return generated_quantities(N_regions=N_regions,
                                    N_years_obs=N_years_obs, N_years=N_years,
                                    N_states=N_states,
                                    state_region_ind=state_region_ind, N=N,
                                    state_ind=state_ind,
                                    region_ind=region_ind, year_ind=year_ind,
                                    y=y, years=years, counts=counts,
                                    GP_region_std=GP_region_std,
                                    GP_state_std=GP_state_std,
                                    year_std=year_std, state_std=state_std,
                                    region_std=region_std, tot_var=tot_var,
                                    prop_var=prop_var, mu=mu,
                                    length_GP_region_long=length_GP_region_long,
                                    length_GP_state_long=length_GP_state_long,
                                    length_GP_region_short=length_GP_region_short,
                                    length_GP_state_short=length_GP_state_short)
    _f = jit(vmap(_generated_quantities))
    return _f(_samples['GP_region_std'], _samples['GP_state_std'],
              _samples['year_std'], _samples['state_std'],
              _samples['region_std'], _samples['tot_var'],
              _samples['prop_var'], _samples['mu'],
              _samples['length_GP_region_long'],
              _samples['length_GP_state_long'],
              _samples['length_GP_region_short'],
              _samples['length_GP_state_short'])

def parameters_info(*, N_regions, N_years_obs, N_years, N_states,
                       state_region_ind, N, state_ind, region_ind, year_ind,
                       y, years, counts):
    return { 'GP_region_std': { 'shape': [N_years, N_regions] },
             'GP_state_std': { 'shape': [N_years, N_states] },
             'year_std': { 'shape': [N_years_obs] },
             'state_std': { 'shape': [N_states] },
             'region_std': { 'shape': [N_regions] },
             'tot_var': { 'shape': [] },'prop_var': { 'shape': [17] },
             'mu': { 'shape': [] },'length_GP_region_long': { 'shape': [] },
             'length_GP_state_long': { 'shape': [] },
             'length_GP_region_short': { 'shape': [] },
             'length_GP_state_short': { 'shape': [] }, }

