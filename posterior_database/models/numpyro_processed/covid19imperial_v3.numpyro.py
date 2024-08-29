from stannumpyro.distributions import *
from stannumpyro.dppllib import lax_cond, numpyro_cond, lax_while_loop, lax_fori_loop, fori_loop, foreach_loop, lax_foreach_loop, jit, sample, param, observe, factor, array, zeros, ones, empty, matmul, true_divide, floor_divide, transpose, dtype_long, dtype_float, vmap, get_element
from stannumpyro.stanlib import cumulative_sum_vector, dot_product_vector_vector, exp_vector, log_real, rep_matrix_int_int_int, rep_vector_real_int, sub_col_matrix_int_int_int, tail_vector_int


def convert_inputs(inputs):
    N0 = inputs['N0']
    M = inputs['M']
    N = array(inputs['N'], dtype=dtype_long)
    N2 = inputs['N2']
    cases = array(inputs['cases'], dtype=dtype_long)
    deaths = array(inputs['deaths'], dtype=dtype_long)
    f = array(inputs['f'], dtype=dtype_float)
    P = inputs['P']
    X = array(inputs['X'], dtype=dtype_float)
    EpidemicStart = array(inputs['EpidemicStart'], dtype=dtype_long)
    pop = array(inputs['pop'], dtype=dtype_float)
    SI = array(inputs['SI'], dtype=dtype_float)
    return {'N0': N0, 'M': M, 'N': N, 'N2': N2, 'cases': cases, 'deaths':
        deaths, 'f': f, 'P': P, 'X': X, 'EpidemicStart': EpidemicStart,
        'pop': pop, 'SI': SI}


def transformed_data(*, N0, M, N, N2, cases, deaths, f, P, X, EpidemicStart,
    pop, SI):
    SI_rev = empty([N2], dtype=dtype_float)

    @jit
    def _fori__1(i, _acc__2):
        SI_rev = _acc__2
        SI_rev = SI_rev.at[i - 1].set(SI[N2 - i + 1 - 1])
        return SI_rev
    SI_rev = lax_fori_loop(1, N2 + 1, _fori__1, SI_rev)
    f_rev = empty([M, N2], dtype=dtype_float)

    @jit
    def _fori__3(m, _acc__4):
        f_rev = _acc__4

        @jit
        def _fori__5(i, _acc__6):
            f_rev = _acc__6
            f_rev = f_rev.at[m - 1, i - 1].set(f[N2 - i + 1 - 1, m - 1])
            return f_rev
        f_rev = lax_fori_loop(1, N2 + 1, _fori__5, f_rev)
        return f_rev
    f_rev = lax_fori_loop(1, M + 1, _fori__3, f_rev)
    return {'SI_rev': SI_rev, 'f_rev': f_rev}


def model(*, N0, M, N, N2, cases, deaths, f, P, X, EpidemicStart, pop, SI,
    SI_rev, f_rev):
    mu = sample('mu', lower_constrained_improper_uniform(0, shape=[M]))
    alpha_hier = sample('alpha_hier', lower_constrained_improper_uniform(0,
        shape=[P]))
    kappa = sample('kappa', lower_constrained_improper_uniform(0, shape=[]))
    y = sample('y', lower_constrained_improper_uniform(0, shape=[M]))
    phi = sample('phi', lower_constrained_improper_uniform(0, shape=[]))
    tau = sample('tau', lower_constrained_improper_uniform(0, shape=[]))
    ifr_noise = sample('ifr_noise', lower_constrained_improper_uniform(0,
        shape=[M]))
    prediction = rep_matrix_int_int_int(0, N2, M)
    E_deaths = rep_matrix_int_int_int(0, N2, M)
    Rt = rep_matrix_int_int_int(0, N2, M)
    Rt_adj = Rt
    cumm_sum = rep_matrix_int_int_int(0, N2, M)
    alpha = empty([P], dtype=dtype_float)

    @jit
    def _fori__7(i, _acc__8):
        alpha = _acc__8
        alpha = alpha.at[i - 1].set(alpha_hier[i - 1] - true_divide(
            log_real(array(1.05, dtype=dtype_float)), array(6.0, dtype=
            dtype_float)))
        return alpha
    alpha = lax_fori_loop(1, P + 1, _fori__7, alpha)

    @jit
    def _fori__9(m, _acc__10):
        E_deaths, Rt, Rt_adj, cumm_sum, prediction = _acc__10
        prediction = prediction.at[1 - 1:N0, m - 1].set(rep_vector_real_int
            (y[m - 1], N0))
        cumm_sum = cumm_sum.at[2 - 1:N0, m - 1].set(cumulative_sum_vector(
            prediction[2 - 1:N0, m - 1]))
        Rt = Rt.at[:, m - 1].set(mu[m - 1] * exp_vector(matmul(-X[m - 1],
            alpha)))
        Rt_adj = Rt_adj.at[1 - 1:N0, m - 1].set(Rt[1 - 1:N0, m - 1])

        @jit
        def _fori__11(i, _acc__12):
            Rt_adj, cumm_sum, prediction = _acc__12
            convolution = dot_product_vector_vector(sub_col_matrix_int_int_int
                (prediction, 1, m, i - 1), tail_vector_int(SI_rev, i - 1))
            cumm_sum = cumm_sum.at[i - 1, m - 1].set(cumm_sum[i - 1 - 1, m -
                1] + prediction[i - 1 - 1, m - 1])
            Rt_adj = Rt_adj.at[i - 1, m - 1].set(true_divide(pop[m - 1] -
                cumm_sum[i - 1, m - 1], pop[m - 1]) * Rt[i - 1, m - 1])
            prediction = prediction.at[i - 1, m - 1].set(Rt_adj[i - 1, m - 
                1] * convolution)
            return Rt_adj, cumm_sum, prediction
        Rt_adj, cumm_sum, prediction = lax_fori_loop(N0 + 1, N2 + 1,
            _fori__11, (Rt_adj, cumm_sum, prediction))
        E_deaths = E_deaths.at[1 - 1, m - 1].set(array(1e-15, dtype=
            dtype_float) * prediction[1 - 1, m - 1])

        @jit
        def _fori__13(i, _acc__14):
            E_deaths = _acc__14
            E_deaths = E_deaths.at[i - 1, m - 1].set(ifr_noise[m - 1] *
                dot_product_vector_vector(sub_col_matrix_int_int_int(
                prediction, 1, m, i - 1), tail_vector_int(f_rev[m - 1], i - 1))
                )
            return E_deaths
        E_deaths = lax_fori_loop(2, N2 + 1, _fori__13, E_deaths)
        return E_deaths, Rt, Rt_adj, cumm_sum, prediction
    E_deaths, Rt, Rt_adj, cumm_sum, prediction = lax_fori_loop(1, M + 1,
        _fori__9, (E_deaths, Rt, Rt_adj, cumm_sum, prediction))
    observe('_tau__15', exponential(array(0.03, dtype=dtype_float)), tau)

    def _fori__16(m, _acc__17):
        observe(f'_y__{m}__18', exponential(true_divide(1, tau)), y[m - 1])
        return None
    _ = fori_loop(1, M + 1, _fori__16, None)
    observe('_phi__19', normal(0, 5), phi)
    observe('_kappa__20', normal(0, array(0.5, dtype=dtype_float)), kappa)
    observe('_mu__21', normal(array(3.28, dtype=dtype_float), kappa), mu)
    observe('_alpha_hier__22', gamma(array(0.1667, dtype=dtype_float), 1),
        alpha_hier)
    observe('_ifr_noise__23', normal(1, array(0.1, dtype=dtype_float)),
        ifr_noise)

    def _fori__24(m, _acc__25):
        observe(f'_deaths__{m}__26', neg_binomial_2(E_deaths[EpidemicStart[
            m - 1] - 1:N[m - 1], m - 1], phi), deaths[EpidemicStart[m - 1] -
            1:N[m - 1], m - 1])
        return None
    _ = fori_loop(1, M + 1, _fori__24, None)


def generated_quantities(*, N0, M, N, N2, cases, deaths, f, P, X,
    EpidemicStart, pop, SI, SI_rev, f_rev, mu, alpha_hier, kappa, y, phi,
    tau, ifr_noise):
    prediction = rep_matrix_int_int_int(0, N2, M)
    E_deaths = rep_matrix_int_int_int(0, N2, M)
    Rt = rep_matrix_int_int_int(0, N2, M)
    Rt_adj = Rt
    cumm_sum = rep_matrix_int_int_int(0, N2, M)
    alpha = empty([P], dtype=dtype_float)

    @jit
    def _fori__27(i, _acc__28):
        alpha = _acc__28
        alpha = alpha.at[i - 1].set(alpha_hier[i - 1] - true_divide(
            log_real(array(1.05, dtype=dtype_float)), array(6.0, dtype=
            dtype_float)))
        return alpha
    alpha = lax_fori_loop(1, P + 1, _fori__27, alpha)

    @jit
    def _fori__29(m, _acc__30):
        E_deaths, Rt, Rt_adj, cumm_sum, prediction = _acc__30
        prediction = prediction.at[1 - 1:N0, m - 1].set(rep_vector_real_int
            (y[m - 1], N0))
        cumm_sum = cumm_sum.at[2 - 1:N0, m - 1].set(cumulative_sum_vector(
            prediction[2 - 1:N0, m - 1]))
        Rt = Rt.at[:, m - 1].set(mu[m - 1] * exp_vector(matmul(-X[m - 1],
            alpha)))
        Rt_adj = Rt_adj.at[1 - 1:N0, m - 1].set(Rt[1 - 1:N0, m - 1])

        @jit
        def _fori__31(i, _acc__32):
            Rt_adj, cumm_sum, prediction = _acc__32
            convolution = dot_product_vector_vector(sub_col_matrix_int_int_int
                (prediction, 1, m, i - 1), tail_vector_int(SI_rev, i - 1))
            cumm_sum = cumm_sum.at[i - 1, m - 1].set(cumm_sum[i - 1 - 1, m -
                1] + prediction[i - 1 - 1, m - 1])
            Rt_adj = Rt_adj.at[i - 1, m - 1].set(true_divide(pop[m - 1] -
                cumm_sum[i - 1, m - 1], pop[m - 1]) * Rt[i - 1, m - 1])
            prediction = prediction.at[i - 1, m - 1].set(Rt_adj[i - 1, m - 
                1] * convolution)
            return Rt_adj, cumm_sum, prediction
        Rt_adj, cumm_sum, prediction = lax_fori_loop(N0 + 1, N2 + 1,
            _fori__31, (Rt_adj, cumm_sum, prediction))
        E_deaths = E_deaths.at[1 - 1, m - 1].set(array(1e-15, dtype=
            dtype_float) * prediction[1 - 1, m - 1])

        @jit
        def _fori__33(i, _acc__34):
            E_deaths = _acc__34
            E_deaths = E_deaths.at[i - 1, m - 1].set(ifr_noise[m - 1] *
                dot_product_vector_vector(sub_col_matrix_int_int_int(
                prediction, 1, m, i - 1), tail_vector_int(f_rev[m - 1], i - 1))
                )
            return E_deaths
        E_deaths = lax_fori_loop(2, N2 + 1, _fori__33, E_deaths)
        return E_deaths, Rt, Rt_adj, cumm_sum, prediction
    E_deaths, Rt, Rt_adj, cumm_sum, prediction = lax_fori_loop(1, M + 1,
        _fori__29, (E_deaths, Rt, Rt_adj, cumm_sum, prediction))
    prediction0 = rep_matrix_int_int_int(0, N2, M)
    E_deaths0 = rep_matrix_int_int_int(0, N2, M)
    cumm_sum0 = rep_matrix_int_int_int(0, N2, M)

    @jit
    def _fori__35(m, _acc__36):
        E_deaths0, cumm_sum0, prediction0 = _acc__36

        @jit
        def _fori__37(i, _acc__38):
            cumm_sum0 = _acc__38
            cumm_sum0 = cumm_sum0.at[i - 1, m - 1].set(cumm_sum0[i - 1 - 1,
                m - 1] + y[m - 1])
            return cumm_sum0
        cumm_sum0 = lax_fori_loop(2, N0 + 1, _fori__37, cumm_sum0)
        prediction0 = prediction0.at[1 - 1:N0, m - 1].set(rep_vector_real_int
            (y[m - 1], N0))

        @jit
        def _fori__39(i, _acc__40):
            cumm_sum0, prediction0 = _acc__40
            convolution0 = 0

            @jit
            def _fori__41(j, _acc__42):
                convolution0 = _acc__42
                convolution0 = convolution0 + prediction0[j - 1, m - 1] * SI[
                    i - j - 1]
                return convolution0
            convolution0 = lax_fori_loop(1, i - 1 + 1, _fori__41, convolution0)
            cumm_sum0 = cumm_sum0.at[i - 1, m - 1].set(cumm_sum0[i - 1 - 1,
                m - 1] + prediction0[i - 1 - 1, m - 1])
            prediction0 = prediction0.at[i - 1, m - 1].set(true_divide(pop[
                m - 1] - cumm_sum0[i - 1, m - 1], pop[m - 1]) * mu[m - 1] *
                convolution0)
            return cumm_sum0, prediction0
        cumm_sum0, prediction0 = lax_fori_loop(N0 + 1, N2 + 1, _fori__39, (
            cumm_sum0, prediction0))
        E_deaths0 = E_deaths0.at[1 - 1, m - 1].set(uniform_rng(array(1e-16,
            dtype=dtype_float), array(1e-15, dtype=dtype_float)))

        @jit
        def _fori__43(i, _acc__44):
            E_deaths0 = _acc__44

            @jit
            def _fori__45(j, _acc__46):
                E_deaths0 = _acc__46
                E_deaths0 = E_deaths0.at[i - 1, m - 1].set(E_deaths0[i - 1,
                    m - 1] + prediction0[j - 1, m - 1] * f[i - j - 1, m - 1
                    ] * ifr_noise[m - 1])
                return E_deaths0
            E_deaths0 = lax_fori_loop(1, i - 1 + 1, _fori__45, E_deaths0)
            return E_deaths0
        E_deaths0 = lax_fori_loop(2, N2 + 1, _fori__43, E_deaths0)
        return E_deaths0, cumm_sum0, prediction0
    E_deaths0, cumm_sum0, prediction0 = lax_fori_loop(1, M + 1, _fori__35,
        (E_deaths0, cumm_sum0, prediction0))
    return {'prediction': prediction, 'E_deaths': E_deaths, 'Rt': Rt,
        'Rt_adj': Rt_adj, 'cumm_sum': cumm_sum, 'alpha': alpha,
        'prediction0': prediction0, 'E_deaths0': E_deaths0, 'cumm_sum0':
        cumm_sum0}


def map_generated_quantities(_samples, *, N0, M, N, N2, cases, deaths, f, P,
    X, EpidemicStart, pop, SI, SI_rev, f_rev):

    def _generated_quantities(mu, alpha_hier, kappa, y, phi, tau, ifr_noise):
        return generated_quantities(N0=N0, M=M, N=N, N2=N2, cases=cases,
            deaths=deaths, f=f, P=P, X=X, EpidemicStart=EpidemicStart, pop=
            pop, SI=SI, SI_rev=SI_rev, f_rev=f_rev, mu=mu, alpha_hier=
            alpha_hier, kappa=kappa, y=y, phi=phi, tau=tau, ifr_noise=ifr_noise
            )
    _f = jit(vmap(_generated_quantities))
    return _f(_samples['mu'], _samples['alpha_hier'], _samples['kappa'],
        _samples['y'], _samples['phi'], _samples['tau'], _samples['ifr_noise'])


def parameters_info(*, N0, M, N, N2, cases, deaths, f, P, X, EpidemicStart,
    pop, SI, SI_rev, f_rev):
    return {'mu': {'shape': [M]}, 'alpha_hier': {'shape': [P]}, 'kappa': {
        'shape': []}, 'y': {'shape': [M]}, 'phi': {'shape': []}, 'tau': {
        'shape': []}, 'ifr_noise': {'shape': [M]}}
