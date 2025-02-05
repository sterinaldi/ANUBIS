import numpy as np
from scipy.stats import dirichlet, qmc, multivariate_normal as mn
from scipy.special import logsumexp
from emcee import EnsembleSampler
from emcee.moves import GaussianMove, StretchMove

from anubis.exceptions import ANUBISException
from anubis._likelihood import _population_log_likelihood, _joint_population_log_likelihood
from figaro.mixture import DPGMM, HDPGMM, mixture, _update_alpha
from figaro.decorators import probit
from figaro.transform import transform_to_probit
from figaro.utils import rejection_sampler
from figaro._likelihood import evaluate_mixture_MC_draws, evaluate_mixture_MC_draws_1d
from figaro._numba_functions import logsumexp_jit

np.seterr(divide = 'ignore')

class uniform:
    """
    Class with the same methods as figaro.mixture.mixture (in particular, pdf and marginalise)
    
    Arguments:
        np.ndarray bounds: 2d-array with bounds
        bool probit:       probit transformation (for compatibility)
    
    Returns:
        uniform: instance of uniform_model class
    """
    def __init__(self, bounds, probit):
        self.bounds       = bounds
        self.probit       = probit
        self.volume       = np.prod(np.diff(self.bounds, axis = 1))
        self.dim          = len(self.bounds)
        self.alpha_factor = 1.
    
    def __call__(self, x):
        return self.pdf(x)
    
    def pdf(self, x):
        x = np.atleast_1d(x)
        return np.ones(len(x))/self.volume

    def logpdf(self, x):
        x = np.atleast_1d(x)
        return -np.ones(len(x))*np.log(self.volume)
    
    def rvs(self, size = 1.):
        size = int(size)
        return np.random.uniform(low = self.bounds[:,0], high = self.bounds[:,1], size = (size, len(self.bounds)))
        
    def marginalise(self, axis):
        if len(axis) == 0:
            return uniform(self.bounds, self.probit)
        return uniform(np.delete(self.bounds, np.atleast_1d(axis), axis = 0), self.probit)

class nonpar_model:
    """
    Wrapper for the figaro.mixture.mixture class with additional methods to include the observed/intrinsic pdf/logpdf automatically.
    
    Arguments:
        figaro.mixture.mixture mixture: non-parametric model
        bool hierarchical:              whether the model comes from a hierarchical inference or not
        callable selfunc:               selection function (if required)
    """
    def __init__(self, mixture, hierarchical, selection_function = None):
        self.mixture = mixture
        self.hierarchical = hierarchical
        self.selfunc = selection_function
        self.probit = self.mixture.probit
        if self.selfunc is not None:
            if self.hierarchical:
                self.alpha = self.mixture.alpha_factor
            else:
                self.alpha = np.mean(self.selfunc(self.mixture.rvs(size = int((self.mixture.dim+1)*1e5))))
    
    def __call__(self, x):
        return self.pdf(x)

    def pdf(self, x):
        """
        pdf of the intrinsic distribution
        
        Arguments:
            np.ndarray x: point to evaluate the mixture at
        
        Returns:
            np.ndarray: mixture.pdf(x)
        """
        if self.selfunc is None or self.hierarchical:
            return self.mixture.pdf(x)
        else:
            with np.errstate(divide = 'ignore', invalid = 'ignore'):
                return np.nan_to_num(self.mixture.pdf(x), nan = 0., posinf = 0., neginf = 0.)
    
    def logpdf(self, x):
        """
        logpdf of the intrinsic distribution
        
        Arguments:
            np.ndarray x: point to evaluate the mixture at
        
        Returns:
            np.ndarray: mixture.logpdf(x)
        """
        if self.selfunc is None or self.hierarchical:
            return self.mixture.logpdf(x)
        else:
            with np.errstate(divide = 'ignore', invalid = 'ignore'):
                return np.nan_to_num(self.mixture.logpdf(x), nan = -np.inf, posinf = -np.inf)
    
    def pdf_observed(self, x):
        """
        pdf of the observed distribution
        
        Arguments:
            np.ndarray x: point to evaluate the mixture at
        
        Returns:
            np.ndarray: mixture.pdf(x)
        """
        if self.selfunc is None or not self.hierarchical:
            return self.mixture.pdf(x)
        else:
            with np.errstate(divide = 'ignore', invalid = 'ignore'):
                return np.nan_to_num(self.mixture.pdf(x)*self.selfunc(x)*self.alpha, nan = 0., posinf = 0., neginf = 0.)
    
    def logpdf_observed(self, x):
        """
        logpdf of the observed distribution
        
        Arguments:
            np.ndarray x: point to evaluate the mixture at
        
        Returns:
            np.ndarray: mixture.logpdf(x)
        """
        if self.selfunc is None or not self.hierarchical:
            return self.mixture.logpdf(x)
        else:
            with np.errstate(divide = 'ignore', invalid = 'ignore'):
                return np.nan_to_num(self.mixture.logpdf(x) + np.log(self.selfunc(x)) + np.log(self.alpha), nan = -np.inf, posinf = -np.inf)

class par_model:
    """
    Class to store a parametric model.
    
    Arguments:
        callable model:    model pdf
        iterable pars:     parameters of the model
        np.ndarray bounds: bounds (FIGARO)
        bool probit:       whether to use the probit transformation or not (FIGARO compatibility)
        callable selfunc:  selection function
        float norm:        normalisation constant for the observed distribution
    
    Returns:
        par_model: instance of model class
    """
    def __init__(self, model,
                       pars,
                       bounds,
                       probit,
                       hierarchical,
                       selection_function = None,
                       inj_pdf = None,
                       n_total_inj = None,
                       norm = None,
                       ):
        self.model        = model
        self.hierarchical = hierarchical
        self.pars         = pars
        self.bounds       = np.atleast_2d(bounds)
        self.dim          = len(self.bounds)
        self.probit       = probit
        self.selfunc      = selection_function
        self.samples      = None
        self.inj_pdf      = inj_pdf
        if self.inj_pdf is not None:
            self.inj_pdf  = self.inj_pdf.flatten()
        self.n_total_inj  = n_total_inj
        if norm is not None:
            self.alpha = norm
        else:
            self.alpha = 1.
    
    def _selfunc(func):
        """
        Applies the selection function to to the intrinsic distribution.
        """
        def observed_model(self, x, *args):
            if not self.hierarchical and self.selfunc is not None:
                return func(self, x, *args)*self.selfunc(x)
            else:
                return func(self, x, *args)
        return observed_model
    
    def __call__(self, x):
        return self.pdf(x)
    
    def _compute_alpha_factor(self, pars, shared_pars, n_draws):
        """
        Computes the normalisation of the product p_intr(x|lambda)p_det(x) via monte carlo approximation
        (Eq. 6 of Mandel et al. 2019)
        
        Arguments:
            np.ndarray pars:        parameters of the distribution
            np.ndarray shared pars: shared parameters of the distribution
            int n_draws:            number of draws for the MC integral
        """
        self.alpha = None
        if callable(self.selfunc):
            if self.samples is None:
                self.volume     = np.prod(np.diff(self.bounds, axis = 1))
                self.samples    = qmc.scale(qmc.Halton(len(self.bounds)).random(int(n_draws)), *self.bounds.T)
                self.sf_samples = self.selfunc(self.samples).flatten()
            if pars is not None:
                self.alpha = np.nan_to_num(np.atleast_1d([np.mean(self.model(self.samples, *p, *sp).flatten()*self.sf_samples*self.volume) for p, sp in zip(pars, shared_pars)]), neginf = np.inf, nan = np.inf)
                self.alpha[self.alpha < 1e-3] = np.inf
            else:
                self.alpha = np.atleast_1d(np.mean(self.pdf(self.samples)*self.sf_samples*self.volume))
        else:
            if pars is not None:
                self.alpha = np.nan_to_num(np.atleast_1d([np.sum(self.model(self.selfunc, *p, *sp).flatten()/self.inj_pdf)/self.n_total_inj for p, sp in zip(pars, shared_pars)]), neginf = np.inf, nan = np.inf)
                var = np.nan_to_num(np.atleast_1d([np.sum(self.model(self.selfunc, *p, *sp).flatten()**2/self.inj_pdf**2)/self.n_total_inj**2 - a**2/self.n_total_inj for a, p, sp in zip(self.alpha, pars, shared_pars)]), neginf = np.inf, nan = np.inf)
#                self.alpha[self.alpha < 1e-3] = np.inf
                self.alpha[np.sqrt(var)/self.alpha > 0.05] = np.inf
            else:
                self.alpha = np.atleast_1d(np.sum(self.pdf(self.selfunc).flatten()/self.inj_pdf)/self.n_total_inj)
        if len(self.alpha) == 1:
            self.alpha = self.alpha[0]
    
    def pdf(self, x):
        """
        pdf of the observed distribution
        
        Arguments:
            np.ndarray x: point to evaluate the mixture at
        
        Returns:
            np.ndarray: p_intr.pdf(x)*p_obs(x)/norm
        """
        return self.model(x, *self.pars)

    def logpdf(self, x):
        """
        logpdf of the observed distribution
        
        Arguments:
            np.ndarray x: point to evaluate the mixture at
        
        Returns:
            np.ndarray: p_intr.logpdf(x)
        """
        return np.log(self.model(x, *self.pars))
    
    @_selfunc
    def pdf_observed(self, x):
        """
        pdf of the intrinsic distribution
        
        Arguments:
            np.ndarray x: point to evaluate the mixture at
        
        Returns:
            np.ndarray: p_intr.pdf(x)
        """
        return self.model(x, *self.pars)/self.alpha
    
    def pdf_pars(self, x, pars, shared_pars):
        """
        Observed pdf with different realisations of the parameters theta.
        
        Arguments:
            np.ndarray x:           point to evaluate the mixture at
            np.ndarray pars:        array of parameters
            np.ndarray shared_pars: array of shared parameters
        Returns:
            np.ndarray: p_intr.pdf(x|theta)*p_obs(x)/norm
        """
        if self.alpha is not None:
            if hasattr(self.alpha, '__iter__'):
                return np.array([self._model(x, p, sp)/a for p, sp, a in zip(pars, shared_pars, self.alpha)])
            else:
                return np.array([self._model(x, p, sp)/self.alpha for p, sp in zip(pars, shared_pars)])
        else:
            return np.array([self._model(x, p, sp) for p, sp in zip(pars, shared_pars)])
    
    @_selfunc
    def _model(self, x, pars, shared_pars):
        """
        Decorated intrinsic distribution with explicit dependence of parameters theta
        
        Arguments:
            np.ndarray x:           point to evaluate the mixture at
            np.ndarray pars:        array of parameters
            np.ndarray shared_pars: array of shared parameters
        
        Returns:
            np.ndarray: p_intr.pdf(x|theta)*p_obs(x)
        """
        return self.model(x, *pars, *shared_pars).flatten()

class het_mixture:
    """
    Class to store a single draw from AMM.
    
    Arguments:
        list-of-callables models: models in the mixture
        np.ndarray:               weights
        np.ndarray bounds:        bounds (FIGARO)
        bool augment:             whether the model includes a non-parametric augmentation
        callable selfunc:         selection function
        int n_shared_pars:        number of shared parameters among models
        
    Returns:
        het_mixture: instance of het_mixture class
    """
    def __init__(self, models,
                       weights,
                       bounds,
                       augment,
                       hierarchical,
                       selfunc       = None,
                       n_shared_pars = 0,
                       ):
        # Components
        self.models        = models
        self.hierarchical  = hierarchical
        self.weights       = weights
        self.bounds        = np.atleast_2d(bounds)
        self.dim           = len(self.bounds)
        self.augment       = augment
        self.selfunc       = selfunc
        self.n_shared_pars = int(n_shared_pars)
        # Weights and normalisation
        if self.selfunc is not None:
            if not self.hierarchical:
                self.intrinsic_weights = [wi/mi.alpha for wi, mi in zip(self.weights[self.augment:], self.models[self.augment:])]
                if self.augment:
                    self.intrinsic_weights = [self.weights[0]*self.models[0].alpha] + self.intrinsic_weights
                self.intrinsic_weights = np.array(self.intrinsic_weights/np.sum(self.intrinsic_weights))
                self.observed_weights  = self.weights
            else:
                self.intrinsic_weights = self.weights
                self.observed_weights  = np.array([wi*mi.alpha for wi, mi in zip(self.weights, self.models)])
                #if not np.isfinite(self.observed_weights).all():
                    #print(self.observed_weights, [mi.alpha for mi in self.models])
                self.observed_weights /= np.sum(self.observed_weights)
        else:
            self.intrinsic_weights = self.weights
            self.observed_weights  = self.weights
        self.log_intrinsic_weights = np.log(self.intrinsic_weights)
        self.log_observed_weights  = np.log(self.observed_weights)
        # Parametric models only
        self.parametric_weights     = self.intrinsic_weights[self.augment:]
        self.log_parametric_weights = np.log(self.parametric_weights)
        self.norm_parametric        = np.sum(self.parametric_weights)
        self.log_norm_parametric    = np.log(self.norm_parametric)
        if self.augment:
            self.probit = self.models[0].probit
        else:
            self.probit = False
    
    def __call__(self, x):
        return self.pdf(x)
    
    def pdf(self, x):
        """
        Evaluate mixture at point(s) x
        
        Arguments:
            np.ndarray x: point(s) to evaluate the mixture at
        
        Returns:
            np.ndarray: het_mixture.pdf(x)
        """
        return np.array([wi*mi.pdf(x) for wi, mi in zip(self.intrinsic_weights, self.models)]).sum(axis = 0)
    
    def logpdf(self, x):
        """
        Evaluate log mixture at point(s) x
        
        Arguments:
            np.ndarray x: point(s) to evaluate the mixture at
        
        Returns:
            np.ndarray: het_mixture.logpdf(x)
        """
        return np.array([wi+mi.logpdf(x) for wi, mi in zip(self.log_intrinsic_weights, self.models)]).sum(axis = 0)

    def pdf_observed(self, x):
        """
        Evaluate mixture at point(s) x (observed)
        
        Arguments:
            np.ndarray x: point(s) to evaluate the mixture at
        
        Returns:
            np.ndarray: het_mixture.pdf(x)
        """
        return np.array([wi*mi.pdf_observed(x) for wi, mi in zip(self.observed_weights, self.models)]).sum(axis = 0)
    
    def logpdf_observed(self, x):
        """
        Evaluate log mixture at point(s) x (observed)
        
        Arguments:
            np.ndarray x: point(s) to evaluate the mixture at
        
        Returns:
            np.ndarray: het_mixture.logpdf(x)
        """
        return np.array([wi+mi.logpdf(x) for wi, mi in zip(self.log_observed_weights, self.models)]).sum(axis = 0)
    
    def pdf_parametric(self, x):
        """
        Evaluate mixture at point(s) x (parametric models only)
        
        Arguments:
            np.ndarray x: point(s) to evaluate the mixture at
        
        Returns:
            np.ndarray: het_mixture.pdf(x)
        """
        return np.array([wi*mi.pdf(x)/self.norm_parametric for wi, mi in zip(self.parametric_weights, self.models[self.augment:])]).sum(axis = 0)

    def logpdf_parametric(self, x):
        """
        Evaluate log mixture at point(s) x (parametric models only)
        
        Arguments:
            np.ndarray x: point(s) to evaluate the mixture at
        
        Returns:
            np.ndarray: het_mixture.logpdf(x)
        """
        return np.array([wi+mi.logpdf(x)-self.log_norm_parametric for wi, mi in zip(self.log_parametric_weights, self.models[self.augment:])]).sum(axis = 0)

#-----------------#
# Inference class #
#-----------------#

class AMM:
    """
    Class to infer a distribution given a set of samples.
    
    Arguments:
        list-of-callables:          models
        iterable bounds:            boundaries of the rectangle over which the distribution is defined. It should be in the format [[xmin, xmax],[ymin, ymax],...]
        iterable pars:              fixed parameters of the parametric model(s)
        iterable shared_pars:       shared fixed parameters of the parametric model(s)
        iterable par_bounds:        boundaries of the allowed values for the parameters. It should be in the format [[[xmin, xmax],[ymin, ymax]],[[xmin, xmax]],...]
        iterable shared_par_bounds: boundaries of the allowed values for the shared parameters. See above for the format.
        iterable prior_pars:        NIW prior parameters (k, L, nu, mu)
        callable selfunc:           selection function (if required)
        double n_draws_pars:        number of draws for MC integral over parameters
        double n_draws_norm:        number of draws for normalisation MC integral over parameters
        double alpha0:              initial guess for concentration parameter
        np.ndarray gamma0:          Dirichlet Distribution prior
        bool probit:                whether to use the probit transformation for the DPGMM
        bool augment:               whether to include the non-parametric channel
        int n_reassignments:        number of reassignments
        np.ndarray norm:            normalisation constant for the parametric observed distributions. Use None if not available
        int n_steps_mcmc:           number of steps for the mcmc sampler before drawing a sample
    
    Returns:
        AMM: instance of AMM class
    """
    def __init__(self, models,
                       bounds,
                       pars               = None,
                       shared_pars        = None,
                       par_bounds         = None,
                       shared_par_bounds  = None,
                       prior_pars         = None,
                       selection_function = None,
                       inj_pdf            = None,
                       n_total_inj        = None,
                       n_draws_pars       = None,
                       n_draws_norm       = None,
                       alpha0             = 1.,
                       gamma0             = None,
                       probit             = False,
                       augment            = True,
                       n_reassignments    = None,
                       norm               = None,
                       n_steps_mcmc       = None,
                       ):
        # Settings
        self.bounds       = np.atleast_2d(bounds)
        self.dim          = len(self.bounds)
        self.probit       = probit
        self.augment      = augment
        self.selfunc      = selection_function
        self.inj_pdf      = inj_pdf
        self.n_total_inj  = n_total_inj
        self.hierarchical = False
        # Parametric models
        if pars is None:
            pars = [[] for _ in models]
            self.n_draws_pars = 0
        if shared_pars is None:
            shared_pars = []
        if par_bounds is not None:
            self.par_bounds = [np.atleast_2d(pb) if pb is not None else None for pb in par_bounds]
            if n_draws_pars is not None:
                self.n_draws_pars = int(n_draws_pars)
            else:
                self.n_draws_pars = int(1e3)
        else:
            self.par_bounds   = None
            self.n_draws_pars = 1
        if shared_par_bounds is not None:
            self.shared_par_bounds = np.atleast_2d(shared_par_bounds)
            if n_draws_pars is not None:
                self.n_draws_pars = int(n_draws_pars)
            else:
                self.n_draws_pars = int(1e3)
        else:
            self.shared_par_bounds = None
        if self.selfunc is not None:
            if n_draws_norm is not None:
                self.n_draws_norm = int(n_draws_norm)
            else:
                self.n_draws_norm = int(1e4)
        if norm is None:
            self.norm   = [None for _ in models]
        else:
            self.norm   = norm
        self.par_models = [par_model(mod, list(p) + list(shared_pars), bounds, probit, hierarchical = False, selection_function = self.selfunc, inj_pdf = self.inj_pdf, n_total_inj = n_total_inj, norm = n) for mod, p, n in zip(models, pars, self.norm)]
        # DPGMM initialisation (if required)
        if self.augment:
            self.nonpar = DPGMM(bounds     = bounds,
                                prior_pars = prior_pars,
                                alpha0     = alpha0,
                                probit     = self.probit,
                                )
            self.volume       = np.prod(np.diff(self.nonpar.bounds, axis = 1))
            self.components   = [self.nonpar] + self.par_models
            self.n_components = len(models) + 1
        else:
            self.components   = self.par_models
            self.n_components = len(models)
        # Gibbs sampler
        self.n_reassignments  = n_reassignments
        if gamma0 is None:
            self.gamma0 = np.ones(self.n_components)
        else:
            if not hasattr(gamma0, '__iter__'):
                self.gamma0 = np.ones(self.n_components)*gamma0
            elif len(gamma0) == self.n_components:
                self.gamma0 = gamma0
            else:
                raise Exception("gamma0 must be an array with {0} components or a float.".format(self.n_components))
        if n_steps_mcmc is None:
            self.n_steps_mcmc = int(1e3)
        else:
            self.n_steps_mcmc = int(n_steps_mcmc)
        self.first_run = True
        if self.shared_par_bounds is None:
            self.samplers = [EnsembleSampler(nwalkers    = 1,
                                             ndim        = len(b),
                                             log_prob_fn = _population_log_likelihood,
                                             args        = ([self]),
                                             moves       = GaussianMove((np.diff(b).flatten()/20)**2),
                                             )
                            for b in self.par_bounds]
        else:
            n_pars          = np.sum([len(b) for b in self.par_bounds])+len(self.shared_par_bounds)
            self.all_bounds = np.array([bi for b in self.par_bounds for bi in b] + list(self.shared_par_bounds))
            self.sampler = EnsembleSampler(nwalkers    = 1,
                                           ndim        = n_pars,
                                           log_prob_fn = _joint_population_log_likelihood,
                                           args        = ([self]),
                                           moves       = GaussianMove((np.diff(self.all_bounds).flatten()/20)**2),
                                           )
        # Initialisation
        self.initialise()
    
    def pdf(self, x):
        """
        Evaluate mixture at point(s) x
        
        Arguments:
            np.ndarray x: point(s) to evaluate the mixture at
        
        Returns:
            np.ndarray: mixture.pdf(x)
        """
        return np.array([wi*mi.pdf(x) for wi, mi in zip(self.weights, self.models)]).sum(axis = 0)
    
    def __call__(self, x):
        return self.pdf(x)
    
    def initialise(self, prior_pars = None):
        """
        Initialise the mixture to initial conditions.

        Arguments:
            iterable prior_pars: NIW prior parameters (k, L, nu, mu) for the DPGMM. If None, old parameters are kept
        """
        self.n_pts             = np.zeros(self.n_components)
        self.weights           = self.gamma0/np.sum(self.gamma0)
        self.stored_pts        = {}
        self.stored_pts_probit = {}
        self.assignations      = {}
        # Draw new parameter realisations
        if self.par_bounds is not None or self.shared_par_bounds is not None:
            self.evaluated_logL       = {}
            if self.par_bounds is not None:
                self.par_draws        = [qmc.scale(qmc.Halton(len(b)).random(self.n_draws_pars), *b.T) if b is not None else None for b in self.par_bounds]
            else:
                self.par_draws        = [[[] for _ in range(self.n_draws_pars)] for _ in range(len(self.components[self.augment:]))]
            if self.shared_par_bounds is not None:
                self.shared_par_draws = qmc.scale(qmc.Halton(len(self.shared_par_bounds)).random(self.n_draws_pars), *self.shared_par_bounds.T)
            else:
                self.shared_par_draws = [[] for _ in range(self.n_draws_pars)]
        else:
            self.par_draws        = [[[] for _ in range(self.n_draws_pars)] for _ in range(len(self.components[self.augment:]))]
            self.shared_par_draws = [[] for _ in range(self.n_draws_pars)]
        if self.selfunc is not None:
            [m._compute_alpha_factor(p, self.shared_par_draws, self.n_draws_norm) for m, p, n in zip(self.components[self.augment:], self.par_draws, self.norm) if n is None]
        if self.augment:
            self.nonpar.initialise()
            self.ids_nonpar = {}

    def _assign_to_component(self, x, x_probit, pt_id, id_nonpar = None, reassign = False):
        """
        Assign the sample x to an existing cluster or to a new cluster according to the marginal distribution of cluster assignment.
        
        Arguments:
            np.ndarray x:  sample
            int pt_id:     point id
            int id_nonpar: FIGARO id for the point
            bool reassign: wheter the point is new or is being reassigned
        """
        scores             = np.zeros(self.n_components)
        vals               = np.zeros(shape = (self.n_components, self.n_draws_pars))
        for i in range(self.n_components):
            score, vals[i] = self._log_predictive_likelihood(x, i, pt_id)
            scores[i]      = score + np.log(self.gamma0[i] + self.n_pts[i])
        scores             = np.exp(scores - logsumexp(scores))
        id                 = np.random.choice(self.n_components, p = scores)
        self.n_pts[id]    += 1
        self.weights       = (self.n_pts + self.gamma0)/np.sum(self.n_pts + self.gamma0)
        # If DPGMM, updates mixture
        if self.augment and id == 0:
            if id_nonpar is None:
                self.ids_nonpar[int(pt_id)] = len(list(self.nonpar.stored_pts.keys()))
                self.nonpar.add_new_point(x)
            else:
                self._reassign_point_nonpar(x_probit, id_nonpar)
        # Parameter estimation
        elif self.par_bounds is not None:
            self.evaluated_logL[pt_id] = vals
        self.assignations[pt_id]       = int(id)

    def _reassign_point_nonpar(self, x, id_nonpar):
        """
        Update the probability density reconstruction reassigining an existing sample
        
        Arguments:
            id:        sample id
            id_nonpar: FIGARO id for the point
        """
        self.nonpar._assign_to_cluster(x, id_nonpar)
        self.nonpar.alpha = _update_alpha(self.nonpar.alpha, self.nonpar.n_pts, (np.array(self.nonpar.N_list) > 0).sum(), self.nonpar.alpha_0)

    def _log_predictive_likelihood(self, x, i, pt_id):
        """
        Compute log likelihood of drawing sample x from component i given the samples that are already assigned to that component marginalised over the component parameters.
        
        Arguments:
            np.ndarray x: sample
            int i:        component id
            pt_id:        ANUBIS point ID
        
        Returns:
            double:     marginal log Likelihood
            np.ndarray: individual log Likelihood values for theta_i
        """
        # Non-parametric
        if self.augment and i == 0:
            return self._log_predictive_mixture(x), np.zeros(self.n_draws_pars)
        # Parametric
        else:
            # Fixed parameters or parameterless model
            if (self.par_bounds is None or self.par_bounds[i - self.augment] is None) and self.shared_par_bounds is None:
                return np.log(self.components[i].pdf(x)), np.zeros(self.n_draws_pars)
            # Marginalisation over parameters
            else:
                i_p = i - self.augment
                if not pt_id in list(self.evaluated_logL.keys()):
                    log_p = np.log(self.components[i].pdf_pars(x, self.par_draws[i_p], self.shared_par_draws)).flatten()
                else:
                    log_p = self.evaluated_logL[pt_id][i]
                log_total_p = np.atleast_1d(np.sum([self.evaluated_logL[pt][i] for pt in range(int(np.sum(self.n_pts))) if self.assignations[pt] == i], axis = 0))
                denom       = logsumexp(log_total_p) - np.log(self.n_draws_pars)
                v           = logsumexp(log_p + log_total_p) - np.log(self.n_draws_pars)
                return np.nan_to_num(v - denom, nan = -np.inf, neginf = -np.inf), log_p
    
    @probit
    def _log_predictive_mixture(self, x):
        """
        Compute log likelihood for non-parametric mixture (mixture of predictive likelihood)
        
        Arguments:
            np.ndarray x: sample
        
        Returns:
            double: log Likelihood
        """
        scores = np.zeros(self.nonpar.n_cl + 1)
        for j, i in enumerate(list(np.arange(self.nonpar.n_cl)) + ["new"]):
            if i == "new":
                ss = None
                scores[j] = -np.log(self.volume)
            else:
                ss = self.nonpar.mixture[i]
                if ss.N < 1:
                    scores[j] = 0.
                else:
                    scores[j] = self.nonpar._log_predictive_likelihood(x, ss)
            if ss is None:
                scores[j] += np.log(self.nonpar.alpha) - np.log(self.nonpar.n_pts + self.nonpar.alpha)
            elif ss.N < 1:
                scores[j]  = -np.inf
            else:
                scores[j] += np.log(ss.N) - np.log(self.nonpar.n_pts + self.nonpar.alpha)
        return logsumexp(scores)
    
    def add_new_point(self, x):
        """
        Update the probability density reconstruction adding a new sample
        
        Arguments:
            np.ndarray x: sample
        """
        x = np.atleast_2d(x)
        self.stored_pts[int(np.sum(self.n_pts))] = x
        if self.probit:
            x_probit = transform_to_probit(x, self.bounds)
            self.stored_pts_probit[int(np.sum(self.n_pts))] = x_probit
        else:
            x_probit = x
            self.stored_pts_probit[int(np.sum(self.n_pts))] = x
        self._assign_to_component(x, x_probit, pt_id = int(np.sum(self.n_pts)))
    
    def density_from_samples(self, samples, make_comp = True):
        """
        Reconstruct the probability density from a set of samples.
        
        Arguments:
            iterable samples: samples set
        
        Returns:
            het_mixture: the inferred mixture
        """
        np.random.shuffle(samples)
        if self.n_reassignments is None:
            n_reassignments = 5*len(samples)
        else:
            n_reassignments = self.n_reassignments
        for s in samples:
            self.add_new_point(s)
        # Random Gibbs walk (if required)
        if self.n_components > 1:
            for id in np.random.choice(int(np.sum(self.n_pts)), size = int(n_reassignments), replace = True):
                self._reassign_point(int(id))
            # Reassign all points once
            for id in range(int(np.sum(self.n_pts))):
                self._reassign_point(int(id))
        d = self.build_mixture(make_comp = make_comp)
        self.initialise()
        return d
    
    def _reassign_point(self, id):
        """
        Update the probability density reconstruction reassigining an existing sample
        
        Arguments:
            id: sample id
        """
        x                     = self.stored_pts[id]
        x_probit              = self.stored_pts_probit[id]
        cid                   = self.assignations[id]
        id_nonpar             = None
        self.n_pts[cid]      -= 1
        self.assignations[id] = None
        if self.augment and cid == 0:
            id_nonpar  = self.ids_nonpar[id]
            self.nonpar._remove_from_cluster(x_probit, self.nonpar.assignations[id_nonpar])
        self._assign_to_component(x, x_probit, id, id_nonpar = id_nonpar, reassign = True)
    
    def build_mixture(self, make_comp = True):
        """
        Instances a mixture class representing the inferred distribution
        
        Returns:
            het_mixture: the inferred distribution
        """
        # Parameter estimation
        if self.par_bounds is not None or self.shared_par_bounds is not None:
            # If no parameters are shared among models, the parameter space is separable
            if self.par_bounds is not None and self.shared_par_bounds is None:
                par_vals = []
                # Individual subspaces
                for i in range(len(self.par_models)):
                    if self.par_draws[i] is not None:
                        i_p                  = i + self.augment
                        log_total_p          = np.atleast_1d(np.sum([self.evaluated_logL[pt][i_p] for pt in range(int(np.sum(self.n_pts))) if self.assignations[pt] == i_p], axis = 0))
                        max_p                = self.par_draws[i][np.where(log_total_p == log_total_p.max())].flatten()
                        self.model_to_sample = i
                        if self.first_run:
                            initial_state = np.mean(self.par_bounds[i], axis = 1).flatten()
                        else:
                            initial_state = None
                        self.samplers[i].run_mcmc(initial_state            = max_p,
                                                  nsteps                   = self.n_steps_mcmc,
                                                  progress                 = False,
                                                  skip_initial_state_check = True,
                                                  )
                        par_vals.append(self.samplers[i].get_last_sample()[0][0])
                    else:
                        par_vals.append([])
                shared_par_vals = []
                self.first_run = False
            # In presence of shared parameters, the space is not separable anymore
            else:
                # Joint distribution
                log_total_p  = np.array([self.evaluated_logL[pt][self.assignations[pt]] for pt in range(int(np.sum(self.n_pts))) if pt in self.evaluated_logL.keys()]).sum(axis = 0)
                max_idx      = np.where(log_total_p == log_total_p.max())
                all_par      = [dd[max_idx].flatten() for dd in self.par_draws] + [self.shared_par_draws[max_idx].flatten()]
                max_p        = np.array([par for par_vec in all_par for par in par_vec])
                if self.first_run:
                    initial_state = np.mean(self.all_bounds, axis = 1).flatten()
                else:
                    initial_state = None
                self.sampler.run_mcmc(initial_state            = max_p,
                                      nsteps                   = self.n_steps_mcmc,
                                      progress                 = False,
                                      skip_initial_state_check = True,
                                      )
                pt = self.sampler.get_last_sample()[0][0]
                self.first_run = False
                # Unpack sample
                if self.par_bounds is not None:
                    par_vals = []
                    n_prev_pars = 0
                    for i, b in enumerate(self.par_bounds):
                        if self.par_draws[i] is not None:
                            par_vals.append(pt[n_prev_pars:n_prev_pars+len(b)])
                            n_prev_pars += len(b)
                        else:
                            par_vals.append([])
                else:
                    par_vals = [[] for _ in range(len(self.par_models))]
                shared_par_vals = pt[-len(self.shared_par_bounds):]
            # Build parametric models
            par_models = [par_model(m.model, list(par) + list(shared_par_vals), self.bounds, self.probit, hierarchical = True, selection_function = self.selfunc, inj_pdf = self.inj_pdf, n_total_inj = self.n_total_inj, norm = n) for m, par, n in zip(self.par_models, par_vals, self.norm)]
            # Renormalise the models in presence of selection effects
            if self.selfunc is not None:
                [m._compute_alpha_factor([p], [shared_par_vals], self.n_draws_norm) for m, p, n in zip(par_models, par_vals, self.norm) if n is None]
        # Fixed parameters
        else:
            par_models = self.par_models
        # Non-parametric model (if required)
        if self.augment:
            if self.nonpar.n_pts == 0:
                nonpar = uniform(self.bounds, self.probit)
            else:
                nonpar = self.nonpar.build_mixture(make_comp = make_comp)
            models = [nonpar_model(nonpar, self.hierarchical, self.selfunc)] + par_models
        else:
            models = par_models
        # Number of shared parameters
        if self.shared_par_bounds is not None:
            n_shared_pars = len(self.shared_par_bounds)
        else:
            n_shared_pars = 0
        if (self.selfunc is not None) and self.hierarchical:
            alphas = [m.alpha for m in par_models]
            if self.augment:
                alphas = [nonpar.alpha_factor] + alphas
            n_pts = np.nan_to_num(self.n_pts/np.array(alphas), neginf = 0., posinf = 0., nan = 0.)
        else:
            n_pts = self.n_pts
        return het_mixture(models, dirichlet(n_pts+self.gamma0).rvs()[0], self.bounds, self.augment, selfunc = self.selfunc, n_shared_pars = n_shared_pars, hierarchical = self.hierarchical)
        
class HAMM(AMM):
    """
    Class to infer a distribution given a set of events.
    Child of AMM class.
    
    Arguments:
        list-of-callables:          models
        iterable bounds:            boundaries of the rectangle over which the distribution is defined. It should be in the format [[xmin, xmax],[ymin, ymax],...]
        iterable pars:              fixed parameters of the parametric model(s)
        iterable shared_pars:       shared fixed parameters of the parametric model(s)
        iterable par_bounds:        boundaries of the allowed values for the parameters. It should be in the format [[[xmin, xmax],[ymin, ymax]],[[xmin, xmax]],...]
        iterable shared_par_bounds: boundaries of the allowed values for the shared parameters. See above for the format.
        iterable prior_pars:        IW parameters for (H)DPGMM
        callable selfunc:           selection function (if required)
        double n_draws_pars:        number of draws for MC integral over parameters
        doubne MC_draws:            number of draws for MC integral for (H)DPGMM
        double alpha0:              initial guess for concentration parameter
        np.ndarray gamma0:          Dirichlet Distribution prior
        bool probit:                whether to use the probit transformation for the (H)DPGMM
        bool augment:               whether to include the non-parametric channel
        int n_reassignments:        number of reassignments. Default is reassign 5 times the number of available samples
        np.ndarray norm:            normalisation constant for the parametric observed distributions. Use None if not available
    
    Returns:
        HAMM: instance of HAMM class
    """
    def __init__(self, models,
                       bounds,
                       pars               = None,
                       shared_pars        = None,
                       par_bounds         = None,
                       shared_par_bounds  = None,
                       prior_pars         = None,
                       selection_function = None,
                       inj_pdf            = None,
                       n_total_inj        = None,
                       n_draws_pars       = None,
                       n_draws_norm       = None,
                       MC_draws           = None,
                       alpha0             = 1.,
                       gamma0             = None,
                       probit             = False,
                       augment            = True,
                       n_reassignments    = None,
                       norm               = None,
                       n_steps_mcmc       = None,
                       ):
        # Initialise the parent class
        super().__init__(models             = models,
                         bounds             = bounds,
                         pars               = pars,
                         shared_pars        = shared_pars,
                         par_bounds         = par_bounds,
                         shared_par_bounds  = shared_par_bounds,
                         prior_pars         = None,
                         selection_function = selection_function,
                         inj_pdf            = inj_pdf,
                         n_total_inj        = n_total_inj,
                         n_draws_pars       = n_draws_pars,
                         n_draws_norm       = n_draws_norm,
                         alpha0             = alpha0,
                         gamma0             = gamma0,
                         probit             = probit,
                         augment            = augment,
                         n_reassignments    = n_reassignments,
                         norm               = norm,
                         n_steps_mcmc       = n_steps_mcmc,
                         )
        # Setting the hierarchical flag to True
        self.hierarchical = True
        for model in self.par_models:
            model.hierarchical = True
        # (H)DPGMM initialisation (if required)
        if self.augment:
            self.nonpar      = HDPGMM(bounds             = bounds,
                                      prior_pars         = prior_pars,
                                      alpha0             = alpha0,
                                      probit             = self.probit,
                                      MC_draws           = MC_draws,
                                      selection_function = selection_function,
                                      injection_pdf      = inj_pdf,
                                      total_injections   = n_total_inj,
                                      )
            self.components  = [self.nonpar] + self.par_models
        
    def _log_predictive_likelihood(self, x, i, pt_id):
        """
        Compute log likelihood of drawing the event x from component i given the events that are already assigned to that component marginalised over the component parameters.
        
        Arguments:
            dict x: event
            int i:  component id
            pt_id:  ANUBIS point ID
        
        Returns:
            double:     marginal log Likelihood
            np.ndarray: individual log Likelihood values for theta_i
        """
        # Non-parametric
        if self.augment and i == 0:
            return self._log_predictive_mixture(x), np.zeros(self.n_draws_pars)
        # Parametric
        else:
            # Fixed parameters or parameter-less model
            if (self.par_bounds is None or self.par_bounds[i - self.augment] is None) and self.shared_par_bounds is None:
                return np.log(np.mean(self.components[i].pdf(x['samples']))), np.zeros(self.n_draws_pars)
            # Marginalisation over parameters
            else:
                i_p = i - self.augment
                if not pt_id in list(self.evaluated_logL.keys()):
                    log_p = np.zeros(len(self.par_draws[i_p]))
                    if hasattr(self.components[i].alpha, '__iter__'):
                        for j, (p, sp, n) in enumerate(zip(self.par_draws[i_p], self.shared_par_draws, self.components[i].alpha)):
                            log_p[j] = np.log(np.mean(self.components[i].model(x['samples'], *p, *sp).flatten()/n))
                    else:
                        for j, (p, sp) in enumerate(zip(self.par_draws[i_p], self.shared_par_draws)):
                            log_p[j] = np.log(np.mean(self.components[i].model(x['samples'], *p, *sp).flatten()/self.components[i].alpha))
                else:
                    log_p = self.evaluated_logL[pt_id][i]
                log_total_p = np.atleast_1d(np.sum([self.evaluated_logL[pt][i] for pt in range(int(np.sum(self.n_pts))) if self.assignations[pt] == i], axis = 0))
                denom       = logsumexp(log_total_p)
                v           = logsumexp(log_p + log_total_p)
                return np.nan_to_num(v - denom, nan = -np.inf, neginf = -np.inf), log_p

    def _log_predictive_mixture(self, x, logL_x = None):
        """
        Compute log likelihood for non-parametric mixture (mixture of predictive likelihood)
        
        Arguments:
            dict x: event
        
        Returns:
            double: log Likelihood
        """
        scores = np.zeros(self.nonpar.n_cl + 1)
        if x['logL_x'] is None:
            if self.dim == 1:
                logL_x = evaluate_mixture_MC_draws_1d(self.nonpar.mu_MC, self.nonpar.sigma_MC, x['mix'].means, x['mix'].covs, x['mix'].w) - self.nonpar.log_alpha_factor
            else:
                logL_x = evaluate_mixture_MC_draws(self.nonpar.mu_MC, self.nonpar.sigma_MC, x['mix'].means, x['mix'].covs, x['mix'].w) - self.nonpar.log_alpha_factor
            x['logL_x'] = logL_x
        else:
            logL_x = x['logL_x']
        for j, i in enumerate(list(np.arange(self.nonpar.n_cl)) + ["new"]):
            if i == "new":
                ss     = None
                logL_D = np.zeros(self.nonpar.MC_draws)
            else:
                ss     = self.nonpar.mixture[i]
                logL_D = ss.logL_D
            scores[j] = logsumexp(logL_D + logL_x) - logsumexp(logL_D)
            if ss is None:
                scores[j] += np.log(self.nonpar.alpha) - np.log(self.nonpar.n_pts + self.nonpar.alpha)
            else:
                scores[j] += np.log(ss.N) - np.log(self.nonpar.n_pts + self.nonpar.alpha)
        return logsumexp(scores)
    
    def add_new_point(self, ev):
        """
        Update the probability density reconstruction adding a new event
        
        Arguments:
            np.ndarray ev: event
        """
        x = {'samples': ev[0],
             'mix': np.random.choice(ev[1]),
             'logL_x': None,
             }
        self.stored_pts[int(np.sum(self.n_pts))] = x
        self._assign_to_component(x, pt_id = int(np.sum(self.n_pts)))

    def _assign_to_component(self, x, pt_id, id_nonpar = None, reassign = False):
        """
        Assign the event x to an existing cluster or to a new cluster according to the marginal distribution of cluster assignment.
        
        Arguments:
            dict x:        event
            int pt_id:     point id
            int id_nonpar: FIGARO id for the point
            bool reassign: wheter the point is new or is being reassigned
        """
        scores             = np.zeros(self.n_components)
        vals               = np.zeros(shape = (self.n_components, self.n_draws_pars))
        for i in range(self.n_components):
            score, vals[i] = self._log_predictive_likelihood(x, i, pt_id)
            scores[i]      = score + np.log(self.gamma0[i] + self.n_pts[i])
        if np.sum(self.n_pts) == 0 and self.augment:
            id = 0
        else:
            if len(scores) > 1:
                scores             = np.exp(scores - logsumexp(scores))
            else:
                scores = [1.]
            id                 = np.random.choice(self.n_components, p = scores)
        self.n_pts[id]    += 1
        self.weights       = (self.n_pts + self.gamma0)/np.sum(self.n_pts + self.gamma0)
        # If DPGMM, updates mixture
        if self.augment and id == 0:
            if id_nonpar is None:
                self.ids_nonpar[int(pt_id)] = len(list(self.nonpar.stored_pts.keys()))
                self.nonpar.add_new_point([x['mix']])
            else:
                self._reassign_point_nonpar(x['mix'], id_nonpar, x['logL_x'])
        # Parameter estimation
        elif self.par_bounds is not None:
            self.evaluated_logL[pt_id] = vals
        self.assignations[pt_id]       = int(id)
    
    def _reassign_point(self, id):
        """
        Update the probability density reconstruction reassigining an existing sample
        
        Arguments:
            id: sample id
        """
        x                     = self.stored_pts[id]
        cid                   = self.assignations[id]
        id_nonpar             = None
        self.n_pts[cid]      -= 1
        self.assignations[id] = None
        if self.augment and cid == 0:
            id_nonpar  = self.ids_nonpar[id]
            self.nonpar._remove_from_cluster(x, self.nonpar.assignations[id_nonpar], self.nonpar.evaluated_logL[id_nonpar])
        self._assign_to_component(x, id, id_nonpar = id_nonpar, reassign = True)

    def _reassign_point_nonpar(self, x, id_nonpar, logL_x):
        """
        Update the probability density reconstruction reassigining an existing sample
        
        Arguments:
            x:         sample
            id_nonpar: FIGARO id for the point
            logL_x:    evaluated log likelihood
        """
        self.nonpar._assign_to_cluster(x, id_nonpar, logL_x = logL_x)
        self.nonpar.alpha = _update_alpha(self.nonpar.alpha, self.nonpar.n_pts, (np.array(self.nonpar.N_list) > 0).sum(), self.nonpar.alpha_0)
