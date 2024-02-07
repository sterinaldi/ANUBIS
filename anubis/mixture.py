import numpy as np
from scipy.stats import dirichlet

from figaro.mixture import DPGMM, HDPGMM, mixture, _update_alpha
from figaro.decorators import probit
from figaro.utils import rejection_sampler
from figaro._likelihood import evaluate_mixture_MC_draws, evaluate_mixture_MC_draws_1d
from figaro._numba_functions import logsumexp_jit

np.seterr(divide = 'ignore')

def uniform(x, v):
    return np.ones(np.shape(x))*v

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
                       selfunc = None,
                       norm = None,
                       ):
        self.model    = model
        self.pars     = pars
        self.bounds   = np.atleast_2d(bounds)
        self.dim      = len(self.bounds)
        self.probit   = probit
        self.selfunc  = selfunc
        if norm is not None:
            self.norm = norm
        else:
            self.norm = 1.
    
    def _selfunc(func):
        """
        Applies the selection function to to the intrinsic distribution.
        """
        def observed_model(self, x, *args):
            if self.selfunc is not None:
                return func(self, x, *args)*self.selfunc(x)
            else:
                return func(self, x, *args)
        return observed_model
    
    def __call__(self, x):
        return self.pdf(x)
    
    def _compute_normalisation(self, pars, shared_pars, n_draws):
        """
        Computes the normalisation of the product p_intr(x|lambda)p_det(x) via monte carlo approximation
        
        Arguments:
            np.ndarray pars:        parameters of the distribution
            np.ndarray shared pars: shared parameters of the distribution
            int n_draws:            number of draws for the MC integral
        """
        self.norm     = None
        volume        = np.prod(np.diff(self.bounds, axis = 1))
        samples       = rejection_sampler(int(n_draws), self.selfunc, self.bounds)
        self.sf_norm  = np.mean(self.selfunc(np.random.uniform(low = self.bounds[:,0], high = self.bounds[:,1], size = (n_draws, len(self.bounds))))*volume)
        if pars is not None:
            self.norm = np.atleast_1d([np.mean(self.model(samples, *p, *sp).flatten()*self.sf_norm) for p, sp in zip(pars, shared_pars)])
            self.norm[self.norm == 0.] = np.inf
        else:
            self.norm = np.atleast_1d(np.mean(self.pdf_intrinsic(samples))*self.sf_norm)
        if len(self.norm) == 1:
            self.norm = self.norm[0]
    
    @_selfunc
    def pdf(self, x):
        """
        pdf of the observed distribution
        
        Arguments:
            np.ndarray x: point to evaluate the mixture at
        
        Returns:
            np.ndarray: p_intr.pdf(x)*p_obs(x)/norm
        """
        return self.model(x, *self.pars)/self.norm
    
    def pdf_intrinsic(self, x):
        """
        pdf of the intrinsic distribution
        
        Arguments:
            np.ndarray x: point to evaluate the mixture at
        
        Returns:
            np.ndarray: p_intr.pdf(x)
        """
        return self.model(x, *self.pars)
    
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
        if self.norm is not None:
            if hasattr(self.norm, '__iter__'):
                return np.array([self._model(x, p, sp)/n for p, sp, n in zip(pars, shared_pars, self.norm)])
            else:
                return np.array([self._model(x, p, sp)/self.norm for p, sp in zip(pars, shared_pars)])
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
            np.ndarray: p_intr.pdf(x|theta)*p_obs(x)/norm
        """
        return self.model(x, *pars, *shared_pars).flatten()

class het_mixture:
    """
    Class to store a single draw from HMM.
    
    Arguments:
        list-of-callables models: models in the mixture
        np.ndarray:               weights
        np.ndarray bounds:        bounds (FIGARO)
        bool augment:             whether the model includes a non-parametric augmentation
        callable selfunc:         selection function
        int n_draws:              number of draws for normalisation
        int n_shared_pars:        number of shared parameters among models
        
    Returns:
        het_mixture: instance of het_mixture class
    """
    def __init__(self, models,
                       weights,
                       bounds,
                       augment,
                       selfunc       = None,
                       n_draws       = 1e4,
                       n_shared_pars = 0
                       ):
        # Components
        self.models        = models
        self.weights       = weights
        self.bounds        = np.atleast_2d(bounds)
        self.dim           = len(self.bounds)
        self.augment       = augment
        self.selfunc       = selfunc
        self.n_draws       = int(n_draws)
        self.n_shared_pars = int(n_shared_pars)
        # Weights and normalisation
        if self.selfunc is not None:
            self.intrinsic_weights = [wi/mi.norm for wi, mi in zip(self.weights[self.augment:], self.models[self.augment:])]
            if self.augment:
                if isinstance(self.models[0], mixture):
                    self.intrinsic_weights = [self.weights[0]*np.mean(1./self.selfunc(self.models[0].rvs(self.n_draws)))] + self.intrinsic_weights
                else:
                    self.intrinsic_weights = [self.weights[0]*np.mean(1./self.selfunc(np.random.uniform(low = self.bounds[:,0], high = self.bounds[:,1], size = (self.n_draws, len(self.bounds)))))] + self.intrinsic_weights
            self.intrinsic_weights = np.array(self.intrinsic_weights/np.sum(self.intrinsic_weights))
        else:
            self.intrinsic_weights = self.weights
        self.norm_intrinsic = np.sum(self.intrinsic_weights[self.augment:])
        if self.augment:
            self.probit = self.models[0].probit
        else:
            self.probit = False
    
    def __call__(self, x):
        return self.pdf(x)
    
    def pdf(self, x):
        """
        Evaluate mixture at point(s) x (observed)
        
        Arguments:
            np.ndarray x: point(s) to evaluate the mixture at
        
        Returns:
            np.ndarray: het_mixture.pdf(x)
        """
        return np.array([wi*mi.pdf(x) for wi, mi in zip(self.weights, self.models)]).sum(axis = 0)
    
    def logpdf(self, x):
        """
        Evaluate log mixture at point(s) x (observed)
        
        Arguments:
            np.ndarray x: point(s) to evaluate the mixture at
        
        Returns:
            np.ndarray: het_mixture.logpdf(x)
        """
        return np.log(self.pdf(x))

    def pdf_intrinsic(self, x):
        """
        Evaluate mixture at point(s) x (intrinsic)
        
        Arguments:
            np.ndarray x: point(s) to evaluate the mixture at
        
        Returns:
            np.ndarray: het_mixture.pdf(x)
        """
        return np.array([wi*mi.pdf_intrinsic(x)/self.norm_intrinsic for wi, mi in zip(self.intrinsic_weights[self.augment:], self.models[self.augment:])]).sum(axis = 0)
    
    def logpdf_intrinsic(self, x):
        """
        Evaluate log mixture at point(s) x (intrinsic)
        
        Arguments:
            np.ndarray x: point(s) to evaluate the mixture at
        
        Returns:
            np.ndarray: het_mixture.logpdf(x)
        """
        return np.log(self.pdf_intrinsic(x))

#-----------------#
# Inference class #
#-----------------#

class HMM:
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
    
    Returns:
        HMM: instance of HMM class
    """
    def __init__(self, models,
                       bounds,
                       pars              = None,
                       shared_pars       = None,
                       par_bounds        = None,
                       shared_par_bounds = None,
                       prior_pars        = None,
                       selfunc           = None,
                       n_draws_pars      = 1e3,
                       n_draws_norm      = 1e4,
                       alpha0            = 1.,
                       gamma0            = None,
                       probit            = False,
                       augment           = True,
                       n_reassignments   = None,
                       norm              = None,
                       ):
        # Settings
        self.bounds       = np.atleast_2d(bounds)
        self.dim          = len(self.bounds)
        self.probit       = probit
        self.augment      = augment
        self.selfunc      = selfunc
        # Parametric models
        if pars is None:
            pars = [[] for _ in models]
            self.n_draws_pars = 0
        if shared_pars is None:
            shared_pars = []
        if par_bounds is not None:
            self.par_bounds = [np.atleast_2d(pb) if pb is not None else None for pb in par_bounds]
            self.n_draws_pars = int(n_draws_pars)
        else:
            self.par_bounds = None
        if shared_par_bounds is not None:
            self.shared_par_bounds = np.atleast_2d(shared_par_bounds)
            self.n_draws_pars = int(n_draws_pars)
        else:
            self.shared_par_bounds = None
        if self.selfunc is not None:
            self.n_draws_norm = int(n_draws_norm)
        if norm is None:
            self.norm   = [None for _ in models]
        else:
            self.norm   = norm
        self.par_models = [par_model(mod, list(p) + list(shared_pars), bounds, probit, selfunc, norm = n) for mod, p, n in zip(models, pars, self.norm)]
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
        self.n_pts        = np.zeros(self.n_components)
        self.weights      = self.gamma0/np.sum(self.gamma0)
        self.stored_pts   = {}
        self.assignations = {}
        # Draw new parameter realisations
        if self.par_bounds is not None or self.shared_par_bounds is not None:
            self.evaluated_logL       = {}
            if self.par_bounds is not None:
                self.par_draws        = [np.random.uniform(low = b[:,0], high = b[:,1], size = (self.n_draws_pars, len(b))) if b is not None else None for b in self.par_bounds]
            else:
                self.par_draws        = [[[] for _ in range(self.n_draws_pars)] for _ in range(len(self.components[self.augment:]))]
            if self.shared_par_bounds is not None:
                self.shared_par_draws = np.random.uniform(low = self.shared_par_bounds[:,0], high = self.shared_par_bounds[:,1], size = (self.n_draws_pars, len(self.shared_par_bounds)))
            else:
                self.shared_par_draws = [[] for _ in range(self.n_draws_pars)]
        if self.selfunc is not None:
            [m._compute_normalisation(p, self.shared_par_draws, self.n_draws_norm) for m, p, n in zip(self.components[self.augment:], self.par_draws, self.norm) if n is None]
        if self.augment:
            self.nonpar.initialise()
            self.ids_nonpar = {}

    def _assign_to_component(self, x, pt_id, id_nonpar = None, reassign = False):
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
        scores             = np.exp(scores - logsumexp_jit(scores))
        id                 = np.random.choice(self.n_components, p = scores)
        self.n_pts[id]    += 1
        self.weights       = (self.n_pts + self.gamma0)/np.sum(self.n_pts + self.gamma0)
        # If DPGMM, updates mixture
        if self.augment and id == 0:
            if id_nonpar is None:
                self.ids_nonpar[int(pt_id)] = len(list(self.nonpar.stored_pts.keys()))
                self.nonpar.add_new_point(x)
            else:
                self._reassign_point_nonpar(x, id_nonpar)
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
                denom       = logsumexp_jit(log_total_p) - np.log(self.n_draws_pars)
                v           = logsumexp_jit(log_p + log_total_p) - np.log(self.n_draws_pars)
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
                scores[j] = self.nonpar._log_predictive_likelihood(x, ss)
            if ss is None:
                scores[j] += np.log(self.nonpar.alpha) - np.log(self.nonpar.n_pts + self.nonpar.alpha)
            elif ss.N < 1:
                scores[j]  = -np.inf
            else:
                scores[j] += np.log(ss.N) - np.log(self.nonpar.n_pts + self.nonpar.alpha)
        return logsumexp_jit(scores)
    
    def add_new_point(self, x):
        """
        Update the probability density reconstruction adding a new sample
        
        Arguments:
            np.ndarray x: sample
        """
        self.stored_pts[int(np.sum(self.n_pts))] = np.atleast_2d(x)
        self._assign_to_component(np.atleast_2d(x), pt_id = int(np.sum(self.n_pts)))
    
    def density_from_samples(self, samples):
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
        for id in np.random.choice(int(np.sum(self.n_pts)), size = n_reassignments, replace = True):
            self._reassign_point(int(id))
        # Reassign all points once
        for id in range(int(np.sum(self.n_pts))):
            self._reassign_point(int(id))
        d = self.build_mixture()
        self.initialise()
        return d
    
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
            self.nonpar._remove_from_cluster(x, self.nonpar.assignations[id_nonpar])
        self._assign_to_component(x, id, id_nonpar = id_nonpar, reassign = True)
    
    def build_mixture(self):
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
                        pars        = self.par_draws[i].T
                        i_p         = i + self.augment
                        log_total_p = np.atleast_1d(np.sum([self.evaluated_logL[pt][i_p] for pt in range(int(np.sum(self.n_pts))) if self.assignations[pt] == i_p], axis = 0))
                        vals        = np.exp(log_total_p - logsumexp_jit(log_total_p))
                        par_vals.append(np.atleast_1d([np.random.choice(p, p = vals) for p in pars]))
                    else:
                        par_vals.append([])
                shared_par_vals = []
            # In presence of shared parameters, the space is not separable anymore
            else:
                # Joint distribution
                log_total_p  = np.array([self.evaluated_logL[pt][self.assignations[pt]] for pt in range(int(np.sum(self.n_pts)))]).sum(0)
                vals         = np.exp(log_total_p - logsumexp_jit(log_total_p))
                id           = np.random.choice(self.n_draws_pars, p = vals)
                if self.par_bounds is not None:
                    par_vals = [self.par_draws[i][id].T if self.par_draws[i] is not None else [] for i in range(len(self.par_models))]
                else:
                    par_vals = [[] for _ in range(len(self.par_models))]
                shared_par_vals = self.shared_par_draws[id]
            par_models = [par_model(m.model, list(par) + list(shared_par_vals), self.bounds, self.probit, self.selfunc, norm = n) for m, par, n in zip(self.par_models, par_vals, self.norm)]
            # Renormalise the models in presence of selection effects
            if self.selfunc is not None:
                [m._compute_normalisation([p], [shared_par_vals], self.n_draws_norm) for m, p, n in zip(par_models, par_vals, self.norm) if n is None]
        # Fixed parameters
        else:
            par_models = self.par_models
        # Non-parametric model (if required)
        if self.augment:
            if self.nonpar.n_pts == 0:
                models = [par_model(uniform, [1./self.volume], self.bounds, self.probit)] + par_models
            else:
                nonpar = self.nonpar.build_mixture()
                models = [nonpar] + par_models
        else:
            models = par_models
        # Number of shared parameters
        if self.shared_par_bounds is not None:
            n_shared_pars = len(self.shared_par_bounds)
        else:
            n_shared_pars = 0
        return het_mixture(models, dirichlet(self.n_pts+self.gamma0).rvs()[0], self.bounds, self.augment, selfunc = self.selfunc, n_shared_pars = n_shared_pars)
        
class HierHMM(HMM):
    """
    Class to infer a distribution given a set of events.
    Child of HMM class.
    
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
        HierHMM: instance of HierHMM class
    """
    def __init__(self, models,
                       bounds,
                       pars              = None,
                       shared_pars       = None,
                       par_bounds        = None,
                       shared_par_bounds = None,
                       prior_pars        = None,
                       selfunc           = None,
                       n_draws_pars      = 1e3,
                       MC_draws          = None,
                       alpha0            = 1.,
                       gamma0            = None,
                       probit            = False,
                       augment           = True,
                       n_reassignments   = None,
                       norm              = None,
                       ):
        # Initialise the parent class
        super().__init__(models            = models,
                         bounds            = bounds,
                         pars              = pars,
                         shared_pars       = shared_pars,
                         par_bounds        = par_bounds,
                         shared_par_bounds = shared_par_bounds,
                         prior_pars        = None,
                         selfunc           = selfunc,
                         n_draws_pars      = n_draws_pars,
                         alpha0            = alpha0,
                         gamma0            = gamma0,
                         probit            = probit,
                         augment           = augment,
                         n_reassignments   = n_reassignments,
                         norm              = norm,
                         )
        # (H)DPGMM initialisation (if required)
        if self.augment:
            self.nonpar      = HDPGMM(bounds     = bounds,
                                      prior_pars = prior_pars,
                                      alpha0     = alpha0,
                                      probit     = self.probit,
                                      MC_draws   = MC_draws,
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
                if self.selfunc is not None:
                    sf = x['selfunc']
                else:
                    sf = 1.
                if not pt_id in list(self.evaluated_logL.keys()):
                    log_p = np.zeros(len(self.par_draws[i_p]))
                    if hasattr(self.components[i].norm, '__iter__'):
                        for j, (p, sp, n) in enumerate(zip(self.par_draws[i_p], self.shared_par_draws, self.components[i].norm)):
                            log_p[j] = np.log(np.mean(self.components[i].model(x['samples'], *p, *sp).flatten()*sf/n))
                    else:
                        for j, (p, sp) in enumerate(zip(self.par_draws[i_p], self.shared_par_draws)):
                            log_p[j] = np.log(np.mean(self.components[i].model(x['samples'], *p, *sp).flatten()*sf/self.components[i].norm))
                else:
                    log_p = self.evaluated_logL[pt_id][i]
                log_total_p = np.atleast_1d(np.sum([self.evaluated_logL[pt][i] for pt in range(int(np.sum(self.n_pts))) if self.assignations[pt] == i], axis = 0))
                denom       = logsumexp_jit(log_total_p)
                v           = logsumexp_jit(log_p + log_total_p)
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
                logL_x = evaluate_mixture_MC_draws_1d(self.nonpar.mu_MC, self.nonpar.sigma_MC, x['mix'].means, x['mix'].covs, x['mix'].w)
            else:
                logL_x = evaluate_mixture_MC_draws(self.nonpar.mu_MC, self.nonpar.sigma_MC, x['mix'].means, x['mix'].covs, x['mix'].w)
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
            scores[j] = logsumexp_jit(logL_D + logL_x) - logsumexp_jit(logL_D)
            if ss is None:
                scores[j] += np.log(self.nonpar.alpha) - np.log(self.nonpar.n_pts + self.nonpar.alpha)
            else:
                scores[j] += np.log(ss.N) - np.log(self.nonpar.n_pts + self.nonpar.alpha)
        return logsumexp_jit(scores)
    
    def add_new_point(self, ev):
        """
        Update the probability density reconstruction adding a new event
        
        Arguments:
            np.ndarray ev: event
        """
        x = {'samples': ev[0][np.random.choice(len(ev[0]), size = 1000)],
             'mix': np.random.choice(ev[1]),
             'logL_x': None,
             }
        if self.selfunc is not None:
            x['selfunc'] = self.selfunc(x['samples'])
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
        scores             = np.exp(scores - logsumexp_jit(scores))
        id                 = np.random.choice(self.n_components, p = scores)
        self.n_pts[id]    += 1
        self.weights       = (self.n_pts + self.gamma0)/np.sum(self.n_pts + self.gamma0)
        # If DPGMM, updates mixture
        if self.augment and id == 0:
            if id_nonpar is None:
                self.ids_nonpar[int(pt_id)] = len(list(self.nonpar.stored_pts.keys()))
                self.nonpar.add_new_point([x['mix']])
            else:
                self._reassign_point_nonpar(x['mix'], id_nonpar)
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
