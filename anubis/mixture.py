import numpy as np

from figaro.mixture import DPGMM, HDPGMM
from figaro.decorators import probit
from figaro.transform import probit_logJ
from figaro.likelihood import evaluate_mixture_MC_draws, evaluate_mixture_MC_draws_1d
from scipy.special import logsumexp

np.seterr(divide = 'ignore')

def uniform(x, v):
    return np.ones(np.shape(x))*v

class par_model:
    """
    Class to store a parametric model.
    
    Arguments:
        :callable model:    model pdf
        :iterable pars:     parameters of the model
        :np.ndarray bounds: bounds (FIGARO)
        :bool probit:       whether to use the probit transformation or not (FIGARO compatibility)
    
    Returns:
        :par_model: instance of model class
    """
    def __init__(self, model, pars, bounds, probit):
        self.model  = model
        self.pars   = pars
        self.bounds = np.atleast_2d(bounds)
        self.dim    = len(self.bounds)
        self.probit = probit
        
    def __call__(self, x):
        return self.pdf(x)

    def pdf(self, x):
        return self.model(x, *self.pars)
    
    def pdf_pars(self, x, pars):
        return self.model(x, *pars)

class het_mixture:
    """
    Class to store a single draw from HMM.
    
    Arguments:
        :list-of-callables models: models in the mixture
        :iterable pars:            list of model parameters. Must be formatted as [[p1, p2, ...], [q1, q2, ...], ...]. Add empty list for no parameters.
        :np.ndarray:               weights
        :np.ndarray bounds:        bounds (FIGARO)
        
    Returns:
        :het_mixture: instance of het_mixture class
    """
    def __init__(self, models, weights, bounds):
        self.models  = models
        self.weights = weights
        self.bounds  = np.atleast_2d(bounds)
        self.dim     = len(self.bounds)
        self.probit  = models[0].probit
    
    def __call__(self, x):
        return self.pdf(x)
    
    def pdf(self, x):
        return np.array([wi*mi.pdf(x) for wi, mi in zip(self.weights, self.models)]).sum(axis = 0)

#-----------------#
# Inference class #
#-----------------#

class HMM:
    """
    Class to infer a distribution given a set of samples.
    
    Arguments:
        :list-of-callbles:    models
        :iterable bounds:     boundaries of the rectangle over which the distribution is defined. It should be in the format [[xmin, xmax],[ymin, ymax],...]
        :iterable prior_pars: NIW prior parameters (k, L, nu, mu)
        :iterable par_bounds: boundaries of the allowed values for the parameters. It should be in the format [[[xmin, xmax],[ymin, ymax]],[[xmin, xmax]],...]
        :double n_draws:      number of draws for MC integral over parameters
        :double alpha0:       initial guess for concentration parameter
        :np.array gamma0:     Dirichlet Distribution prior
    
    Returns:
        :HMM: instance of HMM class
    """
    def __init__(self, models,
                       bounds,
                       pars = None,
                       par_bounds = None,
                       prior_pars = None,
                       n_draws = 1e3,
                       alpha0 = 1.,
                       gamma0 = None,
                       probit = False,
                       ):
                       
        if pars is None:
            pars = [[] for _ in models]
            self.n_draws = 0
        if par_bounds is not None:
            self.par_bounds = np.atleast_3d(par_bounds).reshape(len(models), -1, 2)
            self.n_draws = int(n_draws)
        else:
            self.par_bounds = None
                
        self.par_models = [par_model(mod, p, bounds, probit) for mod, p in zip(models, pars)]
        if self.par_bounds is not None:
            self.par_draws   = np.atleast_2d([np.random.uniform(low = b[:,0], high = b[:,1], size = (self.n_draws, len(b))) for b in self.par_bounds])
            self.log_total_p = np.array([np.zeros(self.n_draws) for _ in range(len(self.par_models))])
            
        self.bounds       = np.atleast_2d(bounds)
        self.dim          = len(self.bounds)
        self.probit       = probit
        self.DPGMM        = DPGMM(bounds = bounds, prior_pars = prior_pars, alpha0 = alpha0, probit = self.probit)
        self.volume       = np.prod(np.diff(self.DPGMM.bounds, axis = 1))
        self.components   = [self.DPGMM] + self.par_models
        self.n_components = len(models) + 1
        
        self.n_pts = np.zeros(self.n_components)
         
        if gamma0 is None:
            # Symmetric prior
            self.gamma0  = np.ones(self.n_components)
        else:
            gamma0 = np.array(gamma0)
            if len(gamma0) == self.n_components:
                self.gamma0 = gamma0
            else:
                raise Exception("gamma0 must be an array with {0} components.".format(self.n_components))
        self.weights = self.gamma0/np.sum(self.gamma0)
    
    def __call__(self, x):
        return self.pdf(x)
    
    def initialise(self):
        self.n_pts     = np.zeros(self.n_components)
        self.weights   = self.gamma0/np.sum(self.gamma0)
        if self.par_bounds is not None:
            self.par_draws   = np.atleast_2d([np.random.uniform(low = b[:,0], high = b[:,1], size = (self.n_draws, len(b))) for b in self.par_bounds])
            self.log_total_p = np.array([np.zeros(self.n_draws) for _ in range(len(self.par_models))])
    
    def _assign_to_component(self, x):
        scores = np.zeros(self.n_components)
        vals = np.zeros(shape = (self.n_components, self.n_draws))
        for i in range(self.n_components):
            score, vals[i] = self._log_predictive_likelihood(x, i)
            scores[i] = score + np.log(self.gamma0[i] + self.n_pts[i])
        scores = np.exp(scores - logsumexp(scores))
        id = np.random.choice(self.n_components, p = scores)
        self.n_pts[id] += 1
        self.weights = (self.n_pts + self.gamma0)/np.sum(self.n_pts + self.gamma0)
        # If DPGMM, updates mixture
        if id == 0:
            self._add_point_to_mixture(x)
        # Parameter estimation
        elif self.par_bounds is not None:
            self.log_total_p[id-1] += vals[id]
    
    def _add_point_to_mixture(self, x):
        self.DPGMM.add_new_point(x)
    
    def _log_predictive_likelihood(self, x, i):
        if i == 0:
            return self._log_predictive_mixture(x), np.zeros(self.n_draws)
        else:
            if self.par_bounds is None:
                return np.log(self.components[i].pdf(x)), np.zeros(self.n_draws)
            else:
                log_p = np.log([self.components[i].pdf_pars(x, pars) for pars in self.par_draws[i-1]]).flatten()
                v     = logsumexp(log_p + self.log_total_p[i-1]) - logsumexp(self.log_total_p[i-1])
                return v, log_p
    
    @probit
    def _log_predictive_mixture(self, x):
        scores = np.zeros(self.DPGMM.n_cl + 1)
        for j, i in enumerate(list(np.arange(self.DPGMM.n_cl)) + ["new"]):
            if i == "new":
                ss = "new"
            else:
                ss = self.DPGMM.mixture[i]
            scores[j] = self.DPGMM._log_predictive_likelihood(x, ss)
            if ss == "new":
                scores[j] += np.log(self.DPGMM.alpha) - np.log(self.DPGMM.n_pts + self.DPGMM.alpha)
            else:
                scores[j] += np.log(ss.N) - np.log(self.DPGMM.n_pts + self.DPGMM.alpha)
        return logsumexp(scores) - probit_logJ(x, self.bounds, self.probit)
    
    def add_new_point(self, x):
        self._assign_to_component(np.atleast_2d(x))
    
    def density_from_samples(self, samples):
        np.random.shuffle(samples)
        for s in samples:
            self.add_new_point(s)
        d = self.build_mixture()
        self.DPGMM.initialise()
        self.initialise()
        return d
    
    def pdf(self, x):
        return np.array([wi*mi.pdf(x) for wi, mi in zip(self.weights, self.models)]).sum(axis = 0)
    
    def build_mixture(self):
        if self.par_bounds is not None:
            par_vals = []
            for i in range(len(self.par_models)):
                pars     = self.par_draws[i].T
                vals     = np.exp(self.log_total_p[i] - logsumexp(self.log_total_p[i]))
                par_vals.append(np.atleast_1d([np.random.choice(p, p = vals) for p in pars]))
            par_models = [par_model(m.model, par, self.bounds, self.probit) for m, par in zip(self.par_models, par_vals)]
        else:
            par_models = self.par_models
        
        if self.DPGMM.n_pts == 0:
            models = [par_model(uniform, [1./self.volume], self.bounds, self.probit)] + par_models
        else:
            models = [self.DPGMM.build_mixture()] + par_models
        return het_mixture(models, self.weights, self.bounds)
        
class HierHMM:
    """
    Class to infer a distribution given a set of events.
    Child of HMM class.
    
    Arguments:
        :list-of-callbles:    models
        :iterable bounds:     boundaries of the rectangle over which the distribution is defined. It should be in the format [[xmin, xmax],[ymin, ymax],...]
        :iterable prior_pars: NIW prior parameters (k, L, nu, mu)
        :iterable par_bounds: boundaries of the allowed values for the parameters. It should be in the format [[[xmin, xmax],[ymin, ymax]],[[xmin, xmax]],...]
        :double n_draws_pars: number of draws for MC integral over parameters
        :double n_draws_evs:  number of draws for MC integral over events
        :double alpha0:       initial guess for concentration parameter
        :np.array gamma0:     Dirichlet Distribution prior
    
    Returns:
        :HierHMM: instance of HierHMM class
    """
    def __init__(self, models,
                       bounds,
                       pars = None,
                       par_bounds = None,
                       prior_pars = None,
                       n_draws_pars = 1e3,
                       n_draws_evs = 1e3,
                       alpha0 = 1.,
                       gamma0 = None,
                       probit = False,
                       ):
        
        super().__init__(models = models,
                         bounds = bounds,
                         pars = pars,
                         par_bounds = par_bounds,
                         prior_pars = prior_pars,
                         n_draws = n_draws_pars,
                         alpha0 = alpha0,
                         gamma0 = gamma0,
                         probit = probit,
                         )
        
        self.DPGMM      = HDPGMM(bounds = bounds, prior_pars = prior_pars, alpha0 = alpha0, probit = self.probit)
        self.components = [self.DPGMM] + self.par_models

    def _add_point_to_mixture(self, x):
        self.DPGMM.add_new_point([x])
        
    def _log_predictive_likelihood(self, x, i):
        if i == 0:
            return self._log_predictive_mixture(x), np.zeros(self.n_draws)
        else:
            samps = x.rvs(self.n_draws_evs)
            if self.par_bounds is None:
                return np.log(np.mean(self.components[i].pdf(samps))), np.zeros(self.n_draws)
            else:
                log_p = np.log([np.mean(self.components[i].pdf_pars(samps, pars)) for pars in self.par_draws[i-1]]).flatten()
                v     = logsumexp(log_p + self.log_total_p[i-1]) - logsumexp(self.log_total_p[i-1])
                return v, log_p

    @probit
    def _log_predictive_mixture(self, x):
        scores = np.zeros(self.DPGMM.n_cl + 1)
        if self.dim == 1:
            logL_x = evaluate_mixture_MC_draws_1d(self.DPGMM.mu_MC, self.DPGMM.sigma_MC, x.means, x.covs, x.w)
        else:
            logL_x = evaluate_mixture_MC_draws(self.DPGMM.mu_MC, self.DPGMM.sigma_MC, x.means, x.covs, x.w)
        for j, i in list(np.arange(self.DPGMM.n_cl)) + ["new"]:
            if i == "new":
                ss     = "new"
                logL_D = np.zeros(self.DPGMM.MC_draws)
            else:
                ss     = self.DPGMM.mixture[i]
                logL_D = ss.logL_D
            scores[j] = logsumexp(logL_D + logL_x) - logsumexp(logL_D)
            if ss == "new":
                scores[j] += np.log(self.DPGMM.alpha) - np.log(self.DPGMM.n_pts + self.DPGMM.alpha)
            else:
                scores[j] += np.log(ss.N) - np.log(self.DPGMM.n_pts + self.DPGMM.alpha)
        return logsumexp(scores)
    
    def add_new_point(self, ev):
        x = np.random.choice(ev)
        self._assign_to_component(x)
