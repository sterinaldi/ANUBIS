import numpy as np
from figaro.mixture import DPGMM
from scipy.special import logsumexp

def uniform(x, v):
    return v

class par_model:
    """
    Class to store a parametric model.
    
    Arguments:
        :callable model: pdf of the model
        :iterable pars:  parameters of the model
    
    Returns:
        :mixture: instance of model class
    """
    def __init__(self, model, pars):
        self.model = model
        self.pars  = pars
    
    def __call__(self, x):
        return self.pdf(x)

    def pdf(self, x):
        return self.model(x, *pars)

class mixture:
    """
    Class to store a single draw from HMM.
    
    Arguments:
        :list-of-callables models: models in the mixture
        :iterable pars:            list of model parameters. Must be formatted as [[p1, p2, ...], [q1, q2, ...], ...]. Add empty list for no parameters.
        :np.ndarray:               weights
        
    Returns:
        :mixture: instance of mixture class
    """
    def __init__(self, models, weights):
        self.models  = models
        self.weights = weights
    
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
        :list-of-callbles: models
        :iterable bounds:  boundaries of the rectangle over which the distribution is defined. It should be in the format [[xmin, xmax],[ymin, ymax],...]
        :iterable prior_pars: NIW prior parameters (k, L, nu, mu)
        :double alpha0:       initial guess for concentration parameter
        :np.array gamma0:     Dirichlet Distribution prior
    
    Returns:
        :HMM: instance of HMM class
    """
    def __init__(self, models,
                       bounds,
                       pars = None,
                       prior_pars = None,
                       alpha0 = 1.,
                       gamma0 = None,
                       ):
        if pars == None:
            pars = [[] for _ in models]
        self.par_models = [par_model(mod, p) for mod, p in zip(models, pars)]
        self.DPGMM  = DPGMM(bounds = bounds, prior_pars = prior_pars, alpha0 = alpha0)
        
        self.components   = [self.DPGMM] + self.par_models
        self.n_components = len(models) + 1
        
        self.n_pts = np.zeros(self.n_models)
         
        if gamma0 is None:
            self.gamma_0  = np.ones(self.n_components)/self.n_components
        else:
            gamma0 = np.array([gamma0])
            if len(gamma0) == self.n_components:
                self.gamma0 = gamma0
            else:
                raise Exception("gamma0 must be an array with {n} components.".format(self.n_components))
        self.weights = self.gamma0
    
    def __call__(self, x):
        return self.pdf(x)
    
    def initialise(self):
        self.n_pts = np.zeros(self.n_models)
        self.w     = self.gamma0
    
    def _assign_to_component(self, x):
        scores = np.zeros(self.n_models)
        for i in enumerate(self.components):
            scores[i]  = self._log_predictive_likelihood(x, i)
            scores[i] += self.gamma0[i] + self.n_pts[i]
        scores = np.exp(scores - logsumexp(scores))
        id = np.random.choice(self.n_components, p = scores)
        self.n_pts[id] += 1
        self.weights = (self.n_pts + self.gamma0)/np.sum(self.n_pts + self.gamma0)
        # If DPGMM, updates mixture
        if id == 0:
            self.DPGMM.add_new_point(x)
    
    def _log_predictive_likelihood(self, x, i):
        if i == 0:
            scores = {}
            for i in list(np.arange(self.DPGMM.n_cl)) + ["new"]:
                if i == "new":
                    ss = "new"
                else:
                    ss = self.DPGMM.mixture[i]
                scores[i] = self.DPGMM._log_predictive_likelihood(x, ss)
                if ss == "new":
                    scores[i] += np.log(self.DPGMM.alpha) - np.log(self.DPGMM.n_pts + self.DPGMM.alpha)
                else:
                    scores[i] += np.log(ss.N) - np.log(self.DPGMM.n_pts + self.DPGMM.alpha)
            scores = [score for score in scores.values()]
            return logsumexp(scores)
        else:
            return np.log(self.components[i](x))
    
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
        if self.DPGMM.n_pts == 0:
            models = [par_models(uniform, [np.prod(np.diff(self.DPGMM.bounds, axis = 1))])] + self.par_models
        else:
            models = [self.DPGMM.build_mixture()] + self.par_models
        return mixture(models, self.weights)
        
