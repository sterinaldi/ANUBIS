import numpy as np

import optparse
import importlib
import warnings

from pathlib import Path
from tqdm import tqdm

from figaro.mixture import DPGMM, HDPGMM
from figaro.transform import transform_to_probit
from figaro.utils import save_options, load_options, get_priors
from figaro.plot import plot_median_cr, plot_multidim
from figaro.load import load_data, load_single_event, load_selection_function, save_density, load_density, supported_pars
from figaro.rate import sample_rate, normalise_alpha_factor, plot_integrated_rate, plot_differential_rate
from figaro.cosmology import _decorator_dVdz, dVdz_approx_planck18, dVdz_approx_planck15
from figaro.marginal import marginalise

import ray
from ray.util import ActorPool

@ray.remote
class worker:
    def __init__(self, bounds,
                       out_folder_plots,
                       out_folder_draws,
                       ext         = 'json',
                       se_sigma    = None,
                       hier_sigma  = None,
                       scale       = None,
                       events      = None,
                       label       = None,
                       unit        = None,
                       save_se     = True,
                       MC_draws    = None,
                       probit      = True,
                       selfunc     = None,
                       inj_pdf     = None,
                       n_total_inj = None,
                       ):
        self.dim                  = bounds.shape[0]
        self.bounds               = bounds
        self.mixture              = DPGMM(self.bounds, probit = probit)
        self.hierarchical_mixture = HDPGMM(self.bounds,
                                           MC_draws           = MC_draws,
                                           probit             = probit,
                                           selection_function = selfunc,
                                           injection_pdf      = inj_pdf,
                                           total_injections   = n_total_inj,
                                           prior_pars         = get_priors(self.bounds,
                                                                           samples      = events,
                                                                           std          = hier_sigma,
                                                                           scale        = scale,
                                                                           probit       = probit,
                                                                           hierarchical = True,
                                                                           ),
                                           )
        self.out_folder_plots = out_folder_plots
        self.out_folder_draws = out_folder_draws
        self.se_sigma         = se_sigma
        self.scale            = scale
        self.save_se          = save_se
        self.label            = label
        self.unit             = unit
        self.probit           = probit
        self.ext              = ext

    def run_event(self, pars):
        # Unpack data
        samples, name, n_draws = pars
        # Copying (issues with shuffling)
        ev = np.copy(samples)
        ev.setflags(write = True)
        # Actual inference
        prior_pars = get_priors(self.bounds, samples = ev, probit = self.probit, std = self.se_sigma, scale = self.scale, hierarchical = False)
        self.mixture.initialise(prior_pars = prior_pars)
        draws      = [self.mixture.density_from_samples(ev, make_comp = False) for _ in range(n_draws)]
        # Plots
        plt_bounds = np.atleast_2d([ev.min(axis = 0), ev.max(axis = 0)]).T
        if self.save_se:
            if self.dim == 1:
                plot_median_cr(draws,
                               samples    = ev,
                               bounds     = plt_bounds[0],
                               out_folder = self.out_folder_plots,
                               name       = name,
                               label      = self.label,
                               unit       = self.unit,
                               subfolder  = True,
                               )
            else:
                plot_multidim(draws,
                              samples    = ev,
                              bounds     = plt_bounds,
                              out_folder = self.out_folder_plots,
                              name       = name,
                              labels     = self.label,
                              units      = self.unit,
                              subfolder  = True,
                              )
        # Saving
        save_density(draws, folder = self.out_folder_draws, name = 'draws_'+name, ext = self.ext)
        return draws

    def draw_hierarchical(self):
        return self.hierarchical_mixture.density_from_samples(self.posteriors, make_comp = False)
    
    def load_posteriors(self, posteriors):
        self.posteriors = np.copy(posteriors)
        self.posteriors.setflags(write = True)
        for i in range(len(self.posteriors)):
            self.posteriors[i].setflags(write = True)
            
