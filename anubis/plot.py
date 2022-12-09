import numpy as np
import warnings

from corner import corner
from pathlib import Path

from figaro.plot import plot_median_cr, plot_multidim, plot_1d_dist

from anubis.utils import get_samples, get_weights, get_samples_and_weights
from anubis.exceptions import ANUBISException

plot_keys = ['pars', 'weights', 'joint', 'all']

def plot_parametric(draws, injected = None, samples = None, selfunc = None, bounds = None, out_folder = '.', name = 'parametric', n_pts = 1000, label = None, unit = None, show = False, save = True, subfolder = False, true_value = None, true_value_label = '\mathrm{True\ value}', injected_label = '\mathrm{Simulated}', median_label = '\mathrm{Parametric}', logx = False, logy = False):
    """
    Plot the parametric distribution along with samples from the true distribution (if available).
    Works with 1-dimensional distributions only.

    Arguments:
        :iterable draws:                  container for realisations
        :callable or np.ndarray injected: injected distribution (if available)
        :np.ndarray samples:              samples from the true distribution (if available)
        :callable or np.ndarray selfunc:  selection function (if available)
        :iterable bounds:                 bounds for the recovered distribution. If None, bounds from mixture instances are used.
        :str or Path out_folder:          output folder
        :str name:                        name to be given to outputs
        :int n_pts:                       number of points for linspace
        :str label:                       LaTeX-style quantity label, for plotting purposes
        :str unit:                        LaTeX-style quantity unit, for plotting purposes
        :bool save:                       whether to save the plots or not
        :bool show:                       whether to show the plots during the run or not
        :bool subfolder:                  whether to save the plots in different subfolders (for multiple events)
        :float true_value:                true value to infer
        :str true_value_label:            label to assign to the true value marker
        :str injected_label:              label to assign to the injected distribution
        :str median_label:                label to assign to the median distribution
        :bool logx:                       x log scale
        :bool logy:                       y log scale
    """
    if not (np.array([d.dim == 1 for d in draws])).all():
        warnings.warn("The plot_parametric() method works with 1-dimensional distributions only. No plot produced.")
        return
    
    all_bounds = np.atleast_2d([d.bounds[0] for d in draws])
    x_min = np.max(all_bounds[:,0])
    x_max = np.min(all_bounds[:,1])
    
    probit = np.array([d.probit for d in draws]).any()
    
    if bounds is not None:
        if not bounds[0] >= x_min and probit:
            warnings.warn("The provided lower bound is invalid for at least one draw. {0} will be used instead.".format(x_min))
        else:
            x_min = bounds[0]
        if not bounds[1] <= x_max and probit:
            warnings.warn("The provided upper bound is invalid for at least one draw. {0} will be used instead.".format(x_max))
        else:
            x_max = bounds[1]

    x    = np.linspace(x_min, x_max, n_pts+2)[1:-1]
    dx   = x[1]-x[0]
    probs = np.array([np.sum([d.weights[i+1]*model(x) for i, model in enumerate(d.models[1:])], axis = 0) for d in draws])
    probs = np.array([p/np.sum(p*dx) for p in probs])
    
    plot_1d_dist(x                = x,
                 draws            = probs,
                 injected         = injected,
                 samples          = samples,
                 out_folder       = out_folder,
                 name             = name,
                 label            = label,
                 unit             = unit,
                 show             = show,
                 save             = save,
                 subfolder        = subfolder,
                 true_value       = true_value,
                 true_value_label = true_value_label,
                 injected_label   = injected_label,
                 median_label     = median_label,
                 logx             = logx,
                 logy             = logy,
                 )

def plot_non_parametric(draws, injected = None, samples = None, selfunc = None, bounds = None, out_folder = '.', name = 'DPGMM', n_pts = None, labels = None, units = None, hierarchical = False, show = False, save = True, subfolder = False, true_value = None, true_value_label = '\mathrm{True\ value}', injected_label = '\mathrm{Simulated}', figsize = 7, levels = [0.5, 0.68, 0.9], scatter_points = False):
    """
    Plot the recovered non-parametric distribution along with samples from the true distribution (if available).
    
    Arguments:
        :iterable draws:                  container for mixture instances
        :callable or np.ndarray injected: injected distribution (if available, only for 1D distribution)
        :np.ndarray samples:              samples from the true distribution (if available)
        :callable or np.ndarray selfunc:  selection function (if available)
        :iterable bounds:                 bounds for the recovered distribution. If None, bounds from mixture instances are used.
        :str or Path out_folder:          output folder
        :str name:                        name to be given to outputs
        :int n_pts:                       number of points for linspace
        :str labels:                      LaTeX-style quantity labels, for plotting purposes
        :str units:                       LaTeX-style quantity units, for plotting purposes
        :bool hierarchical:               hierarchical inference, for plotting purposes
        :bool save:                       whether to save the plots or not
        :bool show:                       whether to show the plots during the run or not
        :bool subfolder:                  whether to save the plots in different subfolders (for multiple events)
        :float true_value:                true value to infer
        :str true_value_label:            label to assign to the true value marker
        :str injected_label:              label to assign to the injected distribution
        :double figsize:                  figure size (matplotlib)
        :iterable levels:                 credible levels to plot
        :bool scatter_points:             scatter samples on 2d plots
    """
    
    nonpar = [d.models[0] for d in draws]
    
    if draws[0].dim == 1:
        if n_pts is None:
            n_pts = 1000
        plot_median_cr(draws            = nonpar,
                       injected         = injected,
                       samples          = samples,
                       selfunc          = selfunc,
                       bounds           = bounds,
                       out_folder       = out_folder,
                       name             = name,
                       n_pts            = n_pts,
                       label            = labels,
                       unit             = units,
                       hierarchical     = hierarchical,
                       show             = show,
                       save             = save,
                       subfolder        = subfolder,
                       true_value       = true_value,
                       true_value_label = true_value_label,
                       injected_label   = injected_label,
                       )
    else:
        if injected is not None:
            warnings.warn("Injected distribution can be plotted only on 1-dimensional distributions.")
        if n_pts is None:
            n_pts = 200
        plot_multidim(draws          = nonpar,
                      samples        = samples,
                      bounds         = bounds,
                      out_folder     = out_folder,
                      name           = name,
                      labels         = labels,
                      units          = units,
                      hierarchical   = hierarchical,
                      show           = show,
                      save           = save,
                      subfolder      = subfolder,
                      n_pts          = n_pts,
                      true_value     = true_value,
                      figsize        = figsize,
                      levels         = levels,
                      scatter_points = scatter_points,
                      )

def plot_samples(draws, plot = 'joint', out_folder = '.', pars_labels = None, par_models_labels = None, true_pars = None, true_weights = None, name = None):
    """
    Corner plot with samples (parameters, weights and/or both).
    
    Arguments:
        :iterable draws:               container for realisations
        :str plot:                     whether to plot parameters ('pars'), weights ('weights'), both ('joint') or produce all three plots ('all')
        :str or Path out_folder:       output folder
        :list-of-str pars_labels:      labels for parameters
        :list-of-str par_model_labels: labels for models (for weights)
        :iterable true_pars:           parameters true values
        :iterable true_weights:        weights true values
        :str name:                     name to be given to outputs
    """
    
    if not plot in plot_keys:
        raise ANUBISException("Please provide a plot keyword among these: "+" ".join(["{}".format(key) for key in plot_keys]))
    
    if true_pars is not None:
        if type(true_pars) is float or type(true_pars) is int:
            true_pars = [true_pars]
        else:
            true_pars = list(true_pars)
    
    out_folder = Path(out_folder)
    if not out_folder.exists():
        out_folder.mkdir()
        
    if plot in ['pars', 'all']:
        samples = get_samples(draws)
        if name is not None:
            plot_name = name + '_pars.pdf'
        else:
            plot_name = 'parameters.pdf'
            
        if pars_labels is None or not (len(pars_labels) == samples.shape[-1]):
            parameters_labels = ['$p_{0}$'.format(i+1) for i in range(samples.shape[-1])]
        else:
            parameters_labels = ['${0}$'.format(l) for l in pars_labels]

        c = corner(samples, labels = parameters_labels, truths = true_pars, quantiles = [0.16, 0.5, 0.84], show_titles = True)
        c.savefig(Path(out_folder, plot_name), bbox_inches = 'tight')

    if plot in ['weights', 'all']:
        samples = get_weights(draws)
        if name is not None:
            plot_name = name + '_weights.pdf'
        else:
            plot_name = 'weights.pdf'
            
        if par_models_labels is None or not (len(par_models_labels) == (samples.shape[-1]-1)):
            weights_labels = ['$w_{np}$'] + ['$w_{'+'{0}'.format(i+1)+'}$' for i in range(samples.shape[-1]-1)]
        else:
            weights_labels = ['$w_{np}$'] + ['$w_{'+'{0}'.format(l)+'}$' for l in par_models_labels]

        c = corner(samples, labels = weights_labels, truths = true_weights, quantiles = [0.16, 0.5, 0.84], show_titles = True)
        c.savefig(Path(out_folder, plot_name), bbox_inches = 'tight')

    if plot in ['joint', 'all']:
        samples  = get_samples_and_weights(draws)
        n_models = len(draws[0].models)
        if name is not None:
            plot_name = name + '_joint.pdf'
        else:
            plot_name = 'joint.pdf'
            
        if pars_labels is None or not (len(pars_labels) == (samples.shape[-1]-n_models)):
            parameters_labels = ['$p_{0}$'.format(i+1) for i in range(samples.shape[-1]-n_models)]
        else:
            parameters_labels = ['${0}$'.format(l) for l in pars_labels]
        if par_models_labels is None or not (len(par_models_labels) == (n_models-1)):
            weights_labels = ['$w_{np}$'] + ['$w_{'+'{0}'.format(i+1)+'}$' for i in range(n_models-1)]
        else:
            weights_labels = ['$w_{np}$'] + ['$w_{'+'{0}'.format(l)+'}$' for l in par_models_labels]
        
        joint_labels = parameters_labels + weights_labels
        
        if true_pars is None:
            true_pars = [None for _ in range(len(pars_labels))]
        if true_weights is None:
            true_weights = [None for _ in range(len(weights_labels))]
        true_vals = true_pars + true_weights
        
        c = corner(samples, labels = joint_labels, truths = true_vals, quantiles = [0.16, 0.5, 0.84], show_titles = True)
        c.savefig(Path(out_folder, plot_name), bbox_inches = 'tight')
