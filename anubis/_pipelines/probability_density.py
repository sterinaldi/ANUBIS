import numpy as np

import optparse
import dill
import importlib

from pathlib import Path
from tqdm import tqdm
from warnings import warn

from figaro.utils import save_options, load_options, get_priors
from figaro.load import load_single_event, load_selection_function, supported_extensions, supported_pars

from anubis.mixture import HMM
from anubis.load import load_density, save_density, load_models, load_injected_density
from anubis.plot import plot_parametric, plot_non_parametric, plot_samples, plot_median_cr

import ray
from ray.util import ActorPool

@ray.remote
class worker:
    def __init__(self, models,
                       bounds,
                       pars,
                       shared_pars,
                       par_bounds,
                       shared_par_bounds,
                       selection_function = None,
                       MC_draws_pars      = 1e3,
                       MC_draws_norm      = 5e3,
                       n_reassignments    = None,
                       gamma0             = None,
                       augment            = True,
                       sigma              = None,
                       samples            = None,
                       probit             = True,
                       scale              = None,
                       ):
        if augment:
            prior_pars = get_priors(bounds,
                                    samples      = samples,
                                    std          = sigma,
                                    scale        = scale,
                                    probit       = probit,
                                    hierarchical = False,
                                    )
        else:
            prior_pars = None
        self.mixture = HMM(models             = models,
                           bounds             = bounds,
                           pars               = pars,
                           par_bounds         = par_bounds,
                           shared_pars        = shared_pars,
                           shared_par_bounds  = shared_par_bounds,
                           augment            = augment,
                           gamma0             = gamma0,
                           n_reassignments    = n_reassignments,
                           n_draws_pars       = MC_draws_pars,
                           n_draws_norm       = MC_draws_norm,
                           probit             = probit,
                           prior_pars         = prior_pars,
                           selection_function = selection_function,
                           )
        self.samples = np.copy(samples)
        self.samples.setflags(write = True)

    def draw_sample(self):
        return self.mixture.density_from_samples(self.samples, make_comp = False)

def main():

    parser = optparse.OptionParser(prog = 'anubis-density', description = 'Probability density reconstruction')
    # Input/output
    parser.add_option("-i", "--input", type = "string", dest = "input", help = "File with samples", default = None)
    parser.add_option("-m", "--models", type = "string", dest = "models", help = "File with models (list of dictionaries)", default = None)
    parser.add_option("-b", "--bounds", type = "string", dest = "bounds", help = "Density bounds. Must be a string formatted as '[[xmin, xmax], [ymin, ymax],...]'. Quotation marks are required and scientific notation is accepted", default = None)
    parser.add_option("-o", "--output", type = "string", dest = "output", help = "Output folder. Default: same directory as samples", default = None)
    parser.add_option("--inj_density", type = "string", dest = "inj_density_file", help = "Python module with injected densities - please name the methods 'density', 'density_parametric' and 'density_non_parametric'", default = None)
    parser.add_option("--selfunc", type = "string", dest = "selfunc_file", help = "Python module with selection function - please name the method 'selection_function'", default = None)
    parser.add_option("--parameter", type = "string", dest = "par", help = "GW parameter(s) to be read from file", default = None)
    parser.add_option("--waveform", type = "choice", dest = "wf", help = "Waveform to load from samples file. To be used in combination with --parameter.", choices = ['combined', 'seob', 'imr'], default = 'combined')
    # Plot
    parser.add_option("-p", "--postprocess", dest = "postprocess", action = 'store_true', help = "Postprocessing", default = False)
    parser.add_option("--true_pars", type = "string", dest = "true_pars", help = "True parameter values", default = None)
    parser.add_option("--true_weights", type = "string", dest = "true_weights", help = "True relative weights of parametric models", default = None)
    
    # Settings
    parser.add_option("--no_augment", dest = "augment", action = 'store_false', help = "Disable non-parametric augmentation", default = True)
    parser.add_option("--draws", type = "int", dest = "draws", help = "Number of draws", default = 100)
    parser.add_option("--n_samples_dsp", type = "int", dest = "n_samples_dsp", help = "Number of samples to analyse (downsampling). Default: all", default = -1)
    parser.add_option("--exclude_points", dest = "exclude_points", action = 'store_true', help = "Exclude points outside bounds from analysis", default = False)
    parser.add_option("--cosmology", type = "choice", dest = "cosmology", help = "Set of cosmological parameters. Default values from Planck (2021)", choices = ['Planck18', 'Planck15'], default = 'Planck18')
    parser.add_option("--sigma_prior", dest = "sigma_prior", type = "string", help = "Expected standard deviation (prior) - single value or n-dim values. If None, it is estimated from samples", default = None)
    parser.add_option("--n_reassignment", dest = "n_reassignment", type = "float", help = "Number of reassignments", default = None)
    parser.add_option("--fraction", dest = "fraction", type = "float", help = "Fraction of samples standard deviation for sigma prior. Overrided by sigma_prior.", default = None)
    parser.add_option("--snr_threshold", dest = "snr_threshold", type = "float", help = "SNR threshold for simulated GW datasets", default = None)
    parser.add_option("--far_threshold", dest = "far_threshold", type = "float", help = "FAR threshold for simulated GW datasets", default = None)
    parser.add_option("--no_probit", dest = "probit", action = 'store_false', help = "Disable probit transformation", default = True)
    parser.add_option("--config", dest = "config", type = "string", help = "Config file. Warning: command line options override config options", default = None)
    parser.add_option("-l", "--likelihood", dest = "likelihood", action = 'store_true', help = "Resample posteriors to get likelihood samples (only for GW data)", default = False)
    parser.add_option("--n_parallel", dest = "n_parallel", type = "int", help = "Number of parallel threads", default = 1)
    parser.add_option("--mc_draws_pars", dest = "mc_draws_pars", type = "int", help = "Number of draws for assignment MC integral over model parameters", default = None)
    parser.add_option("--mc_draws_norm", dest = "mc_draws_norm", type = "int", help = "Number of draws for MC normalisation integral", default = None)
    parser.add_option("--gamma0", dest = "gamma0", type = "float", help = "concentration parameter for Dirichlet prior on augmented mixture", default = None)

    (options, args) = parser.parse_args()

    if options.config is not None:
        options = load_options(options, parser)
    # Paths
    if options.input is not None:
        options.input = Path(options.input).resolve()
    elif options.config is not None:
        options.input = Path('.').resolve()
    else:
        raise Exception("Please provide path to samples.")
    if options.models is not None:
        options.models = Path(options.models).resolve()
    elif options.models is not None:
        options.models = Path('.').resolve()
    else:
        raise Exception("Please provide module with parametric model(s) definition.")
    if options.output is not None:
        options.output = Path(options.output).resolve()
        if not options.output.exists():
            options.output.mkdir(parents=True)
    else:
        options.output = options.input.parent
    if options.config is not None:
        options.config = Path(options.config).resolve()
    
    if options.config is None:
        save_options(options, options.output)

    # Read models:
    models, pars, shared_pars, par_bounds, shared_par_bounds = load_models(options.models)
    
    # Read bounds
    if options.bounds is not None:
        options.bounds = np.array(np.atleast_2d(eval(options.bounds)), dtype = np.float64)
    elif options.bounds is None and not options.postprocess:
        raise Exception("Please provide bounds for the inference (use -b '[[xmin,xmax],[ymin,ymax],...]')")
    # Read parameter(s)
    if options.par is not None:
        options.par = options.par.split(',')
        if not np.all([par in supported_pars for par in options.par]):
            raise Exception("Please provide parameters from this list: "+', '.join(supported_pars[:-2]))
    # If provided, load injected density
    inj_density        = None
    inj_parametric     = None
    inj_non_parametric = None
    if options.inj_density_file is not None:
        inj_density, inj_parametric, inj_non_parametric = load_injected_density(options.inj_density_file)
    # If provided, load selecton function
    selfunc = None
    if options.selfunc_file is not None:
        selfunc, _, _, _ = load_selection_function(options.selfunc_file, par = options.par)
        if not callable(selfunc):
            raise Exception("Only .py files with callable approximants are allowed for DPGMM reconstruction")
    # If provided, read true parameters and weights
    if options.true_pars is not None:
        options.true_pars = [float(eval(p)) for p in options.true_pars.split(',')]
    if options.true_weights is not None:
        options.true_weights = [float(eval(w)) for w in options.true_weights.split(',')]
        if options.augment:
            options.true_weights = [1.-np.sum(options.true_weights)] + options.true_weights
        
    if options.input.is_file():
        files = [options.input]
        output_draws = options.output
        subfolder = False
    else:
        files = sum([list(options.input.glob('*.'+ext)) for ext in supported_extensions], [])
        output_draws = Path(options.output, 'draws')
        if not output_draws.exists():
            output_draws.mkdir()
        subfolder = True
    if options.exclude_points:
        print("Ignoring points outside bounds.")

    if not options.postprocess:
        ray.init(num_cpus = options.n_parallel)
    
    for i, file in enumerate(files):
        # Load samples
        samples, name = load_single_event(file,
                                          par           = options.par,
                                          n_samples     = options.n_samples_dsp,
                                          cosmology     = options.cosmology,
                                          waveform      = options.wf,
                                          snr_threshold = options.snr_threshold,
                                          far_threshold = options.far_threshold,
                                          likelihood    = options.likelihood,
                                          )
        try:
            dim = np.shape(samples)[-1]
        except IndexError:
            dim = 1
        if options.exclude_points:
            samples = samples[np.where((np.prod(options.bounds[:,0] < samples, axis = 1) & np.prod(samples < options.bounds[:,1], axis = 1)))]
        else:
            # Check if all samples are within bounds
            if options.probit:
                if not np.all([(samples[:,i] > options.bounds[i,0]).all() and (samples[:,i] < options.bounds[i,1]).all() for i in range(dim)]):
                    raise ValueError("One or more samples are outside the given bounds.")
        # Reconstruction
        if not options.postprocess:
            # Actual analysis
            desc = name + ' ({0}/{1})'.format(i+1, len(files))
            pool = ActorPool([worker.remote(models             = [model['model'] for model in models],
                                            bounds             = options.bounds,
                                            pars               = pars,
                                            shared_pars        = shared_pars,
                                            par_bounds         = par_bounds,
                                            shared_par_bounds  = shared_par_bounds,
                                            selection_function = selfunc,
                                            MC_draws_pars      = options.MC_draws_pars,
                                            MC_draws_norm      = options.MC_draws_norm,
                                            n_reassignments    = options.n_reassignments,
                                            gamma0             = options.gamma0,
                                            augment            = options.augment,
                                            sigma              = options.sigma_prior,
                                            samples            = samples,
                                            probit             = options.probit,
                                            scale              = options.fraction,
                                            )
                              for _ in range(options.n_parallel)])
            draws = []
            for s in tqdm(pool.map_unordered(lambda a, v: a.draw_sample.remote(), [_ for _ in range(options.draws)]), total = options.draws, desc = desc):
                draws.append(s)
            draws = np.array(draws)
            # Save reconstruction
            save_density(draws,
                         models,
                         folder = output_draws,
                         name   = name,
                         )
        else:
            draws = load_density(folder             = output_draws,
                                 name               = name,
                                 models             = models,
                                 selection_function = selfunc,
                                 make_comp          = False,
                                 )
        # Plots
        if dim == 1:
            # Full distribution
            plot_median_cr(draws,
                           samples    = samples,
                           injected   = inj_density,
                           bounds     = options.bounds,
                           out_folder = options.output,
                           name       = 'full_density_'+name,
                           label      = options.symbol,
                           unit       = options.unit,
                           subfolder  = subfolder,
                           )
            # Parametric models
            plot_parametric(draws,
                            injected   = inj_parametric,
                            bounds     = options.bounds,
                            out_folder = options.output,
                            name       = 'parametric_'+name,
                            label      = options.symbol,
                            unit       = options.unit,
                            subfolder  = subfolder,
                            )
        else:
            if options.symbol is not None:
                symbols = options.symbol.split(',')
            else:
                symbols = options.symbol
            if options.unit is not None:
                units = options.unit.split(',')
            else:
                units = options.unit
        if options.augment:
            plot_non_parametric(draws,
                                injected   = inj_non_parametric,
                                bounds     = options.bounds,
                                out_folder = options.output,
                                name       = 'nonparametric_'+name,
                                labels     = symbols,
                                units      = units,
                                subfolder  = subfolder,
                               )
        if len(models) + options.augment == 1:
            plot_type = 'pars'
        else:
            plot_type = 'joint'
        plot_samples(draws,
                     plot         = plot_type,
                     out_folder   = options.output,
                     models       = models,
                     true_pars    = options.true_pars,
                     true_weights = options.true_weights,
                     name         = name,
                     subfolder    = subfolder,
                     )

if __name__ == '__main__':
    main()
