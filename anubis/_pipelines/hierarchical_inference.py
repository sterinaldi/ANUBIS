import numpy as np

import optparse
import importlib
import warnings

from pathlib import Path
from tqdm import tqdm
from copy import deepcopy

from figaro.utils import save_options, load_options, get_priors
from figaro.load import load_single_event, load_selection_function, supported_extensions, supported_pars

from figaro.mixture import DPGMM
from figaro.utils import save_options, load_options, get_priors
from figaro.plot import plot_median_cr, plot_multidim
from figaro.load import load_selection_function, supported_pars, load_single_event, load_data as load_data_figaro, save_density as save_density_figaro
from figaro.rate import sample_rate, normalise_alpha_factor, plot_integrated_rate, plot_differential_rate
from figaro.cosmology import _decorator_dVdz, dVdz_approx_planck18, dVdz_approx_planck15

from anubis.mixture import HAMM
from anubis.load import load_density, load_models, load_injected_density, load_data as load_data_anubis, save_density as save_density_anubis
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
                       out_folder_plots,
                       out_folder_draws,
                       selection_function = None,
                       inj_pdf            = None,
                       n_total_inj        = None,
                       MC_draws_pars      = 1e3,
                       MC_draws_norm      = 5e3,
                       n_reassignments    = None,
                       gamma0             = None,
                       augment            = True,
                       se_sigma           = None,
                       hier_sigma         = None,
                       events             = None,
                       probit             = True,
                       scale              = None,
                       save_se            = False,
                       label              = None,
                       unit               = None
                       ):
    
        # Single-event reconstruction settings
        self.bounds           = np.atleast_2d(bounds)
        self.dim              = bounds.shape[0]
        self.out_folder_plots = out_folder_plots
        self.out_folder_draws = out_folder_draws
        self.se_sigma         = se_sigma
        self.scale            = scale
        self.save_se          = save_se
        self.label            = label
        self.unit             = unit
        self.probit           = probit

        if augment:
            prior_pars = get_priors(self.bounds,
                                    samples      = events,
                                    std          = hier_sigma,
                                    scale        = scale,
                                    probit       = probit,
                                    hierarchical = True,
                                    )
        else:
            prior_pars = None
            
        self.DPGMM   = DPGMM(self.bounds, probit = probit)
        self.mixture = HAMM(models             = models,
                            bounds             = self.bounds,
                            pars               = pars,
                            par_bounds         = par_bounds,
                            shared_pars        = shared_pars,
                            shared_par_bounds  = shared_par_bounds,
                            augment            = augment,
                            gamma0             = gamma0,
                            n_reassignments    = n_reassignments,
                            n_draws_pars       = MC_draws_pars,
                            n_draws_norm       = MC_draws_norm,
                            probit             = self.probit,
                            prior_pars         = prior_pars,
                            selection_function = selection_function,
                            inj_pdf            = inj_pdf,
                            n_total_inj        = n_total_inj,
                            )
    
    def run_event(self, pars):
        # Unpack data
        samples, name, n_draws = pars
        # Copying (issues with shuffling)
        ev = np.copy(samples)
        ev.setflags(write = True)
        # Actual inference
        prior_pars = get_priors(self.bounds,
                                samples      = ev,
                                probit       = self.probit,
                                std          = self.se_sigma,
                                scale        = self.scale,
                                hierarchical = False,
                                )
        self.DPGMM.initialise(prior_pars = prior_pars)
        draws      = [self.DPGMM.density_from_samples(ev, make_comp = False) for _ in range(n_draws)]
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
        save_density_figaro(draws, folder = self.out_folder_draws, name = 'draws_'+name)
        return draws
        
    def draw_hierarchical(self):
        return self.mixture.density_from_samples(self.posteriors, make_comp = False)
    
    def load_posteriors(self, posteriors):
        self.posteriors = deepcopy(posteriors)
        for i in range(len(self.posteriors)):
            self.posteriors[i][0].setflags(write = True)

    def draw_sample(self):
        return self.mixture.density_from_samples(self.samples, make_comp = False)

def main():

    parser = optparse.OptionParser(prog = 'anubis-hierarchical', description = 'Hierarchical probability density reconstruction')
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
    parser.add_option("--name", type = "string", dest = "hier_name", help = "Name to be given to hierarchical inference files. Default: same name as samples folder parent directory", default = None)
    parser.add_option("-p", "--postprocess", dest = "postprocess", action = 'store_true', help = "Postprocessing", default = False)
    parser.add_option("-s", "--save_se", dest = "save_single_event", action = 'store_true', help = "Save single event plots", default = False)
    parser.add_option("--true_pars", type = "string", dest = "true_pars", help = "True parameter values", default = None)
    parser.add_option("--true_weights", type = "string", dest = "true_weights", help = "True relative weights of parametric models", default = None)
    parser.add_option("--hier_samples", type = "string", dest = "hier_samples", help = "Samples from hierarchical distribution (true single-event values, for simulations only)", default = None)
    # Settings
    parser.add_option("--no_augment", dest = "augment", action = 'store_false', help = "Disable non-parametric augmentation", default = True)
    parser.add_option("--draws", type = "int", dest = "draws", help = "Number of draws", default = 1000)
    parser.add_option("--se_draws", type = "int", dest = "se_draws", help = "Number of draws for single-event distribution. Default: same as hierarchical distribution", default = None)
    parser.add_option("--n_samples_dsp", type = "int", dest = "n_samples_dsp", help = "Number of samples to analyse (downsampling). Default: all", default = -1)
    parser.add_option("--exclude_points", dest = "exclude_points", action = 'store_true', help = "Exclude points outside bounds from analysis", default = False)
    parser.add_option("--cosmology", type = "choice", dest = "cosmology", help = "Set of cosmological parameters. Default values from Planck (2021)", choices = ['Planck18', 'Planck15'], default = 'Planck18')
    parser.add_option("-e", "--events", dest = "run_events", action = 'store_false', help = "Skip single-event analysis", default = True)
    parser.add_option("--sigma_prior", dest = "sigma_prior", type = "string", help = "Expected standard deviation (prior) - single value or n-dim values. If None, it is estimated from samples", default = None)
    parser.add_option("--se_sigma_prior", dest = "se_sigma_prior", type = "string", help = "Expected standard deviation (prior) for single-event inference - single value or n-dim values. If None, it is estimated from samples", default = None)
    parser.add_option("--n_reassignments", dest = "n_reassignments", type = "float", help = "Number of reassignments", default = None)
    parser.add_option("--symbol", type = "string", dest = "symbol", help = "LaTeX-style quantity symbol, for plotting purposes", default = None)
    parser.add_option("--unit", type = "string", dest = "unit", help = "LaTeX-style quantity unit, for plotting purposes", default = None)
    parser.add_option("--fraction", dest = "fraction", type = "float", help = "Fraction of samples standard deviation for sigma prior. Overrided by sigma_prior.", default = None)
    parser.add_option("--snr_threshold", dest = "snr_threshold", type = "float", help = "SNR threshold for simulated GW datasets", default = None)
    parser.add_option("--far_threshold", dest = "far_threshold", type = "float", help = "FAR threshold for simulated GW datasets", default = None)
    parser.add_option("--no_probit", dest = "probit", action = 'store_false', help = "Disable probit transformation", default = True)
    parser.add_option("--config", dest = "config", type = "string", help = "Config file. Warning: command line options override config options", default = None)
    parser.add_option("-l", "--likelihood", dest = "likelihood", action = 'store_true', help = "Resample posteriors to get likelihood samples (only for GW data)", default = False)
    parser.add_option("--n_parallel", dest = "n_parallel", type = "int", help = "Number of parallel threads", default = 1)
    parser.add_option("--mc_draws_pars", dest = "MC_draws_pars", type = "int", help = "Number of draws for assignment MC integral over model parameters", default = None)
    parser.add_option("--mc_draws_norm", dest = "MC_draws_norm", type = "int", help = "Number of draws for MC normalisation integral", default = None)
    parser.add_option("--gamma0", dest = "gamma0", type = "float", help = "concentration parameter for Dirichlet prior on augmented mixture", default = None)
    parser.add_option("--rate", dest = "rate", action = 'store_true', help = "Compute rate", default = False)
    parser.add_option("--include_dvdz", dest = "include_dvdz", action = 'store_true', help = "Include dV/dz*(1+z)^{-1} term in selection effects.", default = False)

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
    output_plots = Path(options.output, 'plots')
    if not output_plots.exists():
        output_plots.mkdir()
    output_draws = Path(options.output, 'draws')
    if not output_draws.exists():
        output_draws.mkdir()
    # Read hierarchical name
    if options.hier_name is None:
        options.hier_name = options.output.parts[-1]
    if options.selfunc_file is None:
        hier_name = 'observed_'+options.hier_name
    else:
        hier_name = 'intrinsic_'+options.hier_name
    
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
    # Read number of single-event draws
    if options.se_draws is None:
        options.se_draws = options.draws
    if options.sigma_prior is not None:
        options.sigma_prior = np.array([float(s) for s in options.sigma_prior.split(',')])
    if options.se_sigma_prior is not None:
        options.se_sigma_prior = np.array([float(s) for s in options.se_sigma_prior.split(',')])
    # Cosmology
    if options.cosmology == 'Planck18':
        approx_dVdz = dVdz_approx_planck18
    else:
        approx_dVdz = dVdz_approx_planck15
    # If provided, load injected density
    inj_density        = None
    inj_parametric     = None
    inj_non_parametric = None
    if options.inj_density_file is not None:
        inj_density, inj_parametric, inj_non_parametric = load_injected_density(options.inj_density_file)
    # If provided, load selecton function
    selfunc     = None
    inj_pdf     = None
    n_total_inj = None
    if options.selfunc_file is not None:
        selfunc, inj_pdf, n_total_inj, _ = load_selection_function(options.selfunc_file, par = options.par)
    if options.include_dvdz and callable(selfunc):
        if options.par is None:
            print("Redshift is assumed to be the last parameter")
            z_index = -1
        elif 'z' in options.par:
            z_index = np.where(np.array(options.par)=='z')[0][0]
        else:
            raise Exception("Redshift must be included in the rate analysis")
        dec_selfunc = _decorator_dVdz(selfunc, approx_dVdz, z_index, options.bounds[z_index][1])
    else:
        dec_selfunc = selfunc
    # If provided, load true values
    hier_samples = None
    if options.hier_samples is not None:
        options.hier_samples = Path(options.hier_samples).resolve()
        hier_samples, true_name = load_single_event(options.hier_samples,
                                                    par       = options.par,
                                                    cosmology = options.cosmology,
                                                    waveform  = options.wf,
                                                    )
        if np.shape(hier_samples)[-1] == 1:
            hier_samples = hier_samples.flatten()
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
    # Load samples
    events, names = load_data_figaro(options.input,
                                     par        = options.par,
                                     n_samples  = options.n_samples_dsp,
                                     cosmology  = options.cosmology,
                                     waveform   = options.wf,
                                     likelihood = options.likelihood,
                                     )
    try:
        dim = np.shape(events[0][0])[-1]
    except IndexError:
        dim = 1
    if options.exclude_points:
        print("Ignoring points outside bounds.")
        for i, ev in enumerate(events):
            events[i] = ev[np.where((np.prod(options.bounds[:,0] < ev, axis = 1) & np.prod(ev < options.bounds[:,1], axis = 1)))]
    else:
        # Check if all samples are within bounds
        all_samples = np.atleast_2d(np.concatenate(events))
        if options.probit:
            if not np.all([(all_samples[:,i] > options.bounds[i,0]).all() and (all_samples[:,i] < options.bounds[i,1]).all() for i in range(dim)]):
                raise ValueError("One or more samples are outside the given bounds.")
    # Plot labels
    if dim > 1:
        if options.symbol is not None:
            symbols = options.symbol.split(',')
        else:
            symbols = options.symbol
        if options.unit is not None:
            units = options.unit.split(',')
        else:
            units = options.unit
    else:
        symbols = options.symbol
        units   = options.unit
    
    if not options.postprocess:
        ray.init(num_cpus = options.n_parallel)
        # Reconstruction
        pool = ActorPool([worker.remote(models             = [model['model'] for model in models],
                                        bounds             = options.bounds,
                                        pars               = pars,
                                        shared_pars        = shared_pars,
                                        par_bounds         = par_bounds,
                                        shared_par_bounds  = shared_par_bounds,
                                        out_folder_plots   = output_plots,
                                        out_folder_draws   = output_draws,
                                        selection_function = dec_selfunc,
                                        inj_pdf            = inj_pdf,
                                        n_total_inj        = n_total_inj,
                                        MC_draws_pars      = options.MC_draws_pars,
                                        MC_draws_norm      = options.MC_draws_norm,
                                        n_reassignments    = options.n_reassignments,
                                        gamma0             = options.gamma0,
                                        augment            = options.augment,
                                        se_sigma           = options.se_sigma_prior,
                                        hier_sigma         = options.sigma_prior,
                                        events             = events,
                                        probit             = options.probit,
                                        scale              = options.fraction,
                                        save_se            = options.save_single_event,
                                        label              = symbols,
                                        unit               = units,
                                        )
                            for _ in range(options.n_parallel)])
        
        if options.run_events:
            # Run each single-event analysis
            posteriors = []
            for s in tqdm(pool.map_unordered(lambda a, v: a.run_event.remote(v), [[ev, name, options.se_draws] for ev, name in zip(events, names)]), total = len(events), desc = 'Events'):
                posteriors.append(s)
        # Load data for ANUBIS inference
        posteriors, names = load_data_anubis(path_samples  = options.input,
                                             path_mixtures = output_draws,
                                             par           = options.par,
                                             n_samples     = options.n_samples_dsp,
                                             cosmology     = options.cosmology,
                                             waveform      = options.wf,
                                             likelihood    = options.likelihood,
                                             verbose       = False
                                             )
        if options.exclude_points:
            # No need to re-check that all points are within  bounds
            for i, ev in enumerate(posteriors):
                evi          = ev[0]
                posteriors[i][0] = evi[np.where((np.prod(options.bounds[:,0] < evi, axis = 1) & np.prod(evi < options.bounds[:,1], axis = 1)))]
        # Load posteriors
        for s in pool.map(lambda a, v: a.load_posteriors.remote(v), [posteriors for _ in range(options.n_parallel)]):
            pass
        # Run hierarchical analysis
        draws = []
        for s in tqdm(pool.map_unordered(lambda a, v: a.draw_hierarchical.remote(), [_ for _ in range(options.draws)]), total = options.draws, desc = 'Sampling'):
            draws.append(s)
        draws = np.array(draws)
    #        if options.include_dvdz:
    #            normalise_alpha_factor(draws, dvdz = approx_dVdz, z_index = z_index, z_max = options.bounds[z_index][1])
        # Save draws
        save_density_anubis(draws,
                            models,
                            folder = options.output,
                            name   = hier_name,
                            )
    else:
        draws = load_density(folder             = options.output,
                             name               = hier_name,
                             models             = models,
                             selection_function = dec_selfunc,
                             make_comp          = False,
                            )
    # Plots
    if dim == 1:
        # Full distribution
        plot_median_cr(draws,
                       samples      = hier_samples,
                       injected     = inj_density,
                       bounds       = options.bounds[0],
                       out_folder   = output_plots,
                       name         = 'full_density_'+hier_name,
                       label        = symbols,
                       unit         = units,
                       hierarchical = True,
                       )
        # Parametric models
        plot_parametric(draws,
                        injected   = inj_parametric,
                        bounds     = options.bounds,
                        out_folder = output_plots,
                        name       = 'parametric_'+hier_name,
                        label      = symbols,
                        unit       = units,
                        )
    if options.augment:
        plot_non_parametric(draws,
                            injected     = inj_non_parametric,
                            bounds       = options.bounds,
                            out_folder   = output_plots,
                            name         = 'nonparametric_'+hier_name,
                            labels       = symbols,
                            units        = units,
                            hierarchical = True
                           )
    if len(models) + options.augment == 1:
        plot_type = 'pars'
    else:
        plot_type = 'joint'
    plot_samples(draws,
                 plot         = plot_type,
                 out_folder   = output_plots,
                 models       = models,
                 true_pars    = options.true_pars,
                 true_weights = options.true_weights,
                 name         = options.hier_name,
                 )

if __name__ == '__main__':
    main()
