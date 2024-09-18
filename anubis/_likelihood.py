import numpy as np


def _population_log_likelihood(ll, amm, idx = None, joint = False):
    """
    Population likelihood including selection effects (for MCMC sampling).
    Single component.
    
    Arguments:
        np.ndarray ll:      vector of parameters
        anubis.mixture.AMM: augmented mixture model class storing all the relevant info
        int idx:            label of the mixture component
        bool joint:         whether joint or separable parameter space
    
    Returns:
        double: log likelihood
    """
    if idx is None:
        idx = amm.model_to_sample
    alpha_factor = 1.
    comp_idx = int(idx + amm.augment)
    # Indices of the parameters (needed if joint)
    start = 0
    finish = len(amm.par_bounds[idx])
    if joint:
        start  += int(np.sum([len(amm.par_bounds[i]) for i in range(idx)]))
        finish += start
    if amm.shared_par_bounds is not None:
        start_shared = -len(amm.shared_par_bounds)
    else:
        start_shared = len(ll)
    # Check bounds
    if not np.all([amm.par_bounds[idx][i][0] < ll[start+i] < amm.par_bounds[idx][i][1] for i in range(len(ll[start:finish]))]):
        return -np.inf
    if amm.selfunc is not None:
        amm.components[comp_idx]._compute_alpha_factor([ll[start:finish]], [ll[start_shared:]], amm.n_draws_norm)
        alpha_factor = amm.components[comp_idx].alpha
        if not np.isfinite(alpha_factor):
            return -np.inf
    # Points
    id_points = [pt for pt in range(int(np.sum(amm.n_pts))) if amm.assignations[pt] == comp_idx]
    # Hierarchical
    if amm.hierarchical:
        L_vector = np.array([np.mean(amm.components[comp_idx]._model(amm.stored_pts[id]['samples'], ll[start:finish], ll[start_shared:])) for id in id_points])
    else:
        L_vector = np.array([amm.components[comp_idx]._model(amm.stored_pts[id], ll[start:finish], ll[start_shared:]) for id in id_points])
    logL = np.sum(np.log(L_vector)) - len(id_points)*np.log(alpha_factor)
    if np.isfinite(logL):
        return logL
    else:
        return -np.inf

def _joint_population_log_likelihood(ll, amm):
    """
    Population likelihood including selection effects (for MCMC sampling).
    All components.
    
    Arguments:
        np.ndarray ll:      vector of parameters
        anubis.mixture.AMM: augmented mixture model class storing all the relevant info
    
    Returns:
        double log likelihood
    """
    logL = 0.
    if not np.all([amm.shared_par_bounds[i][0] < ll[-len(amm.shared_par_bounds)+i] < amm.shared_par_bounds[i][1] for i in range(len(ll[-len(amm.shared_par_bounds):]))]):
        return -np.inf
    for idx in range(len(amm.par_models)):
        logL += _population_log_likelihood(ll, amm, idx, joint = True)
        if not np.isfinite(logL):
            return -np.inf
    return np.sum(logL)
