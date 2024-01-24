import numpy as np

def get_samples(draws):
    """
    Extract parameter samples from a list of draws.
    
    Arguments:
        :iterable draws: instances of het_mixture class
    
    Returns:
        :np.ndarray: samples
    """
    ll = [[list(d.models[i+d.augment].pars) for i in range(len(d.models)-d.augment)] for d in draws]
    # Funny way of doing the right thing. I may rewrite it in the future, but it works for now...
    return np.array([s for d in ll for m in d for s in m]).reshape(len(draws), -1)

def get_weights(draws):
    """
    Extract weight samples from a list of draws.
    
    Arguments:
        :iterable draws: instances of het_mixture class
    
    Returns:
        :np.ndarray: samples
    """
    return np.array([d.weights for d in draws])

def get_samples_and_weights(draws):
    """
    Extract parameter and weight samples from a list of draws.
    
    Arguments:
        :iterable draws: instances of het_mixture class
    
    Returns:
        :np.ndarray: samples
    """
    return np.block([get_samples(draws), get_weights(draws)])
