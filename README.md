# ANUBIS - Astrophysical Nonparametrically Upgraded Bayesian Inference of Subpopulations

[ANUBIS](https://github.com/sterinaldi/ANUBIS) implements the inference of a multivariate probability density given a set of samples and one or more parametric models.
The Augmente Mixture Model, or AMM, is a finite mixture of parametric models augmented with DPGMM to capture the eventual features missed by the parametric model:

```math
p(x) \simeq \sum_i^N w_i f_i(x|\Lambda_i) + \big(1-\Sigma_i^N w_i \big) \sum_j^\infty \phi_j \mathcal{N}(x|\mu_j,\sigma_j)
```

Please install this code with `pip install anubis` (stable version) or `pip install .` after cloning this repository (development version, potentially unstable).
