# Heterogeneous mixture model implementation

This code implements the inference of a multivariate probability density given a set of samples and an (incomplete) parametric model.
The Heterogeneous Mixture Model, or HMM, is a finite mixture of parametric models augmented with DPGMM to capture the eventual features missed by the parametric model.

To install this code, run `python setup.py install`. In some cases (like on clusters), it may happen that you do not have the permission to write in the default installation directory. In this case, run `python setup.py install --user`.

This code makes use of [FIGARO](https://github.com/sterinaldi/FIGARO). Be sure of having it installed and updated. **DO NOT USE** `pip install figaro`, since it installs an homonymous package.
