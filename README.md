PyAutoDiff
==========

Automatic differentiation for NumPy (very new, experimental, unreliable, etc... but *promising*)

    import autodiff, numpy as np
    print autodiff.fmin_l_bfgs_b(lambda x: (x + 1) ** 2, [np.zeros(())])
    # -> [array(-1.0)]

Dependencies:
  * numpy
  * Theano [git master](https://github.com/Theano/Theano.git) (*currently no official release will work*)

  pip install -r requirements.txt


Thanks:
  * Travis Oliphant for posting a very early version of [numba](https://github.com/ContinuumIO/numba) which provided the inspiration and starting point for this project.
