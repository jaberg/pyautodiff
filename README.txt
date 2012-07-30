PyAutoDiff
==========

Automatic differentiation for NumPy (very new, experimental, unreliable, etc... but *promising*)

    import autodiff, numpy as np
    print autodiff.fmin_l_bfgs_b(lambda x: (x + 1) ** 2, (np.zeros(()),))
    # -> (array(-1.0),)

Other examples include:
  * [Linear SVM done quick](https://github.com/jaberg/pyautodiff/blob/master/autodiff/tests/test_svm.py)

Dependencies:
  * numpy
  * Theano [git master](https://github.com/Theano/Theano.git) (*currently no official release will work*)

Installation:

    pip install -r requirements.txt
    python setup.py install


Thanks:
  * Travis Oliphant for posting a very early version of [numba](https://github.com/numba/numba) which provided the inspiration and starting point for this project.
