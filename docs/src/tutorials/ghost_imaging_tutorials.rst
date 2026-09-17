Ghost imaging
=============

This tutorial introduces basic ghost imaging utilities integrated in ``mitransient``.

The helper API is available in ``mitransient.gi``:

* ``simulate_bucket_measurements(patterns, target, noise_std=0.0, seed=None)``
* ``reconstruct_ghost_image(patterns, measurements, normalize=True)``
* ``reconstruct_transient_ghost_image(patterns, transient_measurements, normalize=True)``

See the runnable example script:

* ``examples/ghost-imaging/basic_ghost_imaging.py``
