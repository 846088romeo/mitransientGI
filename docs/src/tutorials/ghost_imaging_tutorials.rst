Ghost imaging
=============

This tutorial introduces basic ghost imaging utilities integrated in ``mitransient``.

The helper API is available in ``mitransient.gi``:

* ``simulate_bucket_measurements(patterns, target, noise_std=0.0, seed=None)``
* ``reconstruct_ghost_image(patterns, measurements, normalize=True)``
* ``reconstruct_transient_ghost_image(patterns, transient_measurements, normalize=True)``

See the runnable example script:

* ``examples/ghost-imaging/basic_ghost_imaging.py``

Quantum/transient simulator
---------------------------

The scene-based simulator is available through ``mitransient.qghost`` and as
the ``qghost`` command after installation. It loads a Mitsuba XML scene,
simulates correlated signal/idler pairs, and writes a temporal histogram plus
derived intensity and depth images:

.. code-block:: console

	qghost --xml-scene path/to/scene.xml --spp 400000 --out-dir ghost_output

The simulator expects Mitsuba to be installed and uses the variant selected by
the application. A standalone command selects ``scalar_rgb`` only when no
variant has been selected previously.