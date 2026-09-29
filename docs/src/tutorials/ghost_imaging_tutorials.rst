Ghost imaging
=============

This tutorial introduces the Mitsuba-backed quantum ghost-imaging simulator
integrated in ``mitransient``. It operates on Mitsuba XML scenes and records
correlated signal/idler events in a temporal histogram.

Quantum/transient simulator
---------------------------

The scene-based simulator is available as ``mitransient.qghost``. From a
source checkout, run the module file directly. It loads a Mitsuba XML scene,
simulates correlated signal/idler pairs, and writes a temporal histogram plus
derived intensity and depth images:

.. code-block:: console

	python mitransient/qghost.py --xml-scene path/to/scene.xml --spp 400000 --out-dir ghost_output

The simulator expects Mitsuba to be installed and uses the variant selected by
the application. A standalone command selects ``scalar_rgb`` only when no
variant has been selected previously.