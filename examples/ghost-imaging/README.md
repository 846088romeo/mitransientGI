# Ghost imaging examples

This folder contains examples for ghost imaging simulation and reconstruction.

The quantum/transient simulator lives in ``ghostImaging.qghost`` and is
installed as the ``qghost`` command. It requires a Mitsuba XML scene:

```bash
qghost --xml-scene path/to/scene.xml --spp 400000 --out-dir ghost_output
```

The importable helpers in ``mitransient.gi`` are intentionally separate: they
operate on illumination patterns and bucket measurements without requiring a
Mitsuba scene.

- `basic_ghost_imaging.py`: Minimal end-to-end example using
  `mitransient.gi.simulate_bucket_measurements` and
  `mitransient.gi.reconstruct_ghost_image`.
  