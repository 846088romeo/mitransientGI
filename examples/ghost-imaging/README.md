# Ghost imaging examples

This folder contains examples for ghost imaging simulation and reconstruction.

The quantum/transient simulator lives in ``mitransient/qghost.py``. From a
source checkout, run it with a Mitsuba XML scene:

```bash
python mitransient/qghost.py --xml-scene path/to/scene.xml --spp 400000 --out-dir ghost_output
```

The simulator API is also available from Python:

```python
from mitransient.qghost import simulate, DiskBucket
```

The same implementation can be imported as ``mitransient.qghost`` after the
application has selected a Mitsuba variant.
  