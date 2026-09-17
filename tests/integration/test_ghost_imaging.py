import numpy as np
from importlib.util import module_from_spec, spec_from_file_location
from pathlib import Path


def _ghost_imaging_module():
    module_path = (
        Path(__file__).resolve().parents[2] / "mitransient" / "ghost_imaging.py"
    )
    spec = spec_from_file_location("mitransient_ghost_imaging", module_path)
    module = module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_ghost_imaging_reconstruction_shape_and_signal():
    gi = _ghost_imaging_module()

    rng = np.random.default_rng(3)
    h, w = 16, 16
    n = 1200

    target = np.zeros((h, w), dtype=np.float64)
    target[4:12, 5:11] = 1.0

    patterns = rng.integers(0, 2, size=(n, h, w)).astype(np.float64)
    measurements = gi.simulate_bucket_measurements(patterns, target)
    reconstruction = gi.reconstruct_ghost_image(patterns, measurements)

    assert measurements.shape == (n,)
    assert reconstruction.shape == (h, w)

    inside = reconstruction[4:12, 5:11].mean()
    outside = np.concatenate(
        [reconstruction[:4], reconstruction[12:]], axis=0
    ).mean()
    assert inside > outside


def test_transient_ghost_imaging_reconstruction_shape():
    gi = _ghost_imaging_module()

    rng = np.random.default_rng(7)
    h, w = 8, 8
    n = 400
    t = 5

    target = np.zeros((h, w), dtype=np.float64)
    target[2:6, 2:6] = 1.0
    patterns = rng.integers(0, 2, size=(n, h, w)).astype(np.float64)
    base = gi.simulate_bucket_measurements(patterns, target)
    transient_measurements = np.stack([base * (i + 1) for i in range(t)], axis=1)

    recon = gi.reconstruct_transient_ghost_image(
        patterns, transient_measurements
    )
    assert recon.shape == (h, w, t)
