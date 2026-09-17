import numpy as np


def test_ghost_imaging_reconstruction_shape_and_signal():
    import mitsuba as mi
    mi.set_variant('llvm_ad_rgb')
    import mitransient as mitr

    rng = np.random.default_rng(3)
    h, w = 16, 16
    n = 1200

    target = np.zeros((h, w), dtype=np.float64)
    target[4:12, 5:11] = 1.0

    patterns = rng.integers(0, 2, size=(n, h, w)).astype(np.float64)
    measurements = mitr.gi.simulate_bucket_measurements(patterns, target)
    reconstruction = mitr.gi.reconstruct_ghost_image(patterns, measurements)

    assert measurements.shape == (n,)
    assert reconstruction.shape == (h, w)

    inside = reconstruction[4:12, 5:11].mean()
    outside = np.concatenate(
        [reconstruction[:4], reconstruction[12:]], axis=0
    ).mean()
    assert inside > outside


def test_transient_ghost_imaging_reconstruction_shape():
    import mitsuba as mi
    mi.set_variant('llvm_ad_rgb')
    import mitransient as mitr

    rng = np.random.default_rng(7)
    h, w = 8, 8
    n = 400
    t = 5

    target = np.zeros((h, w), dtype=np.float64)
    target[2:6, 2:6] = 1.0
    patterns = rng.integers(0, 2, size=(n, h, w)).astype(np.float64)
    base = mitr.gi.simulate_bucket_measurements(patterns, target)
    transient_measurements = np.stack([base * (i + 1) for i in range(t)], axis=1)

    recon = mitr.gi.reconstruct_transient_ghost_image(
        patterns, transient_measurements
    )
    assert recon.shape == (h, w, t)
