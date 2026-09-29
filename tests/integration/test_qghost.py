import numpy as np


def _qghost_module():
    import mitsuba as mi

    if mi.variant() is None:
        mi.set_variant("llvm_ad_rgb")

    import mitransient.qghost as qghost

    return qghost


def test_qghost_geometry_and_pair_sampling():
    qghost = _qghost_module()

    rng = np.random.default_rng(4)
    source = np.array([0.0, 0.0, 0.0])
    target = np.array([0.0, 0.0, 2.0])
    d_signal, d_idler, p_signal, p_idler = qghost.sample_correlated_pair(
        rng,
        max_angle_deg=10.0,
        sigma=0.0,
        scene_target=target,
        source=source,
    )

    assert np.isclose(np.linalg.norm(d_signal), 1.0)
    assert np.isclose(np.linalg.norm(d_idler), 1.0)
    assert d_signal[2] < 0.0
    assert d_idler[2] > 0.0
    assert (p_signal, p_idler) in ((1, -1), (-1, 1))

    bucket = qghost.DiskBucket(
        center=source,
        normal=np.array([0.0, 0.0, 1.0]),
        radius=1.0,
    )
    assert bucket.intersect(
        np.array([0.0, 0.0, 1.0]), np.array([0.0, 0.0, -1.0])
    ) == 1.0
    assert bucket.intersect(
        np.array([2.0, 0.0, 1.0]), np.array([0.0, 0.0, -1.0])
    ) is None


def test_qghost_histogram_and_depth_map():
    qghost = _qghost_module()

    hist = qghost.correlate_events_to_histogram(
        signal_events=[(1, 0, 1.0)],
        bucket_events=[2.0],
        width=3,
        height=2,
        time_bins=2,
        t_min=0.0,
        t_max=2.0,
    )
    assert hist.shape == (2, 3, 2)
    assert hist[0, 1, 1] == 1.0

    t_edges = np.array([0.0, 1.0, 2.0])
    depth = qghost.depth_map_from_hist(
        hist,
        t_edges,
        signal_time_offset=1.0,
    )
    assert np.isnan(depth[1, 0])
    assert np.isfinite(depth[0, 1])