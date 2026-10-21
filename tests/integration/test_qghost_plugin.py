import os

import pytest


def test_qghost_plugin_renders_one_bounce_histogram():
    dr = pytest.importorskip("drjit")
    mi = pytest.importorskip("mitsuba")

    if mi.variant() is None:
        mi.set_variant(os.environ.get("MITRANSIENT_VARIANT", "llvm_ad_rgb"))
    if mi.variant().startswith("scalar"):
        pytest.skip("ghost plugins require a non-scalar Mitsuba variant")

    import mitransient

    transform = mi.ScalarTransform4f
    scene = mi.load_dict({
        "type": "scene",
        "integrator": {
            "type": "qghost",
            "spad_z": -1.0,
            "spad_half_width": 0.5,
            "spad_half_height": 0.5,
            "bucket_radius": 1.0,
        },
        "emitter": {
            "type": "spdc_pair",
            "source": [0.0, 0.0, 0.0],
            "target": [0.0, 0.0, 2.0],
            "max_angle_deg": 8.0,
        },
        "sensor": {
            "type": "perspective",
            "to_world": transform.look_at(
                origin=[0.0, 0.0, -1.0],
                target=[0.0, 0.0, 0.0],
                up=[0.0, 1.0, 0.0],
            ),
            "sampler": {
                "type": "independent",
                "sample_count": 4,
            },
            "film": {
                "type": "ghost_film",
                "width": 8,
                "height": 8,
                "event_bins": 512,
                "event_start_opl": 0.0,
                "event_bin_width_opl": 0.01,
                "temporal_bins": 512,
                "start_opl": -1.0,
                "bin_width_opl": 0.01,
            },
        },
        "shape": {
            "type": "sphere",
            "to_world": transform.translate([0.0, 0.0, 2.0]),
            "bsdf": {
                "type": "diffuse",
                "reflectance": 1.0,
            },
        },
    })

    steady, transient = scene.integrator().render(scene, spp=4)
    dr.eval(steady, transient)

    assert transient.shape == (8, 8, 512, 1)
    assert bool(dr.all(dr.isfinite(transient)))
    assert bool(dr.any(transient > 0.0))