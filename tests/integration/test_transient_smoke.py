"""Small end-to-end check for the core transient rendering pipeline."""

import pytest
import os


def test_transient_path_renders_transient_film():
    dr = pytest.importorskip("drjit")
    mi = pytest.importorskip("mitsuba")

    if mi.variant() is None:
        mi.set_variant(os.environ.get("MITRANSIENT_VARIANT", "llvm_ad_rgb"))
    if mi.variant().startswith("scalar"):
        pytest.skip("transient plugins require a non-scalar Mitsuba variant")

    import mitransient as mitr

    transform = mi.ScalarTransform4f
    scene = mi.load_dict(
        {
            "type": "scene",
            "integrator": {
                "type": "transient_path",
                "max_depth": 2,
                "temporal_filter": "box",
            },
            "sensor": {
                "type": "perspective",
                "fov": 45,
                "to_world": transform.look_at(
                    origin=[0, 0, 3],
                    target=[0, 0, 0],
                    up=[0, 1, 0],
                ),
                "sampler": {"type": "independent", "sample_count": 4},
                "film": {
                    "type": "transient_hdr_film",
                    "width": 8,
                    "height": 8,
                    "temporal_bins": 8,
                    "start_opl": 2.0,
                    "bin_width_opl": 0.5,
                    "rfilter": {"type": "box"},
                },
            },
            "shape": {
                "type": "sphere",
                "bsdf": {
                    "type": "diffuse",
                    "reflectance": {"type": "rgb", "value": [0.8, 0.8, 0.8]},
                },
            },
            "light": {
                "type": "rectangle",
                "to_world": transform.translate([0, 1.5, 1.5]).rotate(
                    [1, 0, 0], 180
                ).scale(0.5),
                "emitter": {
                    "type": "area",
                    "radiance": {"type": "rgb", "value": [8, 8, 8]},
                },
            },
        }
    )

    integrator = scene.integrator()
    steady, transient = integrator.render(scene)
    dr.eval(steady, transient)

    assert steady.shape == (8, 8, 3)
    assert transient.shape == (8, 8, 8, 3)
    assert bool(dr.any(transient > 0.0))
    assert mitr.speed_of_light == pytest.approx(299792458.0)