"""Public access to the quantum ghost-imaging simulator.

The implementation remains in :mod:`ghostImaging.qghost` so it can also be
used by the standalone ``qghost`` console command.
"""

from ghostImaging.qghost import (
    C,
    DiskBucket,
    SphereBucket,
    correlate_events_to_histogram,
    depth_map_from_hist,
    estimate_time_window_from_bbox,
    infer_framing_from_bounds,
    main,
    normalize,
    quantize_time,
    sample_correlated_pair,
    sample_cosine_hemisphere,
    save_outputs,
    simulate,
)

__all__ = [
    "C",
    "DiskBucket",
    "SphereBucket",
    "correlate_events_to_histogram",
    "depth_map_from_hist",
    "estimate_time_window_from_bbox",
    "infer_framing_from_bounds",
    "main",
    "normalize",
    "quantize_time",
    "sample_correlated_pair",
    "sample_cosine_hemisphere",
    "save_outputs",
    "simulate",
]