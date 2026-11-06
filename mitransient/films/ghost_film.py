"""Film for statistically correlated SPAD-array and bucket detections."""

from __future__ import annotations

from typing import Sequence

import drjit as dr
import mitsuba as mi
import numpy as np
from scipy import signal as scipy_signal
from tqdm import tqdm


class GhostTransientFilm(mi.Film):
    """Stores two unlabelled detection streams and their temporal correlation."""

    def __init__(self, props: mi.Properties):
        super().__init__(props)
        self.temporal_bins = props.get("temporal_bins", mi.UInt32(256))
        self.start_opl = props.get("start_opl", mi.Float(0.0))
        self.bin_width_opl = props.get("bin_width_opl", mi.Float(0.003))
        self.event_bins = props.get("event_bins", mi.UInt32(self.temporal_bins))
        self.event_start_opl = props.get("event_start_opl", mi.Float(0.0))
        self.event_bin_width_opl = props.get(
            "event_bin_width_opl", mi.Float(self.bin_width_opl)
        )

        if self.sample_border():
            mi.Log(mi.LogLevel.Error,
                   "ghost_film does not support sample_border")

    def base_channels_count(self):
        return 1

    def prepare(self, aovs: Sequence[str]):
        if aovs:
            mi.Log(mi.LogLevel.Warn,
                   "ghost_film ignores AOVs; detections are scalar counts")

        width = int(self.size().x)
        height = int(self.size().y)
        if not hasattr(self, "signal_event_x"):
            self.signal_event_x = []
            self.signal_event_y = []
            self.signal_event_opl = []
            self.signal_event_value = []
            self.bucket_event_opl = []
            self.bucket_event_value = []
            self.debug_signal_storage = mi.TensorXf(
                dr.zeros(mi.Float, width * height), shape=(height, width)
            )
            self.debug_scene_storage = mi.TensorXf(
                dr.zeros(mi.Float, width * height), shape=(height, width)
            )
            self.debug_bucket_storage = mi.TensorXf(
                dr.zeros(mi.Float, 1), shape=(1,)
            )
        self.channels = ["G2"]
        return 1

    def create_block(self):
        raise NotImplementedError(
            "ghost_film accumulates detection streams directly"
        )

    def clear(self):
        self.signal_event_x = []
        self.signal_event_y = []
        self.signal_event_opl = []
        self.signal_event_value = []
        self.bucket_event_opl = []
        self.bucket_event_value = []
        width = int(self.size().x)
        height = int(self.size().y)
        if not hasattr(self, "debug_signal_storage"):
            self.debug_signal_storage = mi.TensorXf(
                dr.zeros(mi.Float, width * height), shape=(height, width)
            )
            self.debug_scene_storage = mi.TensorXf(
                dr.zeros(mi.Float, width * height), shape=(height, width)
            )
            self.debug_bucket_storage = mi.TensorXf(
                dr.zeros(mi.Float, 1), shape=(1,)
            )
        else:
            self.debug_signal_storage.array = dr.zeros(
                mi.Float, self.debug_signal_storage.array.shape
            )
            self.debug_scene_storage.array = dr.zeros(
                mi.Float, self.debug_scene_storage.array.shape
            )
            self.debug_bucket_storage.array = dr.zeros(
                mi.Float, self.debug_bucket_storage.array.shape
            )

    def _event_index(self, opl: mi.Float):
        return dr.floor((opl - self.event_start_opl) / self.event_bin_width_opl)

    def append_events(self, pixel: mi.Point2f, emission_opl:mi.Float,
                      signal_opl: mi.Float, bucket_opl: mi.Float,
                      value: mi.Float, signal_active: mi.Bool,
                      bucket_active: mi.Bool):
        """Store only valid events from one vectorized batch."""
        signal_indices = dr.compress(signal_active)
        bucket_indices = dr.compress(bucket_active)

        if dr.width(signal_indices) > 0:
            signal_x = dr.gather(mi.Float, pixel.x, signal_indices)
            signal_y = dr.gather(mi.Float, pixel.y, signal_indices)

            # 64-bit extraction of the OPL values to avoid precision loss in the correlation
            e_opl_signal = np.asarray(dr.gather(mi.Float, emission_opl, signal_indices), dtype=np.float64)
            s_rel = np.asarray(dr.gather(mi.Float, signal_opl, signal_indices), dtype=np.float64)
            signal_time = e_opl_signal + s_rel
            signal_value = dr.gather(mi.Float, value, signal_indices)
            dr.eval(signal_x, signal_y, signal_value)

            self.signal_event_x.append(np.asarray(signal_x).copy())
            self.signal_event_y.append(np.asarray(signal_y).copy())
            self.signal_event_opl.append(signal_time.copy())
            self.signal_event_value.append(np.asarray(signal_value).copy())

        if dr.width(bucket_indices) > 0:
            e_opl_bucket = np.asarray(dr.gather(mi.Float, emission_opl, bucket_indices), dtype=np.float64)
            b_rel = np.asarray(dr.gather(mi.Float, bucket_opl, bucket_indices), dtype=np.float64)
            bucket_time = e_opl_bucket + b_rel
            bucket_value = dr.gather(mi.Float, value, bucket_indices)
            dr.eval(bucket_value)
            
            self.bucket_event_opl.append(bucket_time.copy())
            self.bucket_event_value.append(np.asarray(bucket_value).copy())

    def add_debug_events(self, pixel: mi.Point2f, signal_active: mi.Bool,
                         scene_active: mi.Bool, bucket_active: mi.Bool):
        """Accumulate stage counts without affecting the correlation."""
        pixel_x = mi.UInt64(dr.floor(pixel.x))
        pixel_y = mi.UInt64(dr.floor(pixel.y))
        width = mi.UInt64(self.size().x)
        height = mi.UInt64(self.size().y)
        valid = ((pixel_x >= 0) & (pixel_x < width) &
                 (pixel_y >= 0) & (pixel_y < height))
        index = pixel_y * width + pixel_x
        dr.scatter_reduce(dr.ReduceOp.Add, self.debug_signal_storage.array,
                          mi.Float(1.0), index, valid & signal_active)
        dr.scatter_reduce(dr.ReduceOp.Add, self.debug_scene_storage.array,
                          mi.Float(1.0), index, valid & scene_active)
        dr.scatter_reduce(dr.ReduceOp.Add, self.debug_bucket_storage.array,
                          mi.Float(1.0), mi.UInt32(0), bucket_active)


    def develop(self, raw: bool = False):
        """Return integrated correlation and ``(y, x, delta_opl, 1)``."""
        height = int(self.size().y)
        width = int(self.size().x)
        temporal_bins = int(np.asarray(self.temporal_bins).item())
        start_opl = float(np.asarray(self.start_opl).item())
        bin_width_opl = float(np.asarray(self.bin_width_opl).item())
        correlation = np.zeros((height, width, temporal_bins), dtype=np.float64)

        if self.signal_event_opl and self.bucket_event_opl:
            signal_x = np.concatenate(self.signal_event_x).astype(np.intp)
            signal_y = np.concatenate(self.signal_event_y).astype(np.intp)
            signal_opl = np.concatenate(self.signal_event_opl)
            signal_value = np.concatenate(self.signal_event_value)
            bucket_opl = np.concatenate(self.bucket_event_opl)
            bucket_value = np.concatenate(self.bucket_event_value)

            order = np.argsort(signal_opl)
            signal_x = signal_x[order]
            signal_y = signal_y[order]
            signal_opl = signal_opl[order]
            signal_value = signal_value[order]
            delta_min = start_opl
            delta_max = start_opl + temporal_bins * bin_width_opl

            chunk_size = 100000
            num_buckets = len(bucket_opl)

            for i in tqdm(range(0, num_buckets, chunk_size), desc="Correlating events", unit="chunk"):
                b_opl_chunk = bucket_opl[i:i + chunk_size]
                b_val_chunk = bucket_value[i:i + chunk_size]

                # Vectorized search for signal events that fall within the correlation window of each bucket event
                los = np.searchsorted(signal_opl, b_opl_chunk - delta_max, side="right")
                his = np.searchsorted(signal_opl, b_opl_chunk - delta_min, side="left")

                # Filter out bucket events that have no corresponding signal events in the correlation window
                counts = his - los
                mask = counts > 0
                if not np.any(mask):
                    continue

                b_opl_valid = b_opl_chunk[mask]
                b_val_valid = b_val_chunk[mask]
                los_valid = los[mask]
                counts_valid = counts[mask]

                # Vectorized expansion of bucket events to match the number of corresponding signal events
                b_opl_exp = np.repeat(b_opl_valid, counts_valid)
                b_val_exp = np.repeat(b_val_valid, counts_valid)

                # Generate indices for the corresponding signal events
                offsets = np.arange(np.sum(counts_valid)) - np.repeat(np.cumsum(counts_valid) - counts_valid, counts_valid)
                sig_idx = np.repeat(los_valid, counts_valid) + offsets

                # 1D arrays of signal events corresponding to the expanded bucket events
                delta = b_opl_exp - signal_opl[sig_idx]
                bins = np.floor((delta - start_opl) / bin_width_opl).astype(np.intp)
                valid = (bins >= 0) & (bins < temporal_bins)

                if np.any(valid):
                    np.add.at(
                        correlation,
                        (signal_y[sig_idx][valid], signal_x[sig_idx][valid], bins[valid]),
                        signal_value[sig_idx][valid] * b_val_exp[valid],
                    )

        transient = mi.TensorXf(correlation[..., None].astype(np.float32))
        steady = mi.TensorXf(np.sum(correlation, axis=2)[..., None].astype(np.float32))
        return steady, transient

    def develop_for(self, raw: bool = False):
        """Return integrated correlation and ``(y, x, delta_opl, 1)``."""
        height = int(self.size().y)
        width = int(self.size().x)
        temporal_bins = int(np.asarray(self.temporal_bins).item())
        start_opl = float(np.asarray(self.start_opl).item())
        bin_width_opl = float(np.asarray(self.bin_width_opl).item())
        correlation = np.zeros((height, width, temporal_bins), dtype=np.float64)

        if self.signal_event_opl and self.bucket_event_opl:
            signal_x = np.concatenate(self.signal_event_x).astype(np.intp)
            signal_y = np.concatenate(self.signal_event_y).astype(np.intp)
            signal_opl = np.concatenate(self.signal_event_opl)
            signal_value = np.concatenate(self.signal_event_value)
            bucket_opl = np.concatenate(self.bucket_event_opl)
            bucket_value = np.concatenate(self.bucket_event_value)

            order = np.argsort(signal_opl)
            signal_x = signal_x[order]
            signal_y = signal_y[order]
            signal_opl = signal_opl[order]
            signal_value = signal_value[order]
            delta_min = start_opl
            delta_max = start_opl + temporal_bins * bin_width_opl

            chunk_size = 50000
            num_buckets = len(bucket_opl)

            for i in range(0, num_buckets, chunk_size):
                b_opl_chuck = bucket_opl[i:i + chunk_size]
                b_val_chunk = bucket_value[i:i + chunk_size]

                los = np.searchsorted(signal_opl, b_opl_chuck - delta_max, side="right")
                his = np.searchsorted(signal_opl, b_opl_chuck - delta_min, side="left")

                for b_idx in range(len(b_opl_chuck)):
                    lo = los[b_idx]
                    hi = his[b_idx]
                    if lo >= hi:
                        continue

                    delta = b_opl_chuck[b_idx] - signal_opl[lo:hi]
                    bins = np.floor((delta - start_opl) / bin_width_opl).astype(np.intp)
                    valid = (bins >= 0) & (bins < temporal_bins)
                    if np.any(valid):
                        np.add.at(
                            correlation,
                            (signal_y[lo:hi][valid], signal_x[lo:hi][valid], bins[valid]),
                            signal_value[lo:hi][valid] * b_val_chunk[b_idx],
                        )

        transient = mi.TensorXf(correlation[..., None])
        steady = mi.TensorXf(np.sum(correlation, axis=2)[..., None])
        return steady, transient

    def to_string(self):
        return (
            "GhostTransientFilm[\n"
            f"  temporal_bins = {self.temporal_bins},\n"
            f"  start_opl = {self.start_opl},\n"
            f"  bin_width_opl = {self.bin_width_opl},\n"
            f"  event_bins = {self.event_bins},\n"
            f"  event_start_opl = {self.event_start_opl},\n"
            f"  event_bin_width_opl = {self.event_bin_width_opl}\n"
            "]"
        )

    def traverse(self, callback):
        super().traverse(callback)
        callback.put("temporal_bins", self.temporal_bins,
                     mi.ParamFlags.NonDifferentiable)
        callback.put("start_opl", self.start_opl,
                     mi.ParamFlags.NonDifferentiable)
        callback.put("bin_width_opl", self.bin_width_opl,
                     mi.ParamFlags.NonDifferentiable)
        callback.put("event_bins", self.event_bins,
                     mi.ParamFlags.NonDifferentiable)
        callback.put("event_start_opl", self.event_start_opl,
                     mi.ParamFlags.NonDifferentiable)
        callback.put("event_bin_width_opl", self.event_bin_width_opl,
                     mi.ParamFlags.NonDifferentiable)


mi.register_film("ghost_film", lambda props: GhostTransientFilm(props))