"""Vectorized statistical ghost-imaging integrator."""

from __future__ import annotations

import drjit as dr
import mitsuba as mi
import numpy as np

from ..films.ghost_film import GhostTransientFilm
from ..emitters.spdc import SPDCPairEmitter
from .common import TransientADIntegrator


class QGhostIntegrator(TransientADIntegrator):
    """Generate correlated rays and record unlabelled detector streams."""

    def __init__(self, props: mi.Properties):
        super().__init__(props)
        self.spad_z = mi.Float(props.get("spad_z", -1.0))
        self.spad_half_width = mi.Float(props.get("spad_half_width", 0.3))
        self.spad_half_height = mi.Float(props.get("spad_half_height", 0.3))
        self.bucket_radius = mi.Float(props.get("bucket_radius", 3.0))
        self.event_weight = mi.Float(props.get("event_weight", 1.0))
        self.pair_rate_hz = mi.Float(props.get("pair_rate_hz", 1e10))
        self.emission_opl_offset = mi.Float(0.0)
        self.debug = mi.Bool(props.get("debug", False))

    def render(self, scene, sensor=0, seed=0, spp=0, develop=True,
               evaluate=True, progress_callback=None):
        if isinstance(sensor, int):
            sensor = scene.sensors()[sensor]
        film = sensor.film()
        if not isinstance(film, GhostTransientFilm):
            raise TypeError("qghost requires ghost_film")

        film.prepare([])
        emitters = scene.emitters()
        if len(emitters) != 1 or not isinstance(emitters[0], SPDCPairEmitter):
            raise TypeError(
                "qghost requires exactly one emitter of type spdc_pair"
            )
        emitter = emitters[0]

        sampler = sensor.sampler().clone()
        sample_count = spp if spp else sampler.sample_count()
        sampler.set_sample_count(sample_count)
        sampler.set_samples_per_wavefront(sample_count)
        # One sample is one generated pair. The film dimensions discretize the
        # signal detector; they do not multiply the number of emitted pairs.
        lane_count = sample_count
        sampler.seed(seed, lane_count)

        d_signal, d_idler, pair_weight = emitter.sample_pair(
            sample=sampler.next_2d(),
            noise_sample_idler=sampler.next_2d(),
            noise_sample_signal=sampler.next_2d(),
            active=mi.Bool(True)
        )
        source = emitter.source
        signal_direction = d_signal.d
        idler_direction = d_idler.d
        signal_z = source.z + self.spad_z
        signal_t = (signal_z - source.z) / signal_direction.z
        signal_p = source + signal_t * signal_direction
        signal_x = ((-(signal_p.x - source.x) + self.spad_half_width) /
                    (2.0 * self.spad_half_width)) * film.size().x
        signal_y = ((-(signal_p.y - source.y) + self.spad_half_height) /
                    (2.0 * self.spad_half_height)) * film.size().y
        signal_active = (signal_t > 1e-6)
        signal_active &= abs(signal_p.x - source.x) <= self.spad_half_width
        signal_active &= abs(signal_p.y - source.y) <= self.spad_half_height

        scene_si = scene.ray_intersect(d_idler)
        scene_active = signal_active & scene_si.is_valid()
        normal = scene_si.sh_frame.n
        normal = dr.select(dr.dot(normal, idler_direction) > 0.0,
                   -normal, normal)
        local_sample = sampler.next_2d(scene_active)
        radius = dr.sqrt(local_sample.x)
        phi = 2.0 * dr.pi * local_sample.y
        helper = dr.select(abs(normal.y) < 0.9,
                           mi.Vector3f(0, 1, 0), mi.Vector3f(1, 0, 0))
        tangent = dr.normalize(dr.cross(helper, normal))
        bitangent = dr.cross(normal, tangent)
        reflected = dr.normalize(
            (radius * dr.cos(phi)) * tangent +
            (radius * dr.sin(phi)) * bitangent +
            dr.sqrt(dr.maximum(0.0, 1.0 - local_sample.x)) * normal
        )

        bucket_normal = dr.normalize(emitter.target - source)
        denominator = dr.dot(bucket_normal, reflected)
        bucket_t = dr.dot(source - scene_si.p, bucket_normal) / denominator
        bucket_p = scene_si.p + bucket_t * reflected
        bucket_active = scene_active & (denominator < -1e-8) & (bucket_t > 1e-6)
        bucket_active &= dr.norm(bucket_p - source) <= self.bucket_radius

        # Give each generated pair an absolute emission OPL. The film does not
        # store a pair id: this clock lets its cross-correlation recover the
        # temporal peak while retaining accidental coincidences statistically.
        # lane = dr.arange(mi.UInt32, sample_count)
        # emission_opl = self.emission_opl_offset + mi.Float(lane) * (
        #     mi.Float(299792458.0) / self.pair_rate_hz
        # )

        # Random sampling U in (0,1]
        u = sampler.next_1d()
        u = dr.maximum(u, 1e-10)

        # Conversion from pair rate to mean OPL separation, then sample an exponential
        c = mi.Float(299792458.0)
        mean_delta_opl = c / self.pair_rate_hz
        delta_opl = -mean_delta_opl * dr.log(u)

        # Calculate the cumulative sum to obtain absolute emission times
        emission_opl = dr.prefix_sum(delta_opl)
        
        event_weight = self.event_weight * pair_weight
        film.append_events(
            mi.Point2f(signal_x, signal_y),
            emission_opl + signal_t,
            emission_opl + scene_si.t + bucket_t,
            event_weight,
            signal_active,
            bucket_active,
        )

        if self.debug:
            film.add_debug_events(
                mi.Point2f(signal_x, signal_y),
                signal_active,
                scene_active,
                bucket_active,
            )
            film.debug_stats = {
                "pairs": sample_count,
                "signal_hits": int(np.asarray(
                    dr.sum(mi.UInt32(signal_active))).item()),
                "scene_hits": int(np.asarray(
                    dr.sum(mi.UInt32(scene_active))).item()),
                "bucket_hits": int(np.asarray(
                    dr.sum(mi.UInt32(bucket_active))).item()),
            }
            dr.eval(film.debug_signal_storage,
                    film.debug_scene_storage,
                    film.debug_bucket_storage)
        if progress_callback:
            progress_callback(1.0)
        return film.develop() if develop else None

    def traverse(self, callback):
        super().traverse(callback)
        callback.put("spad_z", self.spad_z, mi.ParamFlags.NonDifferentiable)
        callback.put("spad_half_width", self.spad_half_width,
                     mi.ParamFlags.NonDifferentiable)
        callback.put("spad_half_height", self.spad_half_height,
                     mi.ParamFlags.NonDifferentiable)
        callback.put("bucket_radius", self.bucket_radius,
                     mi.ParamFlags.NonDifferentiable)
        callback.put("pair_rate_hz", self.pair_rate_hz,
                     mi.ParamFlags.NonDifferentiable)
        callback.put("emission_opl_offset", self.emission_opl_offset,
                 mi.ParamFlags.NonDifferentiable)
        callback.put("event_weight", self.event_weight,
                     mi.ParamFlags.NonDifferentiable)
        callback.put("debug", self.debug, mi.ParamFlags.NonDifferentiable)
    def to_string(self):
        return "QGhostIntegrator[statistical streams, primal]"


mi.register_integrator("qghost", lambda props: QGhostIntegrator(props))