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
        self.signal_jitter_ps = mi.Float(props.get("signal_jitter_ps", 0.0))
        self.bucket_jitter_ps = mi.Float(props.get("bucket_jitter_ps", 0.0))
        self.time_quantization_ps = mi.Float(props.get("time_quantization_ps", 0.0))
        self.pde = mi.Float(props.get("pde", 1.0))

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
        # signal detector.
        lane_count = sample_count
        sampler.seed(seed, lane_count)

        # Sample correlated signal and idler rays from the SPDC emitter. The
        # signal ray is traced to the SPAD, while the idler ray is traced to
        # the scene and then reflected to the bucket detector.
        d_signal, d_idler, pair_weight = emitter.sample_pair(
            sample=sampler.next_2d(),
            noise_sample_idler=sampler.next_2d(),
            noise_sample_signal=sampler.next_2d(),
            active=mi.Bool(True)
        )

        # Compute the intersection of the signal ray with the SPAD plane and
        # convert to film coordinates.
        source = emitter.source
        signal_direction = d_signal.d
        idler_direction = d_idler.d
        signal_z = source.z + self.spad_z
        signal_t = (signal_z - source.z) / signal_direction.z
        signal_p = source + signal_t * signal_direction

        # Convert the signal intersection point to film coordinates.
        signal_x = ((-(signal_p.x - source.x) + self.spad_half_width) /
                    (2.0 * self.spad_half_width)) * film.size().x
        signal_y = ((-(signal_p.y - source.y) + self.spad_half_height) /
                    (2.0 * self.spad_half_height)) * film.size().y
        signal_active = (signal_t > 1e-6)

        # Check SPAD bounds and photon detection efficiency (PDE)
        signal_active &= abs(signal_p.x - source.x) <= self.spad_half_width
        signal_active &= abs(signal_p.y - source.y) <= self.spad_half_height
        signal_active &= sampler.next_1d() < self.pde

        # Trace the idler ray to the scene
        scene_si = scene.ray_intersect(d_idler)
        scene_active = signal_active & scene_si.is_valid()
        normal = scene_si.sh_frame.n
        normal = dr.select(dr.dot(normal, idler_direction) > 0.0,
                   -normal, normal)
        
        # Ensure the normal is pointing in the correct direction
        normal = dr.normalize(normal)
        local_sample = sampler.next_2d(scene_active)
        radius = dr.sqrt(local_sample.x)
        phi = 2.0 * dr.pi * local_sample.y
        helper = dr.select(abs(normal.y) < 0.9,
                           mi.Vector3f(0, 1, 0), mi.Vector3f(1, 0, 0))
        tangent = dr.normalize(dr.cross(helper, normal))
        bitangent = dr.cross(normal, tangent)

        # Sample a reflected ray from the scene intersection point
        reflected = dr.normalize(
            (radius * dr.cos(phi)) * tangent +
            (radius * dr.sin(phi)) * bitangent +
            dr.sqrt(dr.maximum(0.0, 1.0 - local_sample.x)) * normal
        )

        # Trace the reflected ray to the bucket detector and check if it falls within the bucket radius
        bucket_normal = dr.normalize(emitter.target - source)
        denominator = dr.dot(bucket_normal, reflected)
        bucket_t = dr.dot(source - scene_si.p, bucket_normal) / denominator
        bucket_p = scene_si.p + bucket_t * reflected

        # Check if the bucket intersection point is within the bucket radius and if the ray is valid
        bucket_active = scene_active & (denominator < -1e-8) & (bucket_t > 1e-6)
        bucket_active &= dr.norm(bucket_p - source) <= self.bucket_radius
        bucket_active &= sampler.next_1d() < self.pde

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

        # Compute the optical path lengths (OPL) for the signal and bucket detections
        signal_opl = emission_opl + signal_t
        bucket_opl = emission_opl + scene_si.t + bucket_t

        # Apply jitter to the signal and bucket OPLs based on the specified jitter in picoseconds
        signal_jitter_opl = (self.signal_jitter_ps * 1e-12 * c)
        bucket_jitter_opl = (self.bucket_jitter_ps * 1e-12 * c)

        # Calculate the jittered OPLs
        u1_signal, u2_signal = sampler.next_1d(), sampler.next_1d()
        z_signal = (dr.sqrt(-2.0 * dr.log(dr.maximum(u1_signal, 1e-10)))
                     * dr.cos(2.0 * dr.pi * u2_signal))
        u1_bucket, u2_bucket = sampler.next_1d(), sampler.next_1d()
        z_bucket = (dr.sqrt(-2.0 * dr.log(dr.maximum(u1_bucket, 1e-10)))
                     * dr.cos(2.0 * dr.pi * u2_bucket))

        signal_opl += signal_jitter_opl * z_signal
        bucket_opl += bucket_jitter_opl * z_bucket

        # Temporal quatization
        quant_step_opl = self.time_quantization_ps * 1e-12 * c
        quant_active = quant_step_opl > 0.0
        signal_opl = dr.select(quant_active,
                               dr.round(signal_opl / dr.maximum(quant_step_opl, 1e-10)) * quant_step_opl,
                               signal_opl)
        bucket_opl = dr.select(quant_active,
                               dr.round(bucket_opl / dr.maximum(quant_step_opl, 1e-10)) * quant_step_opl,
                               bucket_opl)
        
        event_weight = self.event_weight * pair_weight
        film.append_events(
            pixel=mi.Point2f(signal_x, signal_y),
            signal_opl=signal_opl,
            bucket_opl=bucket_opl,
            value=event_weight,
            signal_active=signal_active,
            bucket_active=bucket_active,
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
        callback.put("signal_jitter_ps", self.signal_jitter_ps, 
                     mi.ParamFlags.NonDifferentiable)
        callback.put("bucket_jitter_ps", self.bucket_jitter_ps, 
                     mi.ParamFlags.NonDifferentiable)
        callback.put("time_quantization_ps", self.time_quantization_ps,
                     mi.ParamFlags.NonDifferentiable)
        callback.put("pde", self.pde, mi.ParamFlags.NonDifferentiable)
    def to_string(self):
        return "QGhostIntegrator[statistical streams, primal]"


mi.register_integrator("qghost", lambda props: QGhostIntegrator(props))