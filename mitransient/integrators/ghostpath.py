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
        self.debug = mi.Bool(props.get("debug", False))
        self.signal_jitter_ps = mi.Float(props.get("signal_jitter_ps", 0.0))
        self.bucket_jitter_ps = mi.Float(props.get("bucket_jitter_ps", 0.0))
        self.time_quantization_ps = mi.Float(props.get("time_quantization_ps", 0.0))
        self.pde = mi.Float(props.get("pde", 1.0))

        # Participating media parameters
        self.sigma_s = mi.Float(props.get("sigma_s", 0.0)) # Scattering coefficient
        self.sigma_a = mi.Float(props.get("sigma_a", 0.0)) # Absorption coefficient
        self.g_hg = mi.Float(props.get("g_hg", 0.0)) # Henyey-Greenstein phase function parameter

    def render(self, scene, sensor=0, seed=0, spp=0, develop=True,
               evaluate=True, progress_callback=None, batch_size=500000):

        if isinstance(sensor, int):
            sensor = scene.sensors()[sensor]
        film = sensor.film()
        if not isinstance(film, GhostTransientFilm):
            raise TypeError("qghost requires ghost_film")

        # Clean up the film and prepare for new rendering
        film.clear()
        film.prepare([])
        emitters = scene.emitters()
        if len(emitters) != 1 or not isinstance(emitters[0], SPDCPairEmitter):
            raise TypeError(
                "qghost requires exactly one emitter of type spdc_pair"
            )
        emitter = emitters[0]

        total_pairs = spp if spp else sensor.sampler().sample_count()
        num_batches = int(np.ceil(total_pairs / batch_size))

        current_opl_offset = 0.0

        c = mi.Float(299792458.0)
        mean_delta_opl = c / self.pair_rate_hz

        sigma_t = self.sigma_s + self.sigma_a
        has_medium = sigma_t > 1e-8
        albedo = dr.select(has_medium, self.sigma_s / dr.maximum(sigma_t, 1e-10), 0.0)

        for batch in range(num_batches):
            current_spp = min(batch_size, total_pairs - batch * batch_size)

            # Set up a sampler for the current batch
            sampler = sensor.sampler().clone()
            sampler.set_sample_count(current_spp)
            sampler.set_samples_per_wavefront(current_spp)
            sampler.seed(seed + batch, current_spp)

            # Sample correlated signal and idler rays from the SPDC emitter. The
            # signal ray is traced to the SPAD, while the idler ray is traced to
            # the scene and then reflected to the bucket detector.
            d_signal, d_idler, pair_weight = emitter.sample_pair(
                sample=sampler.next_2d(),
                noise_sample_idler=sampler.next_2d(),
                noise_sample_signal=sampler.next_2d(),
                active=mi.Bool(True)
            )

            ##############################
            ## -- SIGNAL RAY TRACING -- ##
            ##############################
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

            # Check SPAD bounds and photon detection efficiency (PDE)
            signal_active = (signal_t > 1e-6) & (abs(signal_p.x - source.x) <= self.spad_half_width) & \
                            (abs(signal_p.y - source.y) <= self.spad_half_height) & \
                            (sampler.next_1d() < self.pde)

            ##############################
            ## -- IDLER RAY TRACING -- ##
            ##############################

            # Trace the idler ray to the scene
            scene_si = scene.ray_intersect(d_idler)
            t_surface = dr.select(scene_si.is_valid(), scene_si.t, mi.Float(1e6))

            u_vol = dr.maximum(sampler.next_1d(), 1e-10)
            s_vol = dr.select(has_medium, -dr.log(u_vol) / dr.maximum(sigma_t, 1e-10), mi.Float(1e6))  # Sample a distance in the medium

            is_volume_event = (s_vol < t_surface) & has_medium

            # == SURFACE EVENT ==
            surface_active = scene_si.is_valid() & ~is_volume_event
            normal = scene_si.sh_frame.n
            normal = dr.select(dr.dot(normal, idler_direction) > 0.0,
                    -normal, normal)
        
            # Ensure the normal is pointing in the correct direction
            normal = dr.normalize(normal)
            local_sample = sampler.next_2d(surface_active)
            radius = dr.sqrt(local_sample.x)
            phi = 2.0 * dr.pi * local_sample.y
            helper = dr.select(abs(normal.y) < 0.9,
                            mi.Vector3f(0, 1, 0), mi.Vector3f(1, 0, 0))
            tangent = dr.normalize(dr.cross(helper, normal))
            bitangent = dr.cross(normal, tangent)

            # Sample a reflected ray from the scene intersection point
            reflected_surf = dr.normalize(
                (radius * dr.cos(phi)) * tangent +
                (radius * dr.sin(phi)) * bitangent +
                dr.sqrt(dr.maximum(0.0, 1.0 - local_sample.x)) * normal
            )

            # Trace the reflected ray to the bucket detector and check if it falls within the bucket radius
            bucket_normal = dr.normalize(emitter.target - source)
            denom_surf = dr.dot(bucket_normal, reflected_surf)
            t_bucket_surf = dr.dot(source - scene_si.p, bucket_normal) / denom_surf
            p_bucket_surf = scene_si.p + t_bucket_surf * reflected_surf

            # Check if the bucket intersection point is within the bucket radius and if the ray is valid
            bucket_active_surf = surface_active & (denom_surf < -1e-8) & (t_bucket_surf > 1e-6) & \
                  (dr.norm(p_bucket_surf - source) <= self.bucket_radius) & (sampler.next_1d() < self.pde)

            # Calculate the relative optical path length
            rel_opl_surf = scene_si.t + t_bucket_surf

            # == VOLUME EVENT ==
            p_vol = source + s_vol * idler_direction
            u_vol_dir = sampler.next_2d(is_volume_event)
            z_dir = 1.0 - 2.0 * u_vol_dir.x
            r_dir = dr.sqrt(dr.maximum(0.0, 1.0 - z_dir * z_dir))
            phi_dir = 2.0 * dr.pi * u_vol_dir.y
            d_vol_reflected = mi.Vector3f(r_dir * dr.cos(phi_dir), r_dir * dr.sin(phi_dir), z_dir)

            denom_vol = dr.dot(bucket_normal, d_vol_reflected)
            t_bucket_vol = dr.dot(source - p_vol, bucket_normal) / denom_vol
            p_bucket_vol = p_vol + t_bucket_vol * d_vol_reflected

            bucket_active_vol = is_volume_event & (denom_vol < -1e-8) & (t_bucket_vol > 1e-6) & \
                (dr.norm(p_bucket_vol - source) <= self.bucket_radius) & (sampler.next_1d() < self.pde)

            rel_opl_vol = s_vol + t_bucket_vol
            safe_rel_opl_vol = dr.maximum(rel_opl_vol, 0.0)
            safe_rel_opl_surf = dr.maximum(rel_opl_surf, 0.0)

            attenuation_surface = dr.select(has_medium, dr.exp(-sigma_t * safe_rel_opl_surf), 1.0)
            attenuation_volume = albedo * dr.select(has_medium, dr.exp(-sigma_t * safe_rel_opl_vol), mi.Float(1.0))

            # Combine surface and volume events
            bucket_active = dr.select(is_volume_event, bucket_active_vol, bucket_active_surf)
            bucket_rel_opl = dr.select(is_volume_event, safe_rel_opl_vol, safe_rel_opl_surf)
            weight_attenuation = dr.select(is_volume_event, attenuation_volume, attenuation_surface)
            weight_attenuation = dr.select(dr.isfinite(weight_attenuation), weight_attenuation, mi.Float(0.0))

            # Random sampling U in (0,1]
            u = dr.maximum(sampler.next_1d(), 1e-10)            

            # Conversion from pair rate to mean OPL separation, then sample an exponential
            delta_opl = -mean_delta_opl * dr.log(u)

            # Calculate the cumulative sum to obtain absolute emission times
            emission_opl = mi.Float(current_opl_offset) + dr.prefix_sum(delta_opl)

            last_emission = dr.gather(mi.Float, emission_opl, mi.UInt32(current_spp - 1))
            dr.eval(last_emission)
            current_opl_offset = float(np.asarray(last_emission).item())

            signal_rel_opl = signal_t

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

            signal_rel_opl += signal_jitter_opl * z_signal
            bucket_rel_opl += bucket_jitter_opl * z_bucket

            # Temporal quatization
            quant_step_opl = self.time_quantization_ps * 1e-12 * c
            quant_active = quant_step_opl > 0.0
            signal_rel_opl = dr.select(quant_active,
                                dr.round(signal_rel_opl / dr.maximum(quant_step_opl, 1e-10)) * quant_step_opl,
                                signal_rel_opl)
            bucket_rel_opl = dr.select(quant_active,
                                dr.round(bucket_rel_opl / dr.maximum(quant_step_opl, 1e-10)) * quant_step_opl,
                                bucket_rel_opl)
        
            film.append_events(
                pixel=mi.Point2f(signal_x, signal_y),
                emission_opl=emission_opl,
                signal_opl=signal_rel_opl,
                bucket_opl=bucket_rel_opl,
                value=self.event_weight * pair_weight * weight_attenuation,
                signal_active=signal_active,
                bucket_active=bucket_active,
            )                  

            if self.debug:
                scene_hit_count = scene_si.is_valid()
                film.add_debug_events(
                    mi.Point2f(signal_x, signal_y),
                    signal_active,
                    scene_hit_count,
                    bucket_active,
                )

                current_signal_hits = int(np.asarray(dr.sum(mi.UInt32(signal_active))).item())
                current_scene_hits = int(np.asarray(dr.sum(mi.UInt32(scene_hit_count))).item())
                current_bucket_hits = int(np.asarray(dr.sum(mi.UInt32(bucket_active))).item())

                if not hasattr(film, "debug_stats") or batch == 0:
                    film.debug_stats = {
                        "pairs": 0,
                        "signal_hits": 0,
                        "scene_hits": 0,
                        "bucket_hits": 0,
                    }

                film.debug_stats["pairs"] += current_spp
                film.debug_stats["signal_hits"] += current_signal_hits
                film.debug_stats["scene_hits"] += current_scene_hits
                film.debug_stats["bucket_hits"] += current_bucket_hits
                dr.eval(film.debug_signal_storage,
                        film.debug_scene_storage,
                        film.debug_bucket_storage)
            if progress_callback:
                progress_callback((batch+1)/num_batches)
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
        callback.put("sigma_s", self.sigma_s, mi.ParamFlags.NonDifferentiable)
        callback.put("sigma_a", self.sigma_a, mi.ParamFlags.NonDifferentiable)
        callback.put("g_hg", self.g_hg, mi.ParamFlags.NonDifferentiable)

    def to_string(self):
        return "QGhostIntegrator[statistical streams, primal]"


mi.register_integrator("qghost", lambda props: QGhostIntegrator(props))