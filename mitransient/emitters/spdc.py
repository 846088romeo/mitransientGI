"""SPDC pair source for the forward ghost-imaging integrator."""

from __future__ import annotations

from typing import Tuple

import drjit as dr
import mitsuba as mi


class SPDCPairEmitter(mi.Emitter):
    """Generate anti-correlated signal/idler rays from a point-like source.

    ``sample_pair`` is intentionally a mitransient-specific API: Mitsuba's
    regular emitter interface samples one light path, while SPDC produces two
    correlated arms that must be consumed by the ghost integrator together.
    """

    def __init__(self, props: mi.Properties):
        super().__init__(props)
        self.source = mi.Point3f(
            props.get("source", mi.ScalarPoint3f(0.0))
        )
        self.target = mi.Point3f(
            props.get("target", mi.ScalarPoint3f([0.0, 0.0, 2.5]))
        )
        self.max_angle = dr.deg2rad(
            mi.Float(props.get("max_angle_deg", 18.0))
        )
        self.pair_weight = mi.Float(props.get("pair_weight", 1.0))
        self.m_flags = mi.EmitterFlags.Delta
        self.ray_noise_sigma = mi.Float(props.get("ray_noise_sigma", 0.0))

    def _box_muller(self, sample: mi.Point2f) -> Tuple[mi.Float, mi.Float]:
        """Return a pair of independent standard normal random variables."""
        r = dr.sqrt(-2.0 * dr.log(dr.maximum(sample.x, 1e-10)))
        theta = 2.0 * dr.pi * sample.y
        return r * dr.cos(theta), r * dr.sin(theta)

    def sample_pair(self, 
                    sample: mi.Point2f,
                    noise_sample_idler: mi.Point2f = None,
                    noise_sample_signal: mi.Point2f = None,
                    active: mi.Bool = True) -> Tuple[mi.Ray3f, mi.Ray3f, mi.Float]:
        """Return ``(signal_ray, idler_ray, pair_weight)`` for each lane."""
        forward = dr.normalize(self.target - self.source)
        helper = mi.Vector3f(0, 1, 0)
        helper = dr.select(abs(forward.y) > 0.95,
                           mi.Vector3f(1, 0, 0), helper)
        right = dr.normalize(dr.cross(forward, helper))
        up = dr.normalize(dr.cross(right, forward))

        tangent = dr.tan(self.max_angle)
        ax = (2.0 * sample.x - 1.0) * tangent
        ay = (2.0 * sample.y - 1.0) * tangent

        if noise_sample_idler is not None and noise_sample_signal is not None:
            ni0, ni1 = self._box_muller(noise_sample_idler)
            ns0, ns1 = self._box_muller(noise_sample_signal)

            idler_ax = ax + ni0 * self.ray_noise_sigma
            idler_ay = ay + ni1 * self.ray_noise_sigma
            signal_ax = -ax - ns0 * self.ray_noise_sigma
            signal_ay = -ay - ns1 * self.ray_noise_sigma

            idler_direction = dr.normalize(forward + idler_ax * right + idler_ay * up)
            signal_direction = dr.normalize(-forward + signal_ax * right + signal_ay * up)
        else:
            transverse = ax * right + ay * up
            idler_direction = dr.normalize(forward + transverse)
            signal_direction = dr.normalize(-forward - transverse)

        return (
            mi.Ray3f(self.source, signal_direction),
            mi.Ray3f(self.source, idler_direction),
            self.pair_weight,
        )

    # The ordinary Mitsuba emitter API is not used by qghost, but
    # these methods keep the plugin compatible with scene construction.
    def eval(self, si: mi.SurfaceInteraction3f, active: mi.Bool) -> mi.Spectrum:
        return mi.Spectrum(0.0)

    def sample_ray(self, time, sample1, sample2, sample3, active):
        _, idler_ray, weight = self.sample_pair(sample2, active)
        return idler_ray, mi.Spectrum(weight)

    def sample_direction(self, ref, sample, active):
        raise NotImplementedError("SPDCPairEmitter uses sample_pair()")

    def pdf_direction(self, ref, ds, active):
        return mi.Float(0.0)

    def eval_direction(self, ref, ds, active):
        return mi.Spectrum(0.0)

    def traverse(self, callback):
        super().traverse(callback)
        callback.put("source", self.source, mi.ParamFlags.NonDifferentiable)
        callback.put("target", self.target, mi.ParamFlags.NonDifferentiable)
        callback.put("max_angle", self.max_angle,
                     mi.ParamFlags.NonDifferentiable)
        callback.put("pair_weight", self.pair_weight,
                     mi.ParamFlags.NonDifferentiable)

    def to_string(self):
        return (
            "SPDCPairEmitter[\n"
            f"  source = {self.source},\n"
            f"  target = {self.target},\n"
            f"  max_angle = {dr.rad2deg(self.max_angle)},\n"
            f"  pair_weight = {self.pair_weight}\n"
            "]"
        )


mi.register_emitter("spdc_pair", lambda props: SPDCPairEmitter(props))