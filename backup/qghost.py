#!/usr/bin/env python3
"""Simulador de quantum ghost imaging.

GEOMETRÍA DEL MODELO:
======================
Una fuente de pares correlacionados emite fotones signal (brazo de referencia) e idler (brazo de objeto).

- Fuente: por defecto se auto-reubica detrás de la escena (puede fijarse manualmente desactivando --auto-source).
- Brazo signal: viaja directamente a detector SPAD ubicado en z = spad_z < 0 (referencia temporal fija).
- Brazo idler: viaja a escena (z > 0), dispersa según BSDF Lambertiano, retorna a bucket detector en origen.

Cada fotón signal registra:
  - píxel (px, py) en SPAD basado en posición de impacto.
  - tiempo t_signal ≈ |spad_z| / c (distancia fija).

Cada fotón idler registra:
  - tiempo t_idler = (distancia a escena + dispersión + retorno) / c.

Coincidencia: dt = t_idler - t_signal codifica profundidad en la escena.
Ghost imaging: correlación temporal entre píxels distintos revela estructura de la escena.

PARÁMETROS CLAVE:
=================
- --spad-z: posición del detector de referencia (debe ser < 0).
- --max-angle-deg: apertura angular del cono de pares alrededor de scene_target.
- --ray-noise-sigma: ruido Gaussiano en ángulos de pares (añade descorrelación realista).
- --t-min, --t-max: ventana temporal para binning.
- --bucket-radius: radio del detector de retorno (ubicado en origen).

SALIDAS:
========
- histogram.npy: histograma 3D [altura, ancho, bins_temporales] de coincidencias.
- depth_map.png: profundidad estimada como 0.5 * t_peak * c.
- integrated_intensity.png: integración temporal = imagen fantasma clásica.
- center_histogram.png: curva temporal en píxel central.

"""
import argparse
import math
import os
from dataclasses import dataclass
from typing import List, Optional, Tuple, Union
import time
import mitsuba as mi

import numpy as np

try:
    import matplotlib.pyplot as plt
except Exception:
    plt = None

C = 299_792_458.0  # m/s - velocidad de la luz en vacío.

# Umbrales numéricos para estabilidad.
EPSILON_NORM = 1e-8  # Mínima norma de vector para considerar válida una dirección.
EPSILON_INTERSECT = 1e-6  # Distancia mínima de intersección (evita self-intersection).
EPSILON_OFFSET = 1e-5  # Offset de punto de inicio al lanzar rayo (evita shadow acne).
EPSILON_DOT = 1e-8  # Mínimo valor de dot product para considerar paralelo.
MAX_CONE_ANGLE_DEG = 89.0  # Evita tan(>=90°) inestable o indefinido.


def normalize(v: np.ndarray) -> np.ndarray:
    """Normaliza un vector. Si el vector es cero, lo devuelve sin cambios."""
    n = np.linalg.norm(v)
    if n == 0:
        return v
    return v / n

@dataclass
class SphereBucket:
    center: np.ndarray
    radius: float

    def intersect(self, ray_o: np.ndarray, ray_d: np.ndarray) -> Optional[float]:
        """Intersección rayo-esfera. Retorna distancia t al primer hit válido o None.
        
        Asume: ray_d está normalizado o tiene norma consistente.
        """
        oc = ray_o - self.center
        a = float(np.dot(ray_d, ray_d))
        b = 2.0 * float(np.dot(oc, ray_d))
        c = float(np.dot(oc, oc) - self.radius * self.radius)
        disc = b * b - 4 * a * c
        if disc < 0:
            return None
        s = math.sqrt(disc)
        t0 = (-b - s) / (2 * a)
        t1 = (-b + s) / (2 * a)
        valid = [t for t in (t0, t1) if t > EPSILON_INTERSECT]
        return min(valid) if valid else None

@dataclass
class DiskBucket:
    center: np.ndarray
    normal: np.ndarray
    radius: float

    def __post_init__(self):
        self.normal = normalize(self.normal)

    def basis(self) -> Tuple[np.ndarray, np.ndarray]:
        n = self.normal
        helper = np.array([0.0, 1.0, 0.0]) if abs(n[1]) < 0.9 else np.array([1.0, 0.0, 0.0])
        t = normalize(np.cross(helper, n))
        b = normalize(np.cross(n, t))
        return t, b

    def intersect(self, ray_o: np.ndarray, ray_d: np.ndarray) -> Optional[float]:
        """
        Disco plano. La normal debe apuntar desde el detector hacia la escena.
        Un rayo que vuelve desde la escena tendrá dot(normal, ray_d) < 0.
        """
        denom = float(np.dot(self.normal, ray_d))

        # Solo aceptar rayos que llegan por la cara frontal del detector
        if denom >= -EPSILON_DOT:
            return None

        t = float(np.dot(self.center - ray_o, self.normal) / denom)
        if t <= EPSILON_INTERSECT:
            return None

        p = ray_o + t * ray_d
        if np.linalg.norm(p - self.center) > self.radius:
            return None

        return t

    def sample_point(self, rng: np.random.Generator) -> Tuple[np.ndarray, float]:
        """
        Muestrea uniformemente un punto sobre el disco.
        Devuelve punto y área del detector.
        """
        u1 = rng.random()
        u2 = rng.random()

        r = self.radius * math.sqrt(u1)
        phi = 2.0 * math.pi * u2

        t, b = self.basis()
        p = self.center + r * math.cos(phi) * t + r * math.sin(phi) * b
        area = math.pi * self.radius * self.radius

        return p, area

def sample_cosine_hemisphere(normal: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    """Muestrea una dirección hemisférica con pdf proporcional a cos(theta).

    Muestreo típico de una superficie Lambertiana.
    """
    u1 = rng.random()
    u2 = rng.random()
    r = math.sqrt(u1)
    phi = 2.0 * math.pi * u2
    x = r * math.cos(phi)
    y = r * math.sin(phi)
    z = math.sqrt(max(0.0, 1.0 - u1))

    n = normalize(normal)
    helper = np.array([0.0, 1.0, 0.0]) if abs(n[1]) < 0.9 else np.array([1.0, 0.0, 0.0])
    t = normalize(np.cross(helper, n))
    b = normalize(np.cross(n, t))
    d = x * t + y * b + z * n
    return normalize(d)


def intersect_plane_z(ray_o: np.ndarray, ray_d: np.ndarray, z_plane: float) -> Optional[Tuple[float, np.ndarray]]:
    """Interseca un rayo con un plano horizontal z = constante.

    Retorna None cuando:
    - el rayo es casi paralelo al plano
    - el cruce queda detrás del origen del rayo
    """
    if abs(ray_d[2]) < EPSILON_DOT:
        return None
    t = (z_plane - ray_o[2]) / ray_d[2]
    if t <= EPSILON_INTERSECT:
        return None
    p = ray_o + t * ray_d
    return t, p


def make_mitsuba_ray(ray_o: np.ndarray, ray_d: np.ndarray):
    """Construye un mi.Ray3f."""
    o = mi.Point3f(float(ray_o[0]), float(ray_o[1]), float(ray_o[2]))
    d = mi.Vector3f(float(ray_d[0]), float(ray_d[1]), float(ray_d[2]))

    try:
        return mi.Ray3f(o, d)
    except TypeError:
        pass

    # Fallback para builds que requieren parámetros con nombre.
    return mi.Ray3f(o=o, d=d, time=0.0)


def sample_correlated_pair(
    rng: np.random.Generator,
    max_angle_deg: float,
    sigma: float,
    scene_target: np.ndarray,
    source: np.ndarray,
) -> Tuple[np.ndarray, np.ndarray, int, int]:
    """Genera un par de rayos correlacionados (signal, idler) para ghost imaging.
    
    Geometry:
    - source está en origen (z=0).
    - SPAD (signal arm) está en z < 0 (referencia, tiempo de vuelo corto).
    - Escena (idler arm) está en z > 0 (interacción, tiempo de vuelo variable).
    - El par es anti-correlacionado: signal y idler apuntan en direcciones opuestas + ruido Gaussiano.
    
    Args:
        rng: generador de números aleatorios.
        max_angle_deg: apertura angular máxima del cono (en grados).
        sigma: desviación estándar del ruido Gaussiano en radianes (añade descorrelación).
        scene_target: punto central hacia el cual apunta el brazo de idler.
        source: posición de la fuente de pares (típicamente origen).
    
    Returns:
        (d_signal, d_idler, p_signal, p_idler): direcciones normalizadas de los rayos
        signal e idler, y sus estados de polarización (+1 / -1). Cada par
        lleva polarizaciones opuestas (uno +1 y otro -1), con asignación aleatoria
        de cuál brazo recibe +1.
    """
    # Estabilidad numérica: un cono >= 90° no es un cono frontal válido para este modelo.
    theta_max = math.radians(min(max_angle_deg, MAX_CONE_ANGLE_DEG))
    ax = rng.uniform(-math.tan(theta_max), math.tan(theta_max))
    ay = rng.uniform(-math.tan(theta_max), math.tan(theta_max))

    forward = normalize(scene_target - source)  # Dirección central hacia escena.
    if np.linalg.norm(forward) < EPSILON_NORM:
        forward = np.array([0.0, 0.0, 1.0], dtype=float)

    # Construcción robusta de base ortonormal alrededor de forward.
    # Evita degenerar cuando forward es casi paralelo al eje Y.
    helper = np.array([0.0, 1.0, 0.0], dtype=float)
    if abs(float(np.dot(forward, helper))) > 0.95:
        helper = np.array([1.0, 0.0, 0.0], dtype=float)

    right = normalize(np.cross(forward, helper))
    if np.linalg.norm(right) < EPSILON_NORM:
        right = np.array([1.0, 0.0, 0.0], dtype=float)

    up = normalize(np.cross(right, forward))
    if np.linalg.norm(up) < EPSILON_NORM:
        up = np.array([0.0, 0.0, 1.0], dtype=float)

    transverse = ax * right + ay * up  # Desviación transversal dentro del cono.

    d_idler = normalize(forward + transverse)  # Brazo idler hacia escena con desviación angular.
    d_signal = normalize(-forward - transverse)  # Brazo signal hacia SPAD, anti-correlacionado con idler.

    if sigma > 0.0:
        ni = rng.normal(0.0, sigma, size=2)  # Ruido Gaussiano en ángulos para descorrelación realista.
        ns = rng.normal(0.0, sigma, size=2)

        d_idler = normalize(forward + (ax + ni[0]) * right + (ay + ni[1]) * up)
        d_signal = normalize(-forward + (-(ax + ns[0])) * right + (-(ay + ns[1])) * up)

    # DEBUG: ANTERIOR CONSTRUCCIÓN DE DIRECCIONES SIN BASE ORTONORMAL ROBUSTA
    # dir_base = forward + ax * right + ay * up  # Dirección base con desviación angular.
    
    # # Pares anti-correlacionados: signal va hacia SPAD (z < 0, opuesto a dir_base),
    # # idler va hacia escena (z > 0, dir_base). El ruido Gaussiano añade descorrelación.
    # d_idler = normalize(dir_base + rng.normal(0.0, sigma, size=3))
    # d_signal = normalize(-dir_base + rng.normal(0.0, sigma, size=3))

    # Polarización
    if rng.random() < 0.5:
        p_signal, p_idler = 1, -1
    else:
        p_signal, p_idler = -1, 1

    return d_signal, d_idler, p_signal, p_idler

def infer_framing_from_bounds(
    bmin: np.ndarray,
    bmax: np.ndarray,
    source: np.ndarray,
    margin: float = 1.0,
    min_angle_deg: float = 0.1,
    max_angle_deg: float = MAX_CONE_ANGLE_DEG,
) -> Tuple[np.ndarray, float]:
    """Versión de framing usando bounds explícitos en lugar de scene.bbox()."""
    target = 0.5 * (bmin + bmax)
    forward = normalize(target - source)
    if np.linalg.norm(forward) < EPSILON_NORM:
        raise ValueError(f"Source {source} está demasiado cerca del centro de bounds {target}")

    max_angle = 0.0
    for x in (bmin[0], bmax[0]):
        for y in (bmin[1], bmax[1]):
            for z in (bmin[2], bmax[2]):
                corner = np.array([x, y, z], dtype=float)
                v = corner - source
                vn = np.linalg.norm(v)
                if vn <= EPSILON_NORM:
                    continue
                d = v / vn
                cosang = np.clip(float(np.dot(forward, d)), -1.0, 1.0)
                max_angle = max(max_angle, math.degrees(math.acos(cosang)))

    angle_deg = float(np.clip(max_angle * margin, min_angle_deg, max_angle_deg))
    return target, angle_deg


def estimate_time_window_from_bbox(
    scene,
    source: np.ndarray,
    spad_z: float,
    max_angle_deg: float,
    margin: float = 1.10,
) -> Tuple[float, float, dict]:
    """Estima una ventana temporal [t_min, t_max] para dt a partir de la bbox de la escena.

        Modelo aproximado:
            t_signal in [abs(spad_z)/C, abs(spad_z)/(C*cos(theta_max))]
      t_idler  ~= 2 * d / C  (ida a escena + retorno al bucket en origen)
      dt       = t_idler - t_signal

    Para robustez, se usa distancia mínima y máxima a las esquinas de la bbox,
    se expande por un factor `margin`.
    """
    bbox = scene.bbox()
    bmin = np.array([float(bbox.min.x), float(bbox.min.y), float(bbox.min.z)], dtype=float)
    bmax = np.array([float(bbox.max.x), float(bbox.max.y), float(bbox.max.z)], dtype=float)

    corners = [
        np.array([x, y, z], dtype=float)
        for x in (bmin[0], bmax[0])
        for y in (bmin[1], bmax[1])
        for z in (bmin[2], bmax[2])
    ]

    dists = [float(np.linalg.norm(c - source)) for c in corners]
    d_min = min(dists)
    d_max = max(dists)

    # El tiempo del brazo signal puede variar mucho si el cono angular es amplio.
    # Con theta->90°, el trayecto hasta z=spad_z se hace casi tangencial y crece.
    theta = math.radians(min(float(max_angle_deg), MAX_CONE_ANGLE_DEG))
    cos_theta = max(math.cos(theta), 1e-3)
    t_signal_min = abs(float(spad_z)) / C
    t_signal_max = abs(float(spad_z)) / (C * cos_theta)

    dt_min_raw = 2.0 * d_min / C - t_signal_max
    dt_max_raw = 2.0 * d_max / C - t_signal_min

    center = 0.5 * (dt_min_raw + dt_max_raw)
    half = 0.5 * (dt_max_raw - dt_min_raw)
    half = half * float(max(1.0, margin))

    dt_min = center - half
    dt_max = center + half

    info = {
        "d_min": d_min,
        "d_max": d_max,
        "z_min_used": float(bmin[2]),
        "z_max_used": float(bmax[2]),
        "t_signal_min": t_signal_min,
        "t_signal_max": t_signal_max,
        "dt_min_raw": dt_min_raw,
        "dt_max_raw": dt_max_raw,
        "theta_used_deg": math.degrees(theta),
    }
    return dt_min, dt_max, info

def quantize_time(t: float, step: float) -> float:
    """Cuantiza un tiempo t al bin temporal más cercano."""
    if step == 0:
        return t
    return round(t / step) * step

def correlate_events_to_histogram(
        signal_events: List[Tuple[int, int, float]],  # (px, py, t_signal)
        bucket_events: List[float],  # t_idler
        width: int,
        height: int,
        time_bins: int,
        t_min: float,
        t_max: float,
):
    """Correlaciona eventos de signal e idler para construir el histograma 3D.

    Para cada evento de signal (px, py, t_signal) y cada evento de bucket (t_idler):
        - Calcula dt = t_idler - t_signal
        - Si dt está dentro de [t_min, t_max), asigna al bin temporal correspondiente.
    """
    hist = np.zeros((height, width, time_bins), dtype=np.float64)
    signal_times = np.array([t for _, _, t in signal_events])
    signal_px = np.array([px for px, _, _ in signal_events])
    signal_py = np.array([py for _, py, _ in signal_events])

    order = np.argsort(signal_times)
    signal_times = signal_times[order]
    signal_px = signal_px[order]
    signal_py = signal_py[order]

    for t_bucket in bucket_events:
        lo = np.searchsorted(signal_times, t_bucket - t_max, side="right")
        hi = np.searchsorted(signal_times, t_bucket - t_min, side="left")

        if lo >= hi:
            continue

        dts = t_bucket - signal_times[lo:hi]
        fracs = (dts - t_min) / (t_max - t_min)
        bin_idxs = np.floor(fracs * time_bins).astype(int)
        bin_idxs = np.clip(bin_idxs, 0, time_bins - 1)

        pxs = signal_px[lo:hi]
        pys = signal_py[lo:hi]

        for px, py, bin_idx in zip(pxs, pys, bin_idxs):
            hist[py, px, bin_idx] += 1.0

    return hist

def simulate(
    scene,
    bucket: Union[SphereBucket, DiskBucket],
    source: np.ndarray,
    spp: int,
    width: int,
    height: int,
    time_bins: int,
    t_min: float,
    t_max: float,
    spad_z_plane: float,
    spad_half_width: float,
    spad_half_height: float,
    max_angle_deg: float,
    seed: int,
    ray_noise_sigma: float,
    pair_rate_hz: float,
    signal_jitter_ps: float = 0.0,
    bucket_jitter_ps: float = 0.0,
    time_quantization_ps: float = 0.0,
    debug: bool = False,
    ignore_polarization: bool = False,
    scene_target: np.ndarray = np.array([0.0, 0.0, 2.5]),
) -> Tuple[np.ndarray, np.ndarray, dict]:
    """Simulación Monte Carlo de ghost imaging con pares etiquetados.
    
    Flujo de cada par (signal, idler):
    1. Muestrear par anti-correlacionado desde la fuente.
    2. Signal va a SPAD (z < 0): registro de píxel (px, py) y tiempo t_signal.
    3. Idler va a escena, dispersa según BSDF Lambertiano, y retorna a bucket en origen.
    4. Si ambos eventos ocurren, registrar dt = t_idler - t_signal en el bin temporal.
    5. Histograma 3D: [altura, ancho, bins_temporales] acumula coincidencias temporales.
    
    Validaciones y edge cases:
    - SPAD fuera de límites -> rechazar.
    - Escena sin hit -> rechazar.
    - Bucket sin hit -> rechazar.
    - dt fuera de [t_min, t_max) -> rechazar y contar.
    - Binning robusto con clipping defensivo.
    
    Returns:
        (hist, t_edges, meta): histograma 3D, bordes temporales, y metadatos de conteos.
    """
    # ---------------------------------------------------------------
    # Estado global de la simulación
    # ---------------------------------------------------------------
    rng = np.random.default_rng(seed)
    hist = np.zeros((height, width, time_bins), dtype=np.float64)
    counts = {
        "pairs": spp,
        "signal_hits": 0,
        "scene_hits": 0,
        "lambertian_sampled": 0,  # Muestreo lambertiano exitoso.
        "bucket_no_hit": 0,  # Rayo lambertiano pero bucket.intersect() retorna None.
        "bucket_hits": 0,
        "binned": 0,
        "out_of_range": 0,  # Eventos con dt fuera de [t_min, t_max).
        "mitsuba_invalid": 0,
        "mitsuba_exceptions": 0,
        "rejected_by_dot": 0,  # Rayos rechazados por dot(d_refl, ni) <= 0
    }
    counts["rejected_by_polarization"] = 0

    signal_events = []  # Lista de eventos (px, py, t_signal, weight) para correlación posterior.
    bucket_events = []  # Lista de tiempos t_idler para correlación posterior.

    signal_jitter_sec = signal_jitter_ps * 1e-12
    bucket_jitter_sec = bucket_jitter_ps * 1e-12
    time_quantization_sec = time_quantization_ps * 1e-12

    t_emit = 0.0  # Tiempo de emisión del siguiente par, para timestamps absolutos.

    start_time = time.time()
    dt_min_seen = float("inf")
    dt_max_seen = float("-inf")

    debug_scene_hits = np.zeros((height, width), dtype=np.int32) if debug else None

    for i in range(spp):
        # -----------------------------------------------------------
        # 0) Progreso periódico en consola
        # -----------------------------------------------------------
        if (i + 1) % 100000 == 0:
            frac = (i + 1) / spp
            elapsed = time.time() - start_time
            print(
                f"[{i+1}/{spp}] {100.0*frac:5.1f}%  "
                f"spad={counts['signal_hits']/(i+1):.4f}  "
                f"scene={counts['scene_hits']/(i+1):.4f}  "
                f"bucket={counts['bucket_hits']/(i+1):.6f}  "
                f"elapsed={elapsed:.1f}s  "
            )

        dt_emit = rng.exponential(1.0 / pair_rate_hz) if pair_rate_hz > 0 else 0.0
        t_emit += dt_emit

        d_signal, d_idler, p_signal, p_idler = sample_correlated_pair(
            rng,
            max_angle_deg=max_angle_deg,
            sigma=ray_noise_sigma,
            scene_target=scene_target,
            source=source,
        )

        # -----------------------------------------------------------
        # 1) Brazo de referencia (signal): cruce con plano SPAD
        # -----------------------------------------------------------
        spad_hit = intersect_plane_z(source, d_signal, z_plane=spad_z_plane)
        if spad_hit is None:
            continue
        ts, ps = spad_hit

        # El SPAD está centrado en (source.x, source.y) sobre el plano z=spad_z_plane.
        # Usar coordenadas globales aquí sesga el muestreo cuando source no está en el origen.
        spad_local_x = float(ps[0] - source[0])
        spad_local_y = float(ps[1] - source[1])

        if abs(spad_local_x) > spad_half_width or abs(spad_local_y) > spad_half_height:
            continue
        counts["signal_hits"] += 1

        # Convertir posición local en SPAD a coordenadas de píxel.
        # u,v en [0,1] -> px,py discretos.
        image_x = -spad_local_x
        image_y = -spad_local_y

        u = (image_x + spad_half_width) / (2.0 * spad_half_width)
        v = (image_y + spad_half_height) / (2.0 * spad_half_height)

        px = min(width - 1, max(0, int(np.floor(u * width))))  # Clipping defensivo para evitar índices fuera de rango.
        py = min(height - 1, max(0, int(np.floor(v * height))))

        t_signal = ts / C
        t_abs_signal = t_emit + t_signal + rng.normal(0.0, signal_jitter_sec)  # Timestamp absoluto con jitter.
        t_abs_signal = quantize_time(t_abs_signal, time_quantization_sec)  # Cuantización temporal.
        signal_events.append((px, py, t_abs_signal))


        # -----------------------------------------------------------
        # 2) Brazo de objeto (idler): primer impacto en escena
        # -----------------------------------------------------------
        try:
            mi_ray = make_mitsuba_ray(source, d_idler)
            si = scene.ray_intersect(mi_ray)
            if not si.is_valid():
                counts["mitsuba_invalid"] += 1
                continue

            ti = float(si.t)
            pi = np.array([si.p.x, si.p.y, si.p.z], dtype=float)
            ni = np.array([si.sh_frame.n.x, si.sh_frame.n.y, si.sh_frame.n.z], dtype=float)
            
            # Asegurar que la normal apunta hacia el rayo entrante (hacia afuera de la superficie).
            if np.dot(ni, d_idler) > 0:
                ni = -ni
            
        except Exception as e:
            counts["mitsuba_exceptions"] += 1
            if debug and counts["mitsuba_exceptions"] <= 3:
                print(f"Mitsuba intersect exception (sample {i}): {type(e).__name__}: {e}")
            continue
 
        counts["scene_hits"] += 1

        if debug:
            debug_scene_hits[py, px] += 1

        # -----------------------------------------------------------
        # 3) Rebote difuso Lambertiano y prueba de retorno al bucket
        # -----------------------------------------------------------
        # DEBUG: LAMBERTIANO REALISTA CON NORMAL CORRECTA
        d_refl = sample_cosine_hemisphere(ni, rng)
        counts["lambertian_sampled"] += 1

        # Reject if we shoot below the surface
        if np.dot(d_refl, ni) <= 0.0:
            counts["rejected_by_dot"] += 1
            continue

        # See whether the scattered ray returns to the bucket.
        tb = bucket.intersect(pi + EPSILON_OFFSET * ni, d_refl)
        if tb is None:
            counts["bucket_no_hit"] += 1
            continue

        counts["bucket_hits"] += 1

        t_idler = (ti + tb) / C
        t_abs_idler = t_emit + t_idler + rng.normal(0.0, bucket_jitter_sec)  # Timestamp absoluto con jitter.
        t_abs_idler = quantize_time(t_abs_idler, time_quantization_sec)
        # Filtrado por polarización en el bucket: solo los rayos con +1 pasan.
        if ignore_polarization or p_idler == 1:
            bucket_events.append((t_abs_idler))
        else:
            counts["rejected_by_polarization"] += 1
            continue

        dt = t_abs_idler - t_abs_signal

        dt_min_seen = min(dt_min_seen, dt)
        dt_max_seen = max(dt_max_seen, dt)

    hist = correlate_events_to_histogram(
        signal_events=signal_events,
        bucket_events=bucket_events,
        width=width,
        height=height,
        time_bins=time_bins,
        t_min=t_min,
        t_max=t_max,
    )

    counts["binned"] = int(hist.sum())
    counts["signal_hits"] = len(signal_events)
    counts["bucket_hits"] = len(bucket_events)

    # Bordes del eje temporal (uniforme) para export y post-procesado.
    t_edges = np.linspace(t_min, t_max, time_bins + 1)
    meta = {
        "counts": counts,
        "spad_z": spad_z_plane,
        "spad_half_width": spad_half_width,
        "spad_half_height": spad_half_height,
        "time_edges": t_edges,
        "debug_scene_hits": debug_scene_hits,
        "observed_dt_range": (dt_min_seen if math.isfinite(dt_min_seen) else None, 
                               dt_max_seen if math.isfinite(dt_max_seen) else None),
    }
    if math.isfinite(dt_min_seen) and math.isfinite(dt_max_seen):
        print(f"Seen dt range: [{dt_min_seen*1e9:.3f} ns, {dt_max_seen*1e9:.3f} ns]")
    else:
        print("Seen dt range: no bucket-return events")
    return hist, t_edges, meta


def depth_map_from_hist(
    hist: np.ndarray,
    t_edges: np.ndarray,
    signal_time_offset: float = 0.0,
) -> np.ndarray:
    """Mapea tiempo-de-pico a profundidad.

    El histograma se construye sobre dt = t_idler - t_signal. Para recuperar una
    estimación de distancia física (ida/vuelta del brazo idler), se corrige con el
    offset temporal del brazo de referencia:

        t_idler ~= dt + signal_time_offset
        depth ~= 0.5 * t_idler * C

    Args:
        hist: histograma [H, W, T].
        t_edges: bordes temporales del eje dt.
        signal_time_offset: tiempo de vuelo del brazo signal (s), típicamente abs(spad_z)/C.
    """
    centers = 0.5 * (t_edges[:-1] + t_edges[1:])
    idx = np.argmax(hist, axis=2)
    peak_dt = centers[idx]
    peak_w = np.max(hist, axis=2)
    t_idler_est = peak_dt + signal_time_offset
    depth = 0.5 * t_idler_est * C
    # Evita profundidades negativas por offset o bins de dt negativos.
    depth[depth < 0] = np.nan
    depth[peak_w <= 0] = np.nan
    return depth


def save_outputs(hist: np.ndarray, t_edges: np.ndarray, out_dir: str, spad_z: float, hist_grid_size: int = 4) -> None:
    """Guarda arrays y visualizaciones derivadas del histograma.

    Archivos:
    - histogram.npy / time_edges.npy: datos crudos
    - integrated_intensity.png: integral temporal por píxel
    - depth_map.png: profundidad por tiempo de pico
    - center_histogram.png: traza temporal de un píxel de referencia
    - histograms_grid.png: grid de histogramas temporales distribuidos sobre la escena
    """
    os.makedirs(out_dir, exist_ok=True)
    np.save(os.path.join(out_dir, "histogram.npy"), hist)
    np.save(os.path.join(out_dir, "time_edges.npy"), t_edges)

    if plt is None:
        return

    intensity = np.sum(hist, axis=2)
    signal_time_offset = abs(spad_z) / C
    depth = depth_map_from_hist(hist, t_edges, signal_time_offset=signal_time_offset)
    center_curve = hist[hist.shape[0] // 2, hist.shape[1] // 2, :]

    # Auto-crop visualización al bbox útil para evitar que la escena quede en una esquina pequeña.
    mask = intensity > 0
    if np.any(mask):
        ys, xs = np.where(mask)
        margin = max(2, int(round(0.05 * max(hist.shape[0], hist.shape[1]))))
        y0 = max(0, int(ys.min()) - margin)
        y1 = min(hist.shape[0], int(ys.max()) + 1 + margin)
        x0 = max(0, int(xs.min()) - margin)
        x1 = min(hist.shape[1], int(xs.max()) + 1 + margin)
    else:
        y0, y1, x0, x1 = 0, hist.shape[0], 0, hist.shape[1]

    intensity_view = intensity[y0:y1, x0:x1]
    depth_view = depth[y0:y1, x0:x1]

    fig = plt.figure(figsize=(6, 5))
    plt.imshow(intensity_view, origin="lower")
    plt.colorbar(label="Coincidences")
    plt.title("Integrated ghost image (auto-cropped)")
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, "integrated_intensity.png"), dpi=150)
    plt.close(fig)

    fig = plt.figure(figsize=(6, 5))
    plt.imshow(depth_view, origin="lower")
    plt.colorbar(label="Estimated depth [m]")
    plt.title("Peak-time depth map (auto-cropped)")
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, "depth_map.png"), dpi=150)
    plt.close(fig)

    fig = plt.figure(figsize=(6, 4))
    centers = 0.5 * (t_edges[:-1] + t_edges[1:]) * 1e9
    plt.plot(centers, center_curve)
    plt.xlabel("Δt [ns]")
    plt.ylabel("Counts")
    plt.title("Temporal histogram at center pixel")
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, "center_histogram.png"), dpi=150)
    plt.close(fig)

    # Grid de histogramas temporales (por defecto 4x4) distribuido sobre el ROI útil.
    n = max(1, int(hist_grid_size))
    ys_grid = np.linspace(y0, max(y0, y1 - 1), n).astype(int)
    xs_grid = np.linspace(x0, max(x0, x1 - 1), n).astype(int)

    fig, axes = plt.subplots(n, n, figsize=(3.0 * n, 2.2 * n), sharex=True, sharey=True)
    if n == 1:
        axes = np.array([[axes]])

    for iy, yy in enumerate(ys_grid):
        for ix, xx in enumerate(xs_grid):
            ax = axes[iy, ix]
            curve = hist[yy, xx, :]
            ax.plot(centers, curve, linewidth=1.0)
            ax.set_title(f"x={xx}, y={yy}", fontsize=8)
            ax.grid(alpha=0.2, linewidth=0.5)
            if iy == n - 1:
                ax.set_xlabel("Δt [ns]")
            if ix == 0:
                ax.set_ylabel("Counts")

    fig.suptitle(f"Temporal histograms grid ({n}x{n})", y=1.02)
    fig.tight_layout()
    fig.savefig(os.path.join(out_dir, "histograms_grid.png"), dpi=150, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    # Configure a standalone invocation without overriding a variant chosen
    # by the caller when this module is imported from another application.
    if mi.variant() is None:
        mi.set_variant("scalar_rgb")

    # -------------------------------------------------------------------
    # Fase A: parseo y validación de argumentos de entrada
    # -------------------------------------------------------------------
    parser = argparse.ArgumentParser(description="Ghost tracer prototype")
    parser.add_argument("--spp", type=int, default=400000, help="number of tagged pairs")
    parser.add_argument("--width", type=int, default=64, help="SPAD image width in pixels")
    parser.add_argument("--height", type=int, default=64, help="SPAD image height in pixels")
    parser.add_argument("--time-bins", type=int, default=200, help="number of temporal bins for dt histogram")
    parser.add_argument("--t-min", type=float, default=-8e-9, help="seconds")
    parser.add_argument("--t-max", type=float, default=20e-9, help="seconds")
    parser.add_argument("--spad-z", type=float, default=-1.0, help="SPAD plane z [m]")
    parser.add_argument("--spad-half-width", type=float, default=0.30, help="SPAD half-width [m]")
    parser.add_argument("--spad-half-height", type=float, default=0.30, help="SPAD half-height [m]")
    parser.add_argument("--max-angle-deg", type=float, default=18.0, help="cone angle for pair sampling")
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--out-dir", type=str, default="ghost_output")
    parser.add_argument("--ray-noise-sigma", type=float, default=0.002,
                    help="Desviación típica del ruido angular gaussiano en radianes")
    parser.add_argument("--xml-scene", type=str, default=None, required=True,
                    help="Ruta a una escena XML de Mitsuba")
    parser.add_argument("--bucket-radius", type=float, default=3.0,
                    help="Radio del bucket detector en metros")
    parser.add_argument("--scene-target", type=float, nargs=3, default=[0.0, 0.0, 2.5],
                    help="Punto al que apunta el cono central del brazo de escena")
    parser.add_argument("--auto-source", action=argparse.BooleanOptionalAction, default=True,
                    help="Reubica automáticamente la fuente/bucket detrás de la escena")
    parser.add_argument("--source-margin", type=float, default=0.5,
                    help="Distancia mínima [m] entre la fuente y la cara frontal más cercana de la escena")
    parser.add_argument("--target-frame-angle-deg", type=float, default=25.0,
                    help="Ángulo objetivo aproximado para encuadrar la escena al inferir fuente (solo auto-source)")
    parser.add_argument("--auto-frame", action=argparse.BooleanOptionalAction, default=True,
                    help="Autoajusta scene-target y max-angle a partir del bounding box de la escena XML")
    parser.add_argument("--frame-margin", type=float, default=1.20,
                    help="Margen multiplicativo del autoencuadre (>1 abre más el cono)")
    parser.add_argument("--auto-time-window", action=argparse.BooleanOptionalAction, default=True,
                    help="Autoajusta t-min/t-max a partir de la bbox de la escena XML")
    parser.add_argument("--time-window-margin", type=float, default=1.10,
                    help="Margen multiplicativo para auto-time-window (>=1)")
    parser.add_argument("--time-window-pad-frac", type=float, default=0.10,
                    help="Fracción del span observado usada como padding por lado en la calibración piloto")
    parser.add_argument("--debug", action="store_true",
                    help="Genera salidas de depuración de intersecciones")
    parser.add_argument("--pair-rate-hz", type=float, default=1e8,
                help="Tasa media de emisión de pares para timestamps absolutos")
    parser.add_argument("--signal-jitter-ps", type=float, default=0.0,
                    help="Jitter temporal RMS del brazo signal en ps")
    parser.add_argument("--bucket-jitter-ps", type=float, default=0.0,
                    help="Jitter temporal RMS del brazo bucket en ps")
    parser.add_argument("--time-quantization-ps", type=float, default=0.0,
                    help="Cuantización temporal del TDC en ps (0 desactiva)")
    parser.add_argument("--hist-grid-size", type=int, default=4,
                    help="Tamaño N del grid NxN de histogramas temporales (0 o menos desactiva)")
    parser.add_argument("--ignore-polarization", action="store_true",
                    help="Desactiva el filtro por polarización en el bucket para pruebas")
    args = parser.parse_args()

    # Validaciones de argumentos
    if args.spp <= 0:
        raise ValueError(f"--spp debe ser > 0, se proporcionó {args.spp}")
    if args.width <= 0 or args.height <= 0:
        raise ValueError(f"--width y --height deben ser > 0, se proporcionó width={args.width}, height={args.height}")
    if args.time_bins <= 0:
        raise ValueError(f"--time-bins debe ser > 0, se proporcionó {args.time_bins}")
    if args.t_min >= args.t_max:
        raise ValueError(f"--t-min debe ser < --t-max, se proporcionó t_min={args.t_min}, t_max={args.t_max}")
    if args.spad_z >= 0:
        raise ValueError(f"--spad-z debe ser < 0 (SPAD en z negativo), se proporcionó {args.spad_z}")
    if args.spad_half_width <= 0 or args.spad_half_height <= 0:
        raise ValueError(f"SPAD half_width/height deben ser > 0, se proporcionó {args.spad_half_width}/{args.spad_half_height}")
    if args.bucket_radius <= 0:
        raise ValueError(f"--bucket-radius debe ser > 0, se proporcionó {args.bucket_radius}")
    if args.ray_noise_sigma < 0:
        raise ValueError(f"--ray-noise-sigma debe ser >= 0, se proporcionó {args.ray_noise_sigma}")
    if args.source_margin <= 0.0:
        raise ValueError(f"--source-margin debe ser > 0.0, se proporcionó {args.source_margin}")
    if not (0.1 <= args.target_frame_angle_deg < MAX_CONE_ANGLE_DEG):
        raise ValueError(
            f"--target-frame-angle-deg debe estar en [0.1, {MAX_CONE_ANGLE_DEG}), "
            f"se proporcionó {args.target_frame_angle_deg}"
        )
    if args.frame_margin <= 1.0:
        raise ValueError(f"--frame-margin debe ser > 1.0 para abrir el cono, se proporcionó {args.frame_margin}")
    if args.time_window_margin < 1.0:
        raise ValueError(f"--time-window-margin debe ser >= 1.0, se proporcionó {args.time_window_margin}")
    if args.time_window_pad_frac < 0.0:
        raise ValueError(f"--time-window-pad-frac debe ser >= 0.0, se proporcionó {args.time_window_pad_frac}")
    

    # -------------------------------------------------------------------
    # Fase B: carga de escena (Mitsuba XML o escena sintética interna)
    # -------------------------------------------------------------------
    try:
        scene = mi.load_file(args.xml_scene)
    except Exception as e:
        raise RuntimeError(f"No se pudo cargar escena Mitsuba desde {args.xml_scene}: {e}")

    scene_target = np.array(args.scene_target, dtype=float)
    max_angle_deg = float(args.max_angle_deg)
    spad_half_width = float(args.spad_half_width)
    spad_half_height = float(args.spad_half_height)
    sim_t_min = float(args.t_min)
    sim_t_max = float(args.t_max)
    source = np.array([0.0, 0.0, 0.0], dtype=float)
    spad_z_plane = source[2] + float(args.spad_z)

    # -------------------------------------------------------------------
    # Fase C: autoajustes geométricos y temporales (si hay escena Mitsuba)
    # -------------------------------------------------------------------
    # Objetivo: minimizar ejecuciones vacías y alinear sampling con la escena real.
    bbox = scene.bbox()
    bmin = np.array([float(bbox.min.x), float(bbox.min.y), float(bbox.min.z)], dtype=float)
    bmax = np.array([float(bbox.max.x), float(bbox.max.y), float(bbox.max.z)], dtype=float)
    center_world = 0.5 * (bmin + bmax)

    if args.auto_source:
        # Recolocar fuente y bucket detrás de la escena según tamaño lateral + margen.
        depth_extent = max(0.0, float(bmax[2] - bmin[2]))
        radial_extent = 0.0
        for x in (bmin[0], bmax[0]):
            for y in (bmin[1], bmax[1]):
                radial_extent = max(radial_extent, float(np.linalg.norm(np.array([x, y], dtype=float) - center_world[:2])))

        target_theta = math.radians(float(args.target_frame_angle_deg))
        backoff_for_fov = radial_extent / max(math.tan(target_theta), 1e-3)

        backoff = max(
            float(args.source_margin),
            backoff_for_fov,
        )
        source = np.array([center_world[0], center_world[1], bmin[2] - backoff], dtype=float)
        print(
            "Auto-source: "
            f"radial_extent={radial_extent:.3f} m, depth_extent={depth_extent:.3f} m, "
            f"target_angle={args.target_frame_angle_deg:.2f} deg, backoff={backoff:.3f} m"
        )

    spad_z_plane = source[2] + float(args.spad_z)

    center = 0.5 * (bmin + bmax)
    center_dist = float(np.linalg.norm(center - source))

    zmin_rel = bmin[2] - source[2]
    zmax_rel = bmax[2] - source[2]
    crosses_z0 = zmin_rel < 0.0 < zmax_rel
    z_min_clip = None

    framing_bmin = bmin.copy()
    framing_bmax = bmax.copy()
    if crosses_z0:
        # Usa solo el volumen frontal para evitar que auto-frame apunte a la parte "detrás" de la fuente.
        z_min_clip = max(0.05, 0.05 * float(bmax[2] - bmin[2]))
        framing_bmin[2] = max(framing_bmin[2], source[2] + z_min_clip)

    # Ángulo mínimo para cubrir bounds efectivos desde la fuente.
    auto_target_raw, required_angle = infer_framing_from_bounds(
        framing_bmin,
        framing_bmax,
        source=source,
        margin=1.0,
        min_angle_deg=0.1,
        max_angle_deg=MAX_CONE_ANGLE_DEG,
    )

    print(
        "Scene pre-check: "
        f"source={source.tolist()}, spad_plane_z={spad_z_plane:.3f}, "
        f"bbox_min={bmin.tolist()}, bbox_max={bmax.tolist()}, "
        f"z_rel=[{zmin_rel:.3f}, {zmax_rel:.3f}] m, center={center.tolist()}, center_dist={center_dist:.3f} m, "
        f"required_angle_deg={required_angle:.2f}, configured_max_angle_deg={max_angle_deg:.2f}"
    )

    if zmax_rel <= 0.0:
        print("WARNING: La escena está completamente detrás de la fuente (z_rel<=0). Podrías no tener impactos de escena.")
    if crosses_z0:
        print(
            "WARNING: La bbox cruza z=0 (la fuente está dentro/cerca del volumen angular). "
            "Se usará solo la parte frontal (z>0) para auto-frame/auto-time-window."
        )

    # Distancia al punto más lejano de la bbox desde la fuente (cota útil para tiempos).
    farthest_dist = max(
        float(np.linalg.norm(np.array([x, y, z], dtype=float) - source))
        for x in (framing_bmin[0], framing_bmax[0])
        for y in (framing_bmin[1], framing_bmax[1])
        for z in (framing_bmin[2], framing_bmax[2])
    )
    if center_dist > 0 and farthest_dist > 0:
        print(
            f"Distancia al centro de la escena: {center_dist:.3f} m, "
            f"distancia al punto más lejano de la bbox efectiva: {farthest_dist:.3f} m"
        )

    # Probe determinista para validar que el pipeline de intersección Mitsuba funciona.
    try:
        probe_dir = normalize(center_world - source)
        probe_ray = make_mitsuba_ray(source, probe_dir)
        probe_si = scene.ray_intersect(probe_ray)
        print(
            "Probe ray: "
            f"valid={bool(probe_si.is_valid())}, "
            f"t={float(probe_si.t) if probe_si.is_valid() else float('nan'):.4f}"
        )
    except Exception as e:
        print(f"WARNING: Probe ray falló ({type(e).__name__}): {e}")

    # Apertura angular final: required_angle expandido por frame_margin.
    framed_angle = float(np.clip(required_angle * max(1.0, float(args.frame_margin)), 0.1, MAX_CONE_ANGLE_DEG))

    # Auto-frame: actualiza scene_target/max_angle. Si está desactivado, al menos corrige max_angle mínimo.
    if args.auto_frame:
        scene_target = auto_target_raw
        max_angle_deg = framed_angle
    else:
        if max_angle_deg < required_angle:
            max_angle_deg = min(MAX_CONE_ANGLE_DEG, required_angle * 1.05)
        scene_target = auto_target_raw

    max_angle_deg = min(max_angle_deg, MAX_CONE_ANGLE_DEG)

    # Ajuste del tamaño de SPAD para no recortar artificialmente el cono de muestreo.
    needed_half_size = abs(float(args.spad_z)) * math.tan(math.radians(max_angle_deg)) * 1.05
    spad_half_width = max(spad_half_width, needed_half_size)
    spad_half_height = max(spad_half_height, needed_half_size)

    print(
        "Frame aplicado: "
        f"scene_target={scene_target.tolist()}, "
        f"max_angle_deg={max_angle_deg:.2f}, "
        f"spad_half_width={spad_half_width:.3f}, "
        f"spad_half_height={spad_half_height:.3f}"
    )

    # Ajuste automático de t_max mínimo recomendado por distancia lejana.
    recommended_t_max = 2.5 * farthest_dist / C
    if farthest_dist > 0 and sim_t_max < recommended_t_max:
        sim_t_max = recommended_t_max
        print(f"Ajuste automático: t_max -> {sim_t_max*1e9:.1f} ns (por distancia máxima de escena)")

    # Ventana temporal automática en dos pasos:
    # 1) estimación geométrica por bbox
    # 2) refinado por corrida piloto con dt observado
    if args.auto_time_window:
        auto_t_min, auto_t_max, time_info = estimate_time_window_from_bbox(
            scene,
            source=source,
            spad_z=args.spad_z,
            max_angle_deg=max_angle_deg,
            margin=args.time_window_margin,
        )
        sim_t_min = auto_t_min
        sim_t_max = auto_t_max
        print(
            "Auto-time-window: "
            f"d_min={time_info['d_min']:.3f} m, d_max={time_info['d_max']:.3f} m, "
            f"z_used=[{time_info['z_min_used']:.3f}, {time_info['z_max_used']:.3f}] m, "
            f"theta_used={time_info['theta_used_deg']:.2f} deg, "
            f"t_signal=[{time_info['t_signal_min']*1e9:.2f}, {time_info['t_signal_max']*1e9:.2f}] ns, "
            f"dt_raw=[{time_info['dt_min_raw']*1e9:.2f}, {time_info['dt_max_raw']*1e9:.2f}] ns, "
            f"window=[{sim_t_min*1e9:.2f}, {sim_t_max*1e9:.2f}] ns"
        )

        # Calibración fina: pequeña corrida piloto para capturar el rango dt real
        # de eventos que sí retornan al bucket en esta geometría concreta.
        pilot_spp = int(max(5000, min(30000, args.spp // 20)))
        if pilot_spp > 0:
            print(f"Pilot auto-time-window: lanzando {pilot_spp} muestras para calibrar dt observado...")
            bucket_pilot_normal = normalize(scene_target - source)
            bucket_pilot = DiskBucket(center=np.array(source, dtype=float), normal=bucket_pilot_normal, radius=args.bucket_radius)
            _, _, pilot_meta = simulate(
                scene=scene,
                bucket=bucket_pilot,
                source=source,
                spp=pilot_spp,
                width=max(16, args.width // 4),
                height=max(16, args.height // 4),
                time_bins=1,
                t_min=-1e-6,
                t_max=1e-6,
                spad_z_plane=spad_z_plane,
                spad_half_width=spad_half_width,
                spad_half_height=spad_half_height,
                max_angle_deg=max_angle_deg,
                seed=args.seed + 101,
                ray_noise_sigma=args.ray_noise_sigma,
                pair_rate_hz=args.pair_rate_hz,
                signal_jitter_ps=args.signal_jitter_ps,
                bucket_jitter_ps=args.bucket_jitter_ps,
                time_quantization_ps=args.time_quantization_ps,
                debug=False,
                ignore_polarization=args.ignore_polarization,
                scene_target=scene_target,
            )
            obs_min, obs_max = pilot_meta.get("observed_dt_range", (None, None))
            if obs_min is not None and obs_max is not None and obs_max > obs_min:
                span = obs_max - obs_min
                # Padding adaptativo: porcentaje del span observado por cada extremo.
                pad = args.time_window_pad_frac * span
                sim_t_min = obs_min - pad
                sim_t_max = obs_max + pad
                print(
                    "Pilot auto-time-window aplicado: "
                    f"observed=[{obs_min*1e9:.2f}, {obs_max*1e9:.2f}] ns, "
                    f"pad={pad*1e9:.2f} ns ({args.time_window_pad_frac*100:.1f}% del span), "
                    f"window=[{sim_t_min*1e9:.2f}, {sim_t_max*1e9:.2f}] ns"
                )
            else:
                print("Pilot auto-time-window: sin eventos de retorno suficientes; se mantiene la ventana por bbox.")

    # -------------------------------------------------------------------
    # Fase D: simulación principal Monte Carlo
    # -------------------------------------------------------------------
    # bucket = SphereBucket(center=np.array(source, dtype=float), radius=args.bucket_radius)
    bucket_normal = normalize(scene_target - source)  # apunta desde detector hacia escena
    bucket = DiskBucket(center=np.array(source, dtype=float), normal=bucket_normal, radius=args.bucket_radius)

    if args.debug:
        bbox = scene.bbox()
        scene_bmin = np.array([float(bbox.min.x), float(bbox.min.y), float(bbox.min.z)], dtype=float)
        scene_bmax = np.array([float(bbox.max.x), float(bbox.max.y), float(bbox.max.z)], dtype=float)

        print("Debug geometry:")
        print(f"  source = {source.tolist()}")
        print(f"  spad_plane_z = {spad_z_plane:.6f} m")
        print(f"  spad_half_size = [{spad_half_width:.6f}, {spad_half_height:.6f}] m")
        print(f"  bucket_center = {bucket.center.tolist()}")
        print(f"  bucket_normal = {bucket.normal.tolist()}")
        print(f"  bucket_radius = {bucket.radius:.6f} m")
        print(f"  scene_bbox_min = {scene_bmin.tolist()}")
        print(f"  scene_bbox_max = {scene_bmax.tolist()}")
        print(f"  scene_target = {scene_target.tolist()}")
        print(f"  max_angle_deg = {max_angle_deg:.6f}")

    hist, t_edges, meta = simulate(
        scene=scene,
        bucket=bucket,
        source=source,
        spp=args.spp,
        width=args.width,
        height=args.height,
        time_bins=args.time_bins,
        t_min=sim_t_min,
        t_max=sim_t_max,
        spad_z_plane=spad_z_plane,
        spad_half_width=spad_half_width,
        spad_half_height=spad_half_height,
        max_angle_deg=max_angle_deg,
        seed=args.seed,
        ray_noise_sigma=args.ray_noise_sigma,
        pair_rate_hz=args.pair_rate_hz,
        signal_jitter_ps=args.signal_jitter_ps,
        bucket_jitter_ps=args.bucket_jitter_ps,
        time_quantization_ps=args.time_quantization_ps,
        debug=args.debug,
        ignore_polarization=args.ignore_polarization,
        scene_target=scene_target,
    )
    # -------------------------------------------------------------------
    # Fase E: guardado de resultados + resumen de conteos
    # -------------------------------------------------------------------
    save_outputs(hist, t_edges, args.out_dir, spad_z=args.spad_z, hist_grid_size=args.hist_grid_size)

    print("Simulation finished")
    print("Counts:")
    for k, v in meta["counts"].items():
        print(f"  {k}: {v}")
    if t_edges.size >= 2:
        dt_bin = float(t_edges[1] - t_edges[0])
        depth_per_bin = 0.5 * C * dt_bin
        print(
            f"Time window: [{float(t_edges[0]) * 1e9:.3f}, {float(t_edges[-1]) * 1e9:.3f}] ns "
            f"(bins={hist.shape[2]})"
        )
        print(f"Time bin width: {dt_bin * 1e12:.3f} ps ({dt_bin * 1e9:.6f} ns)")
        print(f"Depth per bin (c*dt/2): {depth_per_bin * 1e3:.3f} mm")
    print(f"Histogram shape: {hist.shape}")
    print(f"Output directory: {os.path.abspath(args.out_dir)}")
    if plt is None:
        print("matplotlib not available, saved only .npy arrays")

    debug_scene_hits = meta.get("debug_scene_hits")
    if args.debug and plt is not None and isinstance(debug_scene_hits, np.ndarray):
        plt.imshow(debug_scene_hits, origin="lower")
        plt.colorbar(label="Debug intersection count")
        plt.title("Debug: scene-hit count (per SPAD pixel)")
        plt.tight_layout()
        plt.savefig(os.path.join(args.out_dir, "debug_scene_hits.png"), dpi=150)
        plt.close()

    if meta["counts"].get("bucket_hits", 0) > 0 and meta["counts"].get("binned", 0) == 0:
        dt_range = meta.get("observed_dt_range", (None, None))
        if dt_range[0] is not None and dt_range[1] is not None:
            print(
                f"WARNING: Se detectaron coincidencias (bucket_hits={meta['counts']['bucket_hits']}), "
                f"pero ninguna cayó dentro de [t_min={sim_t_min*1e9:.1f} ns, t_max={sim_t_max*1e9:.1f} ns). "
                f"Rango dt observado: [{dt_range[0]*1e9:.1f} ns, {dt_range[1]*1e9:.1f} ns]. "
                f"Ajusta --t-min o --t-max para capturar eventos."
            )
        else:
            print(
                "WARNING: Hay coincidencias detectadas pero ninguna en el rango temporal. "
                "Aumenta --t-max o desplaza --t-min/--t-max."
            )


if __name__ == "__main__":
    main()
