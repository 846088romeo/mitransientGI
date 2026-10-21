#!/usr/bin/env python3
"""Render a Mitsuba XML scene with the statistical ghost integrator."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

C = 299792458.0


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Render an XML scene with qghost"
    )
    parser.add_argument(
        "scene",
        type=Path,
        help="Mitsuba XML scene, for example scenes/mighost_T_small_qghost.xml",
    )
    parser.add_argument(
        "--spp",
        "--pairs",
        dest="spp",
        type=int,
        default=256,
        help="total emitted photon pairs (default: 256)",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("ghost_output"),
        help="directory for NumPy outputs (default: ghost_output)",
    )
    parser.add_argument(
        "--variant",
        default="llvm_ad_rgb",
        help="Mitsuba variant (default: llvm_ad_rgb)",
    )
    parser.add_argument(
        "--grid-size",
        type=int,
        default=4,
        help="histogram grid size (default: 4x4)",
    )
    parser.add_argument(
        "--debug",
        action="store_true",
        help="print stage counts and save signal/scene/bucket debug maps",
    )
    parser.add_argument(
        "--no-auto-window",
        action="store_true",
        help="keep the OPL windows written in the XML",
    )
    return parser.parse_args()


def save_visualizations(output, histogram, film, spad_z, grid_size):
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        print("matplotlib is not installed; skipping PNG visualizations")
        return

    import numpy as np

    data = histogram[..., 0]
    intensity = data.sum(axis=2)
    temporal_bins = data.shape[2]
    start_opl = float(np.asarray(film.start_opl).item())
    bin_width_opl = float(np.asarray(film.bin_width_opl).item())
    delta_opl = start_opl + (np.arange(temporal_bins) + 0.5) * bin_width_opl

    peak_indices = np.argmax(data, axis=2)
    spad_offset = abs(float(np.asarray(spad_z).item()))
    depth = 0.5 * (delta_opl[peak_indices] + spad_offset)
    depth[data.max(axis=2) <= 0] = np.nan

    def save_image(path, image, title, colorbar_label):
        fig, ax = plt.subplots(figsize=(6, 5))
        image_plot = ax.imshow(image, origin="lower")
        ax.set_title(title)
        fig.colorbar(image_plot, ax=ax, label=colorbar_label)
        fig.tight_layout()
        fig.savefig(output / path, dpi=150)
        plt.close(fig)

    save_image(
        "integrated_intensity.png",
        intensity,
        "Integrated ghost image",
        "Correlation counts",
    )
    save_image("depth_map.png", depth, "Peak-time depth map", "Distance [m]")

    centers_ns = delta_opl / 299792458.0 * 1e9
    center_curve = data[data.shape[0] // 2, data.shape[1] // 2]
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.plot(centers_ns, center_curve)
    ax.set_xlabel("Delta t [ns]")
    ax.set_ylabel("Correlation")
    ax.set_title("Temporal histogram at center pixel")
    fig.tight_layout()
    fig.savefig(output / "center_histogram.png", dpi=150)
    plt.close(fig)

    n = max(1, int(grid_size))
    active = intensity > 0
    if np.any(active):
        ys, xs = np.where(active)
        y_min, y_max = int(ys.min()), int(ys.max())
        x_min, x_max = int(xs.min()), int(xs.max())
    else:
        y_min, y_max = 0, data.shape[0] - 1
        x_min, x_max = 0, data.shape[1] - 1

    y_samples = np.linspace(y_min, y_max, n).astype(int)
    x_samples = np.linspace(x_min, x_max, n).astype(int)
    fig, axes = plt.subplots(n, n, figsize=(3.0 * n, 2.2 * n),
                             sharex=True, sharey=True, squeeze=False)
    for iy, y in enumerate(y_samples):
        for ix, x in enumerate(x_samples):
            ax = axes[iy, ix]
            ax.plot(centers_ns, data[y, x])
            ax.set_title(f"x={x}, y={y}", fontsize=8)
            ax.grid(alpha=0.2)
            if iy == n - 1:
                ax.set_xlabel("Delta t [ns]")
            if ix == 0:
                ax.set_ylabel("Counts")
    fig.suptitle(f"Temporal histograms ({n}x{n})")
    fig.tight_layout()
    fig.savefig(output / "histograms_grid.png", dpi=150, bbox_inches="tight")
    plt.close(fig)


def configure_temporal_windows(scene, spp, margin=0.1):
    """Adapt event and correlation OPL windows to the loaded scene."""
    import numpy as np
    import mitsuba as mi

    emitter = scene.emitters()[0]
    integrator = scene.integrator()
    film = scene.sensors()[0].film()
    source = np.asarray(emitter.source, dtype=float).reshape(-1)
    bbox = scene.bbox()
    corners = np.asarray([
        [x, y, z]
        for x in (float(bbox.min.x), float(bbox.max.x))
        for y in (float(bbox.min.y), float(bbox.max.y))
        for z in (float(bbox.min.z), float(bbox.max.z))
    ])
    distances = np.linalg.norm(corners - source, axis=1)
    d_min = float(distances.min())
    d_max = float(distances.max())

    spad_opl = abs(float(np.asarray(integrator.spad_z).item()))
    cone = float(np.asarray(emitter.max_angle).item())
    signal_min = spad_opl
    signal_max = spad_opl / max(np.cos(cone), 1e-3)
    delta_min = 2.0 * d_min - signal_max
    delta_max = 2.0 * d_max - signal_min
    span = max(delta_max - delta_min, float(film.bin_width_opl))
    delta_min -= margin * span
    delta_max += margin * span

    film.start_opl = mi.Float(delta_min)
    film.temporal_bins = mi.UInt32(
        max(1, int(np.ceil((delta_max - delta_min) /
                           float(film.bin_width_opl))))
    )

    emission_max = max(0, spp - 1) * C / float(
        np.asarray(integrator.pair_rate_hz).item()
    )
    event_max = emission_max + max(signal_max, 2.0 * d_max)
    film.event_start_opl = mi.Float(0.0)
    film.event_bins = mi.UInt32(
        max(1, int(np.ceil(event_max /
                           float(film.event_bin_width_opl))) + 1)
    )

    # start_value = float(np.asarray(film.start_opl).item())
    # bin_width_value = float(np.asarray(film.bin_width_opl).item())
    # temporal_bin_count = int(np.asarray(film.temporal_bins).item())
    # event_bin_count = int(np.asarray(film.event_bins).item())
    # delta_end = start_value + temporal_bin_count * bin_width_value
    # print(
    #     "Auto OPL window: "
    #     f"scene_distance=[{d_min:.3f}, {d_max:.3f}] m, "
    #     f"delta_opl=[{start_value:.3f}, {delta_end:.3f}] m, "
    #     f"event_bins={event_bin_count}"
    # )


def main() -> int:
    args = parse_args()
    if args.spp <= 0:
        raise ValueError("--spp must be greater than zero")
    if not args.scene.is_file():
        raise FileNotFoundError(f"Scene not found: {args.scene}")

    repo_root = Path(__file__).resolve().parents[1]
    sys.path.insert(0, str(repo_root))

    import drjit as dr
    import mitsuba as mi

    mi.set_variant(args.variant)
    import mitransient  # noqa: F401  # registers custom plugins
    import numpy as np

    scene = mi.load_file(str(args.scene))
    if not args.no_auto_window:
        configure_temporal_windows(scene, args.spp)
    if args.debug:
        params = mi.traverse(scene.integrator())
        params["debug"] = mi.Bool(True)
        params.update()

    # Previous call was commented out to avoid excessive memory usage with large spp values.
    # steady, transient = scene.integrator().render(scene, spp=args.spp)
    # dr.eval(steady, transient)

    # Batch rendering to avoid excessive memory usage with large spp values

    batch_spp = args.spp  # Maximum number of pairs to render in a single batch (ADJUSTABLE)
    num_batches = args.spp // batch_spp
    remainder = args.spp % batch_spp
    batches = [batch_spp] * num_batches + ([remainder] if remainder > 0 else [])
    
    transient_array = None
    steady_array = None

    print(f"Splitting rendering into {len(batches)} batches.")
    scene.sensors()[0].film().clear()
    pair_rate_hz = float(np.asarray(
        scene.integrator().pair_rate_hz
    ).item())
    emission_offset_opl = 0.0
    
    for i, current_spp in enumerate(batches):
        if args.debug:
            params = mi.traverse(scene.integrator())
            params["debug"] = mi.Bool(True)
            params.update()

        params = mi.traverse(scene.integrator())
        params["emission_opl_offset"] = mi.Float(emission_offset_opl)
        params.update()
            
        # Renderiza el lote actual cambiando la semilla para tener nuevo ruido
        scene.integrator().render(
            scene, spp=current_spp, seed=i, develop=False
        )
        emission_offset_opl += current_spp * C / pair_rate_hz

        print(
            f"\r\033[KBatch {i+1}/{len(batches)} completed.",
            end="",
            flush=True,
        )

    print()

    args.output.mkdir(parents=True, exist_ok=True)
    steady, transient = scene.sensors()[0].film().develop()
    dr.eval(steady, transient)
    transient_array = np.asarray(transient)
    steady_array = np.asarray(steady)
    np.save(args.output / "histogram.npy", transient_array)
    np.save(args.output / "steady.npy", steady_array)
    save_visualizations(
        args.output,
        transient_array,
        scene.sensors()[0].film(),
        scene.integrator().spad_z,
        args.grid_size,
    )

    if args.debug:
        film = scene.sensors()[0].film()
        debug_stats = getattr(film, "debug_stats", {})
        print(f"Debug stats: {debug_stats}")
        np.save(args.output / "debug_signal_hits.npy",
            np.asarray(film.debug_signal_storage))
        np.save(args.output / "debug_scene_hits.npy",
            np.asarray(film.debug_scene_storage))
        np.save(args.output / "debug_bucket_hits.npy",
            np.asarray(film.debug_bucket_storage))

    print(f"Scene: {args.scene}")
    print(f"Variant: {mi.variant()}")
    print(f"Steady shape: {steady_array.shape}")
    print(f"Transient shape: {transient_array.shape}")
    print(f"Finite: {bool(np.all(np.isfinite(transient_array)))}")
    print(f"Positive samples: {bool(np.any(transient_array > 0))}")
    print(f"Total correlation: {float(transient_array.sum())}")
    print(f"Saved: {args.output.resolve()}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())