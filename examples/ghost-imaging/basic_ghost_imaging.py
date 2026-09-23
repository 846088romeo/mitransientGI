import numpy as np
import mitransient as mitr

def main():
    height, width = 32, 32
    num_patterns = 2000
    rng = np.random.default_rng(0)

    target = np.zeros((height, width), dtype=np.float64)
    target[8:24, 12:20] = 1.0

    patterns = rng.integers(0, 2, size=(num_patterns, height, width)).astype(
        np.float64
    )
    measurements = mitr.gi.simulate_bucket_measurements(patterns, target)
    reconstruction = mitr.gi.reconstruct_ghost_image(patterns, measurements)

    print("Target shape:", target.shape)
    print("Patterns shape:", patterns.shape)
    print("Measurements shape:", measurements.shape)
    print("Reconstruction shape:", reconstruction.shape)
    print("Reconstruction max value:", reconstruction.max())

if __name__ == "__main__":
    main()