import json
from pathlib import Path

import cv2 as cv
import matplotlib.pyplot as plt
import numpy as np


ROOT = Path(__file__).resolve().parent


def normalized_log_kernel(sigma):
    if not np.isfinite(sigma) or sigma <= 0:
        raise ValueError("Sigma must be positive and finite")

    half_width = int(np.ceil(3 * sigma))
    coordinates = np.arange(-half_width, half_width + 1, dtype=np.float64)
    x, y = np.meshgrid(coordinates, coordinates)
    squared_distance = x**2 + y**2
    kernel = (
        (squared_distance / sigma**2 - 2)
        * np.exp(-squared_distance / (2 * sigma**2))
        / (2 * np.pi * sigma**2)
    )
    return kernel.astype(np.float32)


def build_scale_space(image, sigmas):
    image = np.asarray(image, dtype=np.float32)
    if image.ndim != 2:
        raise ValueError("Scale-space input must be a grayscale image")
    if len(sigmas) == 0:
        raise ValueError("At least one sigma is required")

    responses = [
        cv.filter2D(image, cv.CV_32F, normalized_log_kernel(sigma))
        for sigma in sigmas
    ]
    return np.stack(responses, axis=-1)


def synthetic_circle(size=71, radius=14):
    if size <= 0 or size % 2 == 0:
        raise ValueError("Image size must be a positive odd integer")
    if radius <= 0 or radius >= size // 2:
        raise ValueError("Radius must be positive and smaller than half the image")

    half_width = size // 2
    coordinates = np.arange(-half_width, half_width + 1)
    x, y = np.meshgrid(coordinates, coordinates)
    return ((x**2 + y**2 > radius**2) * 255).astype(np.float32)


def validate_synthetic_circle():
    radius = 14
    image = synthetic_circle(radius=radius)
    sigmas = np.arange(5, 16, dtype=np.float64)
    scale_space = build_scale_space(image, sigmas)
    row, column, scale_index = np.unravel_index(
        np.argmax(scale_space), scale_space.shape
    )
    selected_sigma = float(sigmas[scale_index])
    estimated_radius = float(np.sqrt(2) * selected_sigma)
    expected_center = image.shape[0] // 2
    center_error = float(np.hypot(column - expected_center, row - expected_center))
    radius_error = abs(estimated_radius - radius)

    if center_error > 1 or radius_error > 1:
        raise RuntimeError("Synthetic-circle scale selection failed")
    if scale_index == 0 or scale_index == len(sigmas) - 1:
        raise RuntimeError("Selected sigma is at the boundary of the search range")

    output_dir = ROOT / "results" / "q1"
    output_dir.mkdir(parents=True, exist_ok=True)
    result = {
        "image_size": list(image.shape),
        "true_center_xy": [expected_center, expected_center],
        "true_radius": radius,
        "sigma_values": sigmas.tolist(),
        "selected_sigma": selected_sigma,
        "estimated_center_xy": [int(column), int(row)],
        "estimated_radius": estimated_radius,
        "center_error_pixels": center_error,
        "radius_error_pixels": radius_error,
        "maximum_allowed_error_pixels": 1.0,
        "center_responses": scale_space[
            expected_center, expected_center, :
        ].tolist(),
        "passed": True,
    }
    (output_dir / "synthetic_circle.json").write_text(
        json.dumps(result, indent=2) + "\n", encoding="utf-8"
    )

    fig, axes = plt.subplots(1, 3, figsize=(14, 4.5), layout="constrained")
    kernel_plot = axes[0].imshow(
        normalized_log_kernel(selected_sigma), cmap="RdBu_r"
    )
    axes[0].set_title(f"Scale-normalized LoG, sigma = {selected_sigma:g}")
    axes[0].set_axis_off()
    fig.colorbar(kernel_plot, ax=axes[0], shrink=0.8)

    axes[1].imshow(image, cmap="gray", vmin=0, vmax=255)
    axes[1].add_patch(
        plt.Circle(
            (column, row),
            estimated_radius,
            fill=False,
            color="tab:red",
            linewidth=1.8,
        )
    )
    axes[1].plot(column, row, "+", color="tab:red")
    axes[1].set_title(f"True radius = {radius}; estimated = {estimated_radius:.2f}")
    axes[1].set_axis_off()

    axes[2].plot(sigmas, result["center_responses"], "o-", color="tab:blue")
    axes[2].axvline(
        radius / np.sqrt(2),
        color="tab:green",
        linestyle="--",
        label="Theoretical sigma",
    )
    axes[2].axvline(
        selected_sigma, color="tab:red", linestyle=":", label="Selected sigma"
    )
    axes[2].set_xlabel("Sigma (pixels)")
    axes[2].set_ylabel("LoG response at true center")
    axes[2].set_title("Scale selection")
    axes[2].legend()
    axes[2].grid(alpha=0.25)
    fig.savefig(output_dir / "synthetic_circle.png", dpi=180)
    plt.close(fig)

    print(f"Detected center: ({column}, {row})")
    print(f"Selected sigma: {selected_sigma:g}")
    print(f"Estimated radius: {estimated_radius:.4f} pixels")
    print(f"True radius: {radius} pixels")
    print(f"Radius error: {radius_error:.4f} pixels")
    print("Synthetic-circle validation passed.")
    return result


if __name__ == "__main__":
    validate_synthetic_circle()
