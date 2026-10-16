import json
from pathlib import Path

import cv2 as cv
import matplotlib.pyplot as plt
import numpy as np
from scipy.ndimage import maximum_filter


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


def scale_space_extrema(scale_space, sigmas, response_threshold):
    if len(sigmas) < 3 or scale_space.shape[2] != len(sigmas):
        raise ValueError("Scale-space needs at least three matching sigma values")
    if response_threshold <= 0:
        raise ValueError("The response threshold must be positive")

    local_maxima = maximum_filter(scale_space, size=3, mode="nearest")
    maxima = (scale_space == local_maxima) & (scale_space >= response_threshold)
    maxima[:, :, 0] = False
    maxima[:, :, -1] = False
    coordinates = np.argwhere(maxima)
    return sorted(
        coordinates, key=lambda point: float(scale_space[tuple(point)]), reverse=True
    )


def yellow_ring_fraction(yellow_mask, x, y, radius, outer_radius_factor):
    height, width = yellow_mask.shape
    half_width = int(np.ceil(outer_radius_factor * radius))
    x0, x1 = max(0, x - half_width), min(width, x + half_width + 1)
    y0, y1 = max(0, y - half_width), min(height, y + half_width + 1)
    rows, columns = np.ogrid[y0:y1, x0:x1]
    squared_distance = (columns - x) ** 2 + (rows - y) ** 2
    ring = (squared_distance > radius**2) & (
        squared_distance <= (outer_radius_factor * radius) ** 2
    )
    return float(yellow_mask[y0:y1, x0:x1][ring].mean())


def detect_sunflowers():
    image_path = ROOT / "images" / "the_berry_farms_sunflower_field.jpeg"
    image = cv.imread(str(image_path), cv.IMREAD_REDUCED_COLOR_4)
    if image is None:
        raise FileNotFoundError(f"Cannot load image: {image_path.name}")

    gray = cv.cvtColor(image, cv.COLOR_BGR2GRAY).astype(np.float32) / 255.0
    sigmas = np.arange(1.0, 12.5, 0.5)
    response_threshold = 0.22
    minimum_yellow_fraction = 0.35
    outer_radius_factor = 2.2
    minimum_separation_factor = 0.6
    hsv_lower = [15, 100, 120]
    hsv_upper = [38, 255, 255]

    scale_space = build_scale_space(gray, sigmas)
    candidates = scale_space_extrema(scale_space, sigmas, response_threshold)
    hsv = cv.cvtColor(image, cv.COLOR_BGR2HSV)
    yellow_mask = cv.inRange(hsv, np.array(hsv_lower), np.array(hsv_upper)) > 0
    height, width = gray.shape
    circles = []

    for row, column, scale_index in candidates:
        x, y = int(column), int(row)
        sigma = float(sigmas[scale_index])
        radius = float(np.sqrt(2) * sigma)
        if x < radius or y < radius or x + radius >= width or y + radius >= height:
            continue

        yellow_fraction = yellow_ring_fraction(
            yellow_mask, x, y, radius, outer_radius_factor
        )
        if yellow_fraction < minimum_yellow_fraction:
            continue
        if any(
            np.hypot(x - circle["x"], y - circle["y"])
            < minimum_separation_factor * (radius + circle["radius"])
            for circle in circles
        ):
            continue

        circles.append(
            {
                "x": x,
                "y": y,
                "radius": radius,
                "sigma": sigma,
                "response": float(scale_space[row, column, scale_index]),
                "yellow_ring_fraction": yellow_fraction,
            }
        )

    circles.sort(key=lambda circle: (-circle["radius"], -circle["response"]))
    if not circles:
        raise RuntimeError("No sunflower blobs passed the detection thresholds")

    result = {
        "image": image_path.name,
        "image_size_hw": [height, width],
        "coordinate_system": "pixels in the reduced image",
        "original_image_scale_factor": 4,
        "blob_polarity": "dark flower centres",
        "radius_definition": "sqrt(2) * sigma; excludes the outer petals",
        "sigma_values": sigmas.tolist(),
        "response_threshold": response_threshold,
        "yellow_hsv_lower": hsv_lower,
        "yellow_hsv_upper": hsv_upper,
        "minimum_yellow_ring_fraction": minimum_yellow_fraction,
        "ring_outer_radius_factor": outer_radius_factor,
        "minimum_separation_factor": minimum_separation_factor,
        "exclude_boundary_scales": True,
        "exclude_clipped_circles": True,
        "candidate_count": len(candidates),
        "detected_count": len(circles),
        "largest_circles": circles[:5],
        "circles": circles,
    }
    output_dir = ROOT / "results" / "q1"
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "sunflower_circles.json").write_text(
        json.dumps(result, indent=2) + "\n", encoding="utf-8"
    )

    rgb = cv.cvtColor(image, cv.COLOR_BGR2RGB)
    fig, axes = plt.subplots(1, 2, figsize=(12, 6), layout="constrained")
    axes[0].imshow(rgb)
    axes[0].set_title("Input sunflower image (360 x 360)")
    axes[1].imshow(rgb)
    axes[1].set_title(f"{len(circles)} dark-centre blobs; five largest in orange")
    for index, circle in enumerate(circles):
        color = "#ff8c00" if index < 5 else "#00ff99"
        axes[1].add_patch(
            plt.Circle(
                (circle["x"], circle["y"]),
                circle["radius"],
                fill=False,
                color=color,
                linewidth=1.2 if index < 5 else 0.7,
            )
        )
        if index < 5:
            axes[1].text(
                circle["x"] + circle["radius"] + 1,
                circle["y"],
                str(index + 1),
                color=color,
                fontsize=9,
                weight="bold",
            )
    for axis in axes:
        axis.set_axis_off()
    fig.savefig(output_dir / "sunflower_detection.png", dpi=180)
    plt.close(fig)

    print(f"Sunflower image: {width} x {height}")
    print(f"Sigma range: {sigmas[0]:g} to {sigmas[-1]:g}, step 0.5")
    print(f"Detected dark-centre blobs: {len(circles)}")
    for index, circle in enumerate(circles[:5], start=1):
        print(
            f"{index}: center=({circle['x']}, {circle['y']}), "
            f"radius={circle['radius']:.2f}, sigma={circle['sigma']:g}"
        )
    return result


if __name__ == "__main__":
    validate_synthetic_circle()
    detect_sunflowers()
