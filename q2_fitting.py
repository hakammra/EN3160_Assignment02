import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


ROOT = Path(__file__).resolve().parent


def generate_points(seed=0):
    rng = np.random.RandomState(seed)
    count = 50
    angles = rng.uniform(0, 2 * np.pi, count)
    radii = 10 + (10 / 16) * rng.randn(count)
    circle = np.column_stack((2 + radii * np.cos(angles), 3 + radii * np.sin(angles)))
    x = np.linspace(-12, 12, count)
    line = np.column_stack((x, -x + 2 + rng.randn(count)))
    return np.vstack((circle, line))


def line_from_points(points):
    direction = points[1] - points[0]
    length = np.linalg.norm(direction)
    if length < 1e-10:
        raise ValueError("The two points must be different")
    normal = np.array([direction[1], -direction[0]]) / length
    model = np.append(normal, normal @ points[0])
    return -model if model[2] < 0 else model


def line_errors(points, model):
    return np.abs(points @ model[:2] - model[2])


def fit_line(points):
    center = points.mean(axis=0)
    _, _, vectors = np.linalg.svd(points - center, full_matrices=False)
    normal = vectors[-1]
    model = np.append(normal, normal @ center)
    return -model if model[2] < 0 else model


def ransac_line(points, threshold=1.4, minimum_inliers=40, iterations=500, seed=1):
    points = np.asarray(points, dtype=float)
    if points.ndim != 2 or points.shape[1] != 2 or len(points) < 2:
        raise ValueError("Provide at least two points with x and y coordinates")
    if not np.isfinite(points).all():
        raise ValueError("Point coordinates must be finite")
    if threshold <= 0 or not 2 <= minimum_inliers <= len(points) or iterations < 1:
        raise ValueError("Invalid RANSAC parameters")

    rng = np.random.default_rng(seed)
    best = None
    best_score = (-1, -np.inf)
    for iteration in range(iterations):
        sample = rng.choice(len(points), 2, replace=False)
        try:
            sample_model = line_from_points(points[sample])
        except ValueError:
            continue
        sample_mask = line_errors(points, sample_model) <= threshold
        if sample_mask.sum() < minimum_inliers:
            continue

        inliers = sample_mask.copy()
        for _ in range(10):
            model = fit_line(points[inliers])
            updated = line_errors(points, model) <= threshold
            if updated.sum() < minimum_inliers or np.array_equal(updated, inliers):
                break
            inliers = updated
        errors = line_errors(points, model)
        inliers = errors <= threshold
        count = int(inliers.sum())
        if count < minimum_inliers:
            continue
        mean_squared_error = float(np.mean(errors[inliers] ** 2))
        score = (count, -mean_squared_error)
        if score > best_score:
            best_score = score
            best = {
                "model": model,
                "sample_model": sample_model,
                "sample_indices": sample,
                "sample_inlier_indices": np.flatnonzero(sample_mask),
                "inlier_indices": np.flatnonzero(inliers),
                "inlier_count": count,
                "inlier_rmse": float(np.sqrt(mean_squared_error)),
                "winning_iteration": iteration + 1,
                "threshold": threshold,
                "minimum_inliers": minimum_inliers,
                "iterations": iterations,
                "seed": seed,
            }
    if best is None:
        raise RuntimeError("No line reached the required consensus")
    return best


def plot_line(axis, model, label, color, linestyle="solid"):
    a, b, d = model
    if abs(b) < 1e-10:
        axis.axvline(d / a, label=label, color=color, linestyle=linestyle)
    else:
        x = np.array([-15, 15])
        axis.plot(x, (d - a * x) / b, label=label, color=color, linestyle=linestyle)


def save_result(path, result):
    values = {
        key: value.tolist() if isinstance(value, np.ndarray) else value
        for key, value in result.items()
    }
    path.write_text(json.dumps(values, indent=2) + "\n", encoding="utf-8")


def run_line_fitting():
    points = generate_points()
    result = ransac_line(points)
    if not np.isclose(np.linalg.norm(result["model"][:2]), 1):
        raise RuntimeError("The line normal must have unit length")

    output_dir = ROOT / "results" / "q2"
    output_dir.mkdir(parents=True, exist_ok=True)
    np.savetxt(output_dir / "points.csv", points, delimiter=",", header="x,y", comments="")
    save_result(output_dir / "line_fit.json", {"data_seed": 0, **result})

    fig, axis = plt.subplots(figsize=(8, 8), layout="constrained")
    axis.scatter(*points.T, color="0.7", s=22, label="All points")
    axis.scatter(
        *points[result["inlier_indices"]].T, s=24, color="tab:blue", label="Line inliers"
    )
    axis.scatter(
        *points[result["sample_indices"]].T,
        marker="x",
        s=90,
        color="tab:red",
        label="Winning line sample",
        zorder=5,
    )
    plot_line(axis, result["sample_model"], "Sample line", "tab:orange", "dotted")
    plot_line(axis, result["model"], "Refined line", "tab:green")
    plot_line(axis, np.array([1, 1, 2]) / np.sqrt(2), "True line", "0.4", "dashed")
    axis.set(xlim=(-15, 15), ylim=(-14, 17), xlabel="x", ylabel="y")
    axis.set_aspect("equal")
    axis.set_title("Line fitting with RANSAC")
    axis.grid(alpha=0.2)
    axis.legend()
    fig.savefig(output_dir / "line_fit.png", dpi=180)
    plt.close(fig)

    print(f"Line model (a, b, d): {result['model']}")
    print(f"Unit normal length: {np.linalg.norm(result['model'][:2]):.6f}")
    print(f"Line inliers: {result['inlier_count']} of {len(points)}")
    print(f"Line inlier RMSE: {result['inlier_rmse']:.4f}")
    return points, result


if __name__ == "__main__":
    run_line_fitting()
