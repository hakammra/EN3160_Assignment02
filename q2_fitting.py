import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from scipy.optimize import minimize


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
    def convert(value):
        if isinstance(value, np.ndarray):
            return value.tolist()
        if isinstance(value, np.generic):
            return value.item()
        raise TypeError(f"Cannot save {type(value).__name__}")

    path.write_text(
        json.dumps(result, indent=2, default=convert) + "\n", encoding="utf-8"
    )


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


def circle_from_points(points):
    offsets = points[1:] - points[0]
    matrix = 2 * offsets
    if np.linalg.cond(matrix) > 1e8:
        raise ValueError("The circle sample is too close to a straight line")
    right = np.sum(offsets**2, axis=1)
    relative_center = np.linalg.solve(matrix, right)
    center = points[0] + relative_center
    radius = np.linalg.norm(relative_center)
    model = np.append(center, radius)
    if not np.isfinite(model).all() or radius < 1e-10:
        raise ValueError("The circle sample is invalid")
    return model


def circle_errors(points, model):
    return np.abs(np.linalg.norm(points - model[:2], axis=1) - model[2])


def fit_circle(points, initial_model):
    def objective(model):
        residuals = np.linalg.norm(points - model[:2], axis=1) - model[2]
        return float(residuals @ residuals)

    result = minimize(
        objective,
        initial_model,
        method="L-BFGS-B",
        bounds=[(None, None), (None, None), (1e-8, None)],
        options={"ftol": 1e-12, "gtol": 1e-7, "maxiter": 300},
    )
    if not result.success or not np.isfinite(result.x).all():
        raise RuntimeError("Circle refinement did not converge")
    return result.x


def ransac_circle(points, threshold=1.25, minimum_inliers=30, iterations=500, seed=2):
    points = np.asarray(points, dtype=float)
    if points.ndim != 2 or points.shape[1] != 2 or len(points) < 3:
        raise ValueError("Provide at least three points with x and y coordinates")
    if not np.isfinite(points).all():
        raise ValueError("Point coordinates must be finite")
    if threshold <= 0 or not 3 <= minimum_inliers <= len(points) or iterations < 1:
        raise ValueError("Invalid RANSAC parameters")

    rng = np.random.default_rng(seed)
    best = None
    best_score = (-1, -np.inf)
    for iteration in range(iterations):
        sample = rng.choice(len(points), 3, replace=False)
        try:
            sample_model = circle_from_points(points[sample])
        except (ValueError, np.linalg.LinAlgError):
            continue
        sample_mask = circle_errors(points, sample_model) <= threshold
        if sample_mask.sum() < minimum_inliers:
            continue

        inliers = sample_mask.copy()
        model = sample_model.copy()
        try:
            for _ in range(10):
                model = fit_circle(points[inliers], model)
                updated = circle_errors(points, model) <= threshold
                if updated.sum() < minimum_inliers or np.array_equal(updated, inliers):
                    break
                inliers = updated
        except RuntimeError:
            continue
        errors = circle_errors(points, model)
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
        raise RuntimeError("No circle reached the required consensus")
    return best


def run_circle_fitting(points, line_result):
    remaining_indices = np.setdiff1d(np.arange(len(points)), line_result["inlier_indices"])
    result = ransac_circle(points[remaining_indices])
    for name in ("sample_indices", "sample_inlier_indices", "inlier_indices"):
        result[name] = remaining_indices[result[name]]
    result["remaining_point_indices"] = remaining_indices
    result["input_point_count"] = len(remaining_indices)
    output_dir = ROOT / "results" / "q2"
    output_dir.mkdir(parents=True, exist_ok=True)
    save_result(output_dir / "circle_fit.json", {"data_seed": 0, **result})

    fig, axis = plt.subplots(figsize=(11, 8), layout="constrained")
    axis.scatter(*points.T, s=24, color="0.7", label="All points")
    axis.scatter(
        *points[line_result["inlier_indices"]].T,
        s=25,
        color="tab:blue",
        label="Line inliers",
    )
    axis.scatter(
        *points[result["inlier_indices"]].T,
        s=25,
        color="tab:orange",
        label="Circle inliers",
    )
    axis.scatter(
        *points[line_result["sample_indices"]].T,
        s=90,
        marker="x",
        color="tab:red",
        label="Winning line sample",
        zorder=5,
    )
    axis.scatter(
        *points[result["sample_indices"]].T,
        s=90,
        marker="x",
        color="tab:purple",
        label="Winning circle sample",
        zorder=5,
    )
    plot_line(axis, line_result["sample_model"], "Sample line", "tab:blue", "dotted")
    plot_line(axis, line_result["model"], "Refined line", "tab:blue")
    for model, label, color, style in (
        (result["sample_model"], "Sample circle", "tab:orange", "dotted"),
        (result["model"], "Refined circle", "tab:orange", "solid"),
        (np.array([2, 3, 10]), "True circle", "0.4", "dashed"),
    ):
        axis.add_patch(
            plt.Circle(
                model[:2], model[2], fill=False, label=label, color=color, linestyle=style
            )
        )
    plot_line(axis, np.array([1, 1, 2]) / np.sqrt(2), "True line", "0.4", "dashed")
    axis.set(xlim=(-15, 15), ylim=(-14, 17), xlabel="x", ylabel="y")
    axis.set_aspect("equal")
    axis.set_title("Line and circle fitting with RANSAC")
    axis.grid(alpha=0.2)
    axis.legend(loc="upper left", bbox_to_anchor=(1.02, 1), fontsize=9)
    fig.savefig(output_dir / "line_and_circle.png", dpi=180)
    plt.close(fig)

    print(f"Remaining points: {len(remaining_indices)}")
    print(f"Circle model (x, y, radius): {result['model']}")
    print(f"Circle inliers: {result['inlier_count']} of {len(remaining_indices)}")
    print(f"Circle inlier RMSE: {result['inlier_rmse']:.4f}")
    return result


def fit_circle_then_line(points):
    circle = ransac_circle(points)
    circle["input_point_count"] = len(points)
    remaining = np.setdiff1d(np.arange(len(points)), circle["inlier_indices"])
    line = None
    failure = None
    if len(remaining) < 40:
        failure = "Too few points remain for the minimum line consensus of 40"
    else:
        try:
            line = ransac_line(points[remaining])
        except RuntimeError as error:
            failure = str(error)
    if line is not None:
        for name in ("sample_indices", "sample_inlier_indices", "inlier_indices"):
            line[name] = remaining[line[name]]
        line["input_point_count"] = len(remaining)
        line["remaining_point_indices"] = remaining
    return {
        "circle": circle,
        "line": line,
        "remaining_point_indices": remaining,
        "line_failure": failure,
    }


def plot_fitting_order(axis, points, line, circle, title):
    axis.scatter(*points.T, s=22, color="0.75", label="All points")
    for result, label, color, sample_color in (
        (line, "Line", "tab:blue", "tab:red"),
        (circle, "Circle", "tab:orange", "tab:purple"),
    ):
        if result is None:
            continue
        axis.scatter(
            *points[result["inlier_indices"]].T,
            s=25,
            color=color,
            label=f"{label} inliers",
        )
        axis.scatter(
            *points[result["sample_indices"]].T,
            s=85,
            marker="x",
            color=sample_color,
            label=f"Winning {label.lower()} sample",
            zorder=5,
        )
        for model_name, model_label, style in (
            ("sample_model", f"Sample {label.lower()}", "dotted"),
            ("model", f"Refined {label.lower()}", "solid"),
        ):
            model = result[model_name]
            if label == "Line":
                plot_line(axis, model, model_label, color, style)
            else:
                axis.add_patch(
                    plt.Circle(
                        model[:2], model[2], fill=False,
                        label=model_label, color=color, linestyle=style,
                    )
                )
    if line is None:
        axis.text(0.5, 0.04, "No line consensus", ha="center", transform=axis.transAxes)
    axis.set(xlim=(-15, 15), ylim=(-14, 17), xlabel="x", ylabel="y", title=title)
    axis.set_aspect("equal")
    axis.grid(alpha=0.2)


def save_order_comparison(points, line, circle, reverse):
    orders = (
        ("Line then circle", line, circle, circle["input_point_count"]),
        ("Circle then line", reverse["line"], reverse["circle"], len(reverse["remaining_point_indices"])),
    )
    summary = []
    evaluation = {}
    for name, line_result, circle_result, remaining_count in orders:
        summary.append({
            "order": name,
            "line_inliers": None if line_result is None else line_result["inlier_count"],
            "circle_inliers": circle_result["inlier_count"],
            "points_for_second_model": remaining_count,
            "line_inlier_rmse": None if line_result is None else line_result["inlier_rmse"],
            "circle_inlier_rmse": circle_result["inlier_rmse"],
            "line_model": None if line_result is None else line_result["model"],
            "circle_model": circle_result["model"],
        })
        line_indices = np.array([]) if line_result is None else np.asarray(line_result["inlier_indices"])
        circle_indices = np.asarray(circle_result["inlier_indices"])
        evaluation[name] = {
            "generated_line_points_in_line_consensus": int(np.sum(line_indices >= 50)),
            "generated_circle_points_in_line_consensus": int(np.sum(line_indices < 50)),
            "generated_circle_points_in_circle_consensus": int(np.sum(circle_indices < 50)),
            "generated_line_points_in_circle_consensus": int(np.sum(circle_indices >= 50)),
        }

    result = {
        "data_seed": 0,
        "point_count": len(points),
        "line_then_circle": {"line": line, "circle": circle},
        "circle_then_line": reverse,
        "summary": summary,
        "post_fit_evaluation": evaluation,
    }
    output_dir = ROOT / "results" / "q2"
    output_dir.mkdir(parents=True, exist_ok=True)
    save_result(output_dir / "fitting_order_comparison.json", result)

    fig, axes = plt.subplots(1, 2, figsize=(14, 8))
    plot_fitting_order(axes[0], points, line, circle, "Line then circle")
    plot_fitting_order(axes[1], points, reverse["line"], reverse["circle"], "Circle then line")
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", ncol=4, fontsize=10)
    fig.tight_layout(rect=(0, 0.12, 1, 1))
    fig.savefig(output_dir / "fitting_order_comparison.png", dpi=180)
    plt.close(fig)
    return result


if __name__ == "__main__":
    points, line_result = run_line_fitting()
    circle_result = run_circle_fitting(points, line_result)
    reverse = fit_circle_then_line(points)
    comparison = save_order_comparison(points, line_result, circle_result, reverse)
    for row in comparison["summary"]:
        print(f"{row['order']}: line inliers {row['line_inliers']}, circle inliers {row['circle_inliers']}")
