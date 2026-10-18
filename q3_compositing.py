import argparse
import json
from pathlib import Path

import cv2 as cv
import matplotlib.pyplot as plt
import numpy as np


ROOT = Path(__file__).resolve().parent
DEFAULT_CORNERS = [[700, 440], [980, 425], [980, 553], [700, 544]]


def validate_corners(corners, shape):
    corners = np.asarray(corners, dtype=np.float32)
    if corners.shape != (4, 2) or not np.isfinite(corners).all():
        raise ValueError("Provide four finite image coordinates")
    height, width = shape[:2]
    if np.any(corners < 0) or np.any(corners[:, 0] >= width) or np.any(corners[:, 1] >= height):
        raise ValueError("The corners must be inside the background image")
    if not cv.isContourConvex(corners.reshape(-1, 1, 2)) or cv.contourArea(corners) < 1:
        raise ValueError("Select the corners around a convex region")
    return corners


def select_corners(image):
    height, width = image.shape[:2]
    scale = min(1000 / width, 700 / height, 1)
    preview = cv.resize(image, None, fx=scale, fy=scale)
    points = []
    title = "Top left, top right, bottom right, bottom left   Enter to save   R to reset"

    def redraw():
        frame = preview.copy()
        for index, point in enumerate(points, 1):
            position = tuple(np.round(np.array(point) * scale).astype(int))
            cv.circle(frame, position, 5, (0, 255, 255), -1)
            cv.putText(frame, str(index), position, cv.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
        if len(points) == 4:
            contour = np.round(np.array(points) * scale).astype(np.int32)
            cv.polylines(frame, [contour], True, (0, 255, 255), 2)
        cv.imshow(title, frame)

    def on_click(event, x, y, flags, data):
        if event == cv.EVENT_LBUTTONDOWN and len(points) < 4:
            points.append([x / scale, y / scale])
            redraw()

    cv.namedWindow(title, cv.WINDOW_AUTOSIZE)
    cv.setMouseCallback(title, on_click)
    redraw()
    try:
        while True:
            key = cv.waitKey(20) & 255
            if key == 27 or cv.getWindowProperty(title, cv.WND_PROP_VISIBLE) < 1:
                raise RuntimeError("Point selection was cancelled")
            if key in (ord("r"), ord("R")):
                points.clear()
                redraw()
            if key in (10, 13) and len(points) == 4:
                return validate_corners(points, image.shape)
    finally:
        cv.destroyAllWindows()


def composite_overlay(background, overlay, corners, opacity=0.75):
    corners = validate_corners(corners, background.shape)
    if not 0 <= opacity <= 1:
        raise ValueError("Opacity must be between zero and one")
    if overlay.ndim != 3 or overlay.shape[2] not in (3, 4):
        raise ValueError("The overlay must be a colour image")
    height, width = overlay.shape[:2]
    source = np.float32([[0, 0], [width - 1, 0], [width - 1, height - 1], [0, height - 1]])
    homography = cv.getPerspectiveTransform(source, corners)
    if not np.isfinite(homography).all() or np.linalg.cond(homography) > 1e12:
        raise ValueError("The selected corners give an unstable homography")
    alpha = overlay[:, :, 3].astype(np.float32) / 255 if overlay.shape[2] == 4 else np.ones((height, width), np.float32)
    colours = overlay[:, :, :3].astype(np.float32) * alpha[:, :, None]
    size = (background.shape[1], background.shape[0])
    warped = cv.warpPerspective(colours, homography, size, flags=cv.INTER_LINEAR)
    coverage = cv.warpPerspective(alpha, homography, size, flags=cv.INTER_LINEAR)
    result = background.astype(np.float32) * (1 - opacity * coverage[:, :, None]) + opacity * warped
    return np.clip(np.round(result), 0, 255).astype(np.uint8), homography, coverage


def run_compositing(select=False, points_path=None):
    background = cv.imread(str(ROOT / "images/wadham.jpg"))
    overlay = cv.imread(str(ROOT / "images/flag.png"), cv.IMREAD_UNCHANGED)
    if background is None or overlay is None:
        raise FileNotFoundError("The background and flag images are required")
    output_dir = ROOT / "results/q3"
    output_dir.mkdir(parents=True, exist_ok=True)
    points_path = Path(points_path) if points_path else output_dir / "wadham_points.json"
    if select:
        corners = select_corners(background)
    elif points_path.exists():
        corners = json.loads(points_path.read_text())["corners"]
    else:
        corners = DEFAULT_CORNERS
    corners = validate_corners(corners, background.shape)
    opacity = 0.75
    composite, homography, coverage = composite_overlay(background, overlay, corners, opacity)
    record = {
        "background": "images/wadham.jpg",
        "overlay": "images/flag.png",
        "corner_order": ["top left", "top right", "bottom right", "bottom left"],
        "corners": corners.tolist(),
        "opacity": opacity,
        "homography": homography.tolist(),
    }
    points_path.write_text(json.dumps(record, indent=2) + "\n", encoding="utf-8")
    cv.imwrite(str(output_dir / "wadham_composite.png"), composite)
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.8), layout="constrained")
    axes[0].imshow(cv.cvtColor(background, cv.COLOR_BGR2RGB))
    closed = np.vstack((corners, corners[0]))
    axes[0].plot(closed[:, 0], closed[:, 1], "yo-", linewidth=1.5)
    for index, point in enumerate(corners, 1):
        axes[0].text(*point, str(index), color="red", fontsize=12)
    axes[0].set_title("Selected wall corners")
    axes[1].imshow(cv.cvtColor(composite, cv.COLOR_BGR2RGB))
    axes[1].set_title("Blended result")
    for axis in axes:
        axis.axis("off")
    fig.savefig(output_dir / "wadham_comparison.jpg", dpi=120, pil_kwargs={"quality": 90})
    plt.close(fig)
    return record


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--select", action="store_true")
    parser.add_argument("--points", type=Path)
    arguments = parser.parse_args()
    run_compositing(arguments.select, arguments.points)
