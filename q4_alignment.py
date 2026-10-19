import json
from pathlib import Path

import cv2 as cv
import matplotlib.pyplot as plt
import numpy as np


ROOT = Path(__file__).resolve().parent
PCB_NAMES = ["AAAA1.jpg", "ADAD1.jpg", "AECC2.jpg", "AECC3.jpg", "AECC4.jpg"]


def detect_sift(image, sift, maximum_side=1000):
    gray = cv.cvtColor(image, cv.COLOR_BGR2GRAY)
    scale = min(maximum_side / max(gray.shape), 1)
    small = cv.resize(gray, None, fx=scale, fy=scale, interpolation=cv.INTER_AREA)
    keypoints, descriptors = sift.detectAndCompute(small, None)
    sx, sy = small.shape[1] / gray.shape[1], small.shape[0] / gray.shape[0]
    keypoints = [cv.KeyPoint(point.pt[0] / sx, point.pt[1] / sy, point.size / np.sqrt(sx * sy), point.angle, point.response, point.octave, point.class_id) for point in keypoints]
    return keypoints, descriptors, gray.shape


def estimate_pcb_homography(image, reference_features, sift, ratio=0.8, threshold=5.0):
    keypoints, descriptors, _ = detect_sift(image, sift)
    reference_keypoints, reference_descriptors, reference_shape = reference_features
    if descriptors is None or reference_descriptors is None:
        raise RuntimeError("SIFT descriptors could not be found")
    matcher = cv.BFMatcher(cv.NORM_L2)
    neighbours = matcher.knnMatch(descriptors, reference_descriptors, k=2)
    reverse = {match.queryIdx: match.trainIdx for match in matcher.match(reference_descriptors, descriptors)}
    matches = [pair[0] for pair in neighbours if len(pair) == 2 and pair[0].distance < ratio * pair[1].distance and reverse[pair[0].trainIdx] == pair[0].queryIdx]
    if len(matches) < 4:
        raise RuntimeError("At least four reliable matches are required")
    source = np.float32([keypoints[match.queryIdx].pt for match in matches])
    destination = np.float32([reference_keypoints[match.trainIdx].pt for match in matches])
    cv.setRNGSeed(0)
    homography, mask = cv.findHomography(source, destination, cv.RANSAC, threshold, maxIters=5000, confidence=0.995)
    if homography is None or mask is None or not np.isfinite(homography).all() or np.linalg.matrix_rank(homography) != 3:
        raise RuntimeError("A valid PCB homography could not be estimated")
    inliers = mask.ravel().astype(bool)
    if inliers.sum() < 4:
        raise RuntimeError("The homography has too few inliers")
    source_coverage = cv.contourArea(cv.convexHull(source[inliers])) / (image.shape[0] * image.shape[1])
    reference_coverage = cv.contourArea(cv.convexHull(destination[inliers])) / (reference_shape[0] * reference_shape[1])
    if min(source_coverage, reference_coverage) < 0.005:
        raise RuntimeError("The inliers do not cover a usable image region")
    projected = cv.perspectiveTransform(source[:, None, :], homography)[:, 0, :]
    errors = np.linalg.norm(projected - destination, axis=1)
    record = {
        "source_keypoints": len(keypoints),
        "reference_keypoints": len(reference_keypoints),
        "ratio_threshold": ratio,
        "ransac_threshold_pixels": threshold,
        "match_count": len(matches),
        "inlier_count": int(inliers.sum()),
        "inlier_rmse_pixels": float(np.sqrt(np.mean(errors[inliers] ** 2))),
        "source_inlier_coverage": source_coverage,
        "reference_inlier_coverage": reference_coverage,
        "source_points": source.tolist(),
        "reference_points": destination.tolist(),
        "inlier_mask": inliers.tolist(),
        "homography": homography.tolist(),
    }
    return record, keypoints, matches


def draw_pcb_matches(image, reference, keypoints, reference_keypoints, matches, inlier_mask, label):
    inliers = [match for match, accepted in zip(matches, inlier_mask) if accepted]
    shown = sorted(inliers, key=lambda match: match.distance)[:80]
    view = cv.drawMatches(image, keypoints, reference, reference_keypoints, shown, None, matchColor=(0, 220, 0), flags=cv.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS)
    if view.shape[1] > 1400:
        scale = 1400 / view.shape[1]
        view = cv.resize(view, None, fx=scale, fy=scale, interpolation=cv.INTER_AREA)
    view = cv.copyMakeBorder(view, 36, 0, 0, 0, cv.BORDER_CONSTANT, value=(255, 255, 255))
    cv.putText(view, label, (12, 25), cv.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 1, cv.LINE_AA)
    return view


def run_matching():
    cv.setNumThreads(2)
    image_dir = ROOT / "images/pcbs"
    output_dir = ROOT / "results/q4"
    output_dir.mkdir(parents=True, exist_ok=True)
    reference = cv.imread(str(image_dir / "AECC1.jpg"))
    if reference is None:
        raise FileNotFoundError("AECC1.jpg is required as the reference image")
    sift = cv.SIFT_create(nfeatures=5000)
    reference_features = detect_sift(reference, sift)
    summary = []
    for name in PCB_NAMES:
        image = cv.imread(str(image_dir / name))
        if image is None:
            raise FileNotFoundError(name)
        record, keypoints, matches = estimate_pcb_homography(image, reference_features, sift)
        record.update({
            "image": name,
            "reference": "AECC1.jpg",
            "source_shape": list(image.shape[:2]),
            "reference_shape": list(reference.shape[:2]),
            "direction": "source PCB to reference PCB",
            "maximum_features": 5000,
            "detector_maximum_side": 1000,
            "match_filter": "ratio test and mutual nearest neighbour",
            "shown_matches": min(80, record["inlier_count"]),
        })
        stem = Path(name).stem
        (output_dir / f"{stem}_homography.json").write_text(json.dumps(record, indent=2) + "\n", encoding="utf-8")
        view = draw_pcb_matches(image, reference, keypoints, reference_features[0], matches, record["inlier_mask"], f"{name} to AECC1.jpg")
        cv.imwrite(str(output_dir / f"{stem}_matches.jpg"), view, [cv.IMWRITE_JPEG_QUALITY, 90])
        summary.append({key: record[key] for key in ("image", "source_keypoints", "reference_keypoints", "match_count", "inlier_count", "inlier_rmse_pixels")})
        print(f"{name}: {record['match_count']} filtered matches and {record['inlier_count']} RANSAC inliers", flush=True)
    (output_dir / "matches_summary.json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    return summary


def warp_and_difference(image, reference, homography):
    homography = np.asarray(homography, dtype=np.float64)
    if homography.shape != (3, 3) or not np.isfinite(homography).all() or np.linalg.matrix_rank(homography) != 3:
        raise ValueError("A finite homography of rank three is required")
    size = (reference.shape[1], reference.shape[0])
    warped = cv.warpPerspective(image, homography, size, flags=cv.INTER_LINEAR)
    support = cv.warpPerspective(np.ones(image.shape[:2], np.float32), homography, size, flags=cv.INTER_LINEAR)
    valid = support >= 1 - 1e-6
    if not valid.any():
        raise ValueError("The warped image has no valid overlap with the reference")
    warped_gray = cv.cvtColor(warped, cv.COLOR_BGR2GRAY)
    reference_gray = cv.cvtColor(reference, cv.COLOR_BGR2GRAY)
    difference = cv.absdiff(warped_gray, reference_gray)
    difference[~valid] = 0
    return warped, difference, valid


def save_alignment_view(reference, warped, difference, valid, name, path):
    fig, axes = plt.subplots(1, 3, figsize=(12, 5), layout="constrained")
    axes[0].imshow(cv.cvtColor(reference, cv.COLOR_BGR2RGB))
    axes[0].set_title("Reference AECC1")
    axes[1].imshow(cv.cvtColor(warped, cv.COLOR_BGR2RGB))
    axes[1].set_title(f"Aligned {name}")
    colormap = plt.get_cmap("gray").with_extremes(bad="lightgray")
    view = axes[2].imshow(np.ma.array(difference, mask=~valid), cmap=colormap, vmin=0, vmax=255)
    axes[2].set_title("Absolute gray difference")
    fig.colorbar(view, ax=axes[2], shrink=0.6, ticks=[0, 128, 255])
    for axis in axes:
        axis.axis("off")
    fig.supxlabel("Gray areas in the difference image have no source image coverage", fontsize=10)
    fig.savefig(path, dpi=110, pil_kwargs={"quality": 88})
    plt.close(fig)


def run_alignment():
    cv.setNumThreads(2)
    image_dir = ROOT / "images/pcbs"
    output_dir = ROOT / "results/q4"
    reference = cv.imread(str(image_dir / "AECC1.jpg"))
    if reference is None:
        raise FileNotFoundError("AECC1.jpg is required as the reference image")
    if any(not (output_dir / f"{Path(name).stem}_homography.json").exists() for name in PCB_NAMES):
        run_matching()
    for name in PCB_NAMES:
        image = cv.imread(str(image_dir / name))
        if image is None:
            raise FileNotFoundError(name)
        stem = Path(name).stem
        record = json.loads((output_dir / f"{stem}_homography.json").read_text())
        warped, difference, valid = warp_and_difference(image, reference, record["homography"])
        cv.imwrite(str(output_dir / f"{stem}_warped.png"), warped)
        cv.imwrite(str(output_dir / f"{stem}_difference.png"), difference)
        cv.imwrite(str(output_dir / f"{stem}_valid_mask.png"), valid.astype(np.uint8) * 255)
        save_alignment_view(reference, warped, difference, valid, stem, output_dir / f"{stem}_alignment.jpg")
    return PCB_NAMES.copy()


if __name__ == "__main__":
    run_matching()
    run_alignment()
