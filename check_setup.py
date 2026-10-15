from pathlib import Path

import cv2 as cv
import matplotlib
import numpy as np
import scipy


ROOT = Path(__file__).resolve().parent
IMAGE_PATHS = (
    "the_berry_farms_sunflower_field.jpeg",
    "flag.png",
    "wadham.jpg",
    "pcbs/AAAA1.jpg",
    "pcbs/ADAD1.jpg",
    "pcbs/AECC1.jpg",
    "pcbs/AECC2.jpg",
    "pcbs/AECC3.jpg",
    "pcbs/AECC4.jpg",
)


def main():
    print(f"NumPy: {np.__version__}")
    print(f"SciPy: {scipy.__version__}")
    print(f"OpenCV: {cv.__version__}")
    print(f"Matplotlib: {matplotlib.__version__}")
    cv.SIFT_create()
    cv.BFMatcher(cv.NORM_L2)

    for relative_path in IMAGE_PATHS:
        path = ROOT / "images" / relative_path
        image = cv.imread(str(path))
        if image is None:
            raise FileNotFoundError(f"Cannot load image: {relative_path}")
        height, width = image.shape[:2]
        print(f"{relative_path}: {width} x {height}")

    sunflower = cv.imread(
        str(ROOT / "images" / IMAGE_PATHS[0]), cv.IMREAD_REDUCED_COLOR_4
    )
    if sunflower is None:
        raise RuntimeError("Cannot load the reduced sunflower image")
    print(f"Reduced sunflower: {sunflower.shape[1]} x {sunflower.shape[0]}")
    print("All assignment inputs and required OpenCV features are available.")


if __name__ == "__main__":
    main()
