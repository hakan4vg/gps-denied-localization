"""GPS-denied drone localization from a nadir camera image and an onboard satellite map."""

import os

# OpenCV's bundled OpenBLAS mis-detects some virtual CPUs (e.g. "QEMU Virtual CPU" on AMD hosts)
# and crashes with "Illegal instruction" in cv2.estimateAffine2D. Haswell kernels run on any
# AVX2 machine. Must be set before cv2 is first imported; an explicit setting wins.
os.environ.setdefault("OPENBLAS_CORETYPE", "Haswell")
