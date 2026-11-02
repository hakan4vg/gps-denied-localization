import os

# Must be set before cv2 or numpy load OpenBLAS; see gpsloc/__init__.py.
os.environ.setdefault("OPENBLAS_CORETYPE", "Haswell")
