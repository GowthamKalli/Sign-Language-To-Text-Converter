# diag_torchlight.py
import torchlight, sys, inspect
print("torchlight:", torchlight)
print("torchlight file:", getattr(torchlight, "__file__", None))
print("has import_class:", hasattr(torchlight, "import_class"))
print("sys.path[:5]:", sys.path[:5])
