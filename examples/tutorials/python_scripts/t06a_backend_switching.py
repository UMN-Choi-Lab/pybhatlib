"""Tutorial T06a: Backend Switching (NumPy vs PyTorch).

pybhatlib uses a backend abstraction that lets the same code run on
NumPy (CPU) or PyTorch (CPU/GPU). All numerical functions accept an
optional `xp` parameter to select the backend.

What you will learn:
  - get_backend: obtain the current backend module
  - set_backend: change the global default
  - array_namespace: auto-detect backend from array type
  - The xp parameter pattern used throughout pybhatlib
  - When PyTorch is useful (autograd, GPU)

Prerequisites: None.
"""
import os, sys
import numpy as np
np.set_printoptions(precision=4, suppress=True)
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", "..", "src"))

from pybhatlib.backend import get_backend, set_backend, array_namespace
from pybhatlib.vecup import vecdup, matdupfull

# ============================================================
#  Step 1: Default backend
# ============================================================
print("=" * 60)
print("  Step 1: Default Backend is NumPy")
print("=" * 60)

xp = get_backend()
print(f"\n  get_backend() -> {type(xp).__name__}")
print(f"  This is the standard NumPy module.")

# ============================================================
#  Step 2: The xp pattern
# ============================================================
print("\n" + "=" * 60)
print("  Step 2: The xp Parameter Pattern")
print("=" * 60)

A = np.array([[1, 2, 3], [2, 4, 5], [3, 5, 6]], dtype=float)

# Explicit xp=numpy
v = vecdup(A, xp=xp)
print(f"\n  vecdup(A, xp=numpy) = {v}")
print(f"  Type: {type(v)}")

# Without xp (uses default)
v2 = vecdup(A)
print(f"\n  vecdup(A) = {v2}  (same result, default backend)")

# ============================================================
#  Step 3: array_namespace — auto-detect
# ============================================================
print("\n" + "=" * 60)
print("  Step 3: array_namespace — Auto-Detection")
print("=" * 60)

arr_np = np.array([1.0, 2.0, 3.0])
detected = array_namespace(arr_np)
print(f"\n  Input: numpy array")
print(f"  array_namespace(arr) -> {type(detected).__name__}")

# ============================================================
#  Step 4: PyTorch backend (if available)
# ============================================================
print("\n" + "=" * 60)
print("  Step 4: PyTorch Backend")
print("=" * 60)

try:
    import torch
    torch_available = True
except ImportError:
    torch_available = False

if torch_available:
    xp_torch = get_backend("torch")
    print(f"\n  PyTorch is available: {torch.__version__}")
    print(f"  get_backend('torch') -> {type(xp_torch).__name__}")

    # Same computation on both backends
    A_np = np.array([[4, 2], [2, 5]], dtype=np.float64)
    A_torch = torch.tensor([[4, 2], [2, 5]], dtype=torch.float64)

    v_np = vecdup(A_np, xp=get_backend("numpy"))
    v_torch = vecdup(A_torch, xp=xp_torch)

    print(f"\n  NumPy result:   {v_np}")
    print(f"  PyTorch result: {v_torch.numpy()}")
    print(f"  Match: {np.allclose(v_np, v_torch.numpy())}")

    # Roundtrip
    M_np = matdupfull(v_np, xp=get_backend("numpy"))
    M_torch = matdupfull(v_torch, xp=xp_torch)
    print(f"\n  NumPy matdupfull:\n{M_np}")
    print(f"  PyTorch matdupfull:\n{M_torch.numpy()}")
    print(f"  Match: {np.allclose(M_np, M_torch.numpy())}")

    # Auto-detect
    detected_torch = array_namespace(A_torch)
    print(f"\n  array_namespace(torch tensor) -> {type(detected_torch).__name__}")
else:
    print(f"\n  PyTorch is not installed.")
    print(f"  Install with: pip install pybhatlib[torch]")
    print(f"  All tutorials work fine with NumPy only.")

# ============================================================
#  Step 5: When to use PyTorch
# ============================================================
print("\n" + "=" * 60)
print("  Step 5: When PyTorch Is Useful")
print("=" * 60)

print("""
  NumPy (default):
  - CPU-only, always available
  - All pybhatlib features supported
  - Best for most use cases

  PyTorch:
  - Automatic differentiation (autograd) for gradient computation
  - GPU acceleration for large-scale problems
  - Useful for:
    * Very large datasets where GPU parallelism helps
    * Research on new gradient methods
    * Integration with PyTorch-based ML pipelines

  The xp pattern means you write code once and it works on both:

    def my_function(A, xp=None):
        if xp is None:
            xp = get_backend()
        # Use xp.array(), xp.zeros(), etc.
        return xp.sum(A)
""")

print(f"  Next: t06b_custom_specs.py — Custom model specifications")
