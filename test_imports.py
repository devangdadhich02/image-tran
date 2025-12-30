"""Quick test to verify all imports and basic functionality work"""

import sys

def test_import(module_name, import_statement, install_cmd=None):
    try:
        exec(import_statement)
        print(f"[OK] {module_name} imported")
        return True
    except ImportError as e:
        print(f"[FAIL] {module_name} import failed: {e}")
        if install_cmd:
            print(f"  Install with: {install_cmd}")
        return False
    except Exception as e:
        print(f"[ERROR] {module_name} failed: {e}")
        return False

print("Testing imports...")
print("=" * 50)

results = []
results.append(test_import("numpy", "import numpy as np", "pip install numpy"))
results.append(test_import("torch", "import torch", "pip install torch"))
results.append(test_import("PIL", "from PIL import Image", "pip install Pillow"))
results.append(test_import("sconf", "from sconf import Config", "pip install sconf"))
results.append(test_import("ResNet_VAE", "from domainbed.algorithms.vae_dg import ResNet_VAE"))
results.append(test_import("VAE_DG", "from domainbed.algorithms.algorithms import VAE_DG"))
results.append(test_import("get_dataset", "from domainbed.datasets import get_dataset"))

print("=" * 50)
if all(results):
    print("\n[SUCCESS] All critical imports successful!")
    sys.exit(0)
else:
    print("\n[WARNING] Some imports failed. Install missing dependencies:")
    print("  pip install -r requirements.txt")
    sys.exit(1)

