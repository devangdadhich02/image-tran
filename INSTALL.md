# Installation Guide

## Issue Resolved

The terminal error was due to **Python version incompatibility**. The original `requirements.txt` was for Python 3.7-3.10, but you're using Python 3.13.

## Quick Install

**For Python 3.13 (Current):**
```bash
# Install all dependencies from requirements.txt (updated for Python 3.13)
pip install -r requirements.txt
```

**For Python 3.10 or earlier:**
```bash
# Use the original requirements
pip install -r requirements_py310.txt
```

## Python Version Check

Check your Python version:
```bash
python --version
```

- **Python 3.13+**: Use `requirements.txt` (updated)
- **Python 3.7-3.10**: Use `requirements_py310.txt` (original)

## Verify Installation

After installing, verify everything works:

```bash
python test_imports.py
```

You should see `[SUCCESS] All critical imports successful!`

## Manual Install (if needed)

If `pip install -r requirements.txt` fails, install key packages manually:

```bash
pip install numpy==1.21.4
pip install torch==1.8.1
pip install torchvision==0.9.1
pip install Pillow==9.0.1
pip install sconf==0.2.3
pip install prettytable==2.1.0
pip install tensorboardX==2.5
```

## Common Issues

### Issue: `ModuleNotFoundError: No module named 'numpy'`
**Solution**: Run `pip install -r requirements.txt`

### Issue: CUDA/GPU not available
**Solution**: The code requires CUDA. Make sure you have:
- NVIDIA GPU with CUDA support
- CUDA toolkit installed
- PyTorch with CUDA support

To check CUDA:
```python
import torch
print(torch.cuda.is_available())  # Should be True
```

### Issue: Python version mismatch
**Solution**: The code was tested with Python 3.8-3.10. If using Python 3.13, you may need to update package versions.

## After Installation

Once dependencies are installed, you can:

1. **Test the fixes**:
   ```bash
   python train_subset.py quick_test --subset_size 500 --steps 2000 --data_dir DR2/
   ```

2. **Run full training**:
   ```bash
   python train_all.py full_training --data_dir DR2/ --steps 10000
   ```

## Next Steps

See `QUICK_START.md` for usage instructions after installation.

