# CPU Support Added

## Changes Made

The codebase has been updated to support **both CPU and GPU training**. Previously, it required CUDA and would fail if CUDA was not available.

## What Changed

### 1. Removed CUDA Requirement
- **Before**: `assert torch.cuda.is_available(), "CUDA is not available"`
- **After**: Warning message if CUDA not available, but continues with CPU

### 2. Device-Aware Code
- All `.cuda()` calls replaced with `.to(device)`
- Device automatically detected: CUDA if available, else CPU
- All tensors moved to the correct device

### 3. Files Modified
- `train_subset.py` - Removed CUDA assertion, added device detection
- `train_all.py` - Removed CUDA assertion, added device detection  
- `domainbed/trainer.py` - All CUDA calls made device-aware

## Usage

### With GPU (CUDA)
```bash
# Automatically uses GPU if available
python train_subset.py quick_test --subset_size 500 --steps 2000 --data_dir DR2/
```

### With CPU Only
```bash
# Will show warning but continue with CPU
python train_subset.py quick_test --subset_size 500 --steps 2000 --data_dir DR2/
```

**Note**: CPU training will be **much slower** than GPU training. For serious training, install CUDA-enabled PyTorch.

## Installing CUDA PyTorch (Optional)

If you want GPU support, install CUDA-enabled PyTorch:

```bash
# For CUDA 11.8
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118

# For CUDA 12.1
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121
```

Check your CUDA version:
```bash
nvidia-smi
```

## Performance

- **GPU (CUDA)**: ~10-100x faster depending on model size
- **CPU**: Works but very slow for training

For testing/debugging, CPU is fine. For actual training, GPU is recommended.

