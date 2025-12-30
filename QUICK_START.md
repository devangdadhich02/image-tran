# Quick Start Guide - β-VAE Fixes

## ⚠️ First: Install Dependencies

**If you see `ModuleNotFoundError`**, install dependencies first:

```bash
# Windows
pip install -r requirements.txt

# Or use the setup script
setup.bat
```

Then verify:
```bash
python test_imports.py
```

See `INSTALL.md` for detailed installation instructions.

## What Was Fixed

✅ **KL Annealing** - Gradually introduces KL regularization to prevent posterior collapse  
✅ **Reconstruction Loss** - Fixed normalization handling and loss scaling  
✅ **Training Loop** - Properly passes step information for annealing  
✅ **Subset Training** - New script for fast iteration  
✅ **Configuration** - Updated with proper hyperparameters  
✅ **CPU Support** - Now works on CPU (no CUDA required, though GPU is faster)  

## Quick Test (5 minutes)

```bash
# Test on a small subset (500 images, 2000 steps)
python train_subset.py quick_test --subset_size 500 --steps 2000 --data_dir DR2/
```

This will:
- Train on 500 images per environment
- Complete in ~5-10 minutes (GPU) or ~30-60 minutes (CPU)
- Save reconstructions to `results/train_output/DR/.../recon_final/`
- Show if the fixes are working
- **Note**: Works on CPU too (just slower). GPU not required.

## Full Training

```bash
# Full training on complete dataset
python train_all.py full_training --data_dir DR2/ --steps 10000
```

## Check Results

After training, check reconstruction quality:
```bash
# Windows
dir results\train_output\DR\<experiment_name>\recon_final

# Linux/Mac
ls results/train_output/DR/<experiment_name>/recon_final
```

Look for:
- ✅ Fine vessel structures visible
- ✅ Clear optic disc
- ✅ Good color matching
- ✅ Not blurry

## Key Files

- `ALL_FIXES_SUMMARY.md` - Complete summary of all fixes
- `FIXES_AND_USAGE.md` - Detailed explanation of all fixes
- `CPU_SUPPORT.md` - CPU/GPU support details
- `train_subset.py` - Fast subset training script
- `config.yaml` - Updated hyperparameters
- `domainbed/algorithms/vae_dg.py` - Main VAE with KL annealing

## Troubleshooting

**Reconstructions too blurry?**
```bash
# Increase KL weight
python train_subset.py test --loss_multiplier_kl=0.0005
```

**Not converging?**
```bash
# Try longer annealing
python train_subset.py test --kl_anneal_end=3000
```

See `FIXES_AND_USAGE.md` for detailed troubleshooting.

