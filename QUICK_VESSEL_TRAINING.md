# Quick Start - Vessel Reconstruction

## 🎯 Goal
Train VAE to reconstruct fundus images **WITH visible vessels**, not just shape.

## ⚡ Quick Commands

### Test (5-10 minutes)
```bash
python train_subset.py test --subset_size 500 --steps 3000 --data_dir DR2/
```

### Full Training (2-4 hours on GPU)
```bash
python train_all.py vessel_training --steps 15000 --data_dir DR2/
```

## ✅ What Was Fixed

1. **Decoder**: 3 layers → **8 layers** (more capacity)
2. **Loss**: BCE only → **BCE + L1** (preserves details)
3. **KL Weight**: 0.0001 → **0.00005** (less compression)
4. **Annealing**: 5000 → **8000 steps** (more learning time)

## 📊 Check Results

After training, check:
```
results/train_output/DR/<experiment_name>/recon_final/
```

**Look for**: Thin vessel lines, branching patterns, sharp edges

## 🔧 If Vessels Don't Appear

```bash
# Reduce KL weight further
python train_subset.py test --loss_multiplier_kl=0.00001 --steps 5000

# Train longer
python train_all.py training --steps 20000 --data_dir DR/
```

## 📚 Full Guide
See `VESSEL_RECONSTRUCTION_GUIDE.md` for complete details.

**Ready to train!** 🚀

