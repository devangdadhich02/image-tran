# β-VAE Fixes and Usage Guide

## Summary of Issues Fixed

This document explains the issues found in the β-VAE implementation and how they were resolved to improve reconstruction quality, especially for fine vessel structures in fundus images.

---

## Problems Identified

### 1. **No KL Regularization**
- **Issue**: `loss_multiplier_kl` was set to `0` in `config.yaml`, meaning the KL divergence term was completely disabled.
- **Impact**: Without KL regularization, the latent space doesn't learn a proper prior distribution, leading to poor reconstruction quality and inability to capture fine details.

### 2. **No KL Annealing**
- **Issue**: Even if KL weight was non-zero, there was no annealing schedule to gradually introduce KL regularization.
- **Impact**: Starting with full KL weight can cause posterior collapse (latent space becomes uninformative) or prevent the model from learning meaningful representations early in training.

### 3. **Reconstruction Loss Issues**
- **Issue**: 
  - Used `reduction='sum'` which scales with batch size and image size, making loss magnitudes inconsistent
  - Compared reconstruction (in [0,1]) directly with normalized inputs (ImageNet normalization: mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
- **Impact**: Loss values were incorrect, leading to poor training dynamics and inability to learn fine details.

### 4. **No Subset Training Script**
- **Issue**: No way to quickly test changes on a small subset of data.
- **Impact**: Slow iteration cycle when debugging and tuning hyperparameters.

---

## Fixes Implemented

### 1. **KL Annealing Implementation** ✅
**File**: `domainbed/algorithms/vae_dg.py`

- Added `compute_kl_weight()` method that implements:
  - **Linear annealing**: Gradually increases KL weight from 0 to full weight over specified steps
  - **Cyclical annealing**: Option to repeat the annealing cycle (useful for long training)
- KL weight is computed based on current training step and applied to the KL divergence term
- Configurable via `config.yaml`:
  - `kl_anneal_start`: Step to start annealing (default: 0)
  - `kl_anneal_end`: Step to reach full weight (default: 5000)
  - `kl_anneal_cyclical`: Enable cyclical annealing (default: False)
  - `kl_anneal_cycle_length`: Cycle length for cyclical mode

**How it works**:
```python
# Early in training (step < kl_anneal_start): KL weight = 0
# During annealing (kl_anneal_start <= step < kl_anneal_end): KL weight linearly increases
# After annealing (step >= kl_anneal_end): KL weight = 1.0 (full weight)
effective_kl_weight = loss_multiplier_kl * kl_weight
```

### 2. **Fixed Reconstruction Loss** ✅
**File**: `domainbed/algorithms/vae_dg.py`

- **Proper normalization handling**: Detects if input is normalized and unnormalizes it before comparison
- **Mean reduction**: Changed from `reduction='sum'` to `reduction='mean'` for better stability
- **Correct comparison**: Reconstruction (in [0,1]) is now compared with unnormalized input (also in [0,1])

**Key changes**:
```python
# Detect normalized inputs
if x.min() < -0.5:
    # Unnormalize ImageNet-normalized inputs
    x_unnorm = (x * std + mean).clamp(0, 1)
else:
    x_unnorm = x.clamp(0, 1)

# Use mean reduction for stability
recon_loss = F.binary_cross_entropy(recon_x, x_unnorm, reduction='mean')
```

### 3. **Step Information Passing** ✅
**Files**: `domainbed/algorithms/algorithms.py`, `domainbed/trainer.py`

- Updated `VAE_DG.update()` to accept and pass `step` parameter to loss function
- Trainer already passes `step` in the inputs dictionary, so no changes needed there
- Removed debug code that was using a fixed batch (now uses normal data iterator)

### 4. **Subset Training Script** ✅
**File**: `train_subset.py`

- New script for fast iteration on a subset of data
- Randomly samples a specified number of images per environment
- Useful for:
  - Quick hyperparameter tuning
  - Debugging training issues
  - Verifying reconstruction improvements

### 5. **Updated Configuration** ✅
**File**: `config.yaml`

- Set `loss_multiplier_kl: 0.0001` (was 0)
- Set `loss_multiplier_y: 0.1` (was 0)
- Added KL annealing parameters with sensible defaults

---

## How to Use

### Quick Start: Subset Training

For fast iteration and testing:

```bash
python train_subset.py test_run --subset_size 500 --steps 2000 --data_dir DR2/
```

**Parameters**:
- `test_run`: Experiment name
- `--subset_size 500`: Use 500 samples per environment (fast)
- `--steps 2000`: Train for 2000 steps
- `--data_dir DR2/`: Path to your data directory

### Full Training

For full dataset training:

```bash
python train_all.py full_training --data_dir DR2/ --steps 10000
```

### Customizing KL Annealing

Edit `config.yaml` or pass via command line:

```bash
python train_subset.py test_run \
    --subset_size 500 \
    --steps 5000 \
    loss_multiplier_kl=0.0001 \
    kl_anneal_start=0 \
    kl_anneal_end=3000
```

**Recommended settings**:
- **Small dataset / fast training**: `kl_anneal_end = 20-30% of total steps`
- **Large dataset / full training**: `kl_anneal_end = 10-20% of total steps`
- **loss_multiplier_kl**: Start with `0.0001` and adjust based on:
  - If reconstructions are blurry: increase (e.g., `0.0005`)
  - If reconstructions lose structure: decrease (e.g., `0.00005`)

---

## Monitoring Training

### Key Metrics to Watch

1. **recon_loss**: Should decrease steadily. If it plateaus, KL weight might be too high.
2. **KLD_loss**: Should gradually increase as annealing progresses. Should stabilize after annealing completes.
3. **total_loss**: Should decrease overall.

### Reconstruction Quality

Check reconstruction images saved in:
- `results/train_output/DR/<experiment_name>/recon_step_<N>/`
- `results/train_output/DR/<experiment_name>/recon_final/`

**What to look for**:
- ✅ **Good**: Fine vessel structures visible, clear optic disc, good color matching
- ❌ **Bad**: Blurry reconstructions, missing vessels, color shifts

### Troubleshooting

**Problem**: Reconstructions are too blurry
- **Solution**: Increase `loss_multiplier_kl` (e.g., to `0.0005`) or increase `kl_anneal_end`

**Problem**: Reconstructions lose structure early
- **Solution**: Decrease `loss_multiplier_kl` (e.g., to `0.00005`) or increase `kl_anneal_start`

**Problem**: KL loss is always 0
- **Solution**: Check that `kl_anneal_start <= current_step < kl_anneal_end` or increase `loss_multiplier_kl`

**Problem**: Model not converging
- **Solution**: 
  - Try subset training first to verify fixes work
  - Adjust learning rate in config
  - Check that input normalization is correct

---

## Architecture Verification

The implementation matches the paper (https://arxiv.org/pdf/2309.11301):
- ✅ ResNet50 encoder
- ✅ Latent dimension: 256
- ✅ FC hidden dimensions: 1024, 1024
- ✅ Decoder with transposed convolutions
- ✅ Sigmoid output activation (reconstruction in [0,1])

---

## Expected Improvements

After these fixes, you should see:

1. **Better reconstruction quality**: Fine vessel structures should be visible
2. **Stable training**: Loss should decrease smoothly without sudden jumps
3. **Proper latent space**: KL divergence should gradually increase and stabilize
4. **Faster iteration**: Subset training allows quick testing of changes

---

## Files Modified

1. `domainbed/algorithms/vae_dg.py` - KL annealing, fixed reconstruction loss
2. `domainbed/algorithms/algorithms.py` - Pass step to loss function
3. `domainbed/trainer.py` - Removed fixed batch debug code
4. `config.yaml` - Updated hyperparameters and added KL annealing config
5. `train_subset.py` - New script for subset training

---

## Testing Checklist

- [ ] Run subset training and verify it completes without errors
- [ ] Check that reconstruction images show improvement over baseline
- [ ] Verify KL loss increases during annealing phase
- [ ] Check that final reconstructions show fine vessel structures
- [ ] Run full training on complete dataset
- [ ] Export final model weights

---

## Next Steps

1. **Start with subset training** to verify everything works
2. **Monitor reconstruction quality** at different steps
3. **Tune hyperparameters** based on reconstruction quality:
   - Adjust `loss_multiplier_kl` if needed
   - Adjust `kl_anneal_end` based on convergence speed
4. **Run full training** once satisfied with subset results
5. **Export model weights** from best checkpoint

---

## Questions?

If you encounter issues:
1. Check the log file in `results/train_output/DR/<experiment_name>/log.txt`
2. Verify your data directory structure matches expected format
3. Ensure CUDA is available and working
4. Try subset training first to isolate issues

Good luck with your training! 🚀

