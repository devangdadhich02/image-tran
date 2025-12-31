# 📸 How to Check Training and View Images

## Quick Check

Run this to see latest training progress and image locations:

```powershell
python check_training_and_show_images.py
```

## Manual Check

1. **Find Latest Training Folder:**
   ```
   results/train_output/DR/paper_architecture_test/
   ```

2. **Check Intermediate Images (during training):**
   ```
   results/train_output/DR/paper_architecture_test/recon_step_500/
   results/train_output/DR/paper_architecture_test/recon_step_1000/
   results/train_output/DR/paper_architecture_test/recon_step_1500/
   ... etc
   ```

3. **Check Final Images (after training completes):**
   ```
   results/train_output/DR/paper_architecture_test/recon_final/
   ```

## Image Files

Each folder contains:
- `*_orig.png` - Original input image
- `*_recon_normalised.png` - **This is the reconstructed image (use this one!)**
- `*_recon_unchanged.png` - Raw reconstruction (before normalization)

## Training Status

Training is running in background. Check progress:
- Look for `recon_step_*` folders (appears every 500 steps)
- Training will create `recon_final/` when complete (after 5000 steps)

## Share Images with Client

1. Open the `recon_final/` folder
2. Find `*_recon_normalised.png` files
3. These are the final reconstructed images
4. Share these with the client

---

**Current Training:** `paper_architecture_test` (5000 steps, 3-layer decoder)

