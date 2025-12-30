# VAE Vessel Reconstruction Training

## Quick Start

### Subset Training (Quick Test)
```bash
python train_subset.py test --subset_size 500 --steps 3000 --data_dir DR/
```

### Full Training
```bash
python train_all.py training --steps 15000 --data_dir DR/
```

## Generated Images Location

After training completes, images will be in:
```
results/train_output/DR/<experiment_name>/client_images/
```

Run this to check status and generate images:
```bash
python check_and_generate_images.py
```

## Important Files

- `train_subset.py` - Subset training script
- `train_all.py` - Full training script
- `verify_training.py` - Verification and export script
- `generate_client_images.py` - Generate client-ready images
- `check_and_generate_images.py` - Quick status check

## Configuration

Edit `config.yaml` to adjust training parameters:
- `loss_multiplier_kl`: KL divergence weight (default: 0.00005)
- `kl_anneal_end`: KL annealing end step (default: 8000)
- `loss_multiplier_y`: Classification loss weight (default: 0.05)

