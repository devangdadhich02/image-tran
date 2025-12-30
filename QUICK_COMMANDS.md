# Quick Command Reference

## Common Commands

### Subset Training (Fast Testing)
```bash
python train_subset.py quick_test --subset_size 500 --steps 2000 --data_dir DR2/
```

### Full Training
```bash
python train_all.py full_training --data_dir DR2/ --steps 10000
```

## Common Arguments

| Argument | Description | Example |
|----------|-------------|---------|
| `--subset_size` | Number of samples per environment | `--subset_size 500` |
| `--steps` | Number of training steps | `--steps 2000` |
| `--data_dir` | Path to data directory | `--data_dir DR2/` |
| `--debug` | Enable debug mode | `--debug` |
| `--help` | Show all options | `--help` |

## Common Mistakes

❌ **Wrong**: `python train_subset.py test --subse 500`  
✅ **Right**: `python train_subset.py test --subset_size 500`

❌ **Wrong**: `python train_subset.py test --subset-size 500`  
✅ **Right**: `python train_subset.py test --subset_size 500` (use underscore, not hyphen)

## Get Help

```bash
# Show all available options
python train_subset.py --help
python train_all.py --help
```

