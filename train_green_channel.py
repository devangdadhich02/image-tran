"""
Train with Green Channel Only (Better Vessel Contrast)
This script trains using only the Green channel from RGB images.
Green channel has the highest vessel contrast in retinal images.
"""
import sys
from pathlib import Path

# Modify transforms to use Green channel
import domainbed.datasets.transforms as DBT
DBT.basic = DBT.green_channel_basic
DBT.aug = DBT.green_channel_aug

# Now run normal training
if __name__ == "__main__":
    # Import after modifying transforms
    from train_subset import main
    
    print("=" * 70)
    print("GREEN CHANNEL TRAINING MODE")
    print("=" * 70)
    print("Using only Green channel (best vessel contrast)")
    print("=" * 70)
    print()
    
    # Run training
    main()

