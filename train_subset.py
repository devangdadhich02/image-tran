"""
Subset Training Script for β-VAE
==================================
This script trains the VAE on a small subset of data for fast iteration and debugging.
Useful for:
- Quick testing of hyperparameter changes
- Debugging training issues
- Verifying reconstruction quality improvements

Usage:
    python train_subset.py experiment_name --subset_size 500 --steps 1000
"""

import argparse
import collections
import random
import sys
from pathlib import Path

import numpy as np
import PIL
import torch
import torch.utils.data
import torchvision
from sconf import Config
from prettytable import PrettyTable

from domainbed.datasets import get_dataset
from domainbed import hparams_registry
from domainbed.lib import misc
from domainbed.lib.writers import get_writer
from domainbed.lib.logger import Logger
from domainbed.trainer import train


def create_subset_dataset(dataset, subset_size_per_env=500, seed=0):
    """
    Create a subset of the dataset for fast training.
    
    Args:
        dataset: Original dataset
        subset_size_per_env: Number of samples per environment to keep
        seed: Random seed for reproducibility
    
    Returns:
        Modified dataset with subset of data
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    
    # Create subset for each environment
    for env_idx, env_dataset in enumerate(dataset.datasets):
        total_size = len(env_dataset)
        subset_size = min(subset_size_per_env, total_size)
        
        # Randomly sample indices
        indices = list(range(total_size))
        random.shuffle(indices)
        subset_indices = indices[:subset_size]
        
        # Create subset
        subset = torch.utils.data.Subset(env_dataset, subset_indices)
        dataset.datasets[env_idx] = subset
        
        print(f"Environment {env_idx} ({dataset.environments[env_idx]}): "
              f"{total_size} -> {subset_size} samples")
    
    return dataset


def main():
    parser = argparse.ArgumentParser(
        description="DR Domain generalization - Subset Training", 
        allow_abbrev=False
    )
    parser.add_argument("name", type=str, help="Experiment name")
    parser.add_argument("configs", nargs="*", help="Config files")
    parser.add_argument("--data_dir", type=str, default="DR/", help="Data directory (use DR/ or DR2/)")
    parser.add_argument("--dataset", type=str, default="DR", help="Dataset name")
    parser.add_argument("--algorithm", type=str, default="VAE_DG", help="Algorithm")
    parser.add_argument(
        "--trial_seed",
        type=int,
        default=0,
        help="Trial number (used for seeding split_dataset and random_hparams).",
    )
    parser.add_argument("--seed", type=int, default=0, help="Seed for everything else")
    parser.add_argument(
        "--steps", type=int, default=2000, help="Number of training steps"
    )
    parser.add_argument(
        "--subset_size",
        type=int,
        default=500,
        help="Number of samples per environment to use (for fast iteration)"
    )
    parser.add_argument(
        "--checkpoint_freq",
        type=int,
        default=200,
        help="Checkpoint every N steps"
    )
    parser.add_argument("--test_envs", type=int, nargs="+", default=None)
    parser.add_argument("--holdout_fraction", type=float, default=0.2)
    parser.add_argument("--model_save", default=None, type=int, help="Model save start step")
    parser.add_argument("--tb_freq", default=10)
    parser.add_argument("--debug", action="store_true", help="Run w/ debug mode")
    parser.add_argument("--show", action="store_true", help="Show args and hparams w/o run")
    parser.add_argument(
        "--evalmode",
        default="fast",
        help="[fast, all]. if fast, ignore train_in datasets in evaluation time.",
    )
    parser.add_argument("--prebuild_loader", action="store_true", help="Pre-build eval loaders")
    
    args, left_argv = parser.parse_known_args()
    args.deterministic = True

    # Check for incomplete arguments (common mistake: --subse instead of --subset_size)
    for arg in left_argv:
        if arg.startswith('--') and '=' not in arg:
            # Check if it looks incomplete (short, not a known boolean flag)
            if len(arg) < 8 and arg not in ['--debug', '--show', '--prebuild_loader']:
                # Try to find similar argument
                all_args = [action.option_strings[0] for action in parser._actions 
                           if action.option_strings and action.option_strings[0].startswith('--')]
                similar = [a for a in all_args if a.replace('--', '').startswith(arg.replace('--', ''))]
                if similar:
                    print(f"\n❌ Error: Incomplete argument '{arg}'")
                    print(f"💡 Did you mean: {similar[0]}?")
                    print(f"\n📖 Example usage:")
                    print(f"  python train_subset.py quick_test --subset_size 500 --steps 2000 --data_dir DR/")
                    sys.exit(1)

    # setup hparams
    hparams = hparams_registry.default_hparams(args.algorithm, args.dataset)

    keys = ["config.yaml"] + args.configs
    keys = [open(key, encoding="utf8") for key in keys]
    hparams = Config(*keys, default=hparams)
    
    # Only update with valid left_argv (filter out any remaining issues)
    try:
        hparams.argv_update(left_argv)
    except ValueError as e:
        print(f"\nError parsing arguments: {e}")
        print(f"Invalid arguments: {left_argv}")
        print(f"\nExample usage:")
        print(f"  python train_subset.py quick_test --subset_size 500 --steps 2000 --data_dir DR/")
        sys.exit(1)

    # setup debug
    if args.debug:
        args.checkpoint_freq = 5
        args.steps = 10
        args.name += "_debug"

    timestamp = misc.timestamp()
    args.unique_name = f"{timestamp}_{args.name}_subset{args.subset_size}"

    # path setup
    args.work_dir = Path("./results")
    args.data_dir = Path(args.data_dir)

    args.out_root = args.work_dir / Path("train_output") / args.dataset
    args.out_dir = args.out_root / args.unique_name
    args.out_dir.mkdir(exist_ok=True, parents=True)

    writer = get_writer(args.out_root / "runs" / args.unique_name)
    logger = Logger.get(args.out_dir / "log.txt")
    if args.debug:
        logger.setLevel("DEBUG")
    cmd = " ".join(sys.argv)
    logger.info(f"Command :: {cmd}")

    logger.nofmt("Environment:")
    logger.nofmt("\tPython: {}".format(sys.version.split(" ")[0]))
    logger.nofmt("\tPyTorch: {}".format(torch.__version__))
    logger.nofmt("\tTorchvision: {}".format(torchvision.__version__))
    logger.nofmt("\tCUDA: {}".format(torch.version.cuda))
    logger.nofmt("\tCUDNN: {}".format(torch.backends.cudnn.version()))
    logger.nofmt("\tNumPy: {}".format(np.__version__))
    logger.nofmt("\tPIL: {}".format(PIL.__version__))

    # Check CUDA availability and warn if not available
    device = "cuda" if torch.cuda.is_available() else "cpu"
    if not torch.cuda.is_available():
        logger.warning("=" * 60)
        logger.warning("WARNING: CUDA is not available. Training will use CPU.")
        logger.warning("CPU training will be MUCH slower. Consider installing CUDA-enabled PyTorch.")
        logger.warning("=" * 60)
    else:
        logger.info(f"CUDA available: Using GPU for training")

    logger.nofmt("Args:")
    for k, v in sorted(vars(args).items()):
        logger.nofmt("\t{}: {}".format(k, v))

    logger.nofmt("HParams:")
    for line in hparams.dumps().split("\n"):
        logger.nofmt("\t" + line)

    if args.show:
        exit()

    # seed
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.backends.cudnn.deterministic = args.deterministic
    torch.backends.cudnn.benchmark = not args.deterministic

    # Get dataset and create subset
    logger.info("=" * 60)
    logger.info("CREATING SUBSET DATASET FOR FAST ITERATION")
    logger.info("=" * 60)
    
    # Monkey-patch the dataset class to return a subset
    from domainbed.datasets import datasets
    original_dataset_class = vars(datasets)[args.dataset]
    
    class SubsetDatasetWrapper(original_dataset_class):
        def __init__(self, root):
            super().__init__(root)
            # Create subset after full dataset is loaded
            logger.info(f"Creating subset with size {args.subset_size} per environment...")
            create_subset_dataset(self, args.subset_size, args.seed)
            # Verify subset was created
            for env_idx, env_dataset in enumerate(self.datasets):
                actual_size = len(env_dataset)
                logger.info(f"Verified: env{env_idx} now has {actual_size} samples")
    
    # Replace the dataset class (keep it replaced for training)
    vars(datasets)[args.dataset] = SubsetDatasetWrapper
    
    # Dummy call to get dataset info (will create subset)
    temp_dataset, _in_splits, _out_splits = get_dataset([0], args, hparams)
    
    # Verify subset sizes in splits
    logger.info("Verifying subset sizes in train splits:")
    for i, (env, _) in enumerate(_in_splits):
        if i not in args.test_envs[0] if args.test_envs else []:
            logger.info(f"  Train split env{i}: {len(env)} samples")
    
    # print dataset information
    logger.nofmt("Dataset (SUBSET):")
    logger.nofmt(f"\t[{args.dataset}] #envs={len(temp_dataset)}, #classes={temp_dataset.num_classes}")
    for i, env_property in enumerate(temp_dataset.environments):
        logger.nofmt(f"\tenv{i}: {env_property} (#{len(temp_dataset[i])})")
    logger.nofmt("")

    n_steps = args.steps
    checkpoint_freq = args.checkpoint_freq or getattr(temp_dataset, 'CHECKPOINT_FREQ', 200)
    logger.info(f"n_steps = {n_steps}")
    logger.info(f"checkpoint_freq = {checkpoint_freq}")

    org_n_steps = n_steps
    n_steps = (n_steps // checkpoint_freq) * checkpoint_freq + 1
    logger.info(f"n_steps is updated to {org_n_steps} => {n_steps} for checkpointing")

    if not args.test_envs:
        args.test_envs = [[te] for te in range(len(temp_dataset))]
    logger.info(f"Target test envs = {args.test_envs}")

    ###########################################################################
    # Run
    ###########################################################################
    all_records = []
    results = collections.defaultdict(list)

    for test_env in args.test_envs:
        res, records = train(
            test_env,
            args=args,
            hparams=hparams,
            n_steps=n_steps,
            checkpoint_freq=checkpoint_freq,
            logger=logger,
            writer=writer,
        )
        all_records.append(records)
        for k, v in res.items():
            results[k].append(v)

    # log summary table
    logger.info("=== Summary ===")
    logger.info(f"Command: {' '.join(sys.argv)}")
    logger.info("Unique name: %s" % args.unique_name)
    logger.info("Out path: %s" % args.out_dir)
    logger.info("Algorithm: %s" % args.algorithm)
    logger.info("Dataset: %s" % args.dataset)

    table = PrettyTable(["Selection"] + temp_dataset.environments + ["Avg."])
    for key, row in results.items():
        row.append(np.mean(row))
        row = [f"{acc:.3%}" for acc in row]
        table.add_row([key] + row)
    logger.nofmt(table)
    
    logger.info("=" * 60)
    logger.info("SUBSET TRAINING COMPLETE")
    logger.info(f"Results saved to: {args.out_dir}")
    logger.info(f"Reconstructions saved to: {args.out_dir / 'recon_final'}")
    logger.info("=" * 60)


if __name__ == "__main__":
    main()

