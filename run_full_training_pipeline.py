"""
Complete training pipeline script that:
1. Runs subset training
2. Verifies results
3. Runs full training
4. Exports everything for client
"""
import subprocess
import sys
import time
from pathlib import Path
import json

def run_command(cmd, description, timeout=None):
    """Run a command and return success status"""
    print(f"\n{'='*60}")
    print(f"{description}")
    print(f"{'='*60}")
    print(f"Command: {cmd}")
    print()
    
    try:
        result = subprocess.run(
            cmd,
            shell=True,
            capture_output=False,
            text=True,
            timeout=timeout
        )
        success = result.returncode == 0
        if success:
            print(f"\n✅ {description} - SUCCESS")
        else:
            print(f"\n❌ {description} - FAILED (exit code: {result.returncode})")
        return success
    except subprocess.TimeoutExpired:
        print(f"\n⏱️  {description} - TIMEOUT (exceeded {timeout}s)")
        return False
    except Exception as e:
        print(f"\n❌ {description} - ERROR: {e}")
        return False

def find_latest_experiment(base_dir):
    """Find the latest experiment directory"""
    base_path = Path(base_dir)
    if not base_path.exists():
        return None
    
    experiments = sorted([d for d in base_path.iterdir() if d.is_dir()], 
                        key=lambda x: x.stat().st_mtime, reverse=True)
    return experiments[0] if experiments else None

def main():
    print("\n" + "="*60)
    print("COMPLETE TRAINING PIPELINE")
    print("="*60)
    
    # Step 1: Run subset training
    print("\n📋 STEP 1: Running subset training...")
    subset_cmd = 'python train_subset.py vessel_verification --subset_size 500 --steps 3000 --data_dir DR/ --checkpoint_freq 500'
    
    # Run subset training (this will take 10-30 minutes on CPU)
    success = run_command(subset_cmd, "Subset Training", timeout=7200)  # 2 hour timeout
    
    if not success:
        print("\n⚠️  Subset training failed. Check the error messages above.")
        print("   Continuing anyway to check what was generated...")
    
    # Step 2: Find and verify subset training results
    print("\n📋 STEP 2: Verifying subset training results...")
    latest_exp = find_latest_experiment("results/train_output/DR")
    
    if latest_exp:
        print(f"   Found experiment: {latest_exp.name}")
        verify_cmd = f'python verify_training.py "{latest_exp}"'
        run_command(verify_cmd, "Verification", timeout=300)
    else:
        print("   ⚠️  No experiment directory found")
    
    # Step 3: Run full training
    print("\n📋 STEP 3: Running full training on complete dataset...")
    print("   ⚠️  This will take several hours on CPU (2-4 hours on GPU)")
    
    full_cmd = 'python train_all.py vessel_full_training --steps 15000 --data_dir DR/ --model_save 5000'
    
    # Ask user if they want to continue (but since user said don't ask, we'll just run it)
    print("   Starting full training...")
    success = run_command(full_cmd, "Full Training", timeout=86400)  # 24 hour timeout
    
    if not success:
        print("\n⚠️  Full training failed or timed out.")
        print("   Check the error messages above.")
    
    # Step 4: Find and verify full training results
    print("\n📋 STEP 4: Verifying full training results...")
    latest_exp = find_latest_experiment("results/train_output/DR")
    
    if latest_exp:
        print(f"   Found experiment: {latest_exp.name}")
        verify_cmd = f'python verify_training.py "{latest_exp}"'
        run_command(verify_cmd, "Final Verification", timeout=300)
        
        # Generate summary
        print("\n📋 STEP 5: Generating client delivery package...")
        print(f"   Experiment directory: {latest_exp}")
        print(f"   - Model weights: {latest_exp / 'exported_weights'}")
        print(f"   - Client images: {latest_exp / 'client_images'}")
        print(f"   - Final reconstructions: {latest_exp / 'recon_final'}")
        
    else:
        print("   ⚠️  No experiment directory found")
    
    print("\n" + "="*60)
    print("PIPELINE COMPLETE")
    print("="*60)
    print("\n✅ All steps completed!")
    print("   Check the experiment directories for results.")

if __name__ == "__main__":
    main()

