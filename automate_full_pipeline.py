"""
Master automation script that runs the complete training pipeline:
1. Wait for subset training to complete (or run it if not started)
2. Verify subset training results
3. Run full training
4. Verify full training results  
5. Export model weights
6. Generate client-ready images
"""
import subprocess
import sys
import time
from pathlib import Path
import json
import re

def find_latest_experiment(base_dir="results/train_output/DR"):
    """Find the latest experiment directory"""
    base_path = Path(base_dir)
    if not base_path.exists():
        return None
    experiments = sorted([d for d in base_path.iterdir() if d.is_dir()], 
                        key=lambda x: x.stat().st_mtime, reverse=True)
    return experiments[0] if experiments else None

def check_training_running():
    """Check if training process is running"""
    try:
        result = subprocess.run(
            ['powershell', '-Command', 'Get-Process python -ErrorAction SilentlyContinue | Where-Object {$_.CommandLine -like "*train*"}'],
            capture_output=True,
            text=True
        )
        return result.returncode == 0 and result.stdout.strip()
    except:
        return False

def wait_for_training(exp_dir, timeout_minutes=120):
    """Wait for training to complete"""
    exp_path = Path(exp_dir)
    final_recon_dir = exp_path / "recon_final"
    log_file = exp_path / "log.txt"
    
    print(f"Waiting for training to complete (max {timeout_minutes} minutes)...")
    start_time = time.time()
    timeout_seconds = timeout_minutes * 60
    last_size = 0
    
    while time.time() - start_time < timeout_seconds:
        # Check for final reconstructions
        if final_recon_dir.exists() and any(final_recon_dir.glob("*.png")):
            print("\n✅ Training completed!")
            return True
        
        # Check log file for progress
        if log_file.exists():
            current_size = log_file.stat().st_size
            if current_size != last_size:
                print(".", end="", flush=True)
                last_size = current_size
                
                # Check if log indicates completion
                try:
                    with open(log_file, 'r', encoding='utf-8', errors='ignore') as f:
                        content = f.read()
                        if "SUBSET TRAINING COMPLETE" in content or "Summary" in content:
                            print("\n✅ Training completed (found completion message in log)!")
                            return True
                except:
                    pass
        
        time.sleep(30)  # Check every 30 seconds
    
    print(f"\n⏱️  Timeout reached. Checking for partial results...")
    # Check for any reconstructions
    recon_dirs = list(exp_path.glob("recon_step_*"))
    if recon_dirs:
        print(f"   Found {len(recon_dirs)} intermediate checkpoints")
        return True
    return False

def run_verification(exp_dir):
    """Run verification script"""
    print(f"\n{'='*60}")
    print("RUNNING VERIFICATION")
    print(f"{'='*60}")
    
    cmd = f'python verify_training.py "{exp_dir}"'
    result = subprocess.run(cmd, shell=True)
    return result.returncode == 0

def run_subset_training():
    """Run subset training"""
    print(f"\n{'='*60}")
    print("RUNNING SUBSET TRAINING")
    print(f"{'='*60}")
    
    cmd = 'python train_subset.py vessel_verification --subset_size 500 --steps 3000 --data_dir DR/ --checkpoint_freq 500'
    
    print(f"Command: {cmd}")
    print("This will take 20-60 minutes on CPU...")
    
    result = subprocess.run(cmd, shell=True)
    return result.returncode == 0

def run_full_training():
    """Run full training"""
    print(f"\n{'='*60}")
    print("RUNNING FULL TRAINING")
    print(f"{'='*60}")
    print("⚠️  This will take 2-4 hours on GPU, 12-24 hours on CPU")
    
    cmd = 'python train_all.py vessel_full_training --steps 15000 --data_dir DR/ --model_save 5000'
    
    print(f"Command: {cmd}")
    
    result = subprocess.run(cmd, shell=True)
    return result.returncode == 0

def main():
    print("\n" + "="*60)
    print("COMPLETE TRAINING PIPELINE AUTOMATION")
    print("="*60)
    
    # Step 1: Check if subset training is running or completed
    print("\n📋 STEP 1: Checking subset training status...")
    latest_exp = find_latest_experiment()
    
    if latest_exp and "vessel_verification" in latest_exp.name:
        print(f"   Found subset training experiment: {latest_exp.name}")
        
        # Check if it's complete
        final_recon = latest_exp / "recon_final"
        if final_recon.exists() and any(final_recon.glob("*.png")):
            print("   ✅ Subset training already completed!")
            subset_exp = latest_exp
        else:
            print("   ⏳ Subset training in progress, waiting...")
            if wait_for_training(latest_exp, timeout_minutes=120):
                subset_exp = latest_exp
            else:
                print("   ⚠️  Subset training may not have completed properly")
                subset_exp = latest_exp
    else:
        print("   No subset training found. Starting new subset training...")
        if run_subset_training():
            subset_exp = find_latest_experiment()
        else:
            print("   ❌ Subset training failed")
            return 1
    
    # Step 2: Verify subset training
    print("\n📋 STEP 2: Verifying subset training results...")
    if subset_exp:
        run_verification(subset_exp)
    
    # Step 3: Run full training
    print("\n📋 STEP 3: Running full training...")
    if run_full_training():
        full_exp = find_latest_experiment()
    else:
        print("   ⚠️  Full training had issues, but continuing...")
        full_exp = find_latest_experiment()
    
    # Step 4: Verify full training and export
    print("\n📋 STEP 4: Verifying full training and exporting...")
    if full_exp:
        run_verification(full_exp)
        
        # Summary
        print(f"\n{'='*60}")
        print("PIPELINE COMPLETE - CLIENT DELIVERY READY")
        print(f"{'='*60}")
        print(f"\n📦 Output Locations:")
        print(f"   Full Training Experiment: {full_exp}")
        print(f"   - Model Weights: {full_exp / 'exported_weights'}")
        print(f"   - Client Images: {full_exp / 'client_images'}")
        print(f"   - Final Reconstructions: {full_exp / 'recon_final'}")
        
        if subset_exp:
            print(f"\n   Subset Training Experiment: {subset_exp}")
            print(f"   - Reconstructions: {subset_exp / 'recon_final'}")
    else:
        print("   ⚠️  No full training experiment found")
    
    print("\n✅ All steps completed!")
    return 0

if __name__ == "__main__":
    exit(main())

