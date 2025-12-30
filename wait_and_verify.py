"""
Wait for training to complete and then run verification and export
"""
import time
import subprocess
from pathlib import Path
import sys

def wait_for_training_completion(exp_dir, max_wait_minutes=120, check_interval=60):
    """Wait for training to complete by checking for final reconstruction directory"""
    print(f"Waiting for training to complete...")
    print(f"Checking every {check_interval} seconds")
    print(f"Maximum wait time: {max_wait_minutes} minutes")
    
    exp_path = Path(exp_dir)
    final_recon_dir = exp_path / "recon_final"
    log_file = exp_path / "log.txt"
    
    start_time = time.time()
    max_wait_seconds = max_wait_minutes * 60
    
    while True:
        elapsed = time.time() - start_time
        
        # Check if final reconstruction exists
        if final_recon_dir.exists() and any(final_recon_dir.glob("*.png")):
            print(f"\n✅ Training completed! Found final reconstructions.")
            return True
        
        # Check if log file exists and has recent activity
        if log_file.exists():
            # Check last modified time
            log_mtime = log_file.stat().st_mtime
            time_since_update = time.time() - log_mtime
            
            if time_since_update > 300:  # 5 minutes of no updates
                print(f"\n⚠️  Log file hasn't been updated in {time_since_update/60:.1f} minutes")
                print(f"   Training may have completed or stopped.")
                # Check if we have any reconstructions
                recon_dirs = list(exp_path.glob("recon_step_*"))
                if recon_dirs:
                    print(f"   Found {len(recon_dirs)} intermediate reconstruction checkpoints")
                    return True
                return False
        
        if elapsed > max_wait_seconds:
            print(f"\n⏱️  Maximum wait time exceeded ({max_wait_minutes} minutes)")
            return False
        
        elapsed_min = int(elapsed / 60)
        print(f"   Waiting... ({elapsed_min} minutes elapsed)", end='\r')
        time.sleep(check_interval)

def main():
    if len(sys.argv) < 2:
        # Find latest experiment
        base_dir = Path("results/train_output/DR")
        if not base_dir.exists():
            print("❌ No experiment directory found")
            return 1
        
        experiments = sorted([d for d in base_dir.iterdir() if d.is_dir()], 
                           key=lambda x: x.stat().st_mtime, reverse=True)
        if not experiments:
            print("❌ No experiments found")
            return 1
        
        exp_dir = experiments[0]
        print(f"Using latest experiment: {exp_dir.name}")
    else:
        exp_dir = Path(sys.argv[1])
    
    if not exp_dir.exists():
        print(f"❌ Experiment directory not found: {exp_dir}")
        return 1
    
    # Wait for training
    completed = wait_for_training_completion(exp_dir, max_wait_minutes=120)
    
    if not completed:
        print("\n⚠️  Training may still be running or may have stopped.")
        print("   Proceeding with verification anyway...")
    
    # Run verification
    print(f"\n{'='*60}")
    print("RUNNING VERIFICATION AND EXPORT")
    print(f"{'='*60}")
    
    verify_cmd = f'python verify_training.py "{exp_dir}"'
    result = subprocess.run(verify_cmd, shell=True)
    
    if result.returncode == 0:
        print("\n✅ Verification completed successfully!")
        print(f"\n📦 Client-ready outputs:")
        print(f"   - Model weights: {exp_dir / 'exported_weights'}")
        print(f"   - Client images: {exp_dir / 'client_images'}")
        print(f"   - Final reconstructions: {exp_dir / 'recon_final'}")
    else:
        print("\n⚠️  Verification had some issues. Check output above.")
    
    return result.returncode

if __name__ == "__main__":
    exit(main())

