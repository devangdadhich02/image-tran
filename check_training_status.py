"""
Quick script to check if training is actually running and making progress
"""
import time
from pathlib import Path
import subprocess

def check_training_progress(exp_dir):
    """Check if training is making progress"""
    exp_path = Path(exp_dir)
    log_file = exp_path / "log.txt"
    
    if not log_file.exists():
        print(f"[ERROR] Log file not found: {log_file}")
        return False
    
    # Get initial size
    initial_size = log_file.stat().st_size
    initial_mtime = log_file.stat().st_mtime
    
    print(f"Checking training progress...")
    print(f"Log file: {log_file}")
    print(f"Initial size: {initial_size} bytes")
    print(f"Last modified: {time.ctime(initial_mtime)}")
    print(f"\nWaiting 60 seconds to check for updates...")
    
    time.sleep(60)
    
    # Check again
    if log_file.exists():
        current_size = log_file.stat().st_size
        current_mtime = log_file.stat().st_mtime
        
        print(f"\nAfter 60 seconds:")
        print(f"Current size: {current_size} bytes")
        print(f"Last modified: {time.ctime(current_mtime)}")
        
        if current_size > initial_size or current_mtime > initial_mtime:
            print(f"\n[OK] Training is making progress!")
            print(f"   Size increased by: {current_size - initial_size} bytes")
            return True
        else:
            print(f"\n[WARNING] Training appears to be stuck - no updates in 60 seconds")
            return False
    else:
        print(f"\n[ERROR] Log file disappeared!")
        return False

def main():
    import sys
    
    # Find latest experiment
    base_dir = Path("results/train_output/DR")
    if not base_dir.exists():
        print("[ERROR] No training directory found")
        return 1
    
    experiments = sorted([d for d in base_dir.iterdir() if d.is_dir()], 
                       key=lambda x: x.stat().st_mtime, reverse=True)
    
    if not experiments:
        print("[ERROR] No experiments found")
        return 1
    
    latest_exp = experiments[0]
    print(f"Latest experiment: {latest_exp.name}\n")
    
    is_running = check_training_progress(latest_exp)
    
    if not is_running:
        print("\n[RECOMMENDATION] Training appears stuck. Consider:")
        print("  1. Restarting training with smaller subset")
        print("  2. Checking system resources (memory, CPU)")
        print("  3. Looking for error messages in the log")
    
    return 0 if is_running else 1

if __name__ == "__main__":
    exit(main())

