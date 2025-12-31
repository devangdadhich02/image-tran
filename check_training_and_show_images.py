"""
Check training progress and show generated images
"""
import sys
import os
from pathlib import Path
import glob

# Set UTF-8 encoding for Windows
if sys.platform == 'win32':
    sys.stdout.reconfigure(encoding='utf-8')
    sys.stderr.reconfigure(encoding='utf-8')

def find_latest_training():
    """Find the most recent training output"""
    results_dir = Path("results/train_output/DR")
    if not results_dir.exists():
        print("❌ No training results found!")
        return None
    
    # Find all experiment directories
    experiments = list(results_dir.glob("*"))
    if not experiments:
        print("❌ No experiments found!")
        return None
    
    # Sort by modification time
    latest = max(experiments, key=lambda p: p.stat().st_mtime)
    return latest

def show_training_info(exp_dir):
    """Show training information"""
    print(f"\n{'='*60}")
    print(f"📁 Latest Training: {exp_dir.name}")
    print(f"{'='*60}\n")
    
    # Check for log file
    log_file = exp_dir / "log.txt"
    if log_file.exists():
        print("✅ Log file found")
        with open(log_file, 'r', encoding='utf-8', errors='ignore') as f:
            lines = f.readlines()
            # Show last 10 lines
            print("\n📋 Recent Log Output:")
            print("-" * 60)
            for line in lines[-10:]:
                print(line.rstrip())
    
    # Check for intermediate reconstructions
    recon_dirs = sorted(exp_dir.glob("recon_step_*"))
    if recon_dirs:
        print(f"\n✅ Found {len(recon_dirs)} intermediate reconstruction checkpoints")
        latest_recon = recon_dirs[-1]
        print(f"📸 Latest: {latest_recon.name}")
        
        # Count images
        images = list(latest_recon.glob("*_recon_normalised.png"))
        if images:
            print(f"   Images: {len(images)}")
            print(f"\n📂 Path: {latest_recon}")
            print(f"\n💡 To view images, open:")
            print(f"   {latest_recon.absolute()}")
    
    # Check for final reconstructions
    final_dir = exp_dir / "recon_final"
    if final_dir.exists():
        images = list(final_dir.glob("*_recon_normalised.png"))
        if images:
            print(f"\n✅ Final reconstructions found: {len(images)} images")
            print(f"📂 Path: {final_dir.absolute()}")
            print(f"\n💡 To view images, open:")
            print(f"   {final_dir.absolute()}")
    
    return exp_dir

def main():
    print("🔍 Checking Training Progress...\n")
    
    exp_dir = find_latest_training()
    if exp_dir:
        show_training_info(exp_dir)
        print(f"\n{'='*60}")
        print("✅ Check complete!")
        print(f"\n💡 Tip: Open the folder path above to view generated images")
    else:
        print("❌ No training found. Make sure training is running.")

if __name__ == "__main__":
    main()

