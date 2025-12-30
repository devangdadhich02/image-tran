"""
Quick script to check training status and generate images if available
Run this periodically to check if training has completed and generate images
"""
import sys
from pathlib import Path
import subprocess

def main():
    print("\n" + "="*60)
    print("CHECKING TRAINING STATUS AND GENERATING IMAGES")
    print("="*60)
    
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
    print(f"\nLatest experiment: {latest_exp.name}")
    
    # Check for final reconstructions
    final_recon = latest_exp / "recon_final"
    if final_recon.exists() and any(final_recon.glob("*.png")):
        print("[OK] Final reconstructions found!")
        
        # Run verification and image generation
        print("\nRunning verification and image generation...")
        result = subprocess.run(
            f'python verify_training.py "{latest_exp}"',
            shell=True
        )
        
        # Also run dedicated image generation
        print("\nGenerating high-quality client images...")
        result2 = subprocess.run(
            f'python generate_client_images.py --exp_dir "{latest_exp}"',
            shell=True
        )
        
        if result.returncode == 0 or result2.returncode == 0:
            print("\n[SUCCESS] Images generated!")
            print(f"   Check: {latest_exp / 'client_images'}")
            return 0
    else:
        # Check for intermediate reconstructions
        recon_dirs = sorted(latest_exp.glob("recon_step_*"), 
                          key=lambda x: int(x.name.split("_")[-1]) if x.name.split("_")[-1].isdigit() else 0,
                          reverse=True)
        
        if recon_dirs:
            print(f"[INFO] Training in progress - found {len(recon_dirs)} intermediate checkpoints")
            print(f"   Latest: {recon_dirs[0].name}")
            
            # Generate images from latest checkpoint
            print("\nGenerating images from latest checkpoint...")
            result = subprocess.run(
                f'python generate_client_images.py --exp_dir "{latest_exp}" --output_dir "{latest_exp / "client_images_intermediate"}"',
                shell=True
            )
            
            if result.returncode == 0:
                print(f"\n[SUCCESS] Intermediate images generated!")
                print(f"   Check: {latest_exp / 'client_images_intermediate'}")
        else:
            print("[INFO] Training just started - no reconstructions yet")
            print("   Check back in 10-20 minutes")
    
    return 0

if __name__ == "__main__":
    exit(main())

