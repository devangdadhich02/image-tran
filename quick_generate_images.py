"""
FAST Image Generation - Minimal Training for Quick Results
Ye script bahut kam steps mein training karke images generate karega
CPU par bhi fast chalega
"""
import sys
import subprocess
import time
import io
from pathlib import Path

# Fix Windows encoding
if sys.platform == 'win32':
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')
    sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding='utf-8', errors='replace')

def print_header(text):
    print("\n" + "="*70)
    print(f"  {text}")
    print("="*70 + "\n")

def check_for_images():
    """Check if reconstruction images exist"""
    base_dir = Path("results/train_output/DR")
    if not base_dir.exists():
        return None, None
    
    experiments = sorted([d for d in base_dir.iterdir() if d.is_dir() and d.name != "runs"], 
                       key=lambda x: x.stat().st_mtime, reverse=True)
    
    for exp in experiments:
        # Check final
        final_recon = exp / "recon_final"
        if final_recon.exists() and any(final_recon.glob("*.png")):
            return exp, final_recon
        
        # Check intermediate - use latest
        recon_dirs = sorted(exp.glob("recon_step_*"), 
                          key=lambda x: int(x.name.split("_")[-1]) if x.name.split("_")[-1].isdigit() else 0,
                          reverse=True)
        if recon_dirs:
            latest = recon_dirs[0]
            if any(latest.glob("*.png")):
                return exp, latest
    
    return None, None

def run_fast_training():
    """Run VERY FAST training - minimal steps just for images"""
    print_header("FAST TRAINING - Minimal Steps for Quick Images")
    print("Configuration:")
    print("   - Subset: 200 samples (very small)")
    print("   - Steps: 500 only (minimal for quick results)")
    print("   - Checkpoint: Every 200 steps")
    print("   - Estimated time: 15-30 minutes on CPU\n")
    print("Training start ho rahi hai...\n")
    print("-" * 70)
    
    # Use minimal parameters for speed
    cmd = [sys.executable, "train_subset.py", "fast_images", 
           "--subset_size", "200", 
           "--steps", "500", 
           "--checkpoint_freq", "200",
           "--data_dir", "DR/"]
    
    try:
        process = subprocess.Popen(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            universal_newlines=True,
            bufsize=1
        )
        
        for line in process.stdout:
            print(line, end='', flush=True)
            time.sleep(0.01)
        
        process.wait()
        print("-" * 70)
        
        if process.returncode == 0:
            print("\nTraining complete!")
            return True
        else:
            print(f"\nTraining error (code: {process.returncode})")
            return False
            
    except Exception as e:
        print(f"\nError: {e}")
        return False

def generate_images():
    """Generate client images"""
    print_header("GENERATING CLIENT IMAGES")
    
    cmd = [sys.executable, "generate_client_images.py"]
    
    try:
        process = subprocess.Popen(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            universal_newlines=True,
            bufsize=1
        )
        
        for line in process.stdout:
            print(line, end='', flush=True)
            time.sleep(0.01)
        
        process.wait()
        return process.returncode == 0
        
    except Exception as e:
        print(f"\nError: {e}")
        return False

def main():
    print_header("QUICK IMAGE GENERATION - Fast Method")
    print("Ye script minimal training karke jaldi images generate karega")
    print("CPU par bhi 15-30 minutes mein complete ho jayega\n")
    
    # Check for existing images
    print("Checking for existing images...")
    exp_dir, recon_dir = check_for_images()
    
    if not exp_dir or not recon_dir:
        print("No images found. Running FAST training...\n")
        if not run_fast_training():
            print("\nTraining failed!")
            return 1
        
        # Check again
        exp_dir, recon_dir = check_for_images()
        if not exp_dir or not recon_dir:
            print("\nTraining complete but no images found!")
            return 1
    else:
        print(f"Found existing images: {recon_dir.name}\n")
    
    # Generate images
    if not generate_images():
        print("\nImage generation failed!")
        return 1
    
    print_header("SUCCESS! IMAGES READY")
    print("Client images successfully generated!")
    print(f"\nLocation: {exp_dir / 'client_images'}\n")
    
    return 0

if __name__ == "__main__":
    try:
        exit(main())
    except KeyboardInterrupt:
        print("\n\nInterrupted by user")
        exit(1)
    except Exception as e:
        print(f"\n\nError: {e}")
        import traceback
        traceback.print_exc()
        exit(1)

