"""
Complete script to run training and generate images with full progress indicators
Terminal busy rahega aur progress dikhayega
"""
import sys
import subprocess
import time
import io
from pathlib import Path

# Fix Windows encoding issues
if sys.platform == 'win32':
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')
    sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding='utf-8', errors='replace')

def print_header(text, color_code=""):
    """Print formatted header"""
    print("\n" + "="*70)
    print(f"  {text}")
    print("="*70 + "\n")

def check_for_images():
    """Check if reconstruction images already exist"""
    base_dir = Path("results/train_output/DR")
    if not base_dir.exists():
        return None, None
    
    experiments = sorted([d for d in base_dir.iterdir() if d.is_dir() and d.name != "runs"], 
                       key=lambda x: x.stat().st_mtime, reverse=True)
    
    for exp in experiments:
        # Check final reconstructions
        final_recon = exp / "recon_final"
        if final_recon.exists() and any(final_recon.glob("*.png")):
            return exp, final_recon
        
        # Check intermediate reconstructions
        recon_dirs = sorted(exp.glob("recon_step_*"), 
                          key=lambda x: int(x.name.split("_")[-1]) if x.name.split("_")[-1].isdigit() else 0,
                          reverse=True)
        if recon_dirs:
            latest = recon_dirs[0]
            if any(latest.glob("*.png")):
                return exp, latest
    
    return None, None

def run_training_with_progress():
    """Run training with real-time progress output"""
    print_header("🚀 TRAINING STARTING", "CYAN")
    print("📋 Training Configuration:")
    print("   - Dataset: DR")
    print("   - Subset size: 500 samples")
    print("   - Steps: 2000")
    print("   - Terminal busy rahega aur progress dikhayega...")
    print("\n⏳ Training chal rahi hai... (ye thoda time lega)\n")
    print("-" * 70)
    
    cmd = [sys.executable, "train_subset.py", "quick_test", 
           "--subset_size", "500", "--steps", "2000", "--data_dir", "DR/"]
    
    try:
        # Run with real-time output - terminal busy rahega
        process = subprocess.Popen(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            universal_newlines=True,
            bufsize=1
        )
        
        # Print every line in real-time - terminal busy rahega
        for line in process.stdout:
            print(line, end='', flush=True)
            # Small delay to keep terminal busy
            time.sleep(0.01)
        
        process.wait()
        print("\n" + "-" * 70)
        
        if process.returncode == 0:
            print("\n✅ Training complete ho gayi!")
            return True
        else:
            print(f"\n❌ Training mein error aaya (exit code: {process.returncode})")
            return False
            
    except Exception as e:
        print(f"\n❌ Training start karne mein error: {e}")
        import traceback
        traceback.print_exc()
        return False

def generate_images_with_progress():
    """Generate images with progress"""
    print_header("🖼️  IMAGE GENERATION STARTING", "CYAN")
    print("⏳ Images generate ho rahi hain...")
    print("   Terminal busy rahega aur progress dikhayega...\n")
    print("-" * 70)
    
    cmd = [sys.executable, "generate_client_images.py"]
    
    try:
        process = subprocess.Popen(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            universal_newlines=True,
            bufsize=1
        )
        
        # Print every line in real-time
        for line in process.stdout:
            print(line, end='', flush=True)
            time.sleep(0.01)
        
        process.wait()
        print("-" * 70)
        
        return process.returncode == 0
        
    except Exception as e:
        print(f"\n❌ Image generation mein error: {e}")
        return False

def main():
    """Main function - sab kuch yahan se chalega"""
    print_header("🎯 CLIENT IMAGES GENERATE KARNE KA COMPLETE PROCESS", "GREEN")
    print("Ye script:")
    print("  1. Training check karegi (agar images nahi hain to training chalayegi)")
    print("  2. Training ke dauran terminal busy rahega aur progress dikhayega")
    print("  3. Training ke baad images generate karegi")
    print("  4. Client ko bhejne ke liye images ready ho jayengi\n")
    
    # Step 1: Check for existing images
    print("🔍 Checking for existing reconstruction images...")
    exp_dir, recon_dir = check_for_images()
    
    if not exp_dir or not recon_dir:
        print("❌ No reconstruction images found")
        print("🚀 Training start kar rahe hain...\n")
        
        # Run training
        if not run_training_with_progress():
            print("\n❌ Training fail ho gayi. Please check errors above.")
            return 1
        
        # Check again after training
        print("\n🔍 Training ke baad images check kar rahe hain...")
        exp_dir, recon_dir = check_for_images()
        
        if not exp_dir or not recon_dir:
            print("\n❌ Training complete hui lekin images nahi mili.")
            print("   Please check training logs for errors.")
            return 1
    else:
        print(f"✅ Existing images mil gayi: {recon_dir.name}")
        print(f"   Experiment: {exp_dir.name}\n")
    
    # Step 2: Generate client images
    if not generate_images_with_progress():
        print("\n❌ Image generation fail ho gayi.")
        return 1
    
    # Success!
    print_header("🎉 SUCCESS! IMAGES READY HAI!", "GREEN")
    print("✅ Client images successfully generate ho gayi!")
    print(f"\n📂 Images yahan milengi:")
    print(f"   {exp_dir / 'client_images'}")
    print("\n📋 Aapko milenge:")
    print("   - original_XX.png - Original fundus images")
    print("   - reconstruction_XX.png - VAE reconstructions")
    print("   - comparison_XX.png - Side-by-side comparisons")
    print("\n🎯 Ab aap client ko images bhej sakte hain!\n")
    
    return 0

if __name__ == "__main__":
    try:
        exit(main())
    except KeyboardInterrupt:
        print("\n\n⚠️  Process user ne interrupt kar diya")
        exit(1)
    except Exception as e:
        print(f"\n\n❌ Unexpected error: {e}")
        import traceback
        traceback.print_exc()
        exit(1)

