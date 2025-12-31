"""
Comprehensive Image Generation Script with Progress Indicators
This script will:
1. Check if training results exist
2. Run training if needed (with progress)
3. Generate client-ready images (with progress)
4. Keep terminal busy and show progress throughout
"""
import sys
import subprocess
import time
from pathlib import Path

def print_header(text):
    """Print a formatted header"""
    print("\n" + "="*70)
    print(f"  {text}")
    print("="*70 + "\n")

def check_training_results():
    """Check if training results exist"""
    print_header("🔍 CHECKING FOR TRAINING RESULTS")
    
    base_dir = Path("results/train_output/DR")
    if not base_dir.exists():
        print("❌ No training output directory found")
        return None, None
    
    # Find all experiments
    experiments = sorted([d for d in base_dir.iterdir() if d.is_dir() and d.name != "runs"], 
                       key=lambda x: x.stat().st_mtime, reverse=True)
    
    if not experiments:
        print("❌ No experiments found")
        return None, None
    
    print(f"📁 Found {len(experiments)} experiment(s)")
    
    # Check each experiment for reconstructions
    for exp in experiments:
        print(f"\n   Checking: {exp.name}")
        
        # Check for final reconstructions
        final_recon = exp / "recon_final"
        if final_recon.exists() and any(final_recon.glob("*.png")):
            png_count = len(list(final_recon.glob("*.png")))
            print(f"      ✅ Found final reconstructions: {png_count} images")
            return exp, final_recon
        
        # Check for intermediate reconstructions
        recon_dirs = sorted(exp.glob("recon_step_*"), 
                          key=lambda x: int(x.name.split("_")[-1]) if x.name.split("_")[-1].isdigit() else 0,
                          reverse=True)
        if recon_dirs:
            latest_recon = recon_dirs[0]
            png_count = len(list(latest_recon.glob("*.png")))
            print(f"      ✅ Found intermediate reconstructions: {png_count} images in {latest_recon.name}")
            return exp, latest_recon
    
    print("\n❌ No reconstruction images found in any experiment")
    return None, None

def run_training_with_progress():
    """Run training with progress indicators"""
    print_header("🚀 STARTING TRAINING")
    
    print("📋 Training Configuration:")
    print("   - Dataset: DR")
    print("   - Subset size: 500 samples")
    print("   - Steps: 2000")
    print("   - This will generate reconstruction images\n")
    
    print("⏳ Starting training (this may take a while)...")
    print("   Terminal will show training progress...\n")
    
    # Run training
    cmd = [
        sys.executable, 
        "train_subset.py", 
        "quick_test", 
        "--subset_size", "500", 
        "--steps", "2000", 
        "--data_dir", "DR/"
    ]
    
    try:
        # Run with real-time output
        process = subprocess.Popen(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            universal_newlines=True,
            bufsize=1
        )
        
        # Print output in real-time
        for line in process.stdout:
            print(line, end='', flush=True)
            time.sleep(0.01)  # Small delay to keep terminal busy
        
        process.wait()
        
        if process.returncode == 0:
            print("\n✅ Training completed successfully!")
            return True
        else:
            print(f"\n❌ Training failed with exit code {process.returncode}")
            return False
            
    except Exception as e:
        print(f"\n❌ Error running training: {e}")
        return False

def generate_images_with_progress():
    """Generate images using the improved script"""
    print_header("🖼️  GENERATING CLIENT IMAGES")
    
    print("⏳ Running image generation script...")
    print("   Terminal will show detailed progress...\n")
    
    cmd = [sys.executable, "generate_client_images.py"]
    
    try:
        process = subprocess.Popen(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            universal_newlines=True,
            bufsize=1
        )
        
        # Print output in real-time
        for line in process.stdout:
            print(line, end='', flush=True)
            time.sleep(0.01)  # Small delay to keep terminal busy
        
        process.wait()
        
        return process.returncode == 0
        
    except Exception as e:
        print(f"\n❌ Error generating images: {e}")
        return False

def main():
    """Main function"""
    print_header("🎯 CLIENT IMAGE GENERATION PIPELINE")
    print("This script will ensure you have training results and generate client-ready images")
    print("with full progress indicators throughout the process.\n")
    
    # Step 1: Check for existing results
    exp_dir, recon_dir = check_training_results()
    
    if not exp_dir or not recon_dir:
        print("\n" + "="*70)
        print("⚠️  No training results found!")
        print("="*70)
        response = input("\n❓ Would you like to run training now? (y/n): ").strip().lower()
        
        if response == 'y' or response == 'yes':
            # Run training
            if not run_training_with_progress():
                print("\n❌ Training failed. Please check the errors above.")
                return 1
            
            # Check again after training
            exp_dir, recon_dir = check_training_results()
            if not exp_dir or not recon_dir:
                print("\n❌ Training completed but no reconstruction images found.")
                print("   Please check the training logs for errors.")
                return 1
        else:
            print("\n💡 To generate images, you need to run training first:")
            print("   python train_subset.py quick_test --subset_size 500 --steps 2000 --data_dir DR/")
            return 1
    
    # Step 2: Generate images
    print("\n" + "="*70)
    print("✅ Training results found! Proceeding to image generation...")
    print("="*70)
    
    if not generate_images_with_progress():
        print("\n❌ Image generation failed. Please check the errors above.")
        return 1
    
    print_header("🎉 SUCCESS!")
    print("Client images have been generated successfully!")
    print("Check the 'client_images' folder in your experiment directory.")
    
    return 0

if __name__ == "__main__":
    try:
        exit(main())
    except KeyboardInterrupt:
        print("\n\n⚠️  Process interrupted by user")
        exit(1)
    except Exception as e:
        print(f"\n\n❌ Unexpected error: {e}")
        import traceback
        traceback.print_exc()
        exit(1)

