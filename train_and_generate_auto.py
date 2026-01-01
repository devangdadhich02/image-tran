"""
Automatic Training and Image Generation Script
- Trains the model
- Generates images automatically
- If vessels not visible, tries Green channel approach
"""
import subprocess
import sys
import time
from pathlib import Path
import os
import io

# Fix Windows console encoding
if sys.platform == 'win32':
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')
    sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding='utf-8', errors='replace')

def print_progress(message):
    """Print with timestamp"""
    print(f"[{time.strftime('%H:%M:%S')}] {message}")

def check_training_complete(experiment_name):
    """Check if training has completed"""
    results_dir = Path("results/train_output/DR")
    if not results_dir.exists():
        return None
    
    # Find the experiment directory
    exp_dirs = [d for d in results_dir.iterdir() if experiment_name in d.name and d.is_dir()]
    if not exp_dirs:
        return None
    
    exp_dir = exp_dirs[0]
    log_file = exp_dir / "log.txt"
    
    if log_file.exists():
        with open(log_file, 'r', encoding='utf-8', errors='ignore') as f:
            content = f.read()
            if "SUBSET TRAINING COMPLETE" in content or "Training complete" in content:
                return exp_dir
    
    return None

def run_training():
    """Run training command"""
    print_progress("[START] Starting training...")
    print_progress("Command: python train_subset.py vessel_quality --subset_size 500 --steps 5000 --data_dir DR/ --checkpoint_freq 500")
    
    # Use Python from venv if available
    venv_python = Path("venv/Scripts/python.exe")
    if venv_python.exists():
        python_exe = str(venv_python)
    else:
        python_exe = sys.executable
    
    cmd = [
        python_exe,
        "train_subset.py",
        "vessel_quality",
        "--subset_size", "500",
        "--steps", "5000",
        "--data_dir", "DR/",
        "--checkpoint_freq", "500"
    ]
    
    try:
        process = subprocess.Popen(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1,
            universal_newlines=True
        )
        
        # Print output in real-time
        for line in process.stdout:
            print(line, end='', flush=True)
        
        process.wait()
        return process.returncode == 0
    except Exception as e:
        print_progress(f"❌ Training error: {e}")
        return False

def generate_images(exp_dir):
    """Generate client images"""
    print_progress("[INFO] Generating images...")
    
    recon_dir = exp_dir / "recon_final"
    if not recon_dir.exists():
        print_progress("[WARN] No recon_final directory found, checking intermediate reconstructions...")
        # Check for intermediate reconstructions
        step_dirs = sorted([d for d in exp_dir.iterdir() if d.is_dir() and d.name.startswith("recon_step_")])
        if step_dirs:
            recon_dir = step_dirs[-1]  # Use latest step
            print_progress(f"Using latest intermediate: {recon_dir.name}")
        else:
            print_progress("[ERROR] No reconstructions found!")
            return False
    
    output_dir = exp_dir / "client_images"
    output_dir.mkdir(exist_ok=True)
    
    # Use Python from venv if available
    venv_python = Path("venv/Scripts/python.exe")
    if venv_python.exists():
        python_exe = str(venv_python)
    else:
        python_exe = sys.executable
    
    cmd = [
        python_exe,
        "generate_client_images.py",
        str(recon_dir),
        str(output_dir)
    ]
    
    try:
        result = subprocess.run(cmd, capture_output=True, text=True, encoding='utf-8', errors='ignore')
        print(result.stdout)
        if result.stderr:
            print(result.stderr)
        return result.returncode == 0
    except Exception as e:
        print_progress(f"[ERROR] Image generation error: {e}")
        return False

def main():
    print("=" * 70)
    print("AUTOMATIC TRAINING & IMAGE GENERATION")
    print("=" * 70)
    print()
    
    # Check if training already completed
    exp_dir = check_training_complete("vessel_quality")
    
    if exp_dir:
        print_progress("[OK] Training already completed!")
        print_progress(f"Experiment directory: {exp_dir}")
    else:
        print_progress("[INFO] Training not found or incomplete. Starting training...")
        success = run_training()
        
        if not success:
            print_progress("[ERROR] Training failed!")
            return
        
        # Wait a bit and check again
        time.sleep(2)
        exp_dir = check_training_complete("vessel_quality")
        
        if not exp_dir:
            # Find latest experiment
            results_dir = Path("results/train_output/DR")
            if results_dir.exists():
                exp_dirs = sorted([d for d in results_dir.iterdir() if d.is_dir() and "vessel_quality" in d.name])
                if exp_dirs:
                    exp_dir = exp_dirs[-1]
                    print_progress(f"Using latest experiment: {exp_dir.name}")
    
    if exp_dir:
        print_progress("[INFO] Generating images...")
        success = generate_images(exp_dir)
        
        if success:
            print()
            print("=" * 70)
            print("[SUCCESS] Images generated!")
            print("=" * 70)
            print(f"Images location: {exp_dir / 'client_images'}")
            print()
            print("Check the images. If vessels are still not visible:")
            print("   1. Try Green channel approach (better vessel contrast)")
            print("   2. Train for more steps (10000+)")
            print("   3. Adjust KL weight in config.yaml")
        else:
            print_progress("[ERROR] Image generation failed!")
    else:
        print_progress("[ERROR] Could not find experiment directory!")

if __name__ == "__main__":
    main()

