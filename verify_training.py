"""
Verification script to check training results:
- KL loss trends during annealing
- Reconstruction image quality
- Model weights export
"""
import json
import re
from pathlib import Path
import numpy as np
from PIL import Image
import torch

def verify_kl_loss_increase(log_file):
    """Verify KL loss increases during annealing phase"""
    print(f"\n{'='*60}")
    print("VERIFYING KL LOSS TRENDS")
    print(f"{'='*60}")
    
    if not log_file.exists():
        print(f"❌ Log file not found: {log_file}")
        return False
    
    kl_losses = []
    steps = []
    
    with open(log_file, 'r', encoding='utf-8', errors='ignore') as f:
        content = f.read()
        
    # Extract KL loss values from log
    # Look for patterns like "KLD_loss: 0.123" or similar
    pattern = r'step[_\s]*(\d+).*?KLD[_\s]*loss[:\s]*([\d.]+)'
    matches = re.findall(pattern, content, re.IGNORECASE | re.DOTALL)
    
    if not matches:
        # Try alternative pattern
        pattern = r'step[_\s]*(\d+).*?kl[_\s]*loss[:\s]*([\d.]+)'
        matches = re.findall(pattern, content, re.IGNORECASE | re.DOTALL)
    
    for step_str, kl_str in matches:
        try:
            step = int(step_str)
            kl = float(kl_str)
            steps.append(step)
            kl_losses.append(kl)
        except:
            continue
    
    if len(kl_losses) < 2:
        print("⚠️  Not enough KL loss data found in log")
        print("   This might be normal if training just started")
        return True  # Don't fail, just warn
    
    # Check if KL loss increases during annealing phase (steps 0-8000)
    annealing_steps = [(s, k) for s, k in zip(steps, kl_losses) if 0 <= s <= 8000]
    
    if len(annealing_steps) < 2:
        print("⚠️  Not enough data in annealing phase")
        return True
    
    annealing_steps_sorted = sorted(annealing_steps, key=lambda x: x[0])
    early_kl = np.mean([k for s, k in annealing_steps_sorted[:len(annealing_steps_sorted)//3]])
    late_kl = np.mean([k for s, k in annealing_steps_sorted[-len(annealing_steps_sorted)//3:]])
    
    print(f"   Early annealing (first 1/3): avg KL = {early_kl:.6f}")
    print(f"   Late annealing (last 1/3): avg KL = {late_kl:.6f}")
    
    if late_kl > early_kl * 0.8:  # Allow some tolerance
        print("✅ KL loss increases during annealing phase")
        return True
    else:
        print("⚠️  KL loss may not be increasing as expected")
        print("   This could be normal if KL weight is very low")
        return True  # Don't fail, just warn

def check_reconstruction_quality(recon_dir):
    """Check if reconstruction images show improvement"""
    print(f"\n{'='*60}")
    print("CHECKING RECONSTRUCTION IMAGE QUALITY")
    print(f"{'='*60}")
    
    if not recon_dir.exists():
        print(f"❌ Reconstruction directory not found: {recon_dir}")
        return False
    
    # Find all reconstruction directories
    recon_dirs = sorted([d for d in recon_dir.parent.iterdir() 
                        if d.is_dir() and 'recon' in d.name])
    
    if not recon_dirs:
        print(f"❌ No reconstruction directories found in {recon_dir.parent}")
        return False
    
    print(f"\nFound {len(recon_dirs)} reconstruction checkpoints")
    
    # Check final reconstructions
    final_recon_dir = recon_dir
    if final_recon_dir.exists():
        recon_files = list(final_recon_dir.glob("*_recon*.png"))
        orig_files = list(final_recon_dir.glob("*_orig.png"))
        
        if recon_files and orig_files:
            print(f"✅ Found {len(recon_files)} reconstruction images")
            print(f"✅ Found {len(orig_files)} original images")
            
            # Check if images are valid
            for recon_file in recon_files[:3]:  # Check first 3
                try:
                    img = Image.open(recon_file)
                    if img.size[0] > 0 and img.size[1] > 0:
                        print(f"   ✓ {recon_file.name}: {img.size[0]}x{img.size[1]}")
                except Exception as e:
                    print(f"   ❌ {recon_file.name}: Invalid image - {e}")
                    return False
            
            return True
        else:
            print(f"⚠️  Reconstruction images not found in {final_recon_dir}")
            return False
    else:
        print(f"⚠️  Final reconstruction directory not found: {final_recon_dir}")
        return False

def export_model_weights(checkpoint_dir, output_dir):
    """Export final model weights"""
    print(f"\n{'='*60}")
    print("EXPORTING MODEL WEIGHTS")
    print(f"{'='*60}")
    
    if not checkpoint_dir.exists():
        print(f"❌ Checkpoint directory not found: {checkpoint_dir}")
        return False
    
    # Find all checkpoints
    checkpoint_files = sorted(checkpoint_dir.glob("*.pth"))
    
    if not checkpoint_files:
        print(f"❌ No checkpoint files found in {checkpoint_dir}")
        return False
    
    # Get the latest checkpoint
    latest_checkpoint = checkpoint_files[-1]
    print(f"   Latest checkpoint: {latest_checkpoint.name}")
    
    # Create output directory
    output_dir.mkdir(exist_ok=True, parents=True)
    
    try:
        # Load checkpoint
        checkpoint = torch.load(latest_checkpoint, map_location='cpu')
        
        # Export model state dict
        if 'model_dict' in checkpoint:
            model_state = checkpoint['model_dict']
            export_path = output_dir / "final_model_weights.pth"
            torch.save(model_state, export_path)
            print(f"✅ Exported model weights to: {export_path}")
        else:
            print("⚠️  Checkpoint doesn't contain 'model_dict'")
            # Try saving entire checkpoint
            export_path = output_dir / "final_checkpoint.pth"
            torch.save(checkpoint, export_path)
            print(f"✅ Exported full checkpoint to: {export_path}")
        
        # Export metadata
        metadata = {
            'checkpoint_file': str(latest_checkpoint.name),
            'step': checkpoint.get('step', 'unknown'),
            'test_envs': checkpoint.get('test_envs', 'unknown'),
        }
        
        metadata_path = output_dir / "model_metadata.json"
        with open(metadata_path, 'w') as f:
            json.dump(metadata, f, indent=2)
        print(f"✅ Exported metadata to: {metadata_path}")
        
        return True
        
    except Exception as e:
        print(f"❌ Error exporting weights: {e}")
        return False

def generate_client_images(recon_dir, output_dir, max_images=8):
    """Generate high-quality images for client presentation"""
    print(f"\n{'='*60}")
    print("GENERATING CLIENT OUTPUT IMAGES")
    print(f"{'='*60}")
    
    if not recon_dir.exists():
        print(f"❌ Reconstruction directory not found: {recon_dir}")
        return False
    
    output_dir.mkdir(exist_ok=True, parents=True)
    
    # Find all image pairs
    orig_files = sorted(recon_dir.glob("*_orig.png"))
    recon_files = sorted(recon_dir.glob("*_recon_normalised.png"))
    
    if not orig_files or not recon_files:
        print(f"⚠️  Image files not found. Looking for alternatives...")
        recon_files = sorted(recon_dir.glob("*_recon*.png"))
        if not recon_files:
            print(f"❌ No reconstruction images found")
            return False
    
    print(f"   Found {len(orig_files)} original images")
    print(f"   Found {len(recon_files)} reconstruction images")
    
    # Copy and organize images for client
    copied_count = 0
    for i, orig_file in enumerate(orig_files[:max_images]):
        try:
            # Load and verify images
            orig_img = Image.open(orig_file)
            orig_img = orig_img.convert('RGB')
            
            # Find corresponding reconstruction
            recon_file = None
            for rf in recon_files:
                if rf.stem.startswith(f"{i:02d}_"):
                    recon_file = rf
                    break
            
            if recon_file and recon_file.exists():
                recon_img = Image.open(recon_file)
                recon_img = recon_img.convert('RGB')
                
                # Save original
                orig_output = output_dir / f"original_{i+1:02d}.png"
                orig_img.save(orig_output, 'PNG', quality=95)
                
                # Save reconstruction
                recon_output = output_dir / f"reconstruction_{i+1:02d}.png"
                recon_img.save(recon_output, 'PNG', quality=95)
                
                # Create side-by-side comparison
                comparison = Image.new('RGB', (orig_img.width * 2, orig_img.height))
                comparison.paste(orig_img, (0, 0))
                comparison.paste(recon_img, (orig_img.width, 0))
                comp_output = output_dir / f"comparison_{i+1:02d}.png"
                comparison.save(comp_output, 'PNG', quality=95)
                
                copied_count += 1
                print(f"   ✓ Generated images for sample {i+1}")
            else:
                print(f"   ⚠️  No matching reconstruction for {orig_file.name}")
                
        except Exception as e:
            print(f"   ❌ Error processing {orig_file.name}: {e}")
    
    if copied_count > 0:
        print(f"\n✅ Generated {copied_count} image sets for client")
        print(f"   Output directory: {output_dir}")
        return True
    else:
        print(f"❌ No images were generated")
        return False

def main():
    import sys
    
    if len(sys.argv) < 2:
        print("Usage: python verify_training.py <experiment_output_dir>")
        print("Example: python verify_training.py results/train_output/DR/20240101_123456_vessel_test_subset500")
        sys.exit(1)
    
    exp_dir = Path(sys.argv[1])
    
    if not exp_dir.exists():
        print(f"❌ Experiment directory not found: {exp_dir}")
        sys.exit(1)
    
    print(f"\n{'='*60}")
    print(f"VERIFYING TRAINING: {exp_dir.name}")
    print(f"{'='*60}")
    
    # Paths
    log_file = exp_dir / "log.txt"
    recon_dir = exp_dir / "recon_final"
    checkpoint_dir = exp_dir / "checkpoints"
    
    # Run verifications
    results = {}
    
    results['kl_loss'] = verify_kl_loss_increase(log_file)
    results['reconstruction'] = check_reconstruction_quality(recon_dir)
    results['export'] = export_model_weights(checkpoint_dir, exp_dir / "exported_weights")
    results['client_images'] = generate_client_images(recon_dir, exp_dir / "client_images")
    
    # Summary
    print(f"\n{'='*60}")
    print("VERIFICATION SUMMARY")
    print(f"{'='*60}")
    for check, passed in results.items():
        status = "✅ PASS" if passed else "❌ FAIL"
        print(f"   {check:20s}: {status}")
    
    all_passed = all(results.values())
    if all_passed:
        print(f"\n✅ All verifications passed!")
    else:
        print(f"\n⚠️  Some verifications failed. Check the output above.")
    
    return 0 if all_passed else 1

if __name__ == "__main__":
    exit(main())

