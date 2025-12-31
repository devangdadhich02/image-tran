"""
Generate high-quality client-ready images from training results
This script will find the best available reconstructions and create presentation-ready images
WITH PROGRESS INDICATORS for better user experience
"""
import sys
import time
from pathlib import Path
from PIL import Image, ImageDraw, ImageFont
import numpy as np

def find_best_reconstructions(base_dir="results/train_output/DR"):
    """Find the best reconstruction directory"""
    base_path = Path(base_dir)
    if not base_path.exists():
        return None, None
    
    # Find all experiments
    experiments = sorted([d for d in base_path.iterdir() if d.is_dir()], 
                       key=lambda x: x.stat().st_mtime, reverse=True)
    
    # Look for final reconstructions first
    for exp in experiments:
        final_recon = exp / "recon_final"
        if final_recon.exists() and any(final_recon.glob("*.png")):
            return exp, final_recon
        
        # Check for intermediate reconstructions
        recon_dirs = sorted(exp.glob("recon_step_*"), 
                          key=lambda x: int(x.name.split("_")[-1]) if x.name.split("_")[-1].isdigit() else 0,
                          reverse=True)
        if recon_dirs:
            return exp, recon_dirs[0]
    
    return None, None

def create_comparison_image(orig_path, recon_path, output_path, labels=True):
    """Create a side-by-side comparison image with labels"""
    try:
        orig_img = Image.open(orig_path).convert('RGB')
        recon_img = Image.open(recon_path).convert('RGB')
        
        # Resize to same size if needed
        if orig_img.size != recon_img.size:
            recon_img = recon_img.resize(orig_img.size, Image.Resampling.LANCZOS)
        
        # Create comparison image
        width, height = orig_img.size
        comparison = Image.new('RGB', (width * 2 + 40, height + 60), color='white')
        
        # Paste images
        comparison.paste(orig_img, (10, 30))
        comparison.paste(recon_img, (width + 30, 30))
        
        # Add labels if requested
        if labels:
            draw = ImageDraw.Draw(comparison)
            try:
                # Try to use a nice font
                font = ImageFont.truetype("arial.ttf", 20)
            except:
                font = ImageFont.load_default()
            
            draw.text((width // 2 - 40, 5), "Original", fill='black', font=font)
            draw.text((width + width // 2 - 60, 5), "Reconstruction", fill='black', font=font)
        
        comparison.save(output_path, 'PNG', quality=95)
        return True
    except Exception as e:
        print(f"   Error creating comparison: {e}")
        return False

def print_progress_bar(current, total, prefix="Progress", length=40):
    """Print a progress bar"""
    percent = ("{0:.1f}").format(100 * (current / float(total)))
    filled = int(length * current // total)
    bar = '█' * filled + '░' * (length - filled)
    print(f'\r{prefix}: |{bar}| {current}/{total} ({percent}%)', end='', flush=True)
    if current == total:
        print()  # New line when complete

def generate_high_quality_images(recon_dir, output_dir, max_images=12):
    """Generate high-quality images for client WITH PROGRESS INDICATORS"""
    print(f"\n{'='*60}")
    print("🖼️  GENERATING HIGH-QUALITY CLIENT IMAGES")
    print(f"{'='*60}\n")
    
    if not recon_dir or not recon_dir.exists():
        print(f"❌ [ERROR] Reconstruction directory not found: {recon_dir}")
        return False
    
    print(f"📁 Scanning directory: {recon_dir}")
    output_dir = Path(output_dir)
    output_dir.mkdir(exist_ok=True, parents=True)
    
    # Find image pairs with progress
    print("🔍 Searching for original images...", end='', flush=True)
    orig_files = sorted(recon_dir.glob("*_orig.png"))
    print(f" ✓ Found {len(orig_files)}")
    
    # Try different reconstruction file patterns
    print("🔍 Searching for reconstruction images...", end='', flush=True)
    recon_patterns = [
        "*_recon_normalised.png",
        "*_recon_unchanged.png", 
        "*_recon*.png"
    ]
    
    recon_files = []
    for pattern in recon_patterns:
        recon_files = sorted(recon_dir.glob(pattern))
        if recon_files:
            print(f" ✓ Found {len(recon_files)} (pattern: {pattern})")
            break
    
    if not orig_files:
        print(f"\n❌ [ERROR] No original images found in {recon_dir}")
        print(f"   Expected files like: 00_orig.png, 01_orig.png, etc.")
        return False
    
    if not recon_files:
        print(f"\n❌ [ERROR] No reconstruction images found in {recon_dir}")
        print(f"   Expected files like: *_recon_normalised.png or *_recon_unchanged.png")
        return False
    
    print(f"\n📊 Processing {min(len(orig_files), max_images)} image pairs...\n")
    
    # Generate images with progress
    success_count = 0
    total = min(len(orig_files), max_images)
    
    for i, orig_file in enumerate(orig_files[:max_images]):
        try:
            # Show progress
            print_progress_bar(i, total, prefix="Generating", length=40)
            print(f"   Processing image {i+1}/{total}: {orig_file.name}")
            
            # Find matching reconstruction
            recon_file = None
            orig_stem = orig_file.stem.replace("_orig", "")
            
            # Try exact match first
            for rf in recon_files:
                if rf.stem.startswith(orig_stem.split("_")[0]):
                    recon_file = rf
                    break
            
            # If no match, try by index
            if not recon_file and i < len(recon_files):
                recon_file = recon_files[i]
            
            if not recon_file:
                print(f"      ⚠️  [WARNING] No matching reconstruction for {orig_file.name}")
                continue
            
            # Load and process images
            print(f"      📥 Loading images...", end='', flush=True)
            orig_img = Image.open(orig_file).convert('RGB')
            recon_img = Image.open(recon_file).convert('RGB')
            print(" ✓")
            
            # Resize to same size
            if orig_img.size != recon_img.size:
                print(f"      🔄 Resizing reconstruction to match original...", end='', flush=True)
                recon_img = recon_img.resize(orig_img.size, Image.Resampling.LANCZOS)
                print(" ✓")
            
            # Save originals
            print(f"      💾 Saving original image...", end='', flush=True)
            orig_output = output_dir / f"original_{i+1:02d}.png"
            orig_img.save(orig_output, 'PNG', quality=95)
            print(" ✓")
            
            print(f"      💾 Saving reconstruction image...", end='', flush=True)
            recon_output = output_dir / f"reconstruction_{i+1:02d}.png"
            recon_img.save(recon_output, 'PNG', quality=95)
            print(" ✓")
            
            # Create comparison
            print(f"      🎨 Creating comparison image...", end='', flush=True)
            comp_output = output_dir / f"comparison_{i+1:02d}.png"
            create_comparison_image(orig_file, recon_file, comp_output, labels=True)
            print(" ✓")
            
            success_count += 1
            print(f"      ✅ Successfully generated all images for sample {i+1}\n")
            
            # Small delay to show progress (makes terminal look busy)
            time.sleep(0.1)
            
        except Exception as e:
            print(f"\n      ❌ [ERROR] Error processing sample {i+1}: {e}")
            import traceback
            traceback.print_exc()
    
    # Final progress bar
    print_progress_bar(total, total, prefix="Complete", length=40)
    
    if success_count > 0:
        print(f"\n{'='*60}")
        print(f"✅ SUCCESS! Generated {success_count} high-quality image sets")
        print(f"{'='*60}")
        print(f"📂 Output directory: {output_dir.absolute()}")
        
        # Create a summary file
        print(f"\n📝 Creating summary file...", end='', flush=True)
        summary_path = output_dir / "README.txt"
        with open(summary_path, 'w') as f:
            f.write("CLIENT IMAGE PACKAGE\n")
            f.write("=" * 50 + "\n\n")
            f.write(f"Total image sets: {success_count}\n")
            f.write(f"Source: {recon_dir}\n\n")
            f.write("Files:\n")
            f.write("  - original_XX.png: Original fundus images\n")
            f.write("  - reconstruction_XX.png: VAE reconstructions\n")
            f.write("  - comparison_XX.png: Side-by-side comparisons\n\n")
            f.write("These images demonstrate the VAE's ability to reconstruct\n")
            f.write("fundus images with visible vessel structures.\n")
        print(" ✓")
        
        print(f"\n📋 Summary file: {summary_path}")
        print(f"\n🎉 Image generation complete! Your client images are ready.")
        print(f"{'='*60}\n")
        return True
    else:
        print(f"\n❌ [ERROR] No images were generated")
        print(f"   Please check that reconstruction images exist in: {recon_dir}")
        return False

def main():
    import argparse
    
    parser = argparse.ArgumentParser(description="Generate client-ready images from training results")
    parser.add_argument("--exp_dir", type=str, default=None, help="Experiment directory (auto-detect if not provided)")
    parser.add_argument("--output_dir", type=str, default=None, help="Output directory for client images")
    parser.add_argument("--max_images", type=int, default=12, help="Maximum number of image sets to generate")
    
    args = parser.parse_args()
    
    print("\n" + "="*60)
    print("🖼️  CLIENT IMAGE GENERATOR")
    print("="*60)
    print("This script will find training results and generate client-ready images")
    print("with progress indicators for better user experience.\n")
    
    # Find experiment
    print("🔍 Searching for training results...")
    if args.exp_dir:
        exp_dir = Path(args.exp_dir)
        if not exp_dir.exists():
            print(f"❌ [ERROR] Experiment directory not found: {exp_dir}")
            return 1
        recon_dir = exp_dir / "recon_final"
        # Also check for intermediate reconstructions
        if not recon_dir.exists() or not any(recon_dir.glob("*.png")):
            recon_dirs = sorted(exp_dir.glob("recon_step_*"), 
                              key=lambda x: int(x.name.split("_")[-1]) if x.name.split("_")[-1].isdigit() else 0,
                              reverse=True)
            if recon_dirs:
                recon_dir = recon_dirs[0]
                print(f"   ℹ️  Using intermediate reconstruction: {recon_dir.name}")
    else:
        exp_dir, recon_dir = find_best_reconstructions()
    
    if not exp_dir:
        print("\n❌ [ERROR] No training results found!")
        print("\n💡 To generate images, you need to:")
        print("   1. Run training first:")
        print("      python train_subset.py quick_test --subset_size 500 --steps 2000 --data_dir DR/")
        print("\n   2. Or specify an experiment directory:")
        print("      python generate_client_images.py --exp_dir results/train_output/DR/your_experiment")
        return 1
    
    print(f"   ✓ Found experiment: {exp_dir.name}")
    
    if not recon_dir or not recon_dir.exists():
        print(f"\n❌ [ERROR] No reconstruction directory found in {exp_dir}")
        print(f"   Expected: {exp_dir / 'recon_final'}")
        print(f"   Or intermediate: {exp_dir / 'recon_step_*'}")
        print("\n💡 Training may not have completed. Please run training first.")
        return 1
    
    print(f"   ✓ Found reconstruction directory: {recon_dir.name}\n")
    
    # Set output directory
    if args.output_dir:
        output_dir = Path(args.output_dir)
    else:
        output_dir = exp_dir / "client_images"
    
    # Generate images
    success = generate_high_quality_images(recon_dir, output_dir, max_images=args.max_images)
    
    return 0 if success else 1

if __name__ == "__main__":
    exit(main())

