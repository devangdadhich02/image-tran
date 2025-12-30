"""
Generate high-quality client-ready images from training results
This script will find the best available reconstructions and create presentation-ready images
"""
import sys
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

def generate_high_quality_images(recon_dir, output_dir, max_images=12):
    """Generate high-quality images for client"""
    print(f"\n{'='*60}")
    print("GENERATING HIGH-QUALITY CLIENT IMAGES")
    print(f"{'='*60}")
    
    if not recon_dir or not recon_dir.exists():
        print(f"[ERROR] Reconstruction directory not found")
        return False
    
    output_dir = Path(output_dir)
    output_dir.mkdir(exist_ok=True, parents=True)
    
    # Find image pairs
    orig_files = sorted(recon_dir.glob("*_orig.png"))
    
    # Try different reconstruction file patterns
    recon_patterns = [
        "*_recon_normalised.png",
        "*_recon_unchanged.png", 
        "*_recon*.png"
    ]
    
    recon_files = []
    for pattern in recon_patterns:
        recon_files = sorted(recon_dir.glob(pattern))
        if recon_files:
            break
    
    if not orig_files:
        print(f"[ERROR] No original images found in {recon_dir}")
        return False
    
    if not recon_files:
        print(f"[ERROR] No reconstruction images found in {recon_dir}")
        return False
    
    print(f"   Found {len(orig_files)} original images")
    print(f"   Found {len(recon_files)} reconstruction images")
    
    # Generate images
    success_count = 0
    
    for i, orig_file in enumerate(orig_files[:max_images]):
        try:
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
                print(f"   [WARNING] No matching reconstruction for {orig_file.name}")
                continue
            
            # Save individual images
            orig_img = Image.open(orig_file).convert('RGB')
            recon_img = Image.open(recon_file).convert('RGB')
            
            # Resize to same size
            if orig_img.size != recon_img.size:
                recon_img = recon_img.resize(orig_img.size, Image.Resampling.LANCZOS)
            
            # Save originals
            orig_output = output_dir / f"original_{i+1:02d}.png"
            orig_img.save(orig_output, 'PNG', quality=95)
            
            recon_output = output_dir / f"reconstruction_{i+1:02d}.png"
            recon_img.save(recon_output, 'PNG', quality=95)
            
            # Create comparison
            comp_output = output_dir / f"comparison_{i+1:02d}.png"
            create_comparison_image(orig_file, recon_file, comp_output, labels=True)
            
            success_count += 1
            print(f"   [OK] Generated images for sample {i+1}")
            
        except Exception as e:
            print(f"   [ERROR] Error processing sample {i+1}: {e}")
    
    if success_count > 0:
        print(f"\n[SUCCESS] Generated {success_count} high-quality image sets")
        print(f"   Output directory: {output_dir}")
        
        # Create a summary file
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
        
        print(f"   Summary file: {summary_path}")
        return True
    else:
        print(f"[ERROR] No images were generated")
        return False

def main():
    import argparse
    
    parser = argparse.ArgumentParser(description="Generate client-ready images from training results")
    parser.add_argument("--exp_dir", type=str, default=None, help="Experiment directory (auto-detect if not provided)")
    parser.add_argument("--output_dir", type=str, default=None, help="Output directory for client images")
    parser.add_argument("--max_images", type=int, default=12, help="Maximum number of image sets to generate")
    
    args = parser.parse_args()
    
    # Find experiment
    if args.exp_dir:
        exp_dir = Path(args.exp_dir)
        recon_dir = exp_dir / "recon_final"
    else:
        exp_dir, recon_dir = find_best_reconstructions()
    
    if not exp_dir:
        print("[ERROR] No training results found!")
        print("   Please run training first, or specify --exp_dir")
        return 1
    
    print(f"Using experiment: {exp_dir.name}")
    
    if not recon_dir or not recon_dir.exists():
        print(f"[ERROR] No reconstruction directory found in {exp_dir}")
        return 1
    
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

