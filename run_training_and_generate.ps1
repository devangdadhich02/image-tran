# PowerShell script to run training and generate images with progress
# This keeps the terminal busy and shows progress throughout

Write-Host ""
Write-Host "================================================================================" -ForegroundColor Cyan
Write-Host "  CLIENT IMAGE GENERATION - FULL PIPELINE" -ForegroundColor Cyan
Write-Host "================================================================================" -ForegroundColor Cyan
Write-Host ""

# Change to project directory
$projectDir = "C:\Users\Devang Dadhich\OneDrive\Desktop\Update_image"
Set-Location $projectDir

# Activate virtual environment if it exists
if (Test-Path "venv\Scripts\Activate.ps1") {
    Write-Host "🔧 Activating virtual environment..." -ForegroundColor Yellow
    & "venv\Scripts\Activate.ps1"
}

# Check if images already exist
Write-Host "🔍 Checking for existing reconstruction images..." -ForegroundColor Yellow
$reconDirs = Get-ChildItem -Path "results\train_output\DR" -Recurse -Directory -Filter "recon_*" -ErrorAction SilentlyContinue
$hasImages = $false

if ($reconDirs) {
    foreach ($dir in $reconDirs) {
        $pngFiles = Get-ChildItem -Path $dir.FullName -Filter "*.png" -ErrorAction SilentlyContinue
        if ($pngFiles) {
            Write-Host "✅ Found existing reconstruction images in: $($dir.Name)" -ForegroundColor Green
            $hasImages = $true
            break
        }
    }
}

if (-not $hasImages) {
    Write-Host "⚠️  No reconstruction images found. Running training..." -ForegroundColor Yellow
    Write-Host ""
    Write-Host "🚀 Starting training (this will take some time)..." -ForegroundColor Cyan
    Write-Host "   Terminal will show progress throughout..." -ForegroundColor Gray
    Write-Host ""
    
    # Run training
    python train_subset.py quick_test --subset_size 500 --steps 2000 --data_dir DR/
    
    if ($LASTEXITCODE -ne 0) {
        Write-Host ""
        Write-Host "❌ Training failed! Please check the errors above." -ForegroundColor Red
        exit 1
    }
    
    Write-Host ""
    Write-Host "✅ Training completed!" -ForegroundColor Green
    Write-Host ""
} else {
    Write-Host "✅ Using existing reconstruction images" -ForegroundColor Green
    Write-Host ""
}

# Generate client images
Write-Host "🖼️  Generating client-ready images..." -ForegroundColor Cyan
Write-Host "   Terminal will show detailed progress..." -ForegroundColor Gray
Write-Host ""

python generate_client_images.py

if ($LASTEXITCODE -ne 0) {
    Write-Host ""
    Write-Host "❌ Image generation failed! Please check the errors above." -ForegroundColor Red
    exit 1
}

Write-Host ""
Write-Host "================================================================================" -ForegroundColor Green
Write-Host "  ✅ SUCCESS! Client images have been generated!" -ForegroundColor Green
Write-Host "================================================================================" -ForegroundColor Green
Write-Host ""
Write-Host "📂 Check the 'client_images' folder in your experiment directory" -ForegroundColor Yellow
Write-Host ""

