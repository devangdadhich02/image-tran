# PowerShell Script to Train for Paper-Quality Images
# Run this script to train with optimal settings

Write-Host "========================================" -ForegroundColor Cyan
Write-Host "  Paper-Quality Training Script" -ForegroundColor Cyan
Write-Host "========================================" -ForegroundColor Cyan
Write-Host ""

# Activate virtual environment
Write-Host "Activating virtual environment..." -ForegroundColor Yellow
& ".\venv\Scripts\Activate.ps1"

Write-Host ""
Write-Host "Starting training with paper-quality settings..." -ForegroundColor Green
Write-Host ""
Write-Host "Settings:" -ForegroundColor Yellow
Write-Host "  - Architecture: 3-layer decoder (matches paper)" -ForegroundColor White
Write-Host "  - Loss: BCE only (matches paper)" -ForegroundColor White
Write-Host "  - Steps: 15000 (for best quality)" -ForegroundColor White
Write-Host "  - Checkpoints: Every 1000 steps" -ForegroundColor White
Write-Host ""

# Ask user for training type
Write-Host "Choose training type:" -ForegroundColor Cyan
Write-Host "  1. Full training (best quality, slower)" -ForegroundColor White
Write-Host "  2. Subset training (faster, good quality)" -ForegroundColor White
$choice = Read-Host "Enter choice (1 or 2)"

if ($choice -eq "1") {
    Write-Host ""
    Write-Host "Starting FULL training..." -ForegroundColor Green
    python train_all.py paper_quality --steps 15000 --data_dir DR/ --checkpoint_freq 1000
} else {
    Write-Host ""
    Write-Host "Starting SUBSET training..." -ForegroundColor Green
    python train_subset.py paper_quality_test --subset_size 1000 --steps 10000 --data_dir DR/ --checkpoint_freq 500
}

Write-Host ""
Write-Host "========================================" -ForegroundColor Cyan
Write-Host "Training complete!" -ForegroundColor Green
Write-Host "========================================" -ForegroundColor Cyan
Write-Host ""
Write-Host "Check images in:" -ForegroundColor Yellow
Write-Host "  results/train_output/DR/paper_quality*/recon_final/" -ForegroundColor White
Write-Host ""

