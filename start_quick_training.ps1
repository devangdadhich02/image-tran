# Quick training script with progress monitoring
Write-Host "Starting quick training for image generation..." -ForegroundColor Green

# Activate virtual environment
& ".\venv\Scripts\Activate.ps1"

# Start training with smaller parameters for faster results
$trainingCmd = "python train_subset.py quick_images --subset_size 200 --steps 1000 --data_dir DR/ --checkpoint_freq 250"

Write-Host "`nCommand: $trainingCmd" -ForegroundColor Yellow
Write-Host "This will take approximately 15-30 minutes on CPU`n" -ForegroundColor Cyan

# Start training in background
Start-Process powershell -ArgumentList "-NoExit", "-Command", "cd '$PWD'; .\venv\Scripts\Activate.ps1; $trainingCmd" -WindowStyle Normal

Write-Host "Training started in new window. Monitor progress with:" -ForegroundColor Green
Write-Host "  python check_and_generate_images.py" -ForegroundColor Yellow
Write-Host "`nOr check log file:" -ForegroundColor Green
Write-Host "  Get-Content results\train_output\DR\<latest_experiment>\log.txt -Tail 20 -Wait" -ForegroundColor Yellow

