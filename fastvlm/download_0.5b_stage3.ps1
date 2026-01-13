# PowerShell script to download FastVLM 0.5B Stage 3 model
# Run this from the fastvlm directory: .\download_0.5b_stage3.ps1

Write-Host "Downloading FastVLM 0.5B Stage 3 model..." -ForegroundColor Yellow
Write-Host "This may take a few minutes (model size: ~1.15 GB)" -ForegroundColor Cyan

# Create checkpoints directory if it doesn't exist
if (-not (Test-Path "checkpoints")) {
    New-Item -ItemType Directory -Path "checkpoints" | Out-Null
}

# Download the model
$url = "https://ml-site.cdn-apple.com/datasets/fastvlm/llava-fastvithd_0.5b_stage3.zip"
$output = "checkpoints\llava-fastvithd_0.5b_stage3.zip"

Write-Host "Downloading from: $url" -ForegroundColor Gray
Invoke-WebRequest -Uri $url -OutFile $output

Write-Host "`nExtracting model..." -ForegroundColor Yellow
Expand-Archive -Path $output -DestinationPath "checkpoints" -Force

Write-Host "Cleaning up zip file..." -ForegroundColor Gray
Remove-Item $output

Write-Host "`n✓ Model downloaded successfully!" -ForegroundColor Green
Write-Host "Model location: checkpoints\llava-fastvithd_0.5b_stage3\" -ForegroundColor Cyan
Write-Host "`nYou can now run:" -ForegroundColor Yellow
Write-Host "  python predict.py --model-path ./checkpoints/llava-fastvithd_0.5b_stage3 --image-file ./llava/serve/examples/waterview.jpg --prompt 'Describe the image.'" -ForegroundColor White





