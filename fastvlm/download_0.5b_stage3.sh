#!/bin/bash
# Bash script to download FastVLM 0.5B Stage 3 model
# Run this from the fastvlm directory: bash download_0.5b_stage3.sh

echo "Downloading FastVLM 0.5B Stage 3 model..."
echo "This may take a few minutes (model size: ~1.15 GB)"

# Create checkpoints directory if it doesn't exist
mkdir -p checkpoints

# Download the model
cd checkpoints
wget https://ml-site.cdn-apple.com/datasets/fastvlm/llava-fastvithd_0.5b_stage3.zip

# Extract the model
echo ""
echo "Extracting model..."
unzip -q llava-fastvithd_0.5b_stage3.zip

# Clean up zip file
rm llava-fastvithd_0.5b_stage3.zip
cd ..

echo ""
echo "✓ Model downloaded successfully!"
echo "Model location: checkpoints/llava-fastvithd_0.5b_stage3/"
echo ""
echo "You can now run:"
echo "  python predict.py --model-path ./checkpoints/llava-fastvithd_0.5b_stage3 --image-file ./llava/serve/examples/waterview.jpg --prompt 'Describe the image.'"





