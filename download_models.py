#!/usr/bin/env python3
"""
Download script for ComfyUI-Vton-Mask models
Downloads the necessary preprocessing models from HuggingFace
"""

import os
import sys
import argparse
from pathlib import Path

try:
    from huggingface_hub import snapshot_download
except ImportError:
    print("huggingface_hub not found. Installing...")
    import subprocess
    subprocess.check_call([sys.executable, "-m", "pip", "install", "huggingface_hub"])
    from huggingface_hub import snapshot_download

def download_models(comfyui_path=None):
    """Download ComfyUI-Vton-Mask models"""
    
    if comfyui_path is None:
        # Try to find ComfyUI path
        possible_paths = [
            "../../models/ComfyUI-Vton-Mask",  # Relative from custom_nodes
            "../../../models/ComfyUI-Vton-Mask",  # Alternative
            "./models/ComfyUI-Vton-Mask"  # Current directory
        ]
        
        for path in possible_paths:
            if os.path.exists(os.path.dirname(path)):
                comfyui_path = path
                break
        
        if comfyui_path is None:
            comfyui_path = "./models/ComfyUI-Vton-Mask"
            
    # Create models directory
    models_dir = Path(comfyui_path)
    models_dir.mkdir(parents=True, exist_ok=True)
    
    print(f"Downloading models to: {models_dir.absolute()}")
    
    try:
        # Download from kg-09/kg-vton-mask repo
        print("Downloading ComfyUI-Vton-Mask models...")
        snapshot_download(
            repo_id="kg-09/kg-vton-mask",
            local_dir=str(models_dir),
            local_dir_use_symlinks=False
        )
        
        print("Models downloaded successfully")
        print(f"Models location: {models_dir.absolute()}")
        print("\nDownloaded files:")
        
        # List downloaded files
        for root, dirs, files in os.walk(models_dir):
            for file in files:
                if file.endswith(('.onnx', '.json', '.pth', '.pt')):
                    rel_path = os.path.relpath(os.path.join(root, file), models_dir)
                    size = os.path.getsize(os.path.join(root, file)) / (1024*1024)
                    print(f"  {rel_path} ({size:.1f}MB)")
        
        print("\nReady to use ComfyUI-Vton-Mask")
        print("Load the comfyui_vton_mask_workflow.json in ComfyUI to get started")
        
    except Exception as e:
        print(f"Error downloading models: {e}")
        print("Try downloading manually from: https://huggingface.co/kg-09/kg-vton-mask")
        return False
    
    return True

def main():
    parser = argparse.ArgumentParser(description="Download ComfyUI-Vton-Mask models")
    parser.add_argument("--path", type=str, help="Path to ComfyUI models directory")
    parser.add_argument("--list", action="store_true", help="List required models without downloading")
    
    args = parser.parse_args()
    
    if args.list:
        print("Required models for ComfyUI-Vton-Mask:")
        print("  humanparsing/parsing_atr.onnx (~300MB)")
        print("  humanparsing/parsing_lip.onnx (~300MB)")
        print("  dwpose model files (~100-200MB)")
        print("\nTotal size: ~500MB-800MB")
        print("Source: https://huggingface.co/kg-09/kg-vton-mask")
        return
    
    print("ComfyUI-Vton-Mask Model Downloader")
    print("=" * 40)
    
    success = download_models(args.path)
    
    if success:
        print("\nDownload complete. You can now use ComfyUI-Vton-Mask in ComfyUI")
    else:
        print("\nDownload failed. Please check your internet connection and try again.")
        sys.exit(1)

if __name__ == "__main__":
    main() 