import torch
import numpy as np
from PIL import Image
import os
from .utils import resize_image, tensor_to_pil, pil_to_tensor
from .preprocess.humanparsing.run_parsing import Parsing
from .preprocess.dwpose import DWposeDetector
from .src.utils_mask import get_mask_location
import folder_paths

def download_models_if_needed():
    """Automatically download models from HuggingFace if not present"""
    models_dir = os.path.join(folder_paths.models_dir, "ComfyUI-Vton-Mask")
    
    # Check if essential model files exist
    required_files = [
        "humanparsing/parsing_atr.onnx",
        "humanparsing/parsing_lip.onnx"
    ]
    
    missing_files = []
    for file_path in required_files:
        full_path = os.path.join(models_dir, file_path)
        if not os.path.exists(full_path):
            missing_files.append(file_path)
    
    if missing_files:
        print("[ComfyUI-Vton-Mask] Models not found, downloading automatically...")
        print(f"[ComfyUI-Vton-Mask] Download location: {models_dir}")
        
        try:
            # Try to import huggingface_hub
            try:
                from huggingface_hub import snapshot_download
            except ImportError:
                print("[ComfyUI-Vton-Mask] Installing huggingface_hub...")
                import subprocess
                import sys
                subprocess.check_call([sys.executable, "-m", "pip", "install", "huggingface_hub"])
                from huggingface_hub import snapshot_download
            
            # Create models directory
            os.makedirs(models_dir, exist_ok=True)
            
            # Download from user's repo
            print("[ComfyUI-Vton-Mask] Downloading models from kg-09/kg-vton-mask...")
            snapshot_download(
                repo_id="kg-09/kg-vton-mask",
                local_dir=models_dir,
                local_dir_use_symlinks=False
            )
            
            print("[ComfyUI-Vton-Mask] Models downloaded successfully")
            return True
            
        except Exception as e:
            print(f"[ComfyUI-Vton-Mask] Error downloading models: {e}")
            print("[ComfyUI-Vton-Mask] Please download models manually from: https://huggingface.co/kg-09/kg-vton-mask")
            print(f"[ComfyUI-Vton-Mask] Place them in: {models_dir}")
            raise Exception(f"Failed to download models: {e}")
    
    return True

class ComfyUIVtonMaskLoader:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "device": ("STRING", {"default": "cpu"}),
            }
        }
    
    RETURN_TYPES = ("COMFYUI_VTON_MASK_MODEL",)
    RETURN_NAMES = ("mask_model",)
    FUNCTION = "load_mask_model"
    CATEGORY = "ComfyUI-Vton-Mask"

    def load_mask_model(self, device="cpu"):
        # Auto-download models if needed
        download_models_if_needed()
        
        # Only load models necessary for masking
        models_dir = os.path.join(folder_paths.models_dir, "ComfyUI-Vton-Mask")
        
        print("[ComfyUI-Vton-Mask] Loading models...")
        try:
            mask_model = {
                'dwprocessor': DWposeDetector(
                    model_root=models_dir, 
                    device=device
                ),
                'parsing_model': Parsing(
                    model_root=models_dir, 
                    device=device
                )
            }
            print("[ComfyUI-Vton-Mask] Models loaded successfully")
            return (mask_model,)
        except Exception as e:
            print(f"[ComfyUI-Vton-Mask] Error loading models: {e}")
            print(f"[ComfyUI-Vton-Mask] Expected model location: {models_dir}")
            print("[ComfyUI-Vton-Mask] Make sure models are properly downloaded")
            raise e

class ComfyUIVtonMaskGenerator:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "mask_model": ("COMFYUI_VTON_MASK_MODEL",),
                "vton_image": ("IMAGE",),
                "category": (["Upper-body", "Lower-body", "Dresses"],),
                "offset_top": ("INT", {"default": 0, "min": -200, "max": 200}),
                "offset_bottom": ("INT", {"default": 0, "min": -200, "max": 200}),
                "offset_left": ("INT", {"default": 0, "min": -200, "max": 200}),
                "offset_right": ("INT", {"default": 0, "min": -200, "max": 200}),
            }
        }

    RETURN_TYPES = ("IMAGE", "IMAGE", "IMAGE")
    RETURN_NAMES = ("masked_image", "mask", "pose_image")
    FUNCTION = "generate_mask"
    CATEGORY = "ComfyUI-Vton-Mask"

    def generate_mask(self, mask_model, vton_image, category, offset_top, offset_bottom, offset_left, offset_right):
        with torch.inference_mode():
            # Convert input image format
            vton_img = tensor_to_pil(vton_image)
            vton_img_det = resize_image(vton_img)
            
            # Generate pose information
            pose_image, keypoints, _, candidate = mask_model['dwprocessor'](np.array(vton_img_det)[:,:,::-1])
            candidate[candidate<0]=0
            candidate = candidate[0]
            
            candidate[:, 0]*=vton_img_det.width
            candidate[:, 1]*=vton_img_det.height
            
            # Process pose image
            pose_image = pose_image[:,:,::-1]
            pose_image = Image.fromarray(pose_image)
            
            # Generate parsing results
            model_parse, _ = mask_model['parsing_model'](vton_img_det)
            
            # Generate mask
            mask, mask_gray = get_mask_location(
                category, 
                model_parse,
                candidate, 
                model_parse.width, 
                model_parse.height,
                offset_top, 
                offset_bottom, 
                offset_left, 
                offset_right
            )
            
            # Resize masks
            mask = mask.resize(vton_img.size)
            mask_gray = mask_gray.resize(vton_img.size)
            # Composite masked image
            masked_vton_img = Image.composite(mask_gray, vton_img, mask)
            return (pil_to_tensor(masked_vton_img), pil_to_tensor(mask.convert("RGB")), pil_to_tensor(pose_image)) 