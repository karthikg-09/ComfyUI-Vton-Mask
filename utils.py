import torch
import numpy as np
from PIL import Image

def resize_image(image, target_size=768):
    """Resize image while maintaining aspect ratio"""
    width, height = image.size
    if width > height:
        new_width = target_size
        new_height = int((height / width) * target_size)
    else:
        new_height = target_size
        new_width = int((width / height) * target_size)
    
    return image.resize((new_width, new_height), Image.LANCZOS)

def tensor_to_pil(tensor_image):
    """Convert tensor to PIL image"""
    # Handle ComfyUI format (batch, height, width, channels)
    if tensor_image.dim() == 4:
        tensor_image = tensor_image.squeeze(0)
    
    # Convert to numpy and scale to 0-255
    np_image = tensor_image.cpu().numpy()
    if np_image.max() <= 1.0:
        np_image = (np_image * 255).astype(np.uint8)
    else:
        np_image = np_image.astype(np.uint8)
    
    return Image.fromarray(np_image)

def pil_to_tensor(pil_image):
    """Convert PIL image to tensor in ComfyUI format (batch, height, width, channels)"""
    np_image = np.array(pil_image)
    
    # Ensure 3 channels
    if len(np_image.shape) == 2:
        np_image = np.stack([np_image] * 3, axis=-1)
    elif np_image.shape[2] == 4:
        np_image = np_image[:, :, :3]
    
    # Normalize to 0-1
    if np_image.dtype == np.uint8:
        np_image = np_image.astype(np.float32) / 255.0
    
    # Convert to tensor - ComfyUI expects (batch, height, width, channels)
    tensor = torch.from_numpy(np_image)
    
    # Add batch dimension: (height, width, channels) -> (1, height, width, channels)
    return tensor.unsqueeze(0) 