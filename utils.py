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
    if tensor_image.dim() == 4:
        tensor_image = tensor_image.squeeze(0)
    
    if tensor_image.dim() == 3 and tensor_image.shape[0] == 3:
        tensor_image = tensor_image.permute(1, 2, 0)
    
    # Convert to numpy and scale to 0-255
    np_image = tensor_image.cpu().numpy()
    if np_image.max() <= 1.0:
        np_image = (np_image * 255).astype(np.uint8)
    else:
        np_image = np_image.astype(np.uint8)
    
    return Image.fromarray(np_image)

def pil_to_tensor(pil_image):
    """Convert PIL image to tensor"""
    np_image = np.array(pil_image)
    
    # Ensure 3 channels
    if len(np_image.shape) == 2:
        np_image = np.stack([np_image] * 3, axis=-1)
    elif np_image.shape[2] == 4:
        np_image = np_image[:, :, :3]
    
    # Normalize to 0-1
    if np_image.dtype == np.uint8:
        np_image = np_image.astype(np.float32) / 255.0
    
    # Convert to tensor and rearrange dimensions
    tensor = torch.from_numpy(np_image)
    if tensor.dim() == 3:
        tensor = tensor.permute(2, 0, 1)
    
    # Add batch dimension
    return tensor.unsqueeze(0) 