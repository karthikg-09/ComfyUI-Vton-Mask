from .kg_vton_mask_nodes import ComfyUIVtonMaskLoader, ComfyUIVtonMaskGenerator

NODE_CLASS_MAPPINGS = {
    "ComfyUIVtonMaskLoader": ComfyUIVtonMaskLoader,
    "ComfyUIVtonMaskGenerator": ComfyUIVtonMaskGenerator
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "ComfyUIVtonMaskLoader": "Load ComfyUI-Vton-Mask Model",
    "ComfyUIVtonMaskGenerator": "Generate ComfyUI-Vton-Mask"
}

__all__ = ['NODE_CLASS_MAPPINGS', 'NODE_DISPLAY_NAME_MAPPINGS'] 