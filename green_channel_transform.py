"""
Green Channel Transform for Retinal Images
Extracts only the Green channel (highest vessel contrast) from RGB images
"""
from torchvision import transforms as T
from PIL import Image
import torch

class ExtractGreenChannel:
    """Extract Green channel from RGB image"""
    def __call__(self, img):
        if isinstance(img, Image.Image):
            # Convert to RGB if not already
            img = img.convert('RGB')
            # Extract Green channel (index 1)
            img = img.split()[1]
            # Convert back to RGB for compatibility (grayscale as RGB)
            img = img.convert('RGB')
        elif isinstance(img, torch.Tensor):
            # If already a tensor, extract Green channel (index 1)
            if img.shape[0] == 3:  # RGB tensor
                img = img[1:2, :, :]  # Keep only Green channel, maintain channel dimension
                img = img.repeat(3, 1, 1)  # Repeat to 3 channels for compatibility
        return img

# Transform with Green channel extraction
green_channel_basic = T.Compose([
    T.Resize((224, 224)),
    ExtractGreenChannel(),
    T.ToTensor(),
])

green_channel_aug = T.Compose([
    T.Resize((224, 224)),
    ExtractGreenChannel(),
    T.ToTensor(),
])

