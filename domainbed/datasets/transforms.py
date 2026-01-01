from torchvision import transforms as T
from PIL import Image
import torch

class ExtractGreenChannel:
    """
    Extract Green channel from RGB image (best vessel contrast in retinal images).
    Red channel is often oversaturated, Blue is dark/noisy.
    Green channel has highest vessel contrast.
    """
    def __call__(self, img):
        if isinstance(img, Image.Image):
            # Convert to RGB if not already
            img = img.convert('RGB')
            # Extract Green channel (index 1)
            green = img.split()[1]
            # Convert back to RGB for compatibility (grayscale as RGB)
            return green.convert('RGB')
        elif isinstance(img, torch.Tensor):
            # If already a tensor, extract Green channel (index 1)
            if img.shape[0] == 3:  # RGB tensor
                green = img[1:2, :, :]  # Keep only Green channel, maintain channel dimension
                return green.repeat(3, 1, 1)  # Repeat to 3 channels for compatibility
        return img

# Standard RGB transforms
basic = T.Compose(
    [
        T.Resize((224, 224)),
        T.ToTensor(),
        #T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ]
)

aug = T.Compose(
    [
        T.Resize((224, 224)),
        T.ToTensor(),
        #T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ]
)

# Green channel transforms (for better vessel visibility)
green_channel_basic = T.Compose(
    [
        T.Resize((224, 224)),
        ExtractGreenChannel(),
        T.ToTensor(),
    ]
)

green_channel_aug = T.Compose(
    [
        T.Resize((224, 224)),
        ExtractGreenChannel(),
        T.ToTensor(),
    ]
)

