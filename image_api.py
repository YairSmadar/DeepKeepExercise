from abc import ABC, abstractmethod
from typing import Union

import numpy as np
import torch
from PIL import Image, ImageFile, UnidentifiedImageError
import torchvision.transforms as transforms


class ImageManager:
    """
    A class to handle image loading and preprocessing for model input.
    """
    def __init__(self, image_source: Union[str, Image.Image, ImageFile.ImageFile, torch.Tensor, np.ndarray]):
        """
        Initialize the ImageManager with an image source.

        Parameters:
        ----------
        image_source : str | Image.Image | torch.Tensor | np.ndarray
            The source of the image, which can be a file path, a PIL image,
            a Torch tensor, or a NumPy array.
        """
        self.image = self._load_image(image_source)

    @staticmethod
    def _load_image(image_source: Union[str, Image.Image, ImageFile.ImageFile, torch.Tensor, np.ndarray]) -> Image.Image:
        """
        Load an image from various formats into a PIL Image.

        Parameters:
        ----------
        image_source : str | Image.Image | torch.Tensor | np.ndarray
            The source of the image.

        Returns:
        -------
        Image.Image
            The loaded image in PIL format.
        """
        if isinstance(image_source, str):
            try:
                return Image.open(image_source).convert('RGB')
            except (FileNotFoundError, UnidentifiedImageError):
                try:
                    array = np.load(image_source)
                    return Image.fromarray(array)
                except Exception:
                    import cv2
                    image = cv2.imread(image_source)
                    if image is not None:
                        return Image.fromarray(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
                    raise ValueError(f"Unable to load image from the provided path {image_source}.")
        elif isinstance(image_source, (Image.Image, ImageFile.ImageFile)):
            return image_source.convert('RGB')
        elif isinstance(image_source, torch.Tensor):
            return transforms.ToPILImage()(image_source).convert('RGB')
        elif isinstance(image_source, np.ndarray):
            return Image.fromarray(image_source).convert('RGB')
        else:
            raise ValueError("Unsupported image format. Supported formats are file paths, "
                             "PIL images, Torch tensors, and NumPy arrays.")

    @abstractmethod
    def preprocess(self) -> torch.Tensor:
        """
        Preprocess the image for model input.

        Returns:
        -------
        torch.Tensor
            The preprocessed image as a tensor, ready for model input.
        """
        pass


class DeepLabV3image(ImageManager):

    def preprocess(self) -> torch.Tensor:
        """
        Preprocess the image for model input.

        Returns:
        -------
        torch.Tensor
            The preprocessed image as a tensor, ready for model input.
        """
        preprocess = transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])
        return preprocess(self.image).unsqueeze(0)  # Add batch dimension