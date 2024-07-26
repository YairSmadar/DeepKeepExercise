from __future__ import annotations

import argparse
import numpy as np

from image_api import DeepLabV3image
from model_api import BaseModelHandler, TorchModelHandler, OnnxModelHandler


class SegmentationModelAI:
    """
    A class to perform inference using a segmentation model.

    Attributes:
    ----------
    model_handler : ModelHandler
        The ModelHandler instance managing the model.
    """

    def __init__(self, model_handler: BaseModelHandler):
        """
        Initialize the SegmentationModelAI with a ModelHandler.

        Parameters:
        ----------
        model_handler : ModelHandler
            The ModelHandler instance managing the model.
        """
        self.model_handler = model_handler

    def __call__(self, image) -> np.ndarray:
        """
        Perform inference on the input image.

        Parameters:
        ----------
        image : str | Image.Image | torch.Tensor | np.ndarray
            The input image in various formats.

        Returns:
        -------
        np.ndarray
            The output of the model after inference.
        """
        image_class_types = \
            {
                'deeplabv3_mobilenet_v3_large': DeepLabV3image
            }

        image_manager = image_class_types[self.model_handler.model_name](image)
        input_tensor = image_manager.preprocess()
        return self.model_handler.run_inference(input_tensor)


def main(args):
    """
    The main function to run the segmentation inference.

    Parameters:
    ----------
    args : argparse.Namespace
        Command-line arguments passed to the script.
    """

    assert not (args.model_path is None and args.model_type == 'onnx'), \
        "When using onnx as model type, the model_path must be provided."

    model_types = \
        {
            'onnx': OnnxModelHandler,
            'torch': TorchModelHandler
        }

    model_handler = model_types[args.model_type](model_path=args.model_path, model_name=args.model_name)
    segmentation_ai = SegmentationModelAI(model_handler)
    output = segmentation_ai(args.image_path)
    print(f"Model Output Shape: {output.shape}")


def parser():
    p = argparse.ArgumentParser(description="Image Segmentation Model Inference")
    p.add_argument("--model_type", type=str, choices=['torch', 'onnx'], required=True,
                   help="Specify the model type: 'torch' for a PyTorch model, 'onnx' for an ONNX model.")
    p.add_argument("--model_path", type=str, required=False, default=None,
                   help="Path to the model file. Required if using an ONNX model.")
    p.add_argument("--model_name", type=str, required=False, default='deeplabv3_mobilenet_v3_large',
                   help="model name. Required if using an torch model.")
    p.add_argument("--image_path", type=str, required=True,
                   help="Path to the input image file or other supported formats.")

    return p.parse_args()


if __name__ == "__main__":
    args = parser()

    main(args)
