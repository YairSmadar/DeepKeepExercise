from __future__ import annotations

import os
import time
from abc import abstractmethod, ABC

import onnxruntime
import torch
import numpy as np
import torchvision
from matplotlib import pyplot as plt

from image_api import DeepLabV3image


class BaseModelHandler(ABC):
    """
    Abstract base class for handling model loading and inference.
    """

    def __init__(self, model_path: str = None, model_name: str = None):
        """

        :param model_path_or_name: For local models - model path
                                   For online model (downloaded model) - model name
        """

        self.model_path = model_path
        self.model_name = model_name
        self.model = self.load_model()

    @abstractmethod
    def load_model(self):
        """
        Load the model. Must be implemented by subclasses.

        Returns:
        -------
        The loaded model.
        """
        pass

    @abstractmethod
    def run_inference(self, input_tensor: torch.Tensor | np.ndarray) -> np.ndarray:
        """
        Run inference using the loaded model. Must be implemented by subclasses.

        Parameters:
        ----------
        input_tensor : torch.Tensor
            The preprocessed input tensor.

        Returns:
        -------
        np.ndarray
            The output of the model after inference.
        """
        pass


class TorchModelHandler(BaseModelHandler):
    """
    Class for handling PyTorch model loading and inference.
    """

    def load_model(self) -> torch.nn.Module:
        """
        Load a pre-trained DeepLabV3 model with a MobileNetV3-Large backbone.

        Returns:
        -------
        torch.nn.Module
            The loaded PyTorch model.
        """
        # Load the pre-trained DeepLabV3 model with MobileNetV3-Large backbone
        model = self._load_model_by_name()
        model.eval()  # Set the model to evaluation mode

        return model

    def _load_model_by_name(self, is_pretrained: bool = True) -> torch.nn.Module:
        """
        Retrieves a specified PyTorch model.

        Args:
            name (str, optional): The name of the model to retrieve. Defaults to 'deeplabv3_mobilenet_v3_large'.

        Returns:
            torch.nn.Module: The specified PyTorch model.
            :param is_pretrained: Whether to use pretrained weights
        """
        models = {
            'deeplabv3_mobilenet_v3_large': torchvision.models.segmentation.deeplabv3_mobilenet_v3_large(
                pretrained=is_pretrained)
        }

        return models[self.model_name]

    def run_inference(self, input_tensor: torch.Tensor | np.ndarray) -> np.ndarray:
        """
        Run inference using the PyTorch model.

        Parameters:
        ----------
        input_tensor : torch.Tensor
            The preprocessed input tensor.

        Returns:
        -------
        np.ndarray
            The output of the model after inference.
        """
        with torch.no_grad():
            output = self.model(input_tensor)['out']

        return output.numpy()


class OnnxModelHandler(BaseModelHandler):
    """
    Class for handling ONNX model loading and inference.
    """

    def load_model(self) -> onnxruntime.InferenceSession | None:
        """
        Load an ONNX model from a file.

        Returns:
            if the model exist: The loaded ONNX model,
            else:  None
        -------
        onnxruntime.InferenceSession
            The loaded ONNX model.
        """
        if self.model_path is None:
            raise ValueError("Model path must be provided for ONNX models.")

        # Load from existing file
        if os.path.exists(self.model_path):
            return onnxruntime.InferenceSession(self.model_path)

        return None

    def run_inference(self, input_tensor: torch.Tensor | np.ndarray) -> np.ndarray:
        """
        Run inference using the ONNX model.

        Parameters:
        ----------
        input_tensor : torch.Tensor
            The preprocessed input tensor.

        Returns:
        -------
        np.ndarray
            The output of the model after inference.
        """
        if isinstance(input_tensor, torch.Tensor):
            # convert input to numpy
            input_tensor = {self.model.get_inputs()[0].name: input_tensor.numpy()}

        return self.model.run(None, input_tensor)[0]

    def convert_to_onnx(self, model: torch.nn.Module,
                        dummy_input_shape: tuple = (1, 3, 224, 224)):
        """
        Converts the PyTorch model to the ONNX format and saves it to the specified path.

        Args:
            output_path (str): The file path to save the ONNX model.
            :param dummy_input_shape: dummy input shape according to the desired ONNX model input shape
            :param output_path: ONNX file output path
            :param model: torch model to convert yo ONNX
        """
        model.eval()
        dummy_input = torch.randn(*dummy_input_shape)
        torch.onnx.export(model, dummy_input, self.model_path,
                          input_names=['input'], output_names=['output'])
        print(f"Model converted to ONNX and saved to {self.model_path}")

        self.model = onnxruntime.InferenceSession(self.model_path)


class ModelComparator:
    """
    A class to compare the outputs of a PyTorch model and its ONNX equivalent.

    Attributes:
        torch_model (torch.nn.Module): The PyTorch model.
        ort_session (onnxruntime.InferenceSession): The ONNX Runtime session for the ONNX model.
    """

    def __init__(self, torch_handler: TorchModelHandler, onnx_handler: OnnxModelHandler):
        """
        Initializes the ModelComparator with the specified PyTorch model and ONNX model path.

        Args:
            torch_model (torch.nn.Module): The PyTorch model.
            onnx_model_path (str): The file path to the ONNX model.
        """
        self.torch_handler = torch_handler
        self.onnx_handler = onnx_handler

    def compare_models(self, image_path: str) -> tuple:
        """
        Compares the outputs of the PyTorch and ONNX models for the given input image.

        Args:
            image_path (str): The file path to the input image.

        Returns:
            tuple: The outputs of the PyTorch model and ONNX model.
        """

        assert self.torch_handler.model_name == self.onnx_handler.model_name, \
            f"torch model name ({self.torch_handler.model_name}) must be the same as " \
            f"onnx model name ({self.onnx_handler.model_name})"

        image_class_types = \
            {
                'deeplabv3_mobilenet_v3_large': DeepLabV3image
            }

        image_manager = image_class_types[self.torch_handler.model_name](image_path)
        input_tensor = image_manager.preprocess()

        # Measure runtime for PyTorch model
        start_time = time.time()
        with torch.no_grad():
            torch_output = self.torch_handler.run_inference(input_tensor)
        torch_runtime = time.time() - start_time
        print(f"PyTorch model runtime: {torch_runtime:.4f} seconds")

        # Measure runtime for ONNX model
        start_time = time.time()

        ort_output = self.onnx_handler.run_inference(input_tensor)
        ort_runtime = time.time() - start_time
        print(f"ONNX model runtime: {ort_runtime:.4f} seconds")

        l2_diff = np.linalg.norm(torch_output - ort_output)
        print(f"L2 difference: {l2_diff}")

        return torch_output, ort_output

    @staticmethod
    def visualize_difference(torch_output: np.ndarray, ort_output: np.ndarray):
        """
        Visualizes the differences between the outputs of the PyTorch and ONNX models.

        Args:
            torch_output (np.ndarray): The output from the PyTorch model.
            ort_output (np.ndarray): The output from the ONNX model.
        """
        # Remove batch dimension and take the class with highest score
        torch_pred = torch_output.argmax(axis=1)[0]
        ort_pred = ort_output.argmax(axis=1)[0]

        # get classes diff
        diff = np.abs(torch_pred - ort_pred)

        plt.figure(figsize=(15, 5))
        plt.subplot(1, 3, 1)
        plt.title('PyTorch Output')
        plt.imshow(torch_pred, cmap='gray')
        plt.subplot(1, 3, 2)
        plt.title('ONNX Output')
        plt.imshow(ort_pred, cmap='gray')
        plt.subplot(1, 3, 3)
        plt.title('Class Difference')
        plt.imshow(diff, cmap='hot')
        plt.colorbar()
        plt.show()
