import os
import time
import unittest
import requests
import numpy as np
import torch
from PIL import Image
from torchvision.models.segmentation import deeplabv3_mobilenet_v3_large
import subprocess

from model_api import OnnxModelHandler, TorchModelHandler, ModelComparator
from fast_api import create_app
from wrapper_class import SegmentationModelAI
import uvicorn
from multiprocessing import Process
from bs4 import BeautifulSoup


class TestModelConversion(unittest.TestCase):
    """Test the conversion of a PyTorch model to ONNX format."""

    def setUp(self):
        self.onnx_path = 'model.onnx'
        self.torch_model = deeplabv3_mobilenet_v3_large(pretrained=True).eval()

    def test_conversion(self):
        onnx_handler = OnnxModelHandler(model_path=self.onnx_path, model_name='deeplabv3_mobilenet_v3_large')
        onnx_handler.convert_to_onnx(model=self.torch_model)
        self.assertTrue(os.path.exists(self.onnx_path), "ONNX model was not created successfully.")

    def tearDown(self):
        if os.path.exists(self.onnx_path):
            os.remove(self.onnx_path)


class TestModelInference(unittest.TestCase):
    """Test inference on both PyTorch and ONNX models."""

    def setUp(self):
        self.dummy_image = torch.randn(1, 3, 224, 224)  # Create a dummy tensor
        self.onnx_path = 'model.onnx'
        self.torch_handler = TorchModelHandler(model_path=self.onnx_path, model_name='deeplabv3_mobilenet_v3_large')
        self.onnx_handler = OnnxModelHandler(model_path=self.onnx_path, model_name='deeplabv3_mobilenet_v3_large')
        self.onnx_handler.convert_to_onnx(model=self.torch_handler.model)

    def test_torch_inference(self):
        output = self.torch_handler.run_inference(self.dummy_image)
        self.assertIsNotNone(output, "No output from PyTorch model.")

    def test_onnx_inference(self):
        output = self.onnx_handler.run_inference(self.dummy_image)
        self.assertIsNotNone(output, "No output from ONNX model.")

    def test_compare_outputs(self):
        comparator = ModelComparator(self.torch_handler, self.onnx_handler)
        torch_out, onnx_out = comparator.compare_models('img.png')
        self.assertEqual(torch_out.shape, onnx_out.shape, "Output shapes from PyTorch and ONNX models differ.")

    def tearDown(self):
        if os.path.exists(self.onnx_path):
            os.remove(self.onnx_path)


class TestAPI(unittest.TestCase):
    api_url = "http://127.0.0.1:8000/predict/"
    image_path = "img.png"  # Ensure this path points to a valid image file
    server_process = None

    @classmethod
    def setUpClass(cls):
        # Ensure any previous server process is terminated
        cls.tearDownClass()
        # Start the FastAPI server in a subprocess
        cls.server_process = subprocess.Popen(['python', '-m', 'fast_api'])
        # Wait for the server to start
        time.sleep(5)  # Adjust this if your server needs more time to start

    @classmethod
    def tearDownClass(cls):
        # Shut down the FastAPI server
        if cls.server_process:
            cls.server_process.terminate()
            cls.server_process.wait()

    def test_api_response(self):
        with open(self.image_path, 'rb') as img:
            response = requests.post(self.api_url, files={'file': img})
        self.assertEqual(response.status_code, 200, "FastAPI response error.")

        # Parse HTML content
        soup = BeautifulSoup(response.content, 'html.parser')
        original_image = soup.find('img', {'alt': 'Original Image'})
        segmentation_result = soup.find('img', {'alt': 'Segmentation Result'})

        self.assertIsNotNone(original_image, "Original Image not found in response.")
        self.assertIsNotNone(segmentation_result, "Segmentation Result not found in response.")

    def test_api_no_file(self):
        response = requests.post('http://127.0.0.1:8000/predict/', files={'file': None})
        self.assertNotEqual(response.status_code, 200, "API did not handle empty file upload properly.")


class TestErrorHandling(unittest.TestCase):
    """Test the system's error handling."""

    def test_invalid_model_path(self):
        with self.assertRaises(AttributeError):
            onnx_handler = OnnxModelHandler(model_path='invalid_path.onnx')
            onnx_handler.run_inference(torch.randn(1, 3, 224, 224))

    def test_invalid_image_path(self):
        torch_handler = TorchModelHandler(model_path='model.onnx', model_name='deeplabv3_mobilenet_v3_large')
        onnx_handler = OnnxModelHandler(model_path='model.onnx', model_name='deeplabv3_mobilenet_v3_large')
        comparator = ModelComparator(torch_handler, onnx_handler)
        with self.assertRaises(ValueError):
            comparator.compare_models('invalid_image.png')


def run_server():
    """Run FastAPI server for testing."""
    app = create_app(SegmentationModelAI(TorchModelHandler(model_name='deeplabv3_mobilenet_v3_large')))
    uvicorn.run(app, host="0.0.0.0", port=8000)


if __name__ == "__main__":
    unittest.main()
