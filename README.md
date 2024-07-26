
---

# Image Segmentation with FastAPI

This project provides a FastAPI application for image segmentation using pre-trained models. It includes functionality for converting PyTorch models to ONNX format, comparing model outputs, and serving predictions via a web interface.

## Table of Contents
1. [Project Structure](#project-structure)
2. [Setup Instructions](#setup-instructions)
3. [Usage](#usage)
   - [Model Conversion and Comparison](#model-conversion-and-comparison)
   - [Running the FastAPI Application](#running-the-fastapi-application)
   - [Interacting with the Web Interface](#interacting-with-the-web-interface)
4. [Files Overview](#files-overview)

## Project Structure

- **fast_api.py**: Contains the FastAPI application for friendly use the segmentation model.
- **image_api.py**: Defines image handling and preprocessing classes.
- **model_api.py**: Handles model loading and inference for PyTorch and ONNX models.
- **pt_to_onnx.py**: Script to convert PyTorch models to ONNX and compare their outputs.
- **wrapper_class.py**: Contains the `SegmentationModelAI` class that wraps model inference.


## Setup Instructions

1. **Clone the repository**:
   ```sh
   git clone https://github.com/YairSmadar/DeepKeepExercise.git
   cd DeepKeepExercise
   ```

2. **Create a virtual environment**:
   - **On Linux/macOS**:
     ```sh
     python -m venv venv
     source venv/bin/activate
     ```
   - **On Windows**:
     ```cmd
     python -m venv venv
     venv\Scripts\activate
     ```

3. **Install dependencies**:
   ```sh
   pip install -r requirements.txt
   ```


## Usage

### Model Conversion and Comparison

To convert a PyTorch model to ONNX format and compare the outputs:

```sh
python pt_to_onnx.py --image sample_image.png --onnx_output_path model.onnx --visualize --model_name deeplabv3_mobilenet_v3_large
```

- **--image**: Path to the input image for comparison.
- **--onnx_output_path**: Path to save the converted ONNX model.
- **--visualize**: Flag to visualize the outputs.
- **--model_name**: Name of the PyTorch model to convert.

### Running the FastAPI Application

To start the FastAPI server:

```sh
python fast_api.py --model_type onnx --model_path path/to/your/model.onnx
```

- **--model_type**: Choose between 'torch' and 'onnx' for the model type.
- **--model_path**: Path to the ONNX model file (required for ONNX models).

### Interacting with the Web Interface

1. **Access the Web Interface**:
   Open your web browser and navigate to `http://127.0.0.1:8000/`.

2. **Upload an Image**:
   Use the provided form to upload an image. The server will process the image and display both the original and segmented images.

## Files Overview

### fast_api.py

Contains the FastAPI app which:
- Serves an HTML form for image upload.
- Processes uploaded images to produce and display segmentation results.

### image_api.py

Defines classes for handling and preprocessing images:
- **ImageManager**: Base class for image handling.
- **DeepLabV3image**: Specific implementation for the DeepLabV3 model.

### model_api.py

Includes classes for managing model loading and inference:
- **BaseModelHandler**: Abstract base class for model handlers.
- **TorchModelHandler**: Handles PyTorch model operations.
- **OnnxModelHandler**: Handles ONNX model operations and conversion.

### pt_to_onnx.py

A script for converting PyTorch models to ONNX format, comparing outputs, and optionally visualizing differences.

### wrapper_class.py

Contains the `SegmentationModelAI` class, which wraps the model handler for unified inference interface.

---