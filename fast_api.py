import argparse
import os.path

from fastapi import FastAPI, File, UploadFile
from fastapi.responses import HTMLResponse
from PIL import Image
import io
import uvicorn
import numpy as np
import base64
from io import BytesIO

from model_api import OnnxModelHandler, TorchModelHandler
from wrapper_class import SegmentationModelAI


def create_app(model_handler: SegmentationModelAI) -> FastAPI:
    """
    Create and configure the FastAPI application.

    Parameters:
    ----------
    model_handler : SegmentationModelAI
        The model handler instance for segmentation.

    Returns:
    -------
    FastAPI
        The configured FastAPI application.
    """
    app = FastAPI()

    @app.get("/")
    def read_root():
        html_content = """
        <html>
            <body>
                <h2>Image Segmentation API</h2>
                <form action="/predict/" enctype="multipart/form-data" method="post">
                    <input name="file" type="file">
                    <input type="submit">
                </form>
            </body>
        </html>
        """
        return HTMLResponse(content=html_content)

    @app.post("/predict/")
    async def predict(file: UploadFile = File(...)):
        contents = await file.read()
        image = Image.open(io.BytesIO(contents))
        image = image.convert('RGB')  # Ensure compatibility

        # Run inference
        output = model_handler(image)
        output = output.squeeze(0)  # Remove batch dimension

        # Ensure the data type is suitable for Image.fromarray
        torch_pred = output.argmax(axis=0).astype(np.uint8)  # Convert to uint8 for image representation

        # Create a color palette for visualization
        # These numbers are large prime-like values chosen to create distinct colors when modulo 255 is applied.
        # The idea is that these large values, when multiplied by class indices and then taken modulo 255,
        # will generate unique and fairly evenly distributed colors across the spectrum.
        palette = np.array([2 ** 25 - 1, 2 ** 15 - 1, 2 ** 21 - 1])

        num_of_classes = output.shape[0]
        colors = np.array([i for i in range(num_of_classes)])[:, None] * palette
        colors = (colors % 255).astype("uint8")

        # Convert the model output to an image
        seg_img = Image.fromarray(torch_pred)
        seg_img = seg_img.convert("P")  # Convert to 'P' mode to apply a palette
        seg_img.putpalette(colors.flatten())  # Flatten the color array

        # Convert both images to base64 for embedding in HTML
        buffered_input = BytesIO()
        image.save(buffered_input, format="PNG")
        input_img_str = base64.b64encode(buffered_input.getvalue()).decode()

        buffered_output = BytesIO()
        seg_img.save(buffered_output, format="PNG")
        output_img_str = base64.b64encode(buffered_output.getvalue()).decode()

        html_content = f"""
        <html>
            <body>
                <h2>Image Segmentation Result</h2>
                <div>
                    <h3>Original Image</h3>
                    <img src="data:image/png;base64,{input_img_str}" alt="Original Image"/>
                </div>
                <div>
                    <h3>Segmentation Result</h3>
                    <img src="data:image/png;base64,{output_img_str}" alt="Segmentation Result"/>
                </div>
                <br>
                <a href="/">Upload another image</a>
            </body>
        </html>
        """
        return HTMLResponse(content=html_content)

    return app


def parser():
    p = argparse.ArgumentParser(description="Run FastAPI for model inference.")
    p.add_argument('--model_type', type=str, choices=['torch', 'onnx'], required=True,
                   help="Specify the model type: 'torch' for a PyTorch model, 'onnx' for an ONNX model.")
    p.add_argument('--model_path', type=str, required=False, default=None,
                   help="Path to the model file. Required if using an ONNX model.")
    p.add_argument("--model_name", type=str, required=False, default='deeplabv3_mobilenet_v3_large',
                   help="Model name. Required if using a torch model.")
    return p.parse_args()


def main():
    """
    Main function to setup SegmentationModelAI and start FastAPI app.
    """
    args = parser()

    assert not (args.model_type == 'onnx' and not os.path.exists(args.model_path)), \
        "When using ONNX as model type, the model_path must be correct.\n" \
        "To convert a PyTorch model to ONNX, use the pt_to_onnx.py script."

    model_types = {
        'onnx': OnnxModelHandler,
        'torch': TorchModelHandler
    }

    model_handler = model_types[args.model_type](model_path=args.model_path, model_name=args.model_name)
    segmentation_ai = SegmentationModelAI(model_handler)

    # Create the FastAPI app
    app = create_app(segmentation_ai)

    # Run the FastAPI app using Uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)


if __name__ == "__main__":
    main()
