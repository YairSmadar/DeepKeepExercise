import argparse
from model_api import TorchModelHandler, OnnxModelHandler, ModelComparator


def parser() -> argparse.Namespace:
    """
    Parses command-line arguments.

    Returns:
        argparse.Namespace: The parsed command-line arguments.
    """
    # Parse command-line arguments
    p = argparse.ArgumentParser(description='Convert PyTorch model to ONNX and compare outputs.')
    p.add_argument('--image', type=str, default='sample_image.png', help='Path to the input image')
    p.add_argument('--onnx_output_path', type=str, default='model.onnx', help='Path to save the ONNX model')
    p.add_argument('--visualize', action='store_true', help='Flag to visualize the outputs')
    p.add_argument('--model_name', type=str, default='deeplabv3_mobilenet_v3_large',
                   help='Required torch model name to be converted to ONNX')

    return p.parse_args()


def main():
    """
    Main function to convert a PyTorch model to ONNX, compare outputs, and optionally visualize differences.
    """
    args = parser()

    # Load the pretrained PyTorch model
    torch_handler = TorchModelHandler(model_path=args.onnx_output_path, model_name=args.model_name)

    # Convert the PyTorch model to ONNX
    onnx_handler = OnnxModelHandler(model_path=args.onnx_output_path, model_name=args.model_name)
    onnx_handler.convert_to_onnx(model=torch_handler.model)

    # Compare the PyTorch model with the ONNX model
    comparator = ModelComparator(torch_handler=torch_handler,
                                 onnx_handler=onnx_handler)
    torch_output, ort_output = comparator.compare_models(args.image)

    # Visualize the difference between outputs if requested
    if args.visualize:
        comparator.visualize_difference(torch_output, ort_output)


if __name__ == "__main__":
    main()
