# Developer Guidelines

This document provides instructions and guidelines for working on the Image Analysis Toolkit.

## Project Structure

- `nodes/`: Contains the implementation of the ComfyUI nodes. Each node is typically in its own file.
- `web/`: Contains frontend assets (JS/CSS) if applicable.
- `tests/`: Contains tests. (Note: If this directory is missing, please create it).
- `requirements.txt`: Python dependencies.

## Coding Standards

### ComfyUI V3 Node Schema

This project uses the ComfyUI V3 Node Schema.
- Nodes should inherit from `io.ComfyNode`.
- Inputs are defined using `define_schema` class method.
- The execution logic is in the `execute` class method.
- Return types and names are defined in `RETURN_TYPES` and `RETURN_NAMES`.

Example:
```python
import comfy.io as io

class MyNode(io.ComfyNode):
    @classmethod
    def define_schema(cls):
        return io.Schema({
            "image": io.Image.Input(),
            # ...
        })
    # ...
```

### Image Data

- Images are passed as PyTorch tensors.
- Shape: `[Batch, Height, Width, Channels]` (B, H, W, C).
- Values are typically normalized float32 between 0.0 and 1.0.
- When processing with OpenCV, convert to NumPy, remove batch dimension (if batch=1), and scale to 0-255 uint8.

## Testing

- Tests are located in the `tests/` directory.
- Run tests using: `python -m pytest`
- **Mocking `comfy`**: The `comfy` package (specifically `comfy.io`) is not available in the development environment. It is mocked in `tests/mocks/comfy` (or should be mocked if writing new tests). Ensure your tests set up `sys.path` or mocks correctly to handle `import comfy.io`.

## Dependencies

- Ensure dependencies are listed in `requirements.txt`.
- Main libraries: `numpy`, `opencv-python`, `torch`, `matplotlib`, `scikit-learn`, `Pillow`.
