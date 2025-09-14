# ComfyUI Development & Custom Nodes Guide

> **🚀 Building custom nodes and extensions for ComfyUI on Windows**
>
> This guide covers ComfyUI development, custom node creation, extension management, and Windows-specific development practices.

## 📋 Table of Contents

- [Quick Start](#-quick-start)
- [Development Environment](#-development-environment)
- [Custom Node Development](#-custom-node-development)
- [Extension Management](#-extension-management)
- [Debugging & Testing](#-debugging--testing)
- [Performance Optimization](#-performance-optimization)
- [Deployment & Distribution](#-deployment--distribution)
- [Troubleshooting](#-troubleshooting)

## 🚀 Quick Start

### Environment Setup
```powershell
# Clone ComfyUI
git clone https://github.com/comfyanonymous/ComfyUI.git
cd ComfyUI

# Create virtual environment
python -m venv venv
.\venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu129
```

### Create Your First Custom Node
```python
# custom_nodes/my_first_node.py
class MyFirstNode:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "text": ("STRING", {"default": "Hello World"}),
            }
        }

    RETURN_TYPES = ("STRING",)
    FUNCTION = "process"
    CATEGORY = "My Nodes"

    def process(self, text):
        return (text.upper(),)
```

### Test Your Node
```powershell
# Start ComfyUI
python main.py

# Open browser to http://127.0.0.1:8188
# Your node should appear in the "My Nodes" category
```

## 🔧 Development Environment

### System Requirements
- **Python**: 3.10-3.12
- **PyTorch**: 2.6.0+ with CUDA support
- **Git**: For version control and cloning repositories
- **VS Code**: Recommended IDE with Python extensions

### Virtual Environment Setup
```powershell
# Create isolated environment
python -m venv comfyui_dev
.\comfyui_dev\Scripts\activate

# Install core dependencies
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu129
pip install accelerate transformers diffusers

# Install development tools
pip install pytest black flake8 mypy
```

### Project Structure
```
ComfyUI/
├── custom_nodes/
│   ├── my_extension/
│   │   ├── __init__.py
│   │   ├── nodes.py
│   │   ├── requirements.txt
│   │   └── README.md
├── models/
├── input/
├── output/
├── web/
└── main.py
```

## 🏗️ Custom Node Development

### Node Anatomy

#### Basic Node Structure
```python
class MyCustomNode:
    # Define input types
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "input_image": ("IMAGE",),
                "strength": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 2.0, "step": 0.1}),
            },
            "optional": {
                "mask": ("MASK",),
            }
        }

    # Define output types
    RETURN_TYPES = ("IMAGE",)
    RETURN_NAMES = ("processed_image",)

    # Node metadata
    FUNCTION = "process_image"
    CATEGORY = "Image Processing"
    DESCRIPTION = "Process images with custom algorithm"

    def process_image(self, input_image, strength, mask=None):
        # Your processing logic here
        processed = self.apply_custom_processing(input_image, strength, mask)
        return (processed,)
```

#### Advanced Node Features
```python
class AdvancedNode:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "model": ("MODEL",),
                "clip": ("CLIP",),
                "vae": ("VAE",),
            },
            "optional": {
                "lora": ("LORASTACK",),
                "control_net": ("CONTROL_NET",),
            },
            "hidden": {
                "unique_id": "UNIQUE_ID",
            }
        }

    RETURN_TYPES = ("MODEL", "CLIP", "VAE")
    RETURN_NAMES = ("model", "clip", "vae")

    FUNCTION = "apply_modifications"
    CATEGORY = "Model Utilities"

    def apply_modifications(self, model, clip, vae, lora=None, control_net=None, unique_id=None):
        # Apply LoRA if provided
        if lora:
            model = self.apply_lora(model, lora)

        # Apply ControlNet if provided
        if control_net:
            model = self.apply_controlnet(model, control_net)

        return (model, clip, vae)
```

### Node Categories

#### Standard Categories
- **Loaders**: Model, CLIP, VAE loading
- **Conditioning**: Text encoding, conditioning manipulation
- **Sampling**: KSampler, schedulers
- **Image**: Image processing, generation
- **Utils**: Helper functions, utilities

#### Custom Categories
```python
# Create your own category
CATEGORY = "My Custom Nodes/Image Processing"

# Or use subcategories
CATEGORY = "AI Tools/Super Resolution"
```

### Input/Output Types

#### Common Input Types
```python
INPUT_TYPES = {
    "required": {
        "text": ("STRING", {"default": "prompt", "multiline": True}),
        "steps": ("INT", {"default": 20, "min": 1, "max": 100, "step": 1}),
        "cfg": ("FLOAT", {"default": 7.0, "min": 1.0, "max": 20.0, "step": 0.5}),
        "seed": ("INT", {"default": 0, "min": 0, "max": 0xffffffffffffffff}),
        "model": ("MODEL",),
        "positive": ("CONDITIONING",),
        "negative": ("CONDITIONING",),
        "latent": ("LATENT",),
        "image": ("IMAGE",),
        "mask": ("MASK",),
    }
}
```

#### Common Output Types
```python
RETURN_TYPES = ("IMAGE", "MASK", "LATENT", "CONDITIONING")
RETURN_NAMES = ("image", "mask", "latent", "conditioning")
```

### Node Registration

#### Automatic Registration
```python
# __init__.py
from .nodes import MyNode1, MyNode2

NODE_CLASS_MAPPINGS = {
    "MyNode1": MyNode1,
    "MyNode2": MyNode2,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "MyNode1": "My Custom Node 1",
    "MyNode2": "My Custom Node 2",
}
```

#### Manual Registration
```python
# In your node file
import comfy.utils

class MyNode:
    # ... node definition ...

# Register the node
comfy.utils.register_node(MyNode, "My Custom Node")
```

## 📦 Extension Management

### Creating Extensions

#### Extension Structure
```
my_comfyui_extension/
├── __init__.py
├── nodes.py
├── requirements.txt
├── pyrightconfig.json
├── README.md
└── examples/
    └── workflow.json
```

#### Extension Metadata
```python
# __init__.py
from .nodes import *

__version__ = "1.0.0"
__description__ = "My ComfyUI extension for custom processing"
__author__ = "Your Name"
__license__ = "MIT"

NODE_CLASS_MAPPINGS = {
    "MyCustomNode": MyCustomNode,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "MyCustomNode": "My Custom Node",
}

WEB_DIRECTORY = "./web"  # Optional: for custom UI components
```

#### Requirements Management
```txt
# requirements.txt
torch>=2.0.0
torchvision>=0.15.0
transformers>=4.21.0
diffusers>=0.14.0
accelerate>=0.16.0
numpy>=1.21.0
Pillow>=9.0.0
```

### Installing Extensions

#### From GitHub
```powershell
# Clone into custom_nodes directory
cd ComfyUI/custom_nodes
git clone https://github.com/username/my-extension.git

# Install dependencies
cd my-extension
pip install -r requirements.txt
```

#### From Local Directory
```powershell
# Copy extension to custom_nodes
cp -r /path/to/my-extension ComfyUI/custom_nodes/

# Install dependencies
cd ComfyUI/custom_nodes/my-extension
pip install -r requirements.txt
```

#### Manager Installation
```python
# Using ComfyUI Manager (if installed)
# Search for your extension
# Click Install
```

### Extension Dependencies

#### Handling Dependencies
```python
# In your node file
try:
    import custom_library
except ImportError:
    raise ImportError("custom_library is required. Please install it with: pip install custom_library")

class MyNode:
    # ... node implementation ...
```

#### Optional Dependencies
```python
# Check for optional dependencies
try:
    import optional_library
    HAS_OPTIONAL = True
except ImportError:
    HAS_OPTIONAL = False

class MyNode:
    @classmethod
    def INPUT_TYPES(cls):
        inputs = {
            "required": {
                "input": ("IMAGE",),
            }
        }

        if HAS_OPTIONAL:
            inputs["optional"] = {
                "optional_param": ("FLOAT", {"default": 1.0}),
            }

        return inputs
```

## 🐛 Debugging & Testing

### Debug Tools

#### Logging
```python
import logging

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class MyNode:
    def process(self, input_data):
        logger.info(f"Processing input with shape: {input_data.shape}")

        try:
            result = self.do_processing(input_data)
            logger.info("Processing completed successfully")
            return (result,)
        except Exception as e:
            logger.error(f"Processing failed: {e}")
            raise
```

#### Node Validation
```python
class MyNode:
    @classmethod
    def VALIDATE_INPUTS(cls, **kwargs):
        # Validate inputs before processing
        if kwargs.get('strength', 1.0) < 0:
            return "Strength must be non-negative"

        if kwargs.get('image') is None:
            return "Image input is required"

        return True  # Inputs are valid
```

### Testing Framework

#### Unit Tests
```python
# tests/test_my_node.py
import pytest
import torch
import numpy as np
from my_nodes import MyNode

class TestMyNode:
    def setup_method(self):
        self.node = MyNode()

    def test_basic_processing(self):
        # Create test input
        test_image = torch.randn(1, 3, 512, 512)

        # Process
        result = self.node.process(test_image)

        # Assert
        assert result[0].shape == test_image.shape
        assert isinstance(result, tuple)

    def test_edge_cases(self):
        # Test with empty input
        with pytest.raises(ValueError):
            self.node.process(torch.empty(0, 3, 512, 512))

        # Test with wrong dimensions
        with pytest.raises(RuntimeError):
            self.node.process(torch.randn(1, 1, 512, 512))
```

#### Integration Tests
```python
# Test with ComfyUI workflow
def test_workflow_integration():
    # Load workflow JSON
    with open('test_workflow.json', 'r') as f:
        workflow = json.load(f)

    # Execute workflow
    results = execute_workflow(workflow)

    # Verify results
    assert 'output_image' in results
    assert results['output_image'].shape[0] == 1
```

### Debug Workflow

#### Step-by-Step Debugging
```python
# Add debug prints
def debug_node_execution(node, inputs):
    print(f"Executing node: {node.__class__.__name__}")
    print(f"Inputs: {list(inputs.keys())}")

    for key, value in inputs.items():
        if hasattr(value, 'shape'):
            print(f"  {key}: shape {value.shape}, dtype {value.dtype}")
        else:
            print(f"  {key}: {type(value)} = {value}")

    result = node.process(**inputs)
    print(f"Output: {type(result)}, length: {len(result) if isinstance(result, tuple) else 'N/A'}")

    return result
```

## ⚡ Performance Optimization

### Memory Management

#### GPU Memory Optimization
```python
import torch

class OptimizedNode:
    @torch.no_grad()
    def process(self, input_tensor):
        # Ensure input is on GPU
        if not input_tensor.is_cuda:
            input_tensor = input_tensor.cuda()

        # Use autocast for mixed precision
        with torch.autocast(device_type="cuda", dtype=torch.float16):
            result = self.compute(input_tensor)

        # Clear cache if needed
        if torch.cuda.memory_allocated() > 0.9 * torch.cuda.get_device_properties(0).total_memory:
            torch.cuda.empty_cache()

        return (result,)
```

#### Memory-Efficient Processing
```python
# Process in chunks for large inputs
def process_large_image(self, image, chunk_size=512):
    h, w = image.shape[2], image.shape[3]
    chunks = []

    for y in range(0, h, chunk_size):
        for x in range(0, w, chunk_size):
            chunk = image[:, :, y:y+chunk_size, x:x+chunk_size]
            processed_chunk = self.process_chunk(chunk)
            chunks.append(processed_chunk)

    # Reassemble chunks
    return self.reassemble_chunks(chunks, h, w)
```

### Computation Optimization

#### Caching
```python
class CachedNode:
    def __init__(self):
        self.cache = {}

    def get_cache_key(self, inputs):
        # Create hashable key from inputs
        return hash(tuple(inputs.values()))

    def process(self, **inputs):
        cache_key = self.get_cache_key(inputs)

        if cache_key in self.cache:
            return self.cache[cache_key]

        result = self.compute(**inputs)
        self.cache[cache_key] = result

        return result
```

#### Async Processing
```python
import asyncio

class AsyncNode:
    async def process_async(self, input_data):
        # Simulate async processing
        await asyncio.sleep(0.1)  # Replace with actual async work
        return self.do_processing(input_data)

    def process(self, input_data):
        # Run async function in sync context
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)

        try:
            result = loop.run_until_complete(self.process_async(input_data))
            return (result,)
        finally:
            loop.close()
```

## 🚀 Deployment & Distribution

### Packaging Extensions

#### Setup.py
```python
# setup.py
from setuptools import setup, find_packages

setup(
    name="comfyui-my-extension",
    version="1.0.0",
    description="My ComfyUI extension",
    author="Your Name",
    packages=find_packages(),
    install_requires=[
        "torch>=2.0.0",
        "torchvision>=0.15.0",
        "comfyui>=0.0.1",
    ],
    python_requires=">=3.10",
)
```

#### PyPI Distribution
```powershell
# Build package
python setup.py sdist bdist_wheel

# Upload to PyPI
twine upload dist/*
```

### GitHub Releases

#### Release Structure
```
Release v1.0.0
├── my_extension_v1.0.0.zip
├── requirements.txt
├── README.md
└── examples/
    ├── workflow1.json
    └── workflow2.json
```

#### Automated Release
```yaml
# .github/workflows/release.yml
name: Release
on:
  push:
    tags:
      - 'v*'

jobs:
  release:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v3
      - name: Create Release
        uses: actions/create-release@v1
        env:
          GITHUB_TOKEN: ${{ secrets.GITHUB_TOKEN }}
        with:
          tag_name: ${{ github.ref }}
          release_name: Release ${{ github.ref }}
```

## 🔧 Troubleshooting

### Common Issues

#### Issue 1: Node Not Appearing
**Problem:** Custom node doesn't show up in ComfyUI

**Solutions:**
```python
# Check __init__.py
NODE_CLASS_MAPPINGS = {
    "MyNode": MyNode,  # Ensure correct mapping
}

# Restart ComfyUI completely
# Check console for import errors
```

#### Issue 2: Import Errors
**Problem:** Missing dependencies

**Solutions:**
```powershell
# Install missing packages
pip install missing_package

# Check Python path
python -c "import sys; print(sys.path)"

# Verify virtual environment
which python
```

#### Issue 3: Memory Errors
**Problem:** CUDA out of memory

**Solutions:**
```python
# Reduce batch size
# Use gradient checkpointing
# Process in smaller chunks
# Clear GPU cache periodically
torch.cuda.empty_cache()
```

#### Issue 4: Performance Issues
**Problem:** Node runs slowly

**Solutions:**
```python
# Profile the code
import cProfile
cProfile.run('node.process(input_data)')

# Use torch.compile for PyTorch 2.0+
@torch.compile
def optimized_function(x):
    return process(x)

# Enable cuDNN benchmarking
torch.backends.cudnn.benchmark = True
```

### Debug Commands

#### ComfyUI Debug Mode
```powershell
# Start with debug logging
python main.py --logging=DEBUG

# Check for errors in console
# Look for stack traces
```

#### Memory Profiling
```python
# Profile memory usage
import tracemalloc

tracemalloc.start()
result = node.process(input_data)
current, peak = tracemalloc.get_traced_memory()
print(f"Current memory usage: {current / 1024**2:.1f} MB")
print(f"Peak memory usage: {peak / 1024**2:.1f} MB")
tracemalloc.stop()
```

---

*ComfyUI development offers powerful capabilities for creating custom AI workflows. This guide provides the foundation for building robust extensions and custom nodes on Windows systems.*