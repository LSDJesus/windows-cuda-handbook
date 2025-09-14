# TensorRT Integration & Compatibility Guide

> **🚀 Optimizing ComfyUI workflows with TensorRT acceleration on Windows**
>
> This guide covers TensorRT integration for ComfyUI, focusing on UNET conversion, LoRA compatibility, and multi-GPU optimization strategies.

## 📋 Table of Contents

- [Quick Start](#-quick-start)
- [System Requirements](#-system-requirements)
- [TensorRT Setup](#-tensorrt-setup)
- [Model Conversion](#-model-conversion)
- [LoRA Integration](#-lora-integration)
- [Multi-GPU Optimization](#-multi-gpu-optimization)
- [Troubleshooting](#-troubleshooting)
- [Performance Tuning](#-performance-tuning)

## 🚀 Quick Start

### System Compatibility Check
```powershell
# Check GPU and driver compatibility
nvidia-smi --query-gpu=name,driver_version --format=csv

# Verify TensorRT installation
where trtexec
trtexec --version
```

### Basic UNET Conversion
```powershell
# Convert PyTorch UNET to TensorRT
python convert_unet.py --model_path model.pth --output_path unet.engine

# Test conversion
python test_engine.py --engine_path unet.engine
```

### Performance Benchmark
```powershell
# Compare PyTorch vs TensorRT
python benchmark.py --pytorch_model model.pth --trt_engine unet.engine
```

## 🔧 System Requirements

### Hardware Requirements
- **GPU**: RTX 30xx/40xx/50xx series (Ampere/Ada/Hopper architecture)
- **VRAM**: Minimum 8GB, recommended 24GB+
- **CPU**: Modern multi-core processor
- **RAM**: 32GB+ for large model conversion

### Software Requirements
- **TensorRT**: 8.6+ (matches CUDA version)
- **PyTorch**: 2.6.0+ with CUDA support
- **CUDA**: 12.1+ (matching TensorRT)
- **Python**: 3.10-3.12
- **ComfyUI**: Latest version with TensorRT support

### Driver Requirements
```powershell
# Check driver version
nvidia-smi --query-gpu=driver_version --format=csv,noheader,nounits

# Minimum versions:
# RTX 30xx: Driver 470+
# RTX 40xx: Driver 520+
# RTX 50xx: Driver 560+
```

## 🛠️ TensorRT Setup

### Installation

#### Method 1: NVIDIA Installer
```powershell
# Download from NVIDIA website
# Install TensorRT runtime and development libraries
# Add to system PATH
```

#### Method 2: Package Manager
```powershell
# Using conda (if applicable)
conda install -c nvidia tensorrt

# Using pip (development packages)
pip install tensorrt
```

### Environment Configuration

#### CUDA and TensorRT Paths
```powershell
# Set environment variables
$env:CUDA_HOME = "C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v12.1"
$env:TENSORRT_HOME = "C:\Program Files\NVIDIA TensorRT"
$env:PATH = "$env:CUDA_HOME\bin;$env:TENSORRT_HOME\bin;$env:PATH"
```

#### Python Integration
```python
# Verify TensorRT Python bindings
import tensorrt as trt
print("TensorRT version:", trt.__version__)

# Check CUDA compatibility
print("CUDA version:", trt.cuda.get_version())
```

### ComfyUI Integration

#### Install TensorRT Extension
```powershell
# Clone ComfyUI TensorRT extension
git clone https://github.com/comfyanonymous/ComfyUI_TensorRT
cd ComfyUI_TensorRT

# Install dependencies
pip install -r requirements.txt
```

#### Enable in ComfyUI
```python
# Add to ComfyUI/custom_nodes/
# Restart ComfyUI
# TensorRT nodes will appear in the interface
```

## 🔄 Model Conversion

### UNET Conversion Process

#### Step 1: Prepare PyTorch Model
```python
import torch
from diffusers import StableDiffusionPipeline

# Load base model
pipe = StableDiffusionPipeline.from_pretrained("runwayml/stable-diffusion-v1-5")
unet = pipe.unet

# Move to GPU
unet = unet.cuda()

# Set to evaluation mode
unet.eval()
```

#### Step 2: Export to ONNX
```python
# Define input dimensions
batch_size = 1
channels = 4
height = 512
width = 512
sequence_length = 77

# Create sample input
sample_input = torch.randn(batch_size, channels, height // 8, width // 8).cuda()

# Export to ONNX
torch.onnx.export(
    unet,
    sample_input,
    "unet.onnx",
    export_params=True,
    opset_version=17,
    do_constant_folding=True,
    input_names=['input'],
    output_names=['output'],
    dynamic_axes={'input': {0: 'batch_size'}, 'output': {0: 'batch_size'}}
)
```

#### Step 3: Convert to TensorRT
```bash
# Using trtexec command line
trtexec \
    --onnx=unet.onnx \
    --saveEngine=unet.engine \
    --fp16 \
    --workspace=8192 \
    --minShapes=input:1x4x64x64 \
    --optShapes=input:1x4x64x64 \
    --maxShapes=input:4x4x64x64
```

#### Step 4: Python Conversion Script
```python
import tensorrt as trt
from tensorrt import Builder, Network, Config

def convert_onnx_to_engine(onnx_path, engine_path, fp16=True):
    logger = trt.Logger(trt.Logger.WARNING)
    builder = trt.Builder(logger)
    network = builder.create_network(1 << int(trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH))
    parser = trt.OnnxParser(network, logger)

    # Parse ONNX
    with open(onnx_path, 'rb') as model:
        if not parser.parse(model.read()):
            for error in range(parser.num_errors):
                print(parser.get_error(error))

    # Create engine
    config = builder.create_builder_config()
    if fp16:
        config.set_flag(trt.BuilderFlag.FP16)

    config.max_workspace_size = 1 << 30  # 1GB

    engine = builder.build_engine(network, config)
    with open(engine_path, 'wb') as f:
        f.write(engine.serialize())

    return engine
```

### Conversion Optimization

#### Precision Options
```bash
# FP32 (highest accuracy, slowest)
trtexec --onnx=unet.onnx --saveEngine=unet_fp32.engine --workspace=4096

# FP16 (balanced accuracy/speed)
trtexec --onnx=unet.onnx --saveEngine=unet_fp16.engine --fp16 --workspace=4096

# INT8 (fastest, lower accuracy)
trtexec --onnx=unet.onnx --saveEngine=unet_int8.engine --int8 --workspace=4096
```

#### Dynamic Shape Optimization
```bash
# Support multiple batch sizes
trtexec \
    --onnx=unet.onnx \
    --saveEngine=unet_dynamic.engine \
    --fp16 \
    --minShapes=input:1x4x64x64 \
    --optShapes=input:2x4x64x64 \
    --maxShapes=input:8x4x64x64
```

## 🎭 LoRA Integration

### The LoRA Challenge

**Problem:** LoRAs must be baked into the base model before TensorRT conversion.

**Why:** TensorRT engines are static and cannot dynamically load LoRAs at runtime.

### Solution: Pre-baking LoRAs

#### Step 1: Load Base Model and LoRA
```python
from diffusers import StableDiffusionPipeline
import torch
from peft import PeftModel

# Load base model
pipe = StableDiffusionPipeline.from_pretrained("runwayml/stable-diffusion-v1-5")

# Load LoRA
pipe.unet = PeftModel.from_pretrained(pipe.unet, "path/to/lora")

# Merge LoRA into base model
pipe.unet = pipe.unet.merge_and_unload()
```

#### Step 2: Convert Merged Model
```python
# The merged model can now be converted to TensorRT
# Follow the conversion process above
torch.onnx.export(
    pipe.unet,
    sample_input,
    "unet_with_lora.onnx",
    # ... other parameters
)
```

### Multiple LoRA Support

#### Strategy 1: Multiple Engines
```python
# Create separate engines for each LoRA
loras = ["lora1", "lora2", "lora3"]

for lora in loras:
    # Load base + LoRA
    pipe = StableDiffusionPipeline.from_pretrained("runwayml/stable-diffusion-v1-5")
    pipe.unet = PeftModel.from_pretrained(pipe.unet, f"path/to/{lora}")

    # Merge and convert
    pipe.unet = pipe.unet.merge_and_unload()
    convert_to_tensorrt(pipe.unet, f"unet_{lora}.engine")
```

#### Strategy 2: LoRA Engine Cache
```python
class LoRAEngineCache:
    def __init__(self):
        self.engines = {}

    def get_engine(self, lora_name):
        if lora_name not in self.engines:
            # Load and convert on demand
            self.engines[lora_name] = self._load_or_create_engine(lora_name)
        return self.engines[lora_name]
```

## 🔀 Multi-GPU Optimization

### GPU Assignment Strategy

#### Primary GPU for Heavy Computation
```python
# Assign TensorRT UNET to fastest GPU
import torch

# RTX 5090 (GPU 0) - Primary
# RTX 3080 Ti (GPU 1) - Secondary

# Move models to appropriate GPUs
pipe = StableDiffusionPipeline.from_pretrained("runwayml/stable-diffusion-v1-5")
pipe.to("cuda:0")  # Primary GPU

# VAE on secondary GPU
pipe.vae.to("cuda:1")

# CLIP on secondary GPU
pipe.text_encoder.to("cuda:1")
```

#### Memory Distribution
```python
# Check GPU memory
for i in range(torch.cuda.device_count()):
    print(f"GPU {i}: {torch.cuda.get_device_properties(i).total_memory / 1024**3:.1f} GB")

# Strategy: Put largest model on GPU with most memory
# RTX 5090: UNET (largest)
# RTX 3080 Ti: VAE, CLIP (smaller)
```

### Cross-GPU Data Transfer

#### Optimized Transfer
```python
# Minimize cross-GPU transfers
def optimized_inference(latent, text_embeddings):
    # Process text on GPU 1
    with torch.cuda.device(1):
        text_embeddings = text_embeddings.to("cuda:1")

    # Process latent on GPU 0
    with torch.cuda.device(0):
        latent = latent.to("cuda:0")

        # UNET inference on GPU 0
        noise_pred = unet(latent, timestep, text_embeddings.to("cuda:0"))

    return noise_pred
```

#### Pipeline Parallelism
```python
class MultiGPUPipeline:
    def __init__(self):
        self.device_0 = torch.device("cuda:0")  # RTX 5090
        self.device_1 = torch.device("cuda:1")  # RTX 3080 Ti

    def __call__(self, latent, text_embeddings, timestep):
        # Text encoding on GPU 1
        text_embeddings = text_embeddings.to(self.device_1)

        # UNET on GPU 0
        latent = latent.to(self.device_0)
        noise_pred = self.unet(latent, timestep, text_embeddings.to(self.device_0))

        return noise_pred
```

## 🔧 Troubleshooting

### Common Issues

#### Issue 1: Engine Creation Fails
**Error:** `Engine creation failed`

**Solutions:**
```bash
# Increase workspace size
trtexec --onnx=unet.onnx --saveEngine=unet.engine --workspace=16384

# Try different precision
trtexec --onnx=unet.onnx --saveEngine=unet.engine --fp32

# Check ONNX model
python -c "import onnxruntime as ort; ort.InferenceSession('unet.onnx')"
```

#### Issue 2: Runtime Incompatibility
**Error:** `head_dim` instability

**Solution:**
```python
# Recompile without SageAttention
# Use standard attention mechanism
pipe = StableDiffusionPipeline.from_pretrained("runwayml/stable-diffusion-v1-5", use_safetensors=True)
```

#### Issue 3: Input Shape Mismatch
**Error:** Dimension errors in ComfyUI

**Solutions:**
```python
# Use consistent input shapes
# BBoxDetectorSEGS instead of SegmDetectorSEGS
# Ensure all inputs have matching dimensions
```

#### Issue 4: Memory Allocation Errors
**Error:** CUDA out of memory

**Solutions:**
```bash
# Reduce max batch size
trtexec --onnx=unet.onnx --saveEngine=unet.engine --maxShapes=input:2x4x64x64

# Use CPU fallback for some operations
export CUDA_MODULE_LOADING=LAZY
```

### Debug Tools

#### Engine Inspection
```python
import tensorrt as trt

def inspect_engine(engine_path):
    with open(engine_path, 'rb') as f:
        engine_data = f.read()

    runtime = trt.Runtime(trt.Logger(trt.Logger.WARNING))
    engine = runtime.deserialize_cuda_engine(engine_data)

    print(f"Max batch size: {engine.max_batch_size}")
    print(f"Max workspace size: {engine.max_workspace_size}")

    for i in range(engine.num_bindings):
        print(f"Binding {i}: {engine.get_binding_name(i)}")
```

#### Performance Profiling
```python
import time
import torch

def profile_inference(model, input_data, iterations=100):
    times = []
    for _ in range(iterations):
        start = time.time()
        with torch.no_grad():
            _ = model(input_data)
        torch.cuda.synchronize()
        times.append(time.time() - start)

    print(f"Average inference time: {sum(times)/len(times):.4f} seconds")
    print(f"FPS: {1 / (sum(times)/len(times)):.2f}")
```

## ⚡ Performance Tuning

### Optimization Strategies

#### Precision Tuning
```bash
# Test different precisions
for precision in ["fp32", "fp16", "int8"]:
    trtexec --onnx=unet.onnx --saveEngine=unet_${precision}.engine --${precision}
    # Benchmark each
```

#### Batch Size Optimization
```bash
# Find optimal batch size
for batch in [1, 2, 4, 8]:
    trtexec --onnx=unet.onnx --saveEngine=unet_b${batch}.engine \
        --minShapes=input:${batch}x4x64x64 \
        --optShapes=input:${batch}x4x64x64 \
        --maxShapes=input:${batch}x4x64x64
```

#### Workspace Optimization
```bash
# Test workspace sizes
for ws in [4096, 8192, 16384]:
    trtexec --onnx=unet.onnx --saveEngine=unet_ws${ws}.engine --workspace=${ws}
```

### Memory Management

#### GPU Memory Monitoring
```python
def monitor_gpu_memory():
    for i in range(torch.cuda.device_count()):
        allocated = torch.cuda.memory_allocated(i) / 1024**3
        reserved = torch.cuda.memory_reserved(i) / 1024**3
        print(f"GPU {i}: {allocated:.2f}GB allocated, {reserved:.2f}GB reserved")
```

#### Memory-Efficient Conversion
```python
# Use gradient checkpointing during conversion
with torch.no_grad():
    torch.onnx.export(model, sample_input, "model.onnx",
                     enable_onnx_checker=False)  # Skip checker for large models
```

### Benchmarking

#### Comprehensive Benchmark
```python
def benchmark_models(pytorch_model, trt_engine, input_data):
    # PyTorch benchmark
    with torch.no_grad():
        start = time.time()
        for _ in range(100):
            _ = pytorch_model(input_data)
        torch.cuda.synchronize()
        pytorch_time = time.time() - start

    # TensorRT benchmark
    # ... TensorRT inference code ...
    trt_time = time.time() - start

    speedup = pytorch_time / trt_time
    print(f"TensorRT speedup: {speedup:.2f}x")
```

---

*TensorRT integration can provide significant performance improvements for ComfyUI workflows. This guide covers the essential techniques for successful conversion and optimization on Windows systems.*