# Advanced Windows CUDA Development Guide

> **🔬 Advanced techniques for CUDA development, profiling, and optimization on Windows**
>
> This guide covers advanced CUDA development topics including profiling, optimization, multi-threading, and performance analysis for Windows systems.

## 📋 Table of Contents

- [Quick Start](#-quick-start)
- [CUDA Profiling](#-cuda-profiling)
- [Performance Optimization](#-performance-optimization)
- [Multi-threading & Concurrency](#-multi-threading--concurrency)
- [Memory Management](#-memory-management)
- [Kernel Optimization](#-kernel-optimization)
- [Debugging CUDA Code](#-debugging-cuda-code)
- [Cross-Platform Development](#-cross-platform-development)
- [Troubleshooting](#-troubleshooting)

## 🚀 Quick Start

### Profiling Setup
```powershell
# Install NVIDIA Nsight Systems
# Download from NVIDIA website
# Install Nsight Compute for kernel profiling

# Basic profiling command
nsys profile --trace=cuda,nvtx --output=profile_output python my_script.py
```

### Performance Baseline
```python
import torch
import time

def benchmark_model(model, input_data, iterations=100):
    times = []
    with torch.no_grad():
        for _ in range(iterations):
            start = time.time()
            _ = model(input_data)
            torch.cuda.synchronize()
            times.append(time.time() - start)

    avg_time = sum(times) / len(times)
    fps = 1 / avg_time
    print(f"Average inference time: {avg_time:.4f}s")
    print(f"FPS: {fps:.2f}")
    return avg_time, fps
```

### Memory Analysis
```python
# Check memory usage
print(f"GPU memory allocated: {torch.cuda.memory_allocated()/1024**3:.2f} GB")
print(f"GPU memory cached: {torch.cuda.memory_reserved()/1024**3:.2f} GB")

# Profile memory usage
with torch.cuda.profiler.profile():
    with torch.cuda.profiler.record_function("model_inference"):
        output = model(input)
```

## 📊 CUDA Profiling

### Nsight Systems

#### Installation & Setup
```powershell
# Download Nsight Systems from NVIDIA website
# Install the Windows version
# Add to PATH or use full path

# Basic profiling
nsys profile --trace=cuda,nvtx,osrt --output=my_profile python script.py

# Profile with specific GPU
nsys profile --gpu-metrics-device=0 --output=gpu_profile python script.py
```

#### Advanced Profiling
```powershell
# Profile with memory tracing
nsys profile --trace=cuda,nvtx,osrt,memory --output=memory_profile python script.py

# Profile specific functions
nsys profile --trace=cuda,nvtx --capture-range=cudaProfilerApi --output=function_profile python script.py

# Profile with sampling
nsys profile --sampling-trigger=1 --sampling-period=100000 --output=sampling_profile python script.py
```

#### NVTX Annotations
```python
import torch
from torch.cuda import nvtx

def profile_function():
    with nvtx.range("data_preprocessing"):
        # Data preprocessing code
        input_data = preprocess_data()

    with nvtx.range("model_inference"):
        with torch.no_grad():
            output = model(input_data)

    with nvtx.range("post_processing"):
        result = postprocess_output(output)

    return result
```

### Nsight Compute

#### Kernel Analysis
```powershell
# Profile specific kernel
ncu --target-processes all --launch-count 1 --section-folder sections python script.py

# Profile with metrics
ncu --metrics sm__warps_active.avg.pct_of_peak_sustained_active \
    --target-processes all python script.py

# Profile memory access patterns
ncu --section MemoryWorkloadAnalysis \
    --target-processes all python script.py
```

#### Performance Analysis
```python
# Use PyTorch profiler with CUDA
import torch.profiler as profiler

with profiler.profile(
    activities=[profiler.ProfilerActivity.CPU, profiler.ProfilerActivity.CUDA],
    record_shapes=True,
    profile_memory=True,
    with_stack=True
) as prof:
    # Your code here
    output = model(input_data)

# Print results
print(prof.key_averages().table(sort_by="cuda_time_total", row_limit=10))
```

## ⚡ Performance Optimization

### Memory Optimization

#### Memory Pool Management
```python
# Enable memory pool
torch.cuda.set_per_process_memory_fraction(0.8)

# Use memory efficient operations
with torch.no_grad():
    # Use in-place operations where possible
    x.add_(1)  # Instead of x = x + 1

    # Use views instead of copies
    y = x.view(-1, 784)  # Instead of y = x.reshape(-1, 784)
```

#### Gradient Checkpointing
```python
import torch.utils.checkpoint as checkpoint

def checkpointed_forward(x):
    # Checkpoint intermediate activations
    def custom_forward(*inputs):
        return model(*inputs)

    return checkpoint.checkpoint(custom_forward, x)

# Use in training
output = checkpointed_forward(input_data)
```

#### Mixed Precision Training
```python
from torch.cuda.amp import autocast, GradScaler

scaler = GradScaler()

def train_step(model, optimizer, input_data, target):
    optimizer.zero_grad()

    with autocast():
        output = model(input_data)
        loss = criterion(output, target)

    scaler.scale(loss).backward()
    scaler.step(optimizer)
    scaler.update()

    return loss.item()
```

### Computation Optimization

#### cuDNN Optimization
```python
# Enable cuDNN benchmarking
torch.backends.cudnn.benchmark = True

# Set deterministic mode for reproducibility
torch.backends.cudnn.deterministic = True

# Use cuDNN for specific operations
torch.backends.cudnn.enabled = True
```

#### Tensor Core Utilization
```python
# Ensure proper data types for Tensor Cores
x = x.to(dtype=torch.float16)  # For Ampere GPUs
y = y.to(dtype=torch.float16)

# Use matrix multiplication that leverages Tensor Cores
z = torch.matmul(x, y)  # Will use Tensor Cores on supported GPUs
```

#### Asynchronous Operations
```python
import torch.cuda.comm as comm

# Asynchronous data transfer
stream = torch.cuda.current_stream()

with torch.cuda.stream(stream):
    # Asynchronous GPU operations
    x_gpu = x.cuda(non_blocking=True)
    y_gpu = y.cuda(non_blocking=True)

# Wait for completion
stream.synchronize()
```

## 🧵 Multi-threading & Concurrency

### PyTorch DataLoader Optimization

#### Multi-worker DataLoader
```python
from torch.utils.data import DataLoader
import multiprocessing

# Optimize number of workers
num_workers = min(4, multiprocessing.cpu_count())

train_loader = DataLoader(
    dataset,
    batch_size=32,
    shuffle=True,
    num_workers=num_workers,
    pin_memory=True,  # Faster GPU transfer
    persistent_workers=True,  # Keep workers alive
    prefetch_factor=2  # Prefetch batches
)
```

#### Custom DataLoader
```python
class OptimizedDataLoader:
    def __init__(self, dataset, batch_size=32):
        self.dataset = dataset
        self.batch_size = batch_size
        self.stream = torch.cuda.current_stream()

    def __iter__(self):
        for i in range(0, len(self.dataset), self.batch_size):
            batch = self.dataset[i:i+self.batch_size]

            # Asynchronous transfer to GPU
            with torch.cuda.stream(self.stream):
                batch_gpu = batch.cuda(non_blocking=True)

            yield batch_gpu
```

### Multi-GPU Training

#### Data Parallel
```python
import torch.nn as nn

# Wrap model for multi-GPU
model = nn.DataParallel(model)

# Move to GPUs
model = model.cuda()

# Training loop
for data, target in train_loader:
    data, target = data.cuda(), target.cuda()
    output = model(data)
    loss = criterion(output, target)
    loss.backward()
    optimizer.step()
```

#### Distributed Data Parallel
```python
import torch.distributed as dist
import torch.multiprocessing as mp

def setup_distributed(rank, world_size):
    dist.init_process_group("nccl", rank=rank, world_size=world_size)

def train_distributed(rank, world_size):
    setup_distributed(rank, world_size)

    model = MyModel().cuda(rank)
    model = nn.parallel.DistributedDataParallel(model, device_ids=[rank])

    # Training loop
    for epoch in range(num_epochs):
        sampler.set_epoch(epoch)
        for data, target in train_loader:
            data, target = data.cuda(rank), target.cuda(rank)
            output = model(data)
            loss = criterion(output, target)
            loss.backward()
            optimizer.step()
```

### Stream Processing

#### CUDA Streams
```python
# Create multiple streams
stream1 = torch.cuda.current_stream()
stream2 = torch.cuda.Stream()

# Concurrent execution
with torch.cuda.stream(stream1):
    # Operations on stream 1
    x = model1(input1)

with torch.cuda.stream(stream2):
    # Operations on stream 2
    y = model2(input2)

# Synchronize
stream1.synchronize()
stream2.synchronize()
```

#### Pipeline Parallelism
```python
class PipelineParallel:
    def __init__(self, model, num_stages):
        self.model = model
        self.num_stages = num_stages
        self.streams = [torch.cuda.Stream() for _ in range(num_stages)]

    def forward(self, x):
        # Split model into stages
        stages = self.split_model()

        # Pipeline execution
        for i, stage in enumerate(stages):
            with torch.cuda.stream(self.streams[i]):
                x = stage(x)

        # Synchronize all streams
        for stream in self.streams:
            stream.synchronize()

        return x
```

## 💾 Memory Management

### Advanced Memory Techniques

#### Memory Pool
```python
# Create custom memory pool
class MemoryPool:
    def __init__(self, size_mb=1024):
        self.size = size_mb * 1024 * 1024
        self.pool = torch.cuda.ByteTensor(self.size)

    def allocate(self, size):
        # Custom allocation logic
        if size > self.size:
            return torch.cuda.ByteTensor(size)
        return self.pool[:size]

# Use memory pool
pool = MemoryPool(2048)  # 2GB pool
tensor = pool.allocate(1024 * 1024)  # 1MB allocation
```

#### Memory Defragmentation
```python
def defragment_memory():
    # Force garbage collection
    import gc
    gc.collect()

    # Clear CUDA cache
    torch.cuda.empty_cache()

    # Compact memory
    torch.cuda.synchronize()

    print(f"Memory after defrag: {torch.cuda.memory_allocated()/1024**3:.2f} GB")

# Call periodically
defragment_memory()
```

#### Memory-mapped Files
```python
import numpy as np

# Memory-map large files
data = np.memmap('large_file.dat', dtype='float32', mode='r', shape=(1000000, 784))

# Convert to tensor without copying
tensor = torch.from_numpy(data)

# Move to GPU in chunks
chunk_size = 10000
for i in range(0, len(tensor), chunk_size):
    chunk = tensor[i:i+chunk_size].cuda()
    # Process chunk
```

### Unified Memory

#### Managed Memory
```python
# Use unified memory for automatic migration
x = torch.cuda.FloatTensor(1000, 1000).cuda()  # Regular GPU memory
y = torch.cuda.FloatTensor(1000, 1000, memory_format=torch.contiguous_format)  # Contiguous

# Check memory format
print(x.is_contiguous())  # True
print(y.is_contiguous())  # True

# Optimize memory access patterns
x = x.contiguous()  # Ensure contiguous memory
```

## 🔧 Kernel Optimization

### Custom CUDA Kernels

#### PyTorch CUDA Extension
```python
# setup.py
from setuptools import setup
from torch.utils.cpp_extension import BuildExtension, CUDAExtension

setup(
    name='my_cuda_extension',
    ext_modules=[
        CUDAExtension(
            'my_cuda_extension',
            ['kernel.cu', 'extension.cpp'],
            extra_compile_args={'cxx': [], 'nvcc': ['-O3', '--use_fast_math']}
        )
    ],
    cmdclass={'build_ext': BuildExtension}
)
```

#### CUDA Kernel Example
```cpp
// kernel.cu
__global__ void my_kernel(float* input, float* output, int size) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;

    if (idx < size) {
        // Your kernel logic here
        output[idx] = input[idx] * 2.0f + 1.0f;
    }
}

torch::Tensor my_cuda_function(torch::Tensor input) {
    auto output = torch::zeros_like(input);

    int size = input.numel();
    int threads = 256;
    int blocks = (size + threads - 1) / threads;

    my_kernel<<<blocks, threads>>>(
        input.data_ptr<float>(),
        output.data_ptr<float>(),
        size
    );

    return output;
}
```

#### Python Interface
```python
# extension.py
import torch
from torch.utils.cpp_extension import load

# Load the extension
my_extension = load(
    name='my_cuda_extension',
    sources=['kernel.cu', 'extension.cpp'],
    verbose=True
)

# Use the function
x = torch.randn(1000, 1000).cuda()
y = my_extension.my_cuda_function(x)
```

### Kernel Launch Optimization

#### Optimal Block Size
```python
def find_optimal_block_size(kernel_function, max_threads=1024):
    """Find optimal block size for kernel"""
    best_time = float('inf')
    best_block_size = 256

    for block_size in [128, 256, 512, 1024]:
        # Time kernel execution
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)

        start.record()
        # Launch kernel with block_size
        kernel_function(block_size)
        end.record()

        torch.cuda.synchronize()
        time = start.elapsed_time(end)

        if time < best_time:
            best_time = time
            best_block_size = block_size

    return best_block_size
```

#### Shared Memory Usage
```cpp
// Kernel with shared memory
__global__ void shared_memory_kernel(float* input, float* output, int size) {
    __shared__ float shared_data[256];

    int tid = threadIdx.x;
    int idx = blockIdx.x * blockDim.x + tid;

    // Load data into shared memory
    if (idx < size) {
        shared_data[tid] = input[idx];
    }
    __syncthreads();

    // Process data in shared memory
    if (idx < size) {
        float result = 0.0f;
        for (int i = 0; i < blockDim.x; i++) {
            result += shared_data[i];
        }
        output[idx] = result;
    }
}
```

## 🐛 Debugging CUDA Code

### CUDA-GDB

#### Setup & Usage
```powershell
# Install CUDA-GDB (included with CUDA toolkit)
# Launch with debugging
cuda-gdb python

# Set breakpoints
(cuda-gdb) break my_kernel
(cuda-gdb) run script.py

# Debug commands
(cuda-gdb) info cuda threads
(cuda-gdb) info cuda blocks
(cuda-gdb) print variable
```

#### Memory Debugging
```python
# Enable CUDA memory checking
import os
os.environ['CUDA_LAUNCH_BLOCKING'] = '1'

# Use torch's CUDA memory debugger
torch.cuda.memory._record_memory_history()

# Check for memory errors
try:
    output = model(input_data)
except RuntimeError as e:
    if "out of memory" in str(e):
        print("GPU memory error detected")
        torch.cuda.empty_cache()
```

### Error Handling

#### CUDA Error Checking
```python
def cuda_check_error():
    """Check for CUDA errors"""
    err = torch.cuda.get_last_error()
    if err != torch.cuda.Error.Success:
        raise RuntimeError(f"CUDA error: {err}")

# Use after CUDA operations
model.cuda()
cuda_check_error()

output = model(input_data)
cuda_check_error()
```

#### Exception Handling
```python
class CUDAErrorHandler:
    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        if exc_type is RuntimeError and "CUDA" in str(exc_val):
            print("CUDA error detected, clearing cache")
            torch.cuda.empty_cache()
            return True  # Suppress the exception
        return False

# Usage
with CUDAErrorHandler():
    # CUDA operations that might fail
    output = model(input_data)
```

## 🌐 Cross-Platform Development

### Platform Detection

#### GPU Architecture Detection
```python
def get_gpu_architecture():
    """Get GPU architecture information"""
    if torch.cuda.is_available():
        device = torch.cuda.current_device()
        props = torch.cuda.get_device_properties(device)

        arch = props.major * 10 + props.minor
        print(f"GPU: {props.name}")
        print(f"Architecture: {arch}")
        print(f"Compute capability: {props.major}.{props.minor}")

        return arch
    return None

# Use for conditional compilation
arch = get_gpu_architecture()
if arch >= 80:  # Ampere or newer
    # Use Ampere-specific optimizations
    pass
```

#### Platform-Specific Code
```python
import platform
import sys

def get_platform_info():
    """Get platform information"""
    info = {
        'system': platform.system(),
        'release': platform.release(),
        'version': platform.version(),
        'machine': platform.machine(),
        'processor': platform.processor(),
        'python_version': sys.version
    }
    return info

# Platform-specific optimizations
platform_info = get_platform_info()
if platform_info['system'] == 'Windows':
    # Windows-specific code
    pass
elif platform_info['system'] == 'Linux':
    # Linux-specific code
    pass
```

### Cross-Platform Builds

#### CMake Configuration
```cmake
# CMakeLists.txt for cross-platform CUDA builds
cmake_minimum_required(VERSION 3.18)
project(MyCUDAProject)

# Find CUDA
find_package(CUDA REQUIRED)

# Platform-specific flags
if(WIN32)
    set(CUDA_NVCC_FLAGS ${CUDA_NVCC_FLAGS};-O3;--use_fast_math;-Xcompiler;/MT)
elseif(UNIX)
    set(CUDA_NVCC_FLAGS ${CUDA_NVCC_FLAGS};-O3;--use_fast_math;-Xcompiler;-fPIC)
endif()

# Build library
cuda_add_library(my_cuda_lib kernel.cu)
```

#### Python Cross-Platform Extension
```python
# setup.py for cross-platform
from setuptools import setup
from torch.utils.cpp_extension import BuildExtension, CUDAExtension
import platform

extra_compile_args = {
    'cxx': [],
    'nvcc': ['-O3', '--use_fast_math']
}

if platform.system() == 'Windows':
    extra_compile_args['cxx'].extend(['/MT', '/std:c++17'])
    extra_compile_args['nvcc'].extend(['-Xcompiler', '/MT'])
else:
    extra_compile_args['cxx'].extend(['-std=c++17', '-fPIC'])
    extra_compile_args['nvcc'].extend(['-Xcompiler', '-fPIC'])

setup(
    name='my_cuda_extension',
    ext_modules=[
        CUDAExtension(
            'my_cuda_extension',
            ['kernel.cu', 'extension.cpp'],
            extra_compile_args=extra_compile_args
        )
    ],
    cmdclass={'build_ext': BuildExtension}
)
```

## 🔧 Troubleshooting

### Common Performance Issues

#### Issue 1: Memory Fragmentation
**Problem:** GPU memory becomes fragmented

**Solutions:**
```python
# Clear cache regularly
torch.cuda.empty_cache()

# Use memory pools
torch.cuda.set_per_process_memory_fraction(0.9)

# Monitor memory usage
print(torch.cuda.memory_summary())
```

#### Issue 2: Kernel Launch Overhead
**Problem:** Too many small kernel launches

**Solutions:**
```python
# Batch operations
# Use larger batch sizes
# Combine multiple operations into single kernels

# Example: Combine multiple operations
def batched_operations(inputs):
    # Instead of multiple kernel calls
    # Combine into single kernel
    return combined_kernel(inputs)
```

#### Issue 3: Synchronization Issues
**Problem:** Improper synchronization causing race conditions

**Solutions:**
```python
# Proper synchronization
torch.cuda.synchronize()

# Use events for timing
start = torch.cuda.Event(enable_timing=True)
end = torch.cuda.Event(enable_timing=True)

start.record()
# Operations
end.record()
end.synchronize()

time = start.elapsed_time(end)
```

#### Issue 4: Precision Issues
**Problem:** Numerical instability with mixed precision

**Solutions:**
```python
# Use appropriate precision
torch.set_default_dtype(torch.float32)

# GradScaler for mixed precision
from torch.cuda.amp import GradScaler
scaler = GradScaler()

# Scale loss
with autocast():
    loss = model(input)
scaled_loss = scaler.scale(loss)
scaled_loss.backward()
scaler.step(optimizer)
scaler.update()
```

### Debug Tools

#### Memory Debugging
```python
# Enable memory debugging
torch.cuda.memory._record_memory_history(enabled='all')

# Get memory history
history = torch.cuda.memory._memory_stats()

# Print memory stats
for key, value in history.items():
    print(f"{key}: {value}")
```

#### Performance Monitoring
```python
# Monitor GPU utilization
def monitor_gpu():
    while True:
        utilization = torch.cuda.utilization()
        memory_used = torch.cuda.memory_allocated() / torch.cuda.get_device_properties(0).total_memory
        print(f"GPU utilization: {utilization}%, Memory: {memory_used:.2f}")
        time.sleep(1)

# Run in background thread
import threading
threading.Thread(target=monitor_gpu, daemon=True).start()
```

---

*Advanced CUDA development requires deep understanding of GPU architecture and optimization techniques. This guide provides the essential knowledge for high-performance CUDA applications on Windows systems.*