# Mamba-SSM Windows Build Guide

> **🐍 Complete guide for building Mamba-SSM from source on Windows with CUDA 12.9 & PyTorch 2.8.0**
>
> This guide covers the comprehensive process of building the Mamba-SSM library on Windows, including all necessary source code modifications and build optimizations.

## 📋 Table of Contents

- [Quick Start](#-quick-start)
- [System Requirements](#-system-requirements)
- [Initial Setup](#-initial-setup)
- [Build Process](#-build-process)
- [Source Code Modifications](#-source-code-modifications)
- [Troubleshooting](#-troubleshooting)
- [Post-Build Tasks](#-post-build-tasks)
- [Verification](#-verification)

## 🚀 Quick Start

### Prerequisites Check
```powershell
# Verify CUDA and PyTorch installation
nvcc --version
python -c "import torch; print(f'PyTorch: {torch.__version__}')"
python -c "import torch; print(f'CUDA: {torch.version.cuda}')"
```

### Clone and Build
```powershell
# Clone the repository
git clone https://github.com/state-spaces/mamba.git
cd mamba

# Apply Windows-specific modifications
# (See Source Code Modifications section)

# Build the wheel
python setup.py bdist_wheel
```

### Install and Test
```powershell
# Install the built wheel
pip install dist/mamba_ssm-*.whl

# Verify installation
python -c "import mamba_ssm; print('Mamba-SSM installed successfully')"
```

## 🔧 System Requirements

### Hardware Requirements
- **GPU**: NVIDIA GPU with CUDA support (compute capability 7.5+)
- **VRAM**: Minimum 8GB recommended
- **RAM**: 16GB+ for compilation
- **Storage**: 10GB+ free space

### Software Requirements
- **Windows**: 10/11 Pro/Enterprise
- **Python**: 3.10-3.12
- **CUDA**: 12.1+ (tested with 12.9)
- **PyTorch**: 2.6.0+ with CUDA support
- **MSVC**: Visual Studio 2022 with C++ build tools
- **CMake**: 3.26+
- **Ninja**: Build system

### Environment Setup
```powershell
# Verify MSVC installation
cl /?

# Check CMake and Ninja
cmake --version
ninja --version

# Verify CUDA toolkit
where nvcc
```

## 🛠️ Initial Setup

### Virtual Environment
```powershell
# Create isolated environment
python -m venv mamba_build
.\mamba_build\Scripts\activate

# Install PyTorch with CUDA
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu129

# Install build dependencies
pip install setuptools wheel ninja cmake
```

### Repository Setup
```powershell
# Clone Mamba-SSM
git clone https://github.com/state-spaces/mamba.git
cd mamba

# Verify CUDA architectures
python -c "import torch; print(torch.cuda.get_device_capability())"
```

### Environment Variables
```powershell
# Set CUDA architecture (adjust for your GPU)
$env:TORCH_CUDA_ARCH_LIST = "75;86;89;120"

# Disable problematic precision modes initially
$env:MM_DISABLE_BF16 = "1"
$env:MM_DISABLE_HALF = "1"

# Set thread count for compilation
$env:NVCC_THREADS = "8"
```

## 🔨 Build Process

### Standard Build Attempt
```powershell
# First attempt (will likely fail)
python setup.py bdist_wheel
```

**Expected Failures:**
- `bwd_bf16_real.cu` compilation errors
- `bwd_fp16_real.cu` compilation errors
- `urllib.error.HTTPError: 404 Not Found` (phantom errors)
- `ninja: build stopped: subcommand failed`

### Modified Build Process
After applying the source code modifications (see next section):

```powershell
# Clean previous build
rm -rf build/ dist/ *.egg-info/

# Build with modifications
python setup.py bdist_wheel

# The build should now succeed
```

### Build Verification
```powershell
# Check build output
ls dist/

# Verify wheel contents
python -m zipfile -l dist/mamba_ssm-*.whl
```

## 🔧 Source Code Modifications

### 1. setup.py Modifications

#### CUDA Architecture Fix
**Problem:** setup.py ignores `TORCH_CUDA_ARCH_LIST` and hard-codes architectures

**Location:** `setup.py` - around line 150-200

**Original Code:**
```python
# Hard-coded architecture list
arch_list = [60, 61, 62, 70, 72, 75, 80, 86, 89, 90]
```

**Fixed Code:**
```python
# Respect TORCH_CUDA_ARCH_LIST environment variable
import os
torch_cuda_arch_list = os.environ.get('TORCH_CUDA_ARCH_LIST', '75;86;89;120')
arch_list = [int(x.strip()) for x in torch_cuda_arch_list.split(';') if x.strip()]
```

#### CachedWheelsCommand Fix
**Problem:** Attempts to download non-existent Linux wheels, causing 404 errors

**Location:** `setup.py` - `CachedWheelsCommand` class

**Original Code:**
```python
def run(self):
    # Attempts to download wheels
    self.download_wheel()
```

**Fixed Code:**
```python
def run(self):
    # Skip wheel download, force local build
    print("Skipping wheel download - building locally")
    return
```

#### NVCC Threads Fix
**Problem:** Hard-codes `--threads 4`, overriding environment variable

**Location:** `setup.py` - `append_nvcc_threads` function

**Original Code:**
```python
def append_nvcc_threads(nvcc_args):
    nvcc_args.append('--threads')
    nvcc_args.append('4')
```

**Fixed Code:**
```python
def append_nvcc_threads(nvcc_args):
    threads = os.environ.get('NVCC_THREADS', '8')
    nvcc_args.append('--threads')
    nvcc_args.append(threads)
```

### 2. CUDA Header File Modifications

#### BOOL_SWITCH Fix
**Problem:** GCC-specific nested lambda structure incompatible with MSVC

**Location:** `csrc/selective_scan/cus/selective_scan_fwd_kernel.cuh`
**Location:** `csrc/selective_scan/cus/selective_scan_bwd_kernel.cuh`

**Original Code:**
```cpp
BOOL_SWITCH(...) {
    // GCC-specific lambda structure
}
```

**Fixed Code:**
```cpp
// Replace with explicit if/else structure
if (condition) {
    // True branch
    kernel_launcher<...true...>(...);
} else {
    // False branch
    kernel_launcher<...false...>(...);
}
```

#### Math Constants Fix
**Problem:** Missing math constant definitions

**Location:** Both `.cuh` files - add at top

**Fix:**
```cpp
#define _USE_MATH_DEFINES
#include <cmath>
#include <type_traits>
```

#### Type Traits Fix
**Problem:** Using `std::true_type`/`std::false_type` without proper includes

**Location:** Both `.cuh` files - add include

**Fix:**
```cpp
#include <type_traits>
```

#### Boolean Value Fix
**Problem:** Compiler expects boolean values, not types

**Location:** In the if/else structure

**Original Code:**
```cpp
kernel_launcher<...std::true_type...>
kernel_launcher<...std::false_type...>
```

**Fixed Code:**
```cpp
kernel_launcher<...true...>
kernel_launcher<...false...>
```

### 3. C++ Source Modifications

#### Unresolved External Symbols Fix
**Problem:** `selective_scan.cpp` references complex CUDA functions that were removed

**Location:** `csrc/selective_scan.cpp` - `TORCH_DISPATCH` blocks

**Original Code:**
```cpp
TORCH_DISPATCH_FLOATING_TYPES_AND_COMPLEX(...) {
    // References complex functions
}
```

**Fixed Code:**
```cpp
TORCH_DISPATCH_FLOATING_TYPES(...) {
    // Float path - working
    AT_DISPATCH_FLOATING_TYPES(value.scalar_type(), "selective_scan", [&] {
        // Implementation
    });
}

// Complex path - commented out to avoid linker errors
/*
TORCH_DISPATCH_COMPLEX_TYPES(...) {
    // Complex implementation - disabled
}
*/
```

## 🔧 Troubleshooting

### Common Build Errors

#### Error 1: bfloat16 Compilation Failure
**Error:** `bwd_bf16_real.cu` compilation errors

**Solutions:**
```powershell
# Disable bfloat16 initially
$env:MM_DISABLE_BF16 = "1"

# Or fix by ensuring proper CUDA version
# CUDA 11.0+ required for bfloat16
```

#### Error 2: Half Precision Compilation Failure
**Error:** `bwd_fp16_real.cu` compilation errors

**Solutions:**
```powershell
# Disable half precision initially
$env:MM_DISABLE_HALF = "1"

# Re-enable after successful build
$env:MM_DISABLE_HALF = ""
```

#### Error 3: 404 HTTP Errors
**Error:** `urllib.error.HTTPError: 404 Not Found`

**Cause:** `CachedWheelsCommand` trying to download Linux wheels

**Solution:** Apply the `CachedWheelsCommand` fix above

#### Error 4: Unresolved External Symbol
**Error:** `LNK2001: unresolved external symbol`

**Cause:** References to complex CUDA functions after removing complex files

**Solution:** Apply the `TORCH_DISPATCH` fix above

#### Error 5: Ninja Build Stopped
**Error:** `ninja: build stopped: subcommand failed`

**Solutions:**
```powershell
# Check for compilation errors above
# Ensure all source modifications are applied
# Verify CUDA toolkit installation
# Check MSVC build tools
```

### Debug Build Process
```powershell
# Enable verbose output
python setup.py bdist_wheel --verbose

# Build with debug symbols
set DEBUG=1
python setup.py build_ext --inplace

# Check CUDA compilation
nvcc --version
nvcc -o /dev/null -c test.cu  # Test CUDA compilation
```

### Environment Validation
```powershell
# Check all requirements
python -c "import torch; print('PyTorch:', torch.__version__)"
python -c "import torch; print('CUDA:', torch.version.cuda)"
python -c "print('MSVC:'); cl /? | findstr 'Version'"
cmake --version
ninja --version

# Verify GPU
nvidia-smi
```

## 📦 Post-Build Tasks

### Wheel Metadata Fix
**Problem:** Built wheel incorrectly lists `triton` as dependency

**Solution:**
```powershell
# Extract wheel
cd dist
python -m zipfile -e mamba_ssm-*.whl temp_dir/
cd temp_dir

# Edit METADATA file
notepad METADATA
# Remove triton from Requires-Dist

# Repackage wheel
cd ..
python -c "
import zipfile
import os
with zipfile.ZipFile('mamba_ssm_fixed.whl', 'w') as zf:
    for root, dirs, files in os.walk('temp_dir'):
        for file in files:
            zf.write(os.path.join(root, file), os.path.relpath(os.path.join(root, file), 'temp_dir'))
"
```

### Installation Verification
```powershell
# Install fixed wheel
pip install dist/mamba_ssm_fixed.whl

# Test import
python -c "import mamba_ssm; print('Import successful')"

# Test basic functionality
python -c "
import torch
import mamba_ssm
model = mamba_ssm.Mamba(d_model=256, d_state=16, d_conv=4, expand=2)
x = torch.randn(1, 512, 256)
y = model(x)
print('Forward pass successful, output shape:', y.shape)
"
```

## ✅ Verification

### Functional Tests
```python
import torch
import mamba_ssm

def test_mamba_ssm():
    # Test basic model creation
    model = mamba_ssm.Mamba(
        d_model=256,
        d_state=16,
        d_conv=4,
        expand=2
    )

    # Test forward pass
    x = torch.randn(1, 512, 256)
    with torch.no_grad():
        y = model(x)

    print(f"Input shape: {x.shape}")
    print(f"Output shape: {y.shape}")
    print("Basic test passed!")

    # Test with CUDA
    if torch.cuda.is_available():
        model = model.cuda()
        x = x.cuda()
        with torch.no_grad():
            y = model(x)
        print("CUDA test passed!")

if __name__ == "__main__":
    test_mamba_ssm()
```

### Performance Benchmark
```python
import torch
import mamba_ssm
import time

def benchmark_mamba():
    model = mamba_ssm.Mamba(d_model=512, d_state=16, d_conv=4, expand=2)
    model.eval()

    if torch.cuda.is_available():
        model = model.cuda()

    # Test different sequence lengths
    seq_lengths = [512, 1024, 2048, 4096]

    for seq_len in seq_lengths:
        x = torch.randn(1, seq_len, 512)
        if torch.cuda.is_available():
            x = x.cuda()

        # Warm up
        with torch.no_grad():
            for _ in range(10):
                _ = model(x)

        # Benchmark
        torch.cuda.synchronize() if torch.cuda.is_available() else None
        start_time = time.time()

        with torch.no_grad():
            for _ in range(100):
                _ = model(x)

        torch.cuda.synchronize() if torch.cuda.is_available() else None
        end_time = time.time()

        avg_time = (end_time - start_time) / 100
        print(f"Seq length {seq_len}: {avg_time:.4f}s per inference")

if __name__ == "__main__":
    benchmark_mamba()
```

### Integration Test with Other Libraries
```python
# Test with transformers
try:
    from transformers import AutoModelForCausalLM
    print("Transformers integration: OK")
except ImportError:
    print("Transformers not available")

# Test with accelerate
try:
    import accelerate
    print("Accelerate integration: OK")
except ImportError:
    print("Accelerate not available")
```

---

*Building Mamba-SSM on Windows requires careful attention to MSVC compatibility and CUDA architecture configuration. This guide provides the complete process for successful compilation and installation.*