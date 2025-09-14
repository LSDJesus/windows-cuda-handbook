# Flash-Attention Windows Build Guide

> **⚡ Complete guide for building Flash-Attention on Windows with CUDA support**
>
> This comprehensive guide covers all known issues and solutions for compiling Flash-Attention from source on Windows systems, including ABI mismatches, download failures, and GPU architecture targeting.

## 📋 Table of Contents

- [Quick Start](#-quick-start)
- [System Detection](#-system-detection)
- [Prerequisites](#-prerequisites)
- [Build Process](#-build-process)
- [Common Issues & Solutions](#-common-issues--solutions)
- [Version Compatibility](#-version-compatibility)
- [Troubleshooting](#-troubleshooting)
- [Performance Optimization](#-performance-optimization)

## 🚀 Quick Start

### Automated Build Script
```powershell
# One-command build for Flash-Attention
.\scripts\build_flash_attention.ps1
```

### TL;DR Build Commands
```cmd
REM Use Standard System build process
call "C:\Program Files\Microsoft Visual Studio\2022\Community\Common7\Tools\Launch-VsDevShell.ps1"
set DISTUTILS_USE_SDK=1
D:\AI\venv\Scripts\activate
pip wheel . --no-build-isolation --no-deps -w dist
```

### Version Recommendations
- **Stable**: `v2.7.4.post1` (recommended for Windows)
- **Latest**: `v2.8.2+cu128torch2.8.0` (pre-built wheel)
- **Development**: `main` branch (unstable on Windows)

## 🔍 System Detection

### Flash-Attention Compatibility Check
```powershell
# Check PyTorch version and CUDA variant
python -c "import torch; print('PyTorch:', torch.__version__)"
python -c "import torch; print('CUDA:', torch.version.cuda)"

# Check GPU architecture
nvidia-smi --query-gpu=name --format=csv,noheader,nounits

# Verify MSVC environment
cl
```

### System Requirements
- **PyTorch**: 2.6.0+ with CUDA support
- **CUDA**: 12.1+ (matching PyTorch CUDA version)
- **Python**: 3.10-3.12 (64-bit)
- **MSVC**: Visual Studio 2022 (14.3+)
- **GPU**: RTX 30xx/40xx/50xx series

## 📋 Prerequisites

### Required Software
- **Visual Studio 2022** with "Desktop development with C++"
- **CUDA Toolkit** (matching PyTorch version)
- **Python virtual environment** with PyTorch installed
- **Git for Windows** for source checkout

### Recommended Setup
```cmd
REM Create and activate venv
python -m venv D:\AI\flash_attn\venv
D:\AI\flash_attn\venv\Scripts\activate

REM Install PyTorch (choose matching CUDA version)
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
pip install packaging
```

## 🛠️ Build Process

### Step 1: Source Code Preparation

1. **Clone Repository:**
   ```cmd
   git clone https://github.com/Dao-AILab/flash-attention.git
   cd flash-attention
   ```

2. **Checkout Stable Version:**
   ```cmd
   REM Use stable version for Windows builds
   git checkout v2.7.4.post1
   ```

3. **Initialize Submodules:**
   ```cmd
   git submodule update --init --recursive
   ```

### Step 2: Environment Setup

1. **Launch Developer Environment:**
   ```cmd
   REM Use Native Tools Command Prompt for VS 2022
   "C:\Program Files\Microsoft Visual Studio\2022\Community\Common7\Tools\Launch-VsDevShell.ps1"
   ```

2. **Set Build Variables:**
   ```cmd
   set DISTUTILS_USE_SDK=1
   set FLASH_ATTN_CUDA_ARCHS=75;80;86;90;120
   ```

3. **Activate Virtual Environment:**
   ```cmd
   D:\AI\flash_attn\venv\Scripts\activate
   ```

### Step 3: Build Wheel

1. **Build Command:**
   ```cmd
   pip wheel . --no-build-isolation --no-deps -w dist
   ```

2. **Alternative Build Methods:**
   ```cmd
   REM Direct setup.py build
   python setup.py bdist_wheel

   REM Using build module
   python -m build --wheel --no-isolation
   ```

### Step 4: Installation

1. **Install Built Wheel:**
   ```cmd
   pip install dist\flash_attn-*.whl
   ```

2. **Verify Installation:**
   ```python
   import flash_attn
   print("Flash-Attention installed successfully")
   ```

## 🚨 Common Issues & Solutions

### Issue 1: ABI Mismatch Error

**Error:** `ImportError: DLL load failed while importing flash_attn_2_cuda`

**Root Cause:** Wheel compiled against different PyTorch version than runtime.

**Solutions:**

1. **Recompile from Source:**
   ```cmd
   REM Ensure venv has matching PyTorch
   pip install torch==2.8.0+cu129 --index-url https://download.pytorch.org/whl/cu129

   REM Rebuild in same environment
   pip wheel . --no-build-isolation --no-deps -w dist
   ```

2. **Use Pre-built Wheel:**
   ```cmd
   REM Find community-provided wheel matching your PyTorch
   pip install flash_attn-2.8.2+cu128torch2.8.0...whl
   ```

### Issue 2: HTTP 404 Download Error

**Error:** `HTTP Error 404: Not Found` during build

**Root Cause:** `setup.py` tries to download Linux wheels on Windows.

**Solution:**

1. **Edit setup.py:**
   ```python
   # In CachedWheelsCommand.run() method
   # Change: if ...: to if True:
   # This forces local build instead of download
   ```

2. **Manual Edit:**
   ```cmd
   REM Open setup.py in editor
   REM Find CachedWheelsCommand class
   REM Modify the conditional to skip download
   ```

### Issue 3: Ninja Build Stopped Error

**Error:** `ninja: build stopped: subcommand failed`

**Root Cause:** Unstable main branch or version conflicts.

**Solutions:**

1. **Use Stable Version:**
   ```cmd
   git checkout v2.7.4.post1
   ```

2. **Clean Build:**
   ```cmd
   REM Clear any cached builds
   rmdir /s /q build
   pip wheel . --no-build-isolation --no-deps -w dist
   ```

### Issue 4: GPU Architecture Mismatch

**Error:** Runtime errors with "invalid argument"

**Root Cause:** Build doesn't target correct GPU architectures.

**Solution:**

1. **Set Architecture Variable:**
   ```cmd
   REM For RTX 30xx series
   set FLASH_ATTN_CUDA_ARCHS=86

   REM For RTX 40xx series
   set FLASH_ATTN_CUDA_ARCHS=89

   REM For RTX 50xx series
   set FLASH_ATTN_CUDA_ARCHS=120

   REM For multiple GPUs
   set FLASH_ATTN_CUDA_ARCHS=86;120
   ```

2. **Verify Architecture:**
   ```cmd
   REM Check your GPU's compute capability
   nvidia-smi --query-gpu=name,compute_cap --format=csv
   ```

## 📊 Version Compatibility

### PyTorch & CUDA Matrix

| PyTorch Version | CUDA Version | Flash-Attention | Status |
|----------------|--------------|----------------|--------|
| 2.6.0+cu121 | 12.1 | v2.7.4.post1 | ✅ Stable |
| 2.7.0+cu121 | 12.1 | v2.7.4.post1 | ✅ Stable |
| 2.8.0+cu121 | 12.1 | v2.8.0+ | ⚠️ Testing |
| 2.8.0+cu129 | 12.9 | v2.8.2+cu128 | ✅ Pre-built |

### GPU Architecture Support

| GPU Series | Compute Capability | FLASH_ATTN_CUDA_ARCHS |
|------------|-------------------|----------------------|
| RTX 20xx | 7.5 | 75 |
| RTX 30xx | 8.6 | 86 |
| RTX 40xx | 8.9 | 89 |
| RTX 50xx | 9.0/12.0 | 120 |

### Known Working Combinations

**Stable Builds:**
- PyTorch 2.6.0+cu121 + Flash-Attention v2.7.4.post1
- PyTorch 2.7.0+cu121 + Flash-Attention v2.7.4.post1

**Pre-built Wheels:**
- PyTorch 2.8.0+cu129 + flash_attn-2.8.2+cu128torch2.8.0

## 🔧 Troubleshooting

### Build Environment Issues

#### MSVC Not Found
```cmd
REM Ensure using correct developer prompt
"C:\Program Files\Microsoft Visual Studio\2022\Community\Common7\Tools\Launch-VsDevShell.ps1"

REM Verify MSVC
cl
```

#### CUDA Not Found
```cmd
REM Check CUDA installation
nvcc --version

REM Verify CUDA in PATH
where nvcc
```

#### Python Version Mismatch
```cmd
REM Check Python architecture
python -c "import platform; print(platform.architecture())"

REM Should be 64-bit
```

### Runtime Issues

#### Import Errors
```python
# Check installation
python -c "import flash_attn; print('OK')"

# Check CUDA availability
python -c "import torch; print(torch.cuda.is_available())"
```

#### Memory Issues
```python
# Test with smaller batch
import torch
import flash_attn

# Reduce batch size if getting CUDA out of memory
x = torch.randn(4, 1024, 64, device='cuda')  # Smaller batch
flash_attn.flash_attn_func(x, x, x)
```

### Debug Information
```cmd
REM Enable verbose build output
set CMAKE_VERBOSE_MAKEFILE=ON
pip wheel . -v --no-build-isolation --no-deps -w dist

REM Check build logs
type build.log
```

## ⚡ Performance Optimization

### Architecture-Specific Builds

1. **Single GPU System:**
   ```cmd
   REM Build only for your GPU architecture
   set FLASH_ATTN_CUDA_ARCHS=86  # RTX 3080 Ti
   ```

2. **Multi-GPU System:**
   ```cmd
   REM Include all architectures
   set FLASH_ATTN_CUDA_ARCHS=86;120  # RTX 3080 + RTX 5090
   ```

### Memory Optimization

1. **Use Appropriate Precision:**
   ```python
   # Use FP16 for better performance
   x = torch.randn(32, 2048, 64, dtype=torch.float16, device='cuda')
   ```

2. **Batch Size Tuning:**
   ```python
   # Find optimal batch size for your GPU
   for batch_size in [8, 16, 32, 64]:
       x = torch.randn(batch_size, 2048, 64, device='cuda')
       # Time the operation
   ```

### Build Optimization

1. **Parallel Builds:**
   ```cmd
   set MAX_JOBS=8
   pip wheel . --no-build-isolation --no-deps -w dist
   ```

2. **Incremental Builds:**
   ```cmd
   REM Don't clean between builds to save time
   python setup.py build_ext --inplace
   ```

## 📚 Additional Resources

### Community Resources
- [Flash-Attention GitHub Issues](https://github.com/Dao-AILab/flash-attention/issues)
- [PyTorch Forums](https://discuss.pytorch.org/)
- [CUDA Developer Forums](https://forums.developer.nvidia.com/)

### Related Guides
- [Windows CUDA Environment Setup](./windows-cuda-environment-hygiene.md)
- [General Build Process](./windows-cuda-build-process.md)
- [xFormers Build Guide](./xformers-windows-build-guide.md)

### Pre-built Wheels
- Search for community-provided wheels on:
  - GitHub Releases
  - PyTorch community forums
  - CUDA development forums

---

*Flash-Attention is a performance-critical component for modern transformer models. Following this guide ensures reliable builds and optimal performance on Windows CUDA systems.*