# xFormers Windows Build Guide

> **🔄 Complete guide for building xFormers on Windows with CUDA support**
>
> This comprehensive guide covers building xFormers from source on Windows systems, including setup.py modifications, GPU architecture targeting, and common build failures.

## 📋 Table of Contents

- [Quick Start](#-quick-start)
- [System Detection](#-system-detection)
- [Prerequisites](#-prerequisites)
- [Source Preparation](#-source-preparation)
- [Build Process](#-build-process)
- [Common Issues & Solutions](#-common-issues--solutions)
- [Version Compatibility](#-version-compatibility)
- [Troubleshooting](#-troubleshooting)
- [Performance Optimization](#-performance-optimization)

## 🚀 Quick Start

### Automated Build Script
```powershell
# One-command build for xFormers
.\scripts\build_xformers.ps1
```

### TL;DR Build Commands
```cmd
REM Launch developer environment
"C:\Program Files\Microsoft Visual Studio\2022\Community\Common7\Tools\Launch-VsDevShell.ps1"

REM Set build variables
set DISTUTILS_USE_SDK=1
set TORCH_CUDA_ARCH_LIST=8.6;12.0

REM Activate venv and build
D:\AI\venv\Scripts\activate
pip wheel . --no-build-isolation --no-deps -w dist
```

### Version Recommendations
- **Stable**: `v0.0.32.post2` (recommended for Windows)
- **Latest**: `v0.0.33+` (may have Python tagging issues)
- **Development**: `main` branch (requires manual fixes)

## 🔍 System Detection

### xFormers Compatibility Check
```powershell
# Check PyTorch and CUDA
python -c "import torch; print('PyTorch:', torch.__version__)"
python -c "import torch; print('CUDA:', torch.version.cuda)"

# Check GPU architecture
nvidia-smi --query-gpu=name --format=csv,noheader,nounits

# Verify MSVC environment
cl

# Check Git (for submodules)
git --version
```

### System Requirements
- **PyTorch**: 2.6.0+ with CUDA support
- **CUDA**: 12.1+ (matching PyTorch CUDA version)
- **Python**: 3.10-3.12 (64-bit)
- **MSVC**: Visual Studio 2022 (14.3+)
- **Git**: For submodule management
- **GPU**: RTX 30xx/40xx/50xx series

## 📋 Prerequisites

### Required Software
- **Visual Studio 2022** with "Desktop development with C++"
- **CUDA Toolkit** (matching PyTorch version)
- **Python virtual environment** with PyTorch installed
- **Git for Windows** for source and submodules

### Recommended Setup
```cmd
REM Create isolated environment
python -m venv D:\AI\xformers\venv
D:\AI\xformers\venv\Scripts\activate

REM Install PyTorch (match your CUDA version)
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
pip install packaging wheel setuptools
```

## 📥 Source Preparation

### Step 1: Repository Setup

1. **Clone Repository:**
   ```cmd
   git clone https://github.com/facebookresearch/xformers.git
   cd xformers
   ```

2. **Checkout Stable Version:**
   ```cmd
   REM Use stable version for Windows compatibility
   git checkout v0.0.32.post2
   ```

3. **Initialize Submodules:**
   ```cmd
   git submodule update --init --recursive
   ```

### Step 2: Handle Long Path Issues

**Problem:** Windows has legacy path length limitations.

**Solution:**
```cmd
REM Enable long paths for Git (one-time setup)
git config --system core.longpaths true

REM If already cloned, clean and retry
REM Delete xformers directory and start over
```

### Step 3: Setup.py Modification (CRITICAL)

**Problem:** xFormers has hardcoded Python version tagging that causes build failures.

**Solution:**

1. **Open setup.py:**
   ```cmd
   REM Use your preferred text editor
   notepad setup.py
   ```

2. **Find the setup() call** (near the end of the file)

3. **Locate the options parameter:**
   ```python
   # FIND THIS LINE:
   options={"bdist_wheel": {"py_limited_api": "cp39"}},

   # REPLACE WITH (match your Python version):
   # For Python 3.12:
   # (remove the entire options line - delete it)

   # For Python 3.11:
   # options={"bdist_wheel": {"py_limited_api": "cp311"}},

   # For Python 3.10:
   # options={"bdist_wheel": {"py_limited_api": "cp310"}},
   ```

4. **Save and close the file**

**Why this matters:** The hardcoded `cp39` tag causes pip to reject the wheel if you're using Python 3.10+.

## 🛠️ Build Process

### Step 1: Environment Setup

1. **Launch Developer Command Prompt:**
   ```cmd
   REM CRITICAL: Use Native Tools Command Prompt
   "C:\Program Files\Microsoft Visual Studio\2022\Community\Common7\Tools\Launch-VsDevShell.ps1"
   ```

2. **Set Essential Environment Variables:**
   ```cmd
   set DISTUTILS_USE_SDK=1
   set TORCH_CUDA_ARCH_LIST=8.6;12.0
   ```

3. **Activate Virtual Environment:**
   ```cmd
   D:\AI\xformers\venv\Scripts\activate
   ```

### Step 2: Build Wheel

1. **Execute Build:**
   ```cmd
   pip wheel . --no-build-isolation --no-deps -w dist
   ```

2. **Monitor Build Progress:**
   - Build typically takes 10-30 minutes
   - Watch for CUDA compilation messages
   - Check for any error messages

3. **Alternative Build Methods:**
   ```cmd
   REM Direct setup.py build
   python setup.py bdist_wheel

   REM Using build module
   python -m build --wheel --no-isolation
   ```

### Step 3: Installation & Verification

1. **Install Built Wheel:**
   ```cmd
   pip install dist\xformers-*.whl
   ```

2. **Verify Installation:**
   ```python
   import xformers
   print("xFormers version:", xformers.__version__)

   # Test basic functionality
   import torch
   from xformers.components import MultiHeadDispatch
   print("xFormers imported successfully")
   ```

## 🚨 Common Issues & Solutions

### Issue 1: Compiler Not Found

**Error:** `Cannot open include file: 'assert.h'` or `cl.exe not found`

**Root Cause:** Not using the correct developer command prompt.

**Solution:**
```cmd
REM MUST use Native Tools Command Prompt
"C:\Program Files\Microsoft Visual Studio\2022\Community\Common7\Tools\Launch-VsDevShell.ps1"

REM Verify environment
cl
where cl
```

### Issue 2: Python Version Tagging Error

**Error:** Wheel installation fails with version mismatch

**Root Cause:** Hardcoded `cp39` in setup.py doesn't match your Python version.

**Solution:** Follow the setup.py modification in the Source Preparation section above.

### Issue 3: Git Submodule Failures

**Error:** `Filename too long` during submodule update

**Root Cause:** Windows path length limitations.

**Solutions:**

1. **Enable Long Paths:**
   ```cmd
   git config --system core.longpaths true
   ```

2. **Clean and Retry:**
   ```cmd
   REM Delete and re-clone
   cd ..
   rmdir /s /q xformers
   git clone https://github.com/facebookresearch/xformers.git
   cd xformers
   git checkout v0.0.32.post2
   git submodule update --init --recursive
   ```

### Issue 4: CUDA Architecture Mismatch

**Error:** Runtime errors or "invalid argument" at runtime

**Root Cause:** Build not targeting correct GPU architectures.

**Solution:**

1. **Set Architecture List:**
   ```cmd
   REM For RTX 30xx series
   set TORCH_CUDA_ARCH_LIST=8.6

   REM For RTX 40xx series
   set TORCH_CUDA_ARCH_LIST=8.9

   REM For RTX 50xx series
   set TORCH_CUDA_ARCH_LIST=12.0

   REM For multiple GPUs (no quotes!)
   set TORCH_CUDA_ARCH_LIST=8.6;12.0
   ```

2. **Verify Your GPU:**
   ```cmd
   nvidia-smi --query-gpu=name,compute_cap --format=csv
   ```

### Issue 5: Build Directory Conflicts

**Error:** CMake errors or build failures

**Root Cause:** Leftover build artifacts from previous attempts.

**Solution:**
```cmd
REM Clean build directory
rmdir /s /q build
rmdir /s /q dist

REM Clear CMake cache
del CMakeCache.txt
del /s CMakeFiles\*

REM Retry build
pip wheel . --no-build-isolation --no-deps -w dist
```

## 📊 Version Compatibility

### PyTorch & CUDA Matrix

| PyTorch Version | CUDA Version | xFormers Version | Status |
|----------------|--------------|-----------------|--------|
| 2.6.0+cu121 | 12.1 | v0.0.32.post2 | ✅ Stable |
| 2.7.0+cu121 | 12.1 | v0.0.32.post2 | ✅ Stable |
| 2.8.0+cu121 | 12.1 | v0.0.33+ | ⚠️ Testing |
| 2.8.0+cu129 | 12.9 | v0.0.33+ | ⚠️ Testing |

### GPU Architecture Support

| GPU Series | Compute Capability | TORCH_CUDA_ARCH_LIST |
|------------|-------------------|---------------------|
| RTX 20xx | 7.5 | 7.5 |
| RTX 30xx | 8.6 | 8.6 |
| RTX 40xx | 8.9 | 8.9 |
| RTX 50xx | 9.0/12.0 | 12.0 |

### Python Version Tagging

| Python Version | py_limited_api Tag | setup.py Modification |
|----------------|-------------------|----------------------|
| 3.10 | cp310 | `options={"bdist_wheel": {"py_limited_api": "cp310"}}` |
| 3.11 | cp311 | `options={"bdist_wheel": {"py_limited_api": "cp311"}}` |
| 3.12 | cp312 | Remove options line entirely |

## 🔧 Troubleshooting

### Build Environment Issues

#### Wrong Command Prompt
```cmd
REM Check if using correct environment
echo %VSINSTALLDIR%
echo %VCINSTALLDIR%

REM Should point to Visual Studio 2022
```

#### Missing CUDA
```cmd
REM Verify CUDA installation
nvcc --version
where nvcc

REM Check CUDA paths
echo %CUDA_HOME%
echo %CUDA_PATH%
```

#### Python Architecture Mismatch
```cmd
REM Check Python architecture
python -c "import platform; print(platform.architecture())"

REM Should be ('64bit', 'WindowsPE')
```

### Runtime Issues

#### Import Errors
```python
# Test import
python -c "import xformers; print('OK')"

# Check for CUDA support
python -c "import torch; print('CUDA available:', torch.cuda.is_available())"
```

#### Version Conflicts
```python
# Check versions
import xformers
import torch
print("xFormers:", xformers.__version__)
print("PyTorch:", torch.__version__)
```

### Debug Build Process
```cmd
REM Enable verbose output
set CMAKE_VERBOSE_MAKEFILE=ON
pip wheel . -v --no-build-isolation --no-deps -w dist

REM Check all log files
dir /s *.log
type build.log
```

## ⚡ Performance Optimization

### Architecture-Specific Builds

1. **Single GPU Optimization:**
   ```cmd
   REM Build only for your specific GPU
   set TORCH_CUDA_ARCH_LIST=8.6  # RTX 3080 Ti
   ```

2. **Multi-GPU Support:**
   ```cmd
   REM Include all your GPUs
   set TORCH_CUDA_ARCH_LIST=8.6;12.0  # RTX 3080 + RTX 5090
   ```

### Build Optimization

1. **Parallel Compilation:**
   ```cmd
   set MAX_JOBS=8
   pip wheel . --no-build-isolation --no-deps -w dist
   ```

2. **Incremental Builds:**
   ```cmd
   REM Use build_ext for faster iteration
   python setup.py build_ext --inplace
   ```

### Memory Optimization

1. **GPU Memory Management:**
   ```python
   # Test with appropriate batch sizes
   import torch
   from xformers.components import MultiHeadDispatch

   # Start with smaller sequences
   x = torch.randn(4, 512, 768, device='cuda')
   attn = MultiHeadDispatch()
   ```

## 📚 Additional Resources

### Official Resources
- [xFormers GitHub Repository](https://github.com/facebookresearch/xformers)
- [xFormers Documentation](https://facebookresearch.github.io/xformers/)
- [PyTorch Documentation](https://pytorch.org/docs/)

### Community Resources
- [PyTorch Forums](https://discuss.pytorch.org/)
- [CUDA Developer Forums](https://forums.developer.nvidia.com/)

### Related Guides
- [Windows CUDA Environment Setup](./windows-cuda-environment-hygiene.md)
- [General Build Process](./windows-cuda-build-process.md)
- [Flash-Attention Build Guide](./flash-attention-windows-guide.md)

---

*xFormers is a critical component for efficient transformer implementations. This guide ensures successful compilation and optimal performance on Windows CUDA systems.*