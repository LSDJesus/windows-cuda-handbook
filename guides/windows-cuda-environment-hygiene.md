# Windows CUDA Build Environment Hygiene Guide

> **🧹 Essential maintenance for stable C++/CUDA compilation on Windows**
>
> This guide covers the foundational system hygiene practices that prevent build failures and ensure reproducible CUDA development environments.

## 📋 Table of Contents

- [Quick Start](#-quick-start)
- [System Detection](#-system-detection)
- [Prerequisites](#-prerequisites)
- [Environment Cleanup](#-environment-cleanup)
- [PATH Management](#-path-management)
- [Virtual Environment Setup](#-virtual-environment-setup)
- [Troubleshooting](#-troubleshooting)
- [Version Compatibility](#-version-compatibility)

## 🚀 Quick Start

### Automated Detection
```powershell
# Run the environment detection script
.\scripts\detect_system.ps1
```

### TL;DR Cleanup Commands
```powershell
# Remove conflicting Python installations
# (Keep only your target version, e.g., 3.12.10)

# Clean PATH environment variables
# Remove old CUDA/Python entries from both User and System PATH

# Disable Windows app execution aliases
# Turn off python.exe and python3.exe aliases in Windows Settings

# Clean orphaned files
# Delete python.exe/python3.exe from C:\WINDOWS\ and C:\WINDOWS\System32\
```

## 🔍 System Detection

### Environment Analysis Script
```powershell
# Check for conflicting installations
python --version
where python
where python3

# Check CUDA installations
nvcc --version
where nvcc

# Check PATH for duplicates
echo $env:PATH
```

### Common Issues to Detect
- Multiple Python versions in PATH
- Conflicting CUDA toolkit installations
- Orphaned symlinks from incomplete uninstalls
- Windows App Execution Aliases enabled
- Mixed 32-bit/64-bit installations

## 📋 Prerequisites

### Required Software
- **Windows 10/11** (64-bit)
- **Single Python version** (3.10+ recommended)
- **Single CUDA Toolkit** (matching PyTorch version)
- **Visual Studio 2022** with C++ build tools
- **Administrator privileges** for system changes

### Recommended Tools
- **uv** for fast Python package management
- **Git for Windows** for source control
- **Windows Terminal** for better shell experience

## 🧹 Environment Cleanup

### Step 1: Python Version Management

**Problem:** Multiple Python versions cause build system confusion.

**Solution:**
1. **Identify all Python installations:**
   ```cmd
   where python
   where python3
   ```

2. **Uninstall extra versions:**
   - Use Windows Settings → Apps → Search for Python
   - Remove all versions except your target (e.g., 3.12.10)
   - Delete any remaining Python directories manually

3. **Clean Python-related environment variables:**
   ```powershell
   # Remove Python from PATH
   $currentPath = [Environment]::GetEnvironmentVariable("PATH", "User")
   $newPath = ($currentPath -split ';' | Where-Object { $_ -notlike "*python*" }) -join ';'
   [Environment]::SetEnvironmentVariable("PATH", $newPath, "User")
   ```

### Step 2: CUDA Toolkit Cleanup

**Problem:** Multiple CUDA versions cause linker confusion.

**Solution:**
1. **Check installed CUDA versions:**
   ```cmd
   nvcc --version
   where nvcc
   ```

2. **Uninstall old versions:**
   - Use Windows Settings → Apps
   - Remove all CUDA versions except your target
   - Keep only CUDA 12.9.1 for PyTorch 2.8.0+cu129

3. **Clean CUDA environment variables:**
   ```powershell
   # Remove old CUDA variables
   [Environment]::SetEnvironmentVariable("CUDA_HOME", $null, "Machine")
   [Environment]::SetEnvironmentVariable("CUDA_PATH", $null, "Machine")
   ```

### Step 3: Windows App Execution Aliases

**Problem:** Windows creates automatic aliases that interfere with Python execution.

**Solution:**
1. **Disable aliases in Windows Settings:**
   - Open Windows Settings → Apps → Advanced app settings
   - Click "App execution aliases"
   - Turn OFF both `python.exe` and `python3.exe` aliases

2. **Verify aliases are disabled:**
   ```cmd
   where python
   # Should NOT show Microsoft Store aliases
   ```

## 🛤️ PATH Management

### Critical PATH Cleanup

**Problem:** Build tools find wrong executables due to PATH pollution.

**Solution:**

1. **Clean User PATH:**
   ```powershell
   # Get current PATH
   $userPath = [Environment]::GetEnvironmentVariable("PATH", "User")

   # Remove problematic entries
   $cleanedPath = $userPath -split ';' | Where-Object {
       $_ -notlike "*python*" -and
       $_ -notlike "*cuda*" -and
       $_ -notlike "*microsoft*" -and
       $_ -notlike "*windowsapps*"
   }

   # Set cleaned PATH
   [Environment]::SetEnvironmentVariable("PATH", ($cleanedPath -join ';'), "User")
   ```

2. **Clean System PATH:**
   ```powershell
   # Similar process for System PATH
   $systemPath = [Environment]::GetEnvironmentVariable("PATH", "Machine")
   # Apply same filtering logic
   ```

3. **Add correct entries:**
   ```powershell
   # Add your target Python
   $pythonPath = "C:\path\to\your\python312"
   $newPath = "$pythonPath;$env:PATH"
   [Environment]::SetEnvironmentVariable("PATH", $newPath, "User")

   # Add your target CUDA
   $cudaPath = "C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v12.9\bin"
   $newPath = "$cudaPath;$env:PATH"
   [Environment]::SetEnvironmentVariable("PATH", $newPath, "User")
   ```

## 🐍 Virtual Environment Setup

### Pristine venv Creation

**Problem:** Global site-packages contamination causes dependency conflicts.

**Solution:**

1. **Create isolated virtual environment:**
   ```cmd
   # Use absolute path to target Python
   "C:\path\to\python312\python.exe" -m venv D:\AI\my_project\venv
   ```

2. **Activate and verify isolation:**
   ```cmd
   D:\AI\my_project\venv\Scripts\activate
   python -c "import sys; print('Python:', sys.executable)"
   python -c "import site; print('Site packages:', site.getsitepackages())"
   ```

3. **Install dependencies locally:**
   ```cmd
   # Never install in global site-packages
   pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu129
   pip install ninja cmake
   ```

## 🔧 Troubleshooting

### Common Issues & Solutions

#### "python command not found"
```cmd
# Check if aliases are disabled
where python

# If showing Microsoft Store, disable aliases in Windows Settings
# Then add your Python to PATH
setx PATH "C:\path\to\python312;%PATH%"
```

#### "nvcc not found" after CUDA install
```cmd
# Check CUDA installation
dir "C:\Program Files\NVIDIA GPU Computing Toolkit"

# Add CUDA to PATH
setx PATH "C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v12.9\bin;%PATH%"
```

#### Build still finds wrong Python
```cmd
# Check which python is being used
python -c "import sys; print(sys.executable)"

# Force specific Python path
"C:\path\to\python312\python.exe" setup.py build_ext --inplace
```

#### CMake finds wrong compiler
```cmd
# Clear CMake cache
del CMakeCache.txt
del /s CMakeFiles\*

# Force Visual Studio generator
cmake -G "Visual Studio 17 2022" -A x64 ..
```

### Diagnostic Commands

```powershell
# Complete environment audit
Write-Host "=== Environment Audit ==="
Write-Host "Python: $(python --version 2>$null)"
Write-Host "CUDA: $(nvcc --version 2>$null | Select-String 'release')"
Write-Host "MSVC: $(cl 2>$null | Select-String 'Version')"
Write-Host "PATH entries: $(($env:PATH -split ';').Count)"
```

## 📊 Version Compatibility

### Supported Configurations

| Component | Version | Notes |
|-----------|---------|-------|
| Windows | 10/11 (64-bit) | Windows 11 recommended |
| Python | 3.10-3.12 | 3.12.10 tested |
| CUDA | 12.1-12.9 | Match PyTorch version |
| PyTorch | 2.6.0+ | cu121/cu129 variants |
| MSVC | 14.3+ | Visual Studio 2022 |

### Known Conflicts

- **Python 3.13+**: Not yet supported by many ML packages
- **CUDA 12.0**: Known issues with some extensions
- **Mixed architectures**: 32-bit/64-bit Python installations
- **Global packages**: Any packages in global site-packages

## 💡 Best Practices

### Environment Management
- **Single Python version** per system
- **One CUDA toolkit** at a time
- **Clean PATH** with no duplicates
- **Isolated venvs** for each project
- **No global packages** for ML development

### Maintenance Routine
```powershell
# Monthly cleanup checklist
# 1. Check for duplicate PATH entries
# 2. Verify Python/CUDA versions
# 3. Clean orphaned files
# 4. Update Visual Studio/Windows SDK
# 5. Test build environment
```

### Prevention Tips
- Always use virtual environments
- Never install ML packages globally
- Keep system PATH clean and minimal
- Use absolute paths in build scripts
- Document your environment setup

---

*This guide ensures your Windows CUDA development environment remains stable and reproducible. A clean system is the foundation of successful compilation.*