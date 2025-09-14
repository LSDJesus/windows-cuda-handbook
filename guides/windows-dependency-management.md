# Windows Python Dependency Management Guide

> **📦 Mastering virtual environments and dependency reproducibility on Windows**
>
> This guide covers advanced Python dependency management techniques for complex ML projects, focusing on virtual environment management, dependency freezing, and reproducible builds.

## 📋 Table of Contents

- [Quick Start](#-quick-start)
- [Understanding Dependency Conflicts](#-understanding-dependency-conflicts)
- [Virtual Environment Management](#-virtual-environment-management)
- [Dependency Analysis](#-dependency-analysis)
- [Reproducible Environments](#-reproducible-environments)
- [Common Issues & Solutions](#-common-issues--solutions)
- [Advanced Techniques](#-advanced-techniques)
- [Troubleshooting](#-troubleshooting)

## 🚀 Quick Start

### Environment Analysis
```powershell
# Check current environment
python -c "import sys; print('Python:', sys.executable)"
python -c "import site; print('Site packages:', site.getsitepackages())"

# Analyze dependencies
uv pip check
uv pip list --format=freeze > current_env.txt
```

### Create Reproducible Environment
```powershell
# Create golden environment
python -m venv golden_env
golden_env\Scripts\activate
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121

# Freeze dependencies
uv pip freeze > golden_requirements.txt

# Recreate environment
python -m venv new_env
new_env\Scripts\activate
uv pip install --no-deps -r golden_requirements.txt
```

## 🔍 Understanding Dependency Conflicts

### Types of Conflicts

#### Logical Conflicts
**Problem:** Package requires `numpy>=1.20` but another requires `numpy<1.20`

**Detection:**
```powershell
uv pip check
# Shows logical incompatibilities
```

**Solution:**
```powershell
# Use --no-deps to bypass resolver
uv pip install --no-deps -r requirements.txt
```

#### Runtime Conflicts
**Problem:** Package works in isolation but fails with others

**Detection:**
```python
# Test imports
import package1
import package2  # Fails here
```

**Solution:**
```powershell
# Create separate environments
python -m venv env1
python -m venv env2
```

#### Platform Conflicts
**Problem:** Package has different dependencies on different platforms

**Detection:**
```powershell
# Check platform-specific requirements
pip show package_name
```

## 🐍 Virtual Environment Management

### Best Practices

#### Environment Isolation
```powershell
# Create project-specific venv
python -m venv D:\AI\my_project\venv

# Use absolute paths
D:\AI\my_project\venv\Scripts\activate

# Verify isolation
python -c "import sys; print(sys.executable)"
```

#### Environment Naming Convention
```powershell
# Use descriptive names
python -m venv venv_torch212_cu121
python -m venv venv_stable_diffusion
python -m venv venv_comfyui
```

#### Environment Documentation
```powershell
# Document environment purpose
echo "PyTorch 2.1.2 with CUDA 12.1" > venv/README.md
echo "Created: $(Get-Date)" >> venv/README.md
```

### Environment Switching

#### Quick Switching Script
```powershell
# Create activation script
function Activate-Venv {
    param([string]$VenvPath)
    & "$VenvPath\Scripts\activate"
}

# Usage
Activate-Venv "D:\AI\project1\venv"
Activate-Venv "D:\AI\project2\venv"
```

#### Environment Variables
```powershell
# Set project-specific variables
$env:PYTHONPATH = "D:\AI\my_project\src"
$env:TORCH_HOME = "D:\AI\models\torch"
```

## 🔍 Dependency Analysis

### Current Environment Analysis

#### Package Inventory
```powershell
# List all packages
uv pip list

# Detailed package info
uv pip show torch torchvision torchaudio

# Check for outdated packages
uv pip list --outdated
```

#### Dependency Tree
```powershell
# Show dependency relationships
uv pip list --format=freeze | head -20

# Check reverse dependencies
pip show torch | Select-String "Requires:"
```

### Conflict Detection

#### Automated Checking
```powershell
# Use uv for advanced conflict detection
uv pip check

# Manual conflict detection
python -c "
import pkg_resources
for dist in pkg_resources.working_set:
    print(f'{dist.project_name}=={dist.version}')
"
```

#### Version Pinning Analysis
```powershell
# Check for unpinned dependencies
uv pip list --format=freeze | Select-String "==" | Measure-Object
uv pip list --format=freeze | Select-String ">=" | Measure-Object
```

## 🔄 Reproducible Environments

### Golden Environment Creation

#### Step 1: Create Base Environment
```powershell
# Start with clean venv
python -m venv golden_env
golden_env\Scripts\activate

# Install core dependencies
uv pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
uv pip install numpy scipy pillow
```

#### Step 2: Install Project Dependencies
```powershell
# Install from requirements or setup.py
uv pip install -e .
# OR
uv pip install -r requirements.txt
```

#### Step 3: Install Local Packages
```powershell
# Install wheels from local builds
uv pip install D:\builds\xformers\dist\xformers-*.whl
uv pip install D:\builds\flash_attn\dist\flash_attn-*.whl
```

#### Step 4: Freeze Environment
```powershell
# Create comprehensive requirements file
uv pip freeze > golden_requirements.txt

# Include local file paths
echo "# Local wheels" >> golden_requirements.txt
echo "xformers @ file:///D:/builds/xformers/dist/xformers-0.0.32.post2-cp312-cp312-win_amd64.whl" >> golden_requirements.txt
```

### Environment Recreation

#### From Frozen Requirements
```powershell
# Create new environment
python -m venv new_env
new_env\Scripts\activate

# Install without dependency resolution
uv pip install --no-deps -r golden_requirements.txt
```

#### Validation
```powershell
# Test all imports
python -c "
import torch
import torchvision
import xformers
import flash_attn
print('All imports successful')
"

# Verify versions
python -c "
import torch
print('PyTorch:', torch.__version__)
print('CUDA:', torch.version.cuda)
"
```

## 🚨 Common Issues & Solutions

### Issue 1: Dependency Resolution Failures

**Error:** `ERROR: Cannot install -r requirements.txt because of dependency conflicts`

**Root Cause:** Strict dependency resolver rejects logical conflicts.

**Solutions:**

1. **Use --no-deps:**
   ```powershell
   uv pip install --no-deps -r requirements.txt
   ```

2. **Manual resolution:**
   ```powershell
   # Install conflicting packages separately
   uv pip install "numpy>=1.20,<1.25"
   uv pip install "scipy>=1.7,<1.11"
   ```

3. **Create compatibility layer:**
   ```powershell
   # Use compatible versions
   uv pip install numpy==1.24.3 scipy==1.10.1
   ```

### Issue 2: Local Package Installation

**Error:** `ERROR: torch-2.1.2+cu121-torch-2.1.2+cu121-win_amd64.whl is not a valid wheel filename`

**Root Cause:** Incorrect wheel filename or path.

**Solutions:**

1. **Verify wheel exists:**
   ```powershell
   Get-ChildItem D:\builds\*.whl
   ```

2. **Use correct path format:**
   ```powershell
   uv pip install "torch @ file:///D:/builds/torch-2.1.2+cu121-torch-2.1.2+cu121-win_amd64.whl"
   ```

3. **Install from directory:**
   ```powershell
   uv pip install D:\builds\torch-2.1.2+cu121-torch-2.1.2+cu121-win_amd64.whl
   ```

### Issue 3: Environment Path Issues

**Error:** `ModuleNotFoundError` after environment switch

**Root Cause:** PYTHONPATH or script path issues.

**Solutions:**

1. **Check executable:**
   ```powershell
   python -c "import sys; print(sys.executable)"
   ```

2. **Reset paths:**
   ```powershell
   $env:PYTHONPATH = ""
   $env:PATH = "C:\Windows\system32;C:\Windows"
   # Re-activate venv
   ```

3. **Use absolute imports:**
   ```python
   # Instead of: from . import module
   # Use: from mypackage import module
   ```

### Issue 4: Git Repository Dependencies

**Error:** `ERROR: No matching distribution found` for Git packages

**Root Cause:** Git dependencies not properly specified.

**Solutions:**

1. **Use correct Git URL format:**
   ```powershell
   uv pip install "package @ git+https://github.com/user/repo.git@v1.0.0"
   ```

2. **Install from local Git clone:**
   ```powershell
   git clone https://github.com/user/repo.git
   cd repo
   uv pip install -e .
   ```

3. **Freeze Git dependencies:**
   ```powershell
   uv pip freeze | Select-String "@ git+" >> requirements.txt
   ```

## 🔧 Advanced Techniques

### Multi-Environment Management

#### Environment Matrix
```powershell
# Create multiple environments for testing
$environments = @(
    @{Name = "torch21_cu118"; PyTorch = "2.1.0+cu118"},
    @{Name = "torch21_cu121"; PyTorch = "2.1.0+cu121"},
    @{Name = "torch22_cu118"; PyTorch = "2.2.0+cu118"}
)

foreach ($env in $environments) {
    python -m venv "venv_$($env.Name)"
    & "venv_$($env.Name)\Scripts\activate"
    uv pip install "torch==$($env.PyTorch)" --index-url https://download.pytorch.org/whl/cu118
    uv pip freeze > "requirements_$($env.Name).txt"
}
```

#### Environment Switching Script
```powershell
# PowerShell profile addition
function Set-ProjectEnv {
    param([string]$ProjectName)
    $venvPath = "D:\AI\$ProjectName\venv"
    if (Test-Path $venvPath) {
        & "$venvPath\Scripts\activate"
        Write-Host "Activated $ProjectName environment"
    } else {
        Write-Host "Environment not found: $venvPath"
    }
}
```

### Dependency Optimization

#### Minimal Requirements
```powershell
# Create minimal requirements for production
uv pip freeze | Where-Object { $_ -notlike "*test*" -and $_ -notlike "*dev*" } > requirements_minimal.txt
```

#### Development vs Production
```powershell
# Development requirements
uv pip freeze > requirements_dev.txt

# Production requirements (subset)
@(
    "torch",
    "torchvision",
    "numpy",
    "pillow"
) | ForEach-Object {
    uv pip show $_ | Select-String "Version:" | ForEach-Object {
        "$_==$($_.Line -replace 'Version: ')" >> requirements_prod.txt
    }
}
```

## 🔧 Troubleshooting

### Environment Diagnostics

#### Complete Environment Audit
```powershell
Write-Host "=== Environment Audit ==="
Write-Host "Python: $(python --version)"
Write-Host "Executable: $(python -c "import sys; print(sys.executable)")"
Write-Host "Site packages: $(python -c "import site; print(site.getsitepackages()[0])")"
Write-Host "PATH entries: $(($env:PATH -split ';').Count)"
Write-Host "PYTHONPATH: $env:PYTHONPATH"
```

#### Package Conflict Analysis
```powershell
# Find duplicate packages
uv pip list --format=json | ConvertFrom-Json | Group-Object name | Where-Object Count -gt 1

# Check for incompatible versions
uv pip check
```

#### Import Path Debugging
```python
import sys
print("Python path:")
for path in sys.path:
    print(f"  {path}")

print("\nPackage locations:")
import torch
print(f"torch: {torch.__file__}")
```

### Recovery Procedures

#### Environment Reset
```powershell
# Complete environment recreation
Remove-Item venv -Recurse -Force
python -m venv venv
venv\Scripts\activate
uv pip install --no-deps -r golden_requirements.txt
```

#### Dependency Cleanup
```powershell
# Remove orphaned packages
uv pip list --format=freeze | ForEach-Object {
    $package = $_ -split '==' | Select-Object -First 1
    try {
        python -c "import $package" 2>$null
    } catch {
        Write-Host "Orphaned: $package"
        uv pip uninstall $package -y
    }
}
```

## 📊 Best Practices Summary

### Environment Management
- **One venv per project** - Avoid dependency conflicts
- **Use absolute paths** - Prevent path resolution issues
- **Document environments** - Track purpose and creation date
- **Regular audits** - Check for conflicts and outdated packages

### Dependency Management
- **Pin versions** - Ensure reproducibility
- **Use --no-deps** - Bypass resolver conflicts
- **Separate concerns** - Dev vs production requirements
- **Test regularly** - Validate environment integrity

### Troubleshooting
- **Check executable** - Verify correct Python/venv
- **Audit paths** - Clean PATH and PYTHONPATH
- **Use uv** - Better dependency resolution
- **Document fixes** - Track successful resolutions

---

*Mastering Python dependency management is crucial for stable ML development. This guide provides the techniques needed for reproducible, conflict-free environments on Windows.*