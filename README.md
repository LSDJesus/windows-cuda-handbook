# Windows CUDA Development Guides

> **🛠️ Comprehensive collection of Windows CUDA development guides for AI/ML libraries**
>
> A curated collection of professional guides covering Windows CUDA development, troubleshooting, and optimization for popular AI/ML libraries including PyTorch, Mamba-SSM, Flash-Attention, xFormers, and more.

## 📦 Prebuilt Wheels

This repository includes prebuilt wheel files for common CUDA-enabled packages. These are stored using **Git LFS** with Google Cloud Storage backend to keep repository sizes manageable.

### 🚫 Automatic Download Disabled

By default, wheel files in the `wheel/` directory are **not downloaded** when cloning this repository. This prevents large downloads for users who only need the documentation.

**Configuration applied:**
- `git config lfs.fetchexclude "wheel/"` - Excludes wheel directory from automatic fetches
- `git config lfs.pullexclude "wheel/"` - Excludes wheel directory from automatic pulls

### 📥 Downloading Wheels When Needed

If you need the prebuilt wheels:

```bash
# Download all wheel files
git lfs pull --include="wheel/"

# Or download everything
git lfs pull
```

To re-enable automatic downloads for wheel files:
```bash
git config lfs.fetchexclude ""
git config lfs.pullexclude ""
```

### 🔧 Repository Setup

```bash
# Clone the repository
git clone https://github.com/LSDJesus/windows-cuda-handbook.git
cd windows-cuda-handbook

# Install Git LFS
git lfs install

# Optional: Download wheel files
git lfs pull --include="wheel/"
```

## 📚 Guide Collection

### Core Windows CUDA Development
- [**Windows CUDA Environment Hygiene**](guides/windows-cuda-environment-hygiene.md) - System cleanup, PATH management, virtual environments
- [**Windows CUDA Build Process**](guides/windows-cuda-build-process.md) - Smart vs Standard build strategies, MSVC setup
- [**Windows Dependency Management**](guides/windows-dependency-management.md) - Virtual environment management, reproducible builds

### Library-Specific Build Guides
- [**Mamba-SSM Windows Build Guide**](guides/mamba-ssm-windows-build-guide.md) - Complete guide for building Mamba-SSM with CUDA 12.9 & PyTorch 2.8.0
- [**Flash-Attention Windows Guide**](guides/flash-attention-windows-guide.md) - Build guide with ABI mismatch solutions
- [**xFormers Windows Build Guide**](guides/xformers-windows-build-guide.md) - Comprehensive xFormers compilation guide
- [**TensorRT Integration Guide**](guides/tensorrt-integration-guide.md) - TensorRT for ComfyUI with LoRA compatibility

### Advanced Topics
- [**ComfyUI Development Guide**](guides/comfyui-development-guide.md) - Custom nodes, extensions, and ComfyUI development
- [**Google Cloud SDK Guide**](guides/google-cloud-sdk-guide.md) - Google Cloud SDK setup and AI Platform workflows
- [**Advanced CUDA Development Guide**](guides/advanced-cuda-development-guide.md) - Profiling, optimization, multi-threading

## 🚀 Quick Start

### For New Windows CUDA Developers
1. Start with [**Windows CUDA Environment Hygiene**](guides/windows-cuda-environment-hygiene.md)
2. Learn the [**Windows CUDA Build Process**](guides/windows-cuda-build-process.md)
3. Master [**Windows Dependency Management**](guides/windows-dependency-management.md)

### For Specific Library Builds
- **Mamba-SSM**: Follow the [**Mamba-SSM Windows Build Guide**](guides/mamba-ssm-windows-build-guide.md)
- **Flash-Attention**: Use the [**Flash-Attention Windows Guide**](guides/flash-attention-windows-guide.md)
- **xFormers**: Refer to the [**xFormers Windows Build Guide**](guides/xformers-windows-build-guide.md)
- **TensorRT**: Check the [**TensorRT Integration Guide**](guides/tensorrt-integration-guide.md)

### For Advanced Development
- **ComfyUI Extensions**: See [**ComfyUI Development Guide**](guides/comfyui-development-guide.md)
- **Cloud ML**: Follow [**Google Cloud SDK Guide**](guides/google-cloud-sdk-guide.md)
- **Performance Optimization**: Study [**Advanced CUDA Development Guide**](guides/advanced-cuda-development-guide.md)

## 💻 System Requirements

### Minimum Requirements
- **OS**: Windows 10/11 Pro/Enterprise
- **Python**: 3.10-3.12
- **CUDA**: 12.1+ (tested with 12.9)
- **MSVC**: Visual Studio 2022 with C++ build tools
- **GPU**: NVIDIA GPU with CUDA support

### Recommended Setup
- **RAM**: 32GB+
- **GPU VRAM**: 8GB+
- **Storage**: 50GB+ free space
- **Internet**: Stable connection for package downloads

## ✅ Prerequisites Checklist

### Environment Setup
- [ ] [Windows CUDA Environment Hygiene](guides/windows-cuda-environment-hygiene.md) completed
- [ ] Python virtual environment configured
- [ ] CUDA toolkit installed and configured
- [ ] MSVC build tools installed
- [ ] PyTorch with CUDA support installed

### Development Tools
- [ ] Git for version control
- [ ] CMake and Ninja build systems
- [ ] PowerShell 7+ for scripting
- [ ] VS Code with Python/C++ extensions (recommended)

## 🎯 Common Use Cases

### Building from Source
```powershell
# General build pattern
git clone <repository>
cd <repository>
python setup.py bdist_wheel
pip install dist/*.whl
```

### Troubleshooting Builds
1. Check [Windows CUDA Build Process](guides/windows-cuda-build-process.md) for common issues
2. Verify environment with [Windows CUDA Environment Hygiene](guides/windows-cuda-environment-hygiene.md)
3. Review dependency conflicts in [Windows Dependency Management](guides/windows-dependency-management.md)

### Performance Optimization
- Use [Advanced CUDA Development Guide](guides/advanced-cuda-development-guide.md) for profiling
- Consider [TensorRT Integration Guide](guides/tensorrt-integration-guide.md) for inference optimization
- Review [Google Cloud SDK Guide](guides/google-cloud-sdk-guide.md) for cloud-based training

## 📁 Guide Categories

### 🏗️ Build & Compilation
- Windows-specific build configurations
- MSVC compatibility fixes
- CUDA architecture optimization
- Cross-platform build strategies

### 🐛 Troubleshooting & Debugging
- Common compilation errors
- Environment configuration issues
- Dependency resolution problems
- Performance bottleneck identification

### ⚡ Performance & Optimization
- CUDA kernel optimization
- Memory management techniques
- Multi-GPU configurations
- Profiling and benchmarking

### ☁️ Cloud & Deployment
- Google Cloud Platform integration
- Container deployment strategies
- Distributed training setup
- Model serving optimization

## 📖 Reading Guide

### For Beginners
Start with the core Windows CUDA development guides to establish a solid foundation before tackling specific library builds.

### For Experienced Developers
Jump directly to library-specific guides or advanced topics based on your current needs and familiarity with Windows CUDA development.

### For Library Maintainers
The advanced guides provide deep insights into optimization techniques and cross-platform compatibility considerations.

## 📋 Additional Resources

- [**Repository Setup Guide**](guides/README.md) - Detailed setup instructions and wheel management
- [**Contributing Guidelines**](guides/README.md#contributing) - How to contribute new guides

## 🤝 Contributing

### Adding New Guides
1. Follow the established format and structure
2. Include comprehensive table of contents
3. Provide working code examples
4. Test on multiple Windows configurations
5. Update this main README with links

### Guide Standards
- Professional, technical writing style
- Comprehensive troubleshooting sections
- Working code examples with explanations
- Version compatibility matrices
- Performance benchmarking where applicable

## 📄 License

This collection of guides is licensed under the MIT License. See [LICENSE](LICENSE) for details.

Individual guides may reference third-party libraries and tools that have their own licenses and terms of use.

## 🙏 Acknowledgments

These guides were developed through extensive testing and troubleshooting of Windows CUDA development workflows. Special thanks to the open-source community for providing the underlying libraries and tools that make this work possible.

---

*This repository serves as a comprehensive resource for Windows CUDA development. Whether you're building AI/ML libraries from source or optimizing existing workflows, these guides provide the knowledge and techniques needed for success.*</content>
<parameter name="filePath">D:\AI\Github Desktop\windows-cuda-handbook\README.md