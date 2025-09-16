## Large Files (Wheels)

This repository uses Git LFS to manage large wheel files in the `wheel/` directory. By default, these files are **not downloaded** when you clone the repository to keep download sizes manageable.

### Configuration Applied

The repository is configured to exclude wheel files from automatic downloads:
- `git config lfs.fetchexclude "wheel/"` - Excludes from automatic fetches
- `git config lfs.pullexclude "wheel/"` - Excludes from automatic pulls

### Downloading Wheel Files

If you need the wheel files, you can download them using one of these methods:

#### Option 1: Download specific files
```bash
# Download all wheel files
git lfs pull

# Download files from a specific directory
git lfs pull --include="wheel/"

# Download a specific file
git lfs pull --include="wheel/some-package.whl"
```

#### Option 2: Include wheels in future pulls
```bash
# Configure to always download wheel files
git config lfs.fetchexclude ""
git config lfs.pullexclude ""
```

#### Option 3: Selective checkout
```bash
# Use sparse checkout to only get certain directories
git sparse-checkout set --no-cone "guides/" "!wheel/"
```

### Repository Structure

- `guides/` - CUDA development guides and documentation
- `wheel/` - Prebuilt wheel files (Git LFS tracked, not auto-downloaded)
- `lfs-server-setup/` - Infrastructure for Git LFS server (not included in clones)

### Setup

1. Clone the repository:
   ```bash
   git clone https://github.com/LSDJesus/windows-cuda-handbook.git
   cd windows-cuda-handbook
   ```

2. Install Git LFS:
   ```bash
   git lfs install
   ```

3. Download wheel files (optional):
   ```bash
   git lfs pull --include="wheel/"
   ```

### Contributing

When adding new wheel files:
1. Ensure they're placed in the `wheel/` directory
2. They will be automatically tracked by Git LFS due to the `.gitattributes` configuration
3. Commit and push as usual - Git LFS will handle the upload to Google Cloud Storage