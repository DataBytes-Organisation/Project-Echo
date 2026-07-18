# Project Setup with `uv`

## Installing `uv`

### Method 1: Install with with curl or powershell to install the install script

`uv` is a command-line tool for easy and quick setting up python dev environment.

**MacOS and Linux:**

```bash
curl -LsSf [https://astral.sh/uv/install.sh](https://astral.sh/uv/install.sh) | sh
```

**Windows:**

```bash
powershell -ExecutionPolicy ByPass -c "irm https://astral.sh/uv/install.ps1 | iex"
```

Please also look at the installation guide from the [official site](https://docs.astral.sh/uv/getting-started/installation/) in case something updated.

### Method 2: Install with pip

```bash
pip install uv
```

### Method 3: Install with Homebrew (macOS)

```bash
brew install uv
```

### Method 4: Install with Cargo (Rust)

```bash
cargo install uv
```

### Method 5: Download from GitHub Releases

Visit [UV's GitHub releases page](https://github.com/astral-sh/uv/releases) and download the appropriate binary for your platform.

## Verifying Installation

After installation, verify UV is working correctly:

```bash
uv --version
```

## Project Setup with UV

### Create Virtual Environment

```bash
uv venv --seed
```

This creates a virtual environment in the `.venv` directory.

### Activate Virtual Environment

**On Unix/macOS:**

```bash
source .venv/bin/activate
```

**On Windows:**
```bash
.venv\Scripts\activate
```

### Install Dependencies

Since the project uses `pyproject.toml`, UV will automatically read the dependencies:

```bash
uv sync
```

Or install all dependencies directly:

```bash
uv pip install audiomentations diskcache hydra-core librosa matplotlib numpy omegaconf pandas soundfile tensorboard tensorboardx torch torchaudio torchvision tqdm umap-learn
```

### 5. Verify Installation

Test that PyTorch is properly installed with CUDA support (if available):

```python
python -c "import torch; print(f'PyTorch version: {torch.__version__}'); print(f'CUDA available: {torch.cuda.is_available()}')"
```

## Additional Setup for Development

### Install Development Dependencies (if you have them)

```bash
uv pip install pytest black isort mypy jupyter
```

## UV Commands Cheat Sheet

| Command | Description |
|---------|-------------|
| `uv pip install <package>` | Install a package |
| `uv pip install -r requirements.txt` | Install from requirements file |
| `uv pip install -e .` | Install current project in editable mode |
| `uv pip freeze` | List installed packages |
| `uv pip uninstall <package>` | Uninstall a package |
| `uv venv` | Create virtual environment |
| `uv pip compile pyproject.toml` | Generate locked requirements |

## Troubleshooting

### Issue: UV not found after installation

**Solution:** Restart your terminal or add UV to your PATH:

```bash
export PATH="$HOME/.local/bin:$PATH"
```

### Issue: Permission errors on Unix systems

**Solution:** Install UV in user space:

```bash
curl -LsSf https://astral.sh/uv/install.sh | sh -s -- --user
```

### Issue: PyTorch CUDA not working

**Solution:** Install PyTorch with specific CUDA version:

```bash
uv pip install torch torchaudio torchvision --index-url https://download.pytorch.org/whl/cu118
```

Replace `cu118` with your CUDA version (cu117, cu121, etc.).

## Migration from pip

UV is designed to be a drop-in replacement for pip. Most pip commands work with UV by simply replacing `pip` with `uv pip`:

```bash
# Old way with pip
pip install torch

# New way with UV
uv pip install torch
```

Your existing `requirements.txt` and `pyproject.toml` files work without modification.
