# Dependency specification

The experiment runner uses Python 3.12. Its third-party Python dependencies are pinned in [`requirements-pytorch.txt`](requirements-pytorch.txt):

| Package | Pinned version | Use |
| --- | --- | --- |
| `numpy` | `2.5.1` | Client data, schedules, and metrics |
| `Pillow` | `12.3.0` | Image support required by `torchvision` |
| `torch` | `2.13.0+cu126` | Models, SGD, tensors, and checkpoint loading |
| `torchvision` | `0.28.0+cu126` | MobileNetV2 and ResNet model definitions |

The `+cu126` wheels are CUDA 12.6 builds obtained through the PyTorch wheel index specified in the requirements file. GPU execution needs an NVIDIA driver compatible with those wheels. The manifest uses `device: auto`, so the code can also select a CPU, but the recorded GPU wall times should not be expected on a CPU. Other modules imported by the repository's Python files are part of the Python standard library. `pip` installs the packages required transitively by the four pinned distributions.

## Install and verify

From the repository root, create a Python 3.12 virtual environment and install the pinned requirements. On Windows PowerShell:

```powershell
py -3.12 -m venv .venv
.\.venv\Scripts\python.exe -m pip install --upgrade pip
.\.venv\Scripts\python.exe -m pip install -r requirements-pytorch.txt
.\.venv\Scripts\python.exe -m pip check
```

On Linux, substitute `python3.12 -m venv .venv` and `.venv/bin/python -m pip ...`; a compatible NVIDIA driver is still needed for GPU execution. Check the resolved versions and selected device with:

```powershell
.\.venv\Scripts\python.exe -c "import sys, numpy, PIL, torch, torchvision; print(sys.version); print('numpy', numpy.__version__, 'Pillow', PIL.__version__); print('torch', torch.__version__, 'torchvision', torchvision.__version__); print('CUDA available', torch.cuda.is_available())"
```

The experiment data partitions and bundled MobileNetV2 checkpoint are runtime inputs, not Python packages. Their locations, download links, and archive checksums are documented in the [README](README.md). Run commands and the eight experiment configurations are also documented there.
