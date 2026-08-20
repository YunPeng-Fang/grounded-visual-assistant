# Environment Setup

This document records the known-working environment strategy for the current
server. Install PyTorch separately from the remaining Python dependencies so a
later `pip install` does not silently replace it with a CUDA build that requires
a newer NVIDIA driver.

## Server Baseline

The server used for the first successful run has:

- 4 x NVIDIA GeForce RTX 3090, 24 GB each
- NVIDIA driver 550.135
- maximum CUDA driver capability reported by `nvidia-smi`: 12.4
- recommended Python version: 3.10
- inference dtype: `float16`

The `CUDA Version` shown by `nvidia-smi` is the newest CUDA runtime accepted by
the driver. It is not necessarily the CUDA version used to compile the installed
PyTorch wheel. Check the latter with `torch.version.cuda`.

## Clean Online Installation

Create and activate the environment:

```bash
conda create -n grounded-vlm python=3.10 -y
conda activate grounded-vlm
python -m pip install --upgrade pip setuptools wheel
```

Install the locked CUDA 12.4 PyTorch build first:

```bash
python -m pip install -r requirements-torch-cu124.txt
```

Then install the model and project dependencies:

```bash
python -m pip install -r requirements.txt
```

Install the interactive demo dependencies only when the Gradio interface is
needed:

```bash
python -m pip install -r requirements-demo.txt
```

The demo requirements include the base file and pin `numpy==1.26.4`,
`gradio==6.20.0`, and `pandas==2.3.3`. The NumPy pin avoids binary ABI
mismatches with compiled scientific packages in the Python 3.10 environment.

Do not replace the first command with an unqualified `pip install torch`. A
newer PyTorch CUDA wheel may require a driver newer than 550.135.

## Repair an Existing Environment

If the environment reports that the NVIDIA driver is too old, remove only the
PyTorch packages and reinstall the locked pair:

```bash
conda activate grounded-vlm
python -m pip uninstall -y torch torchvision torchaudio
python -m pip install --no-cache-dir -r requirements-torch-cu124.txt
python -m pip install -r requirements.txt
```

There is no need to reinstall the Qwen weights.

## Required Verification

Check the driver-facing PyTorch environment before loading the model:

```bash
python -c "import torch; print('torch:', torch.__version__); print('wheel CUDA:', torch.version.cuda); print('CUDA available:', torch.cuda.is_available()); print('GPU count:', torch.cuda.device_count()); print('GPU 0:', torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'none')"
```

For the locked environment, the important values should be similar to:

```text
torch: 2.5.1+cu124
wheel CUDA: 12.4
CUDA available: True
GPU 0: NVIDIA GeForce RTX 3090
```

Verify Qwen3-VL support separately:

```bash
python -c "import transformers; from transformers import Qwen3VLForConditionalGeneration; print('transformers:', transformers.__version__); print('Qwen3-VL import: OK')"
```

The completed Hard-Dev400 run used `transformers==5.13.1`. Keep that version
when reproducing the saved Qwen3-VL baseline; changing it creates a new runtime
condition and must be recorded in `run_config.json`.

Finally, verify physical GPU 3. Inside the process it is intentionally exposed
as logical `cuda:0`:

```bash
CUDA_VISIBLE_DEVICES=3 python -c "import torch; print(torch.cuda.get_device_name(0)); print(torch.cuda.mem_get_info(0))"
```

## Offline Server Installation

Use a networked Linux x86_64 machine with the same Python 3.10 version as the
server. Download all wheels into one directory:

```bash
mkdir -p wheelhouse
python -m pip download -r requirements-torch-cu124.txt -d wheelhouse
python -m pip download -r requirements.txt -d wheelhouse
python -m pip download -r requirements-grounded-sam2.txt -d wheelhouse
python -m pip download -r requirements-demo.txt -d wheelhouse
```

Copy `wheelhouse`, `requirements-torch-cu124.txt`, and `requirements.txt` to the
server, then install without accessing package indexes:

```bash
conda activate grounded-vlm
python -m pip uninstall -y torch torchvision torchaudio
python -m pip install --no-index --find-links wheelhouse -r requirements-torch-cu124.txt
python -m pip install --no-index --find-links wheelhouse -r requirements.txt
python -m pip install --no-index --find-links wheelhouse -r requirements-grounded-sam2.txt
python -m pip install --no-index --find-links wheelhouse -r requirements-demo.txt
```

Do not prepare Linux wheels on Windows unless you explicitly use matching
`pip download --platform` options. Native extensions are platform-specific.

## First Smoke Test

Keep FlashAttention disabled for the first run. The default PyTorch attention
path is easier to diagnose and is sufficient for a single-image baseline.

```bash
CUDA_VISIBLE_DEVICES=3 python scripts/run_vlm_baseline.py \
  --image data/demo_images/test.jpg \
  --question "What objects are visible in this image?" \
  --local-files-only
```

After a successful run, capture the exact environment for reproducibility:

```bash
python -m pip freeze > environment.lock.txt
```

Keep the generated lock file with experiment artifacts rather than using it as
the cross-machine installation file.

## Common Failure Signatures

### `The NVIDIA driver on your system is too old (found version 12040)`

Cause: the PyTorch wheel was built for a CUDA runtime newer than 12.4.

Fix: reinstall from `requirements-torch-cu124.txt` and rerun the verification
commands above.

### `cannot import name Qwen3VLForConditionalGeneration`

Cause: the installed Transformers build does not expose the Qwen3-VL class.

Fix:

```bash
python -m pip install --upgrade "transformers==5.13.1"
```

### Model loads to 100% but GPU 3 has no memory usage

Check `torch.cuda.is_available()` first. Also remember that after setting
`CUDA_VISIBLE_DEVICES=3`, physical GPU 3 appears inside Python as `cuda:0`.

## References

- PyTorch previous versions: https://pytorch.org/get-started/previous-versions/
- Qwen3-VL repository: https://github.com/QwenLM/Qwen3-VL
- Qwen3-VL model: https://huggingface.co/Qwen/Qwen3-VL-8B-Instruct
