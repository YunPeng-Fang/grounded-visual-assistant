# Local Qwen-VL Model Deployment

The current baseline script performs **local inference** with Hugging Face
`transformers`. It does not call a remote inference API.

Set up and verify the CUDA/PyTorch environment first. See
[`environment_setup.md`](environment_setup.md). The known server configuration
requires the CUDA 12.4 PyTorch wheel; installing a newer CUDA wheel causes the
`driver is too old (found version 12040)` error.

However, if `model_id` is a remote repository name such as
`Qwen/Qwen3-VL-8B-Instruct` and the weights are not cached locally,
`from_pretrained` will try to download model files before running inference.

## Recommended Flow

## Recommended Save Path

Do **not** save model weights inside this Git project unless you have a special
reason. The model files are large and should live in a shared model directory.

Recommended path on a Linux server:

```bash
/data/models/Qwen3-VL-8B-Instruct
```

If you do not have permission to write `/data`, use:

```bash
~/models/Qwen3-VL-8B-Instruct
```

The project config should point to this directory after the files are copied.

## Manual Download URLs

Primary model for this project:

- Hugging Face: https://huggingface.co/Qwen/Qwen3-VL-8B-Instruct
- ModelScope: https://modelscope.cn/models/Qwen/Qwen3-VL-8B-Instruct

Stable fallback model:

- Hugging Face: https://huggingface.co/Qwen/Qwen2.5-VL-7B-Instruct
- ModelScope: https://modelscope.cn/models/Qwen/Qwen2.5-VL-7B-Instruct

Smaller debugging model:

- Hugging Face: https://huggingface.co/Qwen/Qwen2.5-VL-3B-Instruct
- ModelScope: https://modelscope.cn/models/Qwen/Qwen2.5-VL-3B-Instruct

For an RTX 3090 server, start with Qwen3-VL-8B if your dependencies support it.
Use Qwen2.5-VL-7B if you want the most stable fallback. Use the 3B model only
if download/storage/debugging speed matters more than answer quality.

Qwen3-VL requires a Transformers build containing
`Qwen3VLForConditionalGeneration`. The completed Hard-Dev400 run was verified
with `transformers==5.13.1`, Python 3.10.20, and PyTorch 2.5.1+cu124. The project
dependency file pins that tested Transformers version for reproducibility.

```bash
python -m pip install --upgrade "transformers==5.13.1"
```

### 1. Download or cache the model once

Option A: use the current Hugging Face CLI directly.

```bash
hf download Qwen/Qwen3-VL-8B-Instruct
```

Option B: download into an explicit directory.

```bash
hf download Qwen/Qwen3-VL-8B-Instruct \
  --local-dir /data/models/Qwen3-VL-8B-Instruct
```

If Hugging Face is slow or blocked, use a local mirror or ModelScope according
to your server environment.

### Windows download command

If PowerShell reports `huggingface-cli` cannot be found, install the current
Hugging Face CLI first:

```powershell
python -m pip install -U huggingface_hub hf_xet
```

Then check the CLI:

```powershell
hf --help
```

Download the full model repository:

```powershell
hf download Qwen/Qwen3-VL-8B-Instruct `
  --local-dir "D:\Models\Qwen3-VL-8B-Instruct"
```

Do not put the model weights inside the Git project unless you really want to.
If you do save under this project, prefer `models`, not `modes`.

### 2. Point the project to the local path

Edit `configs/default.yaml`:

```yaml
model:
  model_id: /data/models/Qwen3-VL-8B-Instruct
  torch_dtype: float16
  device_map: auto
  max_new_tokens: 256
  local_files_only: true
```

If you used the home directory path, use:

```yaml
model:
  model_id: /home/<your_user>/models/Qwen3-VL-8B-Instruct
  torch_dtype: float16
  device_map: auto
  max_new_tokens: 256
  local_files_only: true
```

## What Files Should Exist

After manual download and copy, the model directory should contain files similar
to:

```text
config.json
generation_config.json
preprocessor_config.json
processor_config.json
tokenizer.json
tokenizer_config.json
vocab.json / merges.txt or tokenizer-related files
model-00001-of-xxxxx.safetensors
model-00002-of-xxxxx.safetensors
...
model.safetensors.index.json
```

The exact shard count may change, so do not worry if the number of
`safetensors` files is different. The key point is that config, processor,
tokenizer, and model weight files must be in the same directory.

### 3. Run inference on the idle GPU

```bash
CUDA_VISIBLE_DEVICES=3 python scripts/run_vlm_baseline.py \
  --image data/demo_images/example.jpg \
  --question "What objects are visible in this image?" \
  --local-files-only
```

With `--local-files-only`, the script will fail immediately if the model is not
available locally. This is useful for confirming that no download is happening.

## Notes for RTX 3090

For RTX 3090, start with:

```yaml
torch_dtype: float16
```

Avoid `bfloat16` unless your driver, PyTorch, and GPU runtime are known to
support it well.
