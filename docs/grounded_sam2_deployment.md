# Grounded-SAM-2 Deployment

The project uses the official Grounded-SAM-2 architecture directly:

```text
local Grounding DINO (Hugging Face) -> boxes -> official SAM 2.1 -> masks
```

The official repository is installed in editable mode. Changes made inside its
`sam2` package are therefore visible to this project immediately, which keeps
future research modifications separate from the project adapter and evaluation
code.

## Why This Integration Path

The official repository provides both an original Grounding DINO API demo and a
Hugging Face Grounding DINO demo. This project starts with the Hugging Face path
because it is officially supported and does not require compiling Grounding
DINO's custom Deformable Attention operator. SAM 2.1 still comes directly from
the official Grounded-SAM-2 source tree.

Grounding DINO 1.5/1.6 and DINO-X demos require a cloud API token. They are not
the starting point for an offline server.

## Required Local Assets

Prepare these items on a machine with internet access and upload them to the
server:

```text
/data/projects/grounded-visual-assistant/third_party/Grounded-SAM-2/
/data/models/grounding-dino-base/
/data/models/sam2/sam2.1_hiera_base_plus.pt
```

Official sources:

- Grounded-SAM-2: https://github.com/IDEA-Research/Grounded-SAM-2
- Grounding DINO Base: https://huggingface.co/IDEA-Research/grounding-dino-base
- SAM 2: https://github.com/facebookresearch/sam2
- SAM 2.1 Base+ checkpoint:
  https://dl.fbaipublicfiles.com/segment_anything_2/092824/sam2.1_hiera_base_plus.pt

### Windows download examples

Download the Grounding DINO repository with the current Hugging Face CLI:

```powershell
hf download IDEA-Research/grounding-dino-base `
  --local-dir "D:\Models\grounding-dino-base"
```

Download the SAM 2.1 Base+ checkpoint:

```powershell
New-Item -ItemType Directory -Force "D:\Models\sam2"
curl.exe -L `
  "https://dl.fbaipublicfiles.com/segment_anything_2/092824/sam2.1_hiera_base_plus.pt" `
  -o "D:\Models\sam2\sam2.1_hiera_base_plus.pt"
```

The official Grounded-SAM-2 source has already been placed in this project's
`third_party/Grounded-SAM-2` directory. Upload that directory with the project.
If it is missing on another copy, download a Git clone or ZIP and extract the
full repository; the `sam2` directory and `setup.py` must remain together.

## Server Installation

The existing environment already has the versions required by current SAM 2:

```text
torch 2.5.1+cu124
torchvision 0.20.1+cu124
Python 3.10
```

Keep this verified PyTorch installation. Do not follow the upstream generic
`pip install torch` command because it may reinstall an incompatible CUDA wheel.

If the server can reach PyPI, run from this project root:

```bash
conda activate grounded-vlm
bash scripts/install_grounded_sam2.sh
```

For the offline server, first prepare the extra Linux wheels on a networked
Linux x86_64 machine with Python 3.10:

```bash
python -m pip download -r requirements-grounded-sam2.txt -d wheelhouse
```

Upload `wheelhouse` with the project and install without contacting an index:

```bash
conda activate grounded-vlm
OFFLINE=1 WHEELHOUSE="$PWD/wheelhouse" bash scripts/install_grounded_sam2.sh
```

Do not build this wheelhouse on Windows: `pycocotools` and other binary wheels
must match the server's Linux platform. If all listed packages have already
been installed in the environment, use `SKIP_DEPENDENCIES=1` instead.

The install script uses:

```bash
SAM2_BUILD_CUDA=0 python -m pip install --no-build-isolation --no-deps -e <repo-path>
```

Disabling the optional SAM 2 CUDA extension avoids requiring `nvcc` for the
first smoke test. Official SAM 2 documentation states that the extension mainly
affects some post-processing and is not required for most inference results.
Disabling pip build isolation is also intentional: upstream lists PyTorch as a
build dependency, while this project must reuse the already verified CUDA 12.4
wheel instead of resolving a new wheel in a temporary build environment.
Once the pipeline is stable and a matching CUDA 12.4 toolkit is available, it
can be rebuilt with `SAM2_BUILD_CUDA=1`.

## Configure Paths

Check `configs/grounded_sam2.yaml`:

```yaml
grounding:
  model_id: /data/models/grounding-dino-base
  box_threshold: 0.4
  text_threshold: 0.3
  local_files_only: true

sam2:
  checkpoint: /data/models/sam2/sam2.1_hiera_base_plus.pt
  model_config: configs/sam2.1/sam2.1_hiera_b+.yaml

runtime:
  device: cuda
  dtype: float16
```

The SAM model config is resolved from the editable official package. Do not
replace it with an arbitrary filesystem path unless you also change how Hydra
locates the config package.

## Verify Without Loading Weights

```bash
python scripts/check_grounded_sam2_env.py
```

The report should show:

```text
imports_ok: true
torch: 2.5.1+cu124
torch_cuda: 12.4
cuda_available: true
grounding_model_exists: true
sam2_checkpoint_exists: true
```

## Single-Image Smoke Test

Use physical GPU 3. It appears as logical `cuda:0` inside the process:

```bash
CUDA_VISIBLE_DEVICES=3 python scripts/run_grounded_sam2.py \
  --image data/eval_v0/images/000000230993.jpg \
  --prompt "person. umbrella. backpack." \
  --local-files-only
```

Artifacts are saved under:

```text
outputs/grounded_sam2/000000230993/
  grounding_boxes.jpg
  grounded_sam2_masks.jpg
  masks/
  result.json
```

`result.json` follows the official Grounded-SAM-2 schema and adds mask scores,
stage latency, thresholds, model identities, and peak CUDA memory.

## Oracle Batch Evaluation

After the single-image test passes, validate five unique images:

```bash
CUDA_VISIBLE_DEVICES=3 python scripts/batch_eval_grounded_sam2.py \
  --max-images 5 \
  --local-files-only
```

Then resume the same run over all 100 images:

```bash
CUDA_VISIBLE_DEVICES=3 python scripts/batch_eval_grounded_sam2.py \
  --local-files-only
```

The script uses categories and boxes from each `object_listing` record. It
loads Grounding DINO and SAM 2.1 only once, writes every completed image
immediately, and can resume after interruption. `metrics.json` includes
class-aware box Precision/Recall/F1, mean matched IoU, mAP50, per-category
results, latency, throughput, errors, and peak CUDA memory. The first ten images
also receive overlays and individual mask files; use `--visualize-limit 0` to
disable them.

## Standard COCO Box and Segmentation Metrics

After all 100 images complete, restore the original small instances and mask
annotations for the prompted categories:

```bash
python scripts/build_coco_grounding_gt.py
```

Then evaluate the existing JSONL predictions. This step is CPU-only and does
not reload Grounding DINO or SAM 2:

```bash
python scripts/eval_grounded_sam2_coco.py --require-complete
```

The default mask ranking score is the Grounding DINO confidence, matching the
bbox ranking. `--segmentation-score mask` and `--segmentation-score product`
are available as later calibration experiments; use separate output
directories when comparing them.

## Research Extension Points

The adapter intentionally keeps these components separate:

- target phrase construction
- Grounding DINO box and text thresholds
- detector outputs and confidence calibration
- SAM 2 box prompting
- mask selection and refinement
- evidence serialization

This makes it possible to compare the official baseline with phrase refinement,
adaptive thresholds, mask-guided VLM re-answering, uncertainty calibration, and
custom SAM 2 fine-tuning without rewriting the upstream pipeline.
