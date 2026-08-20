# GitHub Upload Guide

This repository contains local model weights, datasets, cached images, and
experiment outputs that must not be pushed to GitHub. Follow this guide from
the project root after reviewing `.gitignore`.

## 1. Verify excluded assets

The following local directories are intentionally excluded:

- `Qwen3-VL-8B-Instruct/`
- `models/` and `checkpoints/`
- `third_party/`
- `data/raw/` and downloaded dataset images
- `outputs/`, except placeholder files
- temporary exports and downloaded archives

Check representative paths before staging:

```powershell
git check-ignore -v Qwen3-VL-8B-Instruct/model-00001-of-00004.safetensors
git check-ignore -v data/raw/visual_genome/relationships.json
git check-ignore -v outputs/eval_pope_v0/pope-full9000__qwen3-vl-8b-instruct/metrics.json
git check-ignore -v third_party/Grounded-SAM-2
```

Each command should print the matching `.gitignore` rule.

## 2. Initialize and inspect the local repository

```powershell
git init -b main
git add .
git status --short
```

Audit staged file sizes. This command must print nothing:

```powershell
git ls-files | ForEach-Object {
  if (Test-Path -LiteralPath $_) {
    $item = Get-Item -LiteralPath $_
    if ($item.Length -gt 50MB) {
      "{0:N1} MB`t{1}" -f ($item.Length / 1MB), $_
    }
  }
}
```

Also review the complete staged diff:

```powershell
git diff --cached --stat
git diff --cached -- . ':!data/**/*.jsonl'
```

If an unwanted file is staged, add an ignore rule and remove it from the index:

```powershell
git rm --cached -- path/to/file
```

This removes the file from Git tracking without deleting the local copy.

## 3. Run the release checks

```powershell
python -m pytest tests -q
python scripts/launch_demo.py --results-only --server-name 127.0.0.1 --server-port 7860
```

The dashboard command is a manual smoke test. Stop it with `Ctrl+C` after the
Evaluation page loads successfully.

## 4. Create the first commit

Configure Git identity if the machine does not already have one:

```powershell
git config user.name "YOUR NAME"
git config user.email "YOUR_GITHUB_EMAIL"
```

Then commit:

```powershell
git commit -m "Prepare Grounded Visual Assistant for public release"
```

## 5. Create the GitHub repository

On GitHub, create a repository named `grounded-visual-assistant`. Do not add a
README, `.gitignore`, or license on the creation page because the local project
already contains the first two and the license should be chosen explicitly.

For job applications, a public repository is preferable after verifying that
it contains no unpublished paper code, private data, credentials, or assets
whose licenses prohibit redistribution.

## 6. Add the remote and push

Replace `YOUR_GITHUB_USERNAME` with the actual account name:

```powershell
git remote add origin https://github.com/YOUR_GITHUB_USERNAME/grounded-visual-assistant.git
git push -u origin main
```

GitHub password authentication is not supported for command-line pushes. Use a
browser credential flow, Git Credential Manager, SSH, or a GitHub personal
access token when prompted.

## 7. Final GitHub checks

After pushing, verify the repository page:

- README tables and links render correctly.
- No model weights, dataset images, server paths, or credentials are exposed.
- Installation and results-only demo instructions are visible.
- The repository description and topics include `vision-language-model`,
  `visual-grounding`, `segment-anything`, `qwen-vl`, and `multimodal`.
- Add one architecture image and two qualitative examples when their source
  licenses permit redistribution.

Recommended repository description:

> Evidence-grounded visual QA with Qwen3-VL, Grounding DINO, SAM 2.1, locked evaluation, and failure auditing.

## Release boundary

Do not claim that the project trains Qwen3-VL, Grounding DINO, or SAM 2.1 from
scratch. Do not claim SOTA performance or a verified reduction in hallucination.
The supported claim is that the project provides reproducible, pixel-level
evidence localization and evaluates post-hoc verification under frozen gates.
