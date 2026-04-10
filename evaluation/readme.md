# CountBench Evaluation

Counting accuracy evaluation for NUMINA using [GroundingDINO](https://github.com/IDEA-Research/GroundingDINO) open-set object detection.

## Overview

For each generated video, we extract all frames, detect instances of each target noun using GroundingDINO, and compare detected counts against ground-truth targets. The accuracy is averaged per category within each frame, then averaged across frames and prompts.

**Scoring:** For a frame with *N* noun categories, if *k* categories have correct counts, the frame score is *k/N*. Per-prompt accuracy is the mean frame score across all frames. Overall accuracy is the mean across all prompts.

## Setup

### 1. Install GroundingDINO

```bash
git clone https://github.com/IDEA-Research/GroundingDINO.git
cd GroundingDINO
pip install -e .
```

> Make sure `CUDA_HOME` is set and matches your PyTorch CUDA version:
> ```bash
> export CUDA_HOME=/usr/local/cuda-12.x
> python -c "import torch; print(torch.version.cuda)"  # should match
> ```
> Please follow the [GroundingDINO](https://github.com/IDEA-Research/GroundingDINO) repo for detailed installation precautions.

### 2. Download GroundingDINO weights

```bash
mkdir -p weights
wget -P weights/ https://github.com/IDEA-Research/GroundingDINO/releases/download/v0.1.0-alpha/groundingdino_swint_ogc.pth
```


## Input Files

**`prompts.txt`** — One prompt per line:
```
Three cats chasing two dogs.
Four hikers planting two flags.
A lone astronaut floating in space.
```

**`noun_counts.jsonl`** — One JSON dict per line, matching prompts 1:1:
```json
{"cats": 3, "dogs": 2}
{"hikers": 4, "flags": 2}
{"astronaut": 1}
```

A prompt may contain 1, 2, or 3 noun categories. Background elements (sky, beach, jungle, etc.) are excluded.

## Run Evaluation

```bash
python evaluation/eval_counting.py \
    --video_dir path/to/your/videos/ \
    --noun_counts_file evaluation/noun_counts.jsonl \
    --prompt_file evaluation/prompts.txt \
    --gdino_config GroundingDINO/groundingdino/config/GroundingDINO_SwinT_OGC.py \
    --gdino_weights GroundingDINO/weights/groundingdino_swint_ogc.pth \
    --save_results evaluation/eval_results.json
```

### Arguments

| Argument | Default | Description |
|----------|---------|-------------|
| `--video_dir` | (required) | Directory containing generated `.mp4` files |
| `--noun_counts_file` | (required) | JSONL file with target counts |
| `--prompt_file` | `None` | Prompts file (for terminal display) |
| `--gdino_config` | `GroundingDINO/.../SwinT_OGC.py` | GroundingDINO config path |
| `--gdino_weights` | `weights/groundingdino_swint_ogc.pth` | GroundingDINO checkpoint path |
| `--start_idx` | `1` | First prompt index (1-based) |
| `--end_idx` | `None` | Last prompt index (inclusive) |
| `--save_results` | `eval_results.json` | Output JSON path |

## Output

### Terminal output

Per-frame detection results for every video:

```
[1/210] Three cats chasing two dogs
  Targets: {"cats": 3, "dogs": 2}
  Video:   001_Three_cats_chasing_two_dogs.mp4
  Frames:  81
    Frame | cats(3) | dogs(2) | Score
  ----------------------------------------
        1 |    3✓   |    2✓   | 1.00
        2 |    3✓   |    1✗   | 0.50
        3 |    2✗   |    2✓   | 0.50
      ...
  Accuracy: 0.7531  (2 categories)

================================================================
Overall accuracy: 0.7234 (72.3%)
Evaluated: 210 videos
```

### JSON output (`eval_results.json`)

```json
{
  "overall_accuracy": 0.7234,
  "num_prompts": 210,
  "per_prompt": [
    {
      "index": 1,
      "prompt": "Three cats chasing two dogs",
      "targets": {"cats": 3, "dogs": 2},
      "num_frames": 81,
      "per_noun": {
        "cats": [3, 3, 2, 3, ...],
        "dogs": [2, 1, 2, 2, ...]
      },
      "prompt_accuracy": 0.7531
    }
  ]
}
```
