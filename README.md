# Driving Risk Prediction Competition

Implementation for the Codabench challenge “Traffic Accident Risk Prediction in Autonomous Driving Scenarios”:  
https://www.codabench.org/competitions/11247/#/pages-tab

## Repository Layout
- `mmau_train.py`, `mmau_test.py`: dataset loaders for training/testing splits
- `demo_train.py`: main training script (multimodal GRU baseline)
- `demo_test.py`: inference + metric evaluation + submission export
- `model.py`: model definitions (light-weight multi-modal encoder)
- `requirements.txt`: Python dependencies

## Dataset Layout
Download `MMAU_TRAIN_2025` (train) and `MMAU_TEST_2025` (test) from the competition website and place them like:

```
MMAU_TRAIN/
├── detection/
│   ├── 1/
│   ├── 2/
│   └── ...
├── video/
│   ├── 1/
│   ├── 2/
│   └── ...
├── accident_type.txt
└── train.txt

MMAU_TEST/
├── detection/
├── video/
└── testing.txt
```

Each `video/<category>/<video_id>/` contains `images/000001.jpg … 000150.jpg` and `maps/000001.png … 000150.png`. Each detection JSON lives under `detection/<category>/<video_id>/000001.json …`.

## Environment Setup
NVIDIA 50-series GPUs (e.g., 5070 Ti) require CUDA 12.x compatible PyTorch.

```bash
conda create -n mmau_env python=3.10 -y
conda activate mmau_env
pip install -r requirements.txt
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121  # adjust if needed
```

## Training
```bash
python demo_train.py \
    --dataset-root /path/to/MMAU_TRAIN \
    --checkpoint-dir checkpoints \
    --num-epochs 10 \
    --batch-size 1 \
    --learning-rate 1e-4 \
    --max-bbox-n 10
```

Important flags:
- `--dataset-root`: directory containing `train.txt`, `video/`, `detection/`
- `--resume-checkpoint`: resume from `checkpoints/model_epoch_X.pth`
- `--save-interval`: checkpoint frequency
- `--device cuda:0` (optional) to force a device

## Evaluation / Submission Export
`demo_test.py` works for both validation (train split) and official test split.

```bash
# Validate on training split to inspect metrics
python demo_test.py \
    --dataset-root /path/to/MMAU_TRAIN \
    --phase train \
    --checkpoint checkpoints/model_epoch_10.pth \
    --output-path prediction/train_prediction.json \
    --metrics-path prediction/train_metrics.json

# Generate submission on official test split (no labels, so skip metrics)
python demo_test.py \
    --dataset-root /path/to/MMAU_TEST \
    --phase test \
    --checkpoint checkpoints/model_epoch_10.pth \
    --output-path prediction/prediction.json \
    --skip-eval
```

`demo_test.py` outputs per-video 150-length probability sequences keyed by `"<category>/<video_id>"`. This JSON must be zipped before submission:

```bash
cd prediction
zip submission.zip prediction.json
```

Upload `submission.zip` to Codabench.

## Remote Execution Workflow (if you do not have a GPU)
1. Modify/verify the code locally, then push everything to your GitHub repository (include this README).
2. Share the repo URL plus the commands above with a teammate who has a GPU.
3. The teammate clones the repo, follows the Environment / Training / Evaluation sections, and uploads the resulting `submission.zip`.

## Notes / Tips
- Ensure `--max-bbox-n` matches the preprocessing cap on bounding boxes (default 10).  
- Checkpoints are stored as `checkpoints/model_epoch_*.pth` by default.  
- `demo_test.py` can compute AUC/AP/TTA/STTA when labels are available (`--phase train`).  
- Use `--skip-eval` on unlabeled test data to avoid metric warnings.

