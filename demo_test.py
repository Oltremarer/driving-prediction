import argparse
import json
import os

import numpy as np
import torch
from sklearn.metrics import average_precision_score, roc_auc_score
from torch.utils.data import DataLoader
from tqdm import tqdm

from mmau_train import MMAU_TRAIN
from mmau_test import MMAU_TEST
from model import LightDrivingRiskPredictor


def parse_args():
    parser = argparse.ArgumentParser(description="Evaluate and export predictions for MMAU dataset.")
    parser.add_argument("--dataset-root", type=str, default="/home/msi/driving-risk-prediction/MMAU_TRAIN",
                        help="Path to dataset root that contains train.txt/testing.txt and video folder.")
    parser.add_argument("--phase", type=str, choices=["train", "test"], default="test",
                        help="Which split to run on. Use 'train' for internal validation, 'test' for benchmark split.")
    parser.add_argument("--checkpoint", type=str, required=True,
                        help="Path to the trained model checkpoint (.pth).")
    parser.add_argument("--batch-size", type=int, default=1, help="Inference batch size.")
    parser.add_argument("--num-workers", type=int, default=2, help="DataLoader worker count.")
    parser.add_argument("--output-path", type=str, default="prediction/prediction.json",
                        help="Where to save the prediction json.")
    parser.add_argument("--metrics-path", type=str, default=None,
                        help="Optional path to save computed metrics as json.")
    parser.add_argument("--threshold", type=float, default=0.5,
                        help="Risk threshold used for TTA/STTA.")
    parser.add_argument("--max-bbox-n", type=int, default=10,
                        help="Must match the max_N used when training the model.")
    parser.add_argument("--device", type=str, default=None,
                        help="Device to use. Defaults to cuda if available else cpu.")
    parser.add_argument("--skip-eval", action="store_true",
                        help="Skip metric computation (useful for unlabeled test split).")
    parser.add_argument("--data-aug", action="store_true",
                        help="Use data augmentation when reading videos (default: disabled for deterministic eval).")
    return parser.parse_args()


def build_dataset(args):
    if args.phase == "train":
        dataset = MMAU_TRAIN(root_path=args.dataset_root, phase="train", data_aug=args.data_aug)
    else:
        dataset = MMAU_TEST(root_path=args.dataset_root, phase="test", data_aug=args.data_aug)
    return dataset


def load_model(checkpoint_path, device, max_bbox_n):
    model = LightDrivingRiskPredictor(max_bbox_n=max_bbox_n)
    checkpoint = torch.load(checkpoint_path, map_location=device)
    state_dict = checkpoint.get("model_state_dict", checkpoint)
    model.load_state_dict(state_dict)
    model.to(device)
    model.eval()
    return model


def move_batch_to_device(batch, device):
    batch_on_device = {}
    for key, value in batch.items():
        if torch.is_tensor(value):
            batch_on_device[key] = value.to(device)
        else:
            batch_on_device[key] = value
    return batch_on_device


def compute_tta(seq, tai, threshold):
    if tai < 0:
        return None
    seq = np.asarray(seq)
    tai = int(min(max(tai, 0), len(seq) - 1))
    indices = np.where(seq > threshold)[0]
    if len(indices) == 0:
        return 0.0
    t_a = indices[0]
    return max(float(tai - t_a), 0.0)


def compute_stta(seq, tai, threshold):
    if tai < 0:
        return None
    seq = np.asarray(seq)
    tai = int(min(max(tai, 0), len(seq) - 1))
    indices = np.where(seq > threshold)[0]
    for idx in indices:
        if idx > tai:
            break
        window = seq[idx:tai + 1]
        if window.size > 0 and np.all(window > threshold):
            return max(float(tai - idx), 0.0)
    return 0.0


def evaluate_predictions(pred_sequences, labels, tais, threshold):
    video_scores = []
    label_list = []
    tta_scores = []
    stta_scores = []

    for seq, label, tai in zip(pred_sequences, labels, tais):
        seq = np.asarray(seq)
        video_scores.append(float(seq.max()))
        label_list.append(int(label))

        if int(label) == 1 and tai >= 0:
            tta = compute_tta(seq, tai, threshold)
            stta = compute_stta(seq, tai, threshold)
            if tta is not None:
                tta_scores.append(tta)
            if stta is not None:
                stta_scores.append(stta)

    metrics = {}
    unique_labels = np.unique(label_list)
    if unique_labels.size > 1:
        metrics["auc"] = float(roc_auc_score(label_list, video_scores))
        metrics["ap"] = float(average_precision_score(label_list, video_scores))
    else:
        metrics["auc"] = None
        metrics["ap"] = None

    metrics["tta@0.5"] = float(np.mean(tta_scores)) if tta_scores else None
    metrics["stta@0.5"] = float(np.mean(stta_scores)) if stta_scores else None

    return metrics


def main():
    args = parse_args()
    device = torch.device(args.device) if args.device else torch.device(
        "cuda" if torch.cuda.is_available() else "cpu"
    )
    print(f"Running inference on {device}.")

    dataset = build_dataset(args)
    dataloader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=torch.cuda.is_available(),
    )

    model = load_model(args.checkpoint, device, args.max_bbox_n)

    predictions = {}
    seq_cache = []
    labels = []
    tais = []

    with torch.no_grad():
        for batch in tqdm(dataloader, desc="Inferencing"):
            batch_on_device = move_batch_to_device(batch, device)
            outputs = model(batch_on_device)  # (B, F)
            outputs_cpu = outputs.detach().cpu()

            video_names = batch["video_name"]
            label_tensor = batch["label"]
            tai_tensor = batch["tai"]

            for idx in range(outputs_cpu.size(0)):
                video_id = video_names[idx]
                frame_scores = outputs_cpu[idx].tolist()
                frame_scores = [float(x) for x in frame_scores]
                predictions[video_id] = frame_scores

                seq_cache.append(frame_scores)
                labels.append(int(label_tensor[idx]))
                tais.append(int(tai_tensor[idx]))

    output_dir = os.path.dirname(args.output_path)
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)
    with open(args.output_path, "w", encoding="utf-8") as f:
        json.dump(predictions, f, indent=2)
    print(f"Saved predictions to {args.output_path}")

    if not args.skip_eval:
        metrics = evaluate_predictions(seq_cache, labels, tais, args.threshold)
        print("Evaluation metrics:")
        for key, value in metrics.items():
            print(f"  {key}: {value}")

        if args.metrics_path:
            metrics_dir = os.path.dirname(args.metrics_path)
            if metrics_dir:
                os.makedirs(metrics_dir, exist_ok=True)
            with open(args.metrics_path, "w", encoding="utf-8") as f:
                json.dump(metrics, f, indent=2)
            print(f"Saved metrics to {args.metrics_path}")


if __name__ == "__main__":
    main()

