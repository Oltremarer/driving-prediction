import argparse
import os

import torch
from torch import nn
from torch.utils.data import DataLoader
from tqdm import tqdm

from mmau_train import MMAU_TRAIN
from model import LightDrivingRiskPredictor

# def main():
#     # --- 参数设置 ---
#     ROOT_PATH = "/home/msi/driving-risk-prediction/MMAU_TRAIN"
#     # BATCH_SIZE = 4
#     BATCH_SIZE = 1
#     NUM_WORKERS = 1
#     DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
#     print(f"Using {DEVICE} device to run.")

#     # --- 创建数据集和加载器 ---
#     train_dataset = MMAU_TRAIN(root_path=ROOT_PATH, phase="train", data_aug=True)
#     train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=NUM_WORKERS)

#     # --- 实例化模型 ---
#     # 注意这里的max_bbox_n需要和你数据预处理中的max_N保持一致
#     model = DrivingRiskPredictor(max_bbox_n=10).to(DEVICE)

#     # --- 定义损失函数和优化器 ---
#     # 这是一个逐帧的二分类问题（高风险/低风险），Binary Cross Entropy是理想选择
#     # criterion = nn.CrossEntropyLoss()
#     criterion = nn.BCELoss()
#     optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)

#     # 打印模型结构，检查一下
#     # print(model)

#     # --- 简单的训练示例 ---
#     num_epochs = 10
#     for epoch in tqdm(range(num_epochs)):
#         model.train()
#         total_loss = 0
#         for i, batch in enumerate(train_loader):
#             # 将数据移动到GPU
#             for key, value in batch.items():
#                 if isinstance(value, torch.Tensor):
#                     batch[key] = value.to(DEVICE)

#             # 1. 前向传播
#             risk_predictions = model(batch) # (B, F)
            
#             # 2. 计算损失
#             # 我们需要一个目标（ground truth）风险标签，比赛中没有直接提供
#             # 核心思路：对于事故视频，事故发生前的一小段时间(例如t_ai之前)风险应该逐渐升高，事故后为1；正常视频风险始终为0。
#             # 这里我们创建一个简单的"软标签"作为示例
#             labels = batch['label'].float() # 0 for normal, 1 for accident
#             tai = batch['tai']
            
#             # 创建一个(B, F)的目标张量
#             target_risk = torch.zeros_like(risk_predictions)
#             for j in range(len(labels)):
#                 if labels[j] == 1:
#                     # 假设视频有150帧，事故发生在tai帧
#                     accident_frame = int(tai[j])
#                     # 从事故前30帧开始，风险线性增长到1
#                     start_ramp_frame = max(0, accident_frame - 30)
#                     if accident_frame > start_ramp_frame:
#                         ramp = torch.linspace(0, 1, accident_frame - start_ramp_frame).to(DEVICE)
#                         target_risk[j, start_ramp_frame:accident_frame] = ramp
#                     target_risk[j, accident_frame:] = 1.0

#             loss = criterion(risk_predictions, target_risk)
            
#             # 3. 反向传播和优化
#             optimizer.zero_grad()
#             loss.backward()
#             optimizer.step()
            
#             total_loss += loss.item()
#             if (i+1) % 50 == 0:
#                 print(f"Epoch [{epoch+1}/{num_epochs}], Step [{i+1}/{len(train_loader)}], Loss: {loss.item():.4f}")

#         print(f"Epoch [{epoch+1}/{num_epochs}] finished. Average Loss: {total_loss / len(train_loader):.4f}")

    # --- 推理和生成提交文件示例 ---
    # (假设你已经有了 test_loader)
    # test_dataset = MMAU_TRAIN(root_path=ROOT_PATH, phase="test", data_aug=True)
    # test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=4)

    # model.eval()
    # predictions_dict = {}
    # with torch.no_grad():
    #     for batch in test_loader: # test_loader的batch_size应为1
    #         # 将数据移动到GPU
    #         for key, value in batch.items():
    #             if isinstance(value, torch.Tensor):
    #                 batch[key] = value.to(DEVICE)

    #         risk_scores = model(batch) # (1, F)
            
    #         video_id_full = batch['video_name'][0] # '1/009334'
    #         video_id_key = f"video_{batch['accident_id'][0]}" # 'video_9334'
            
    #         predictions_dict[video_id_key] = {
    #             "video_id": video_id_full,
    #             "risk": risk_scores.squeeze().cpu().tolist() # 转换成list
    #         }

    #     import json
    #     with open("prediction.json", "w") as f:
    #         json.dump(predictions_dict, f, indent=4)

def parse_args():
    parser = argparse.ArgumentParser(description="Train LightDrivingRiskPredictor.")
    parser.add_argument("--dataset-root", type=str, default="/home/msi/driving-risk-prediction/MMAU_TRAIN",
                        help="Path to MMAU_TRAIN root folder.")
    parser.add_argument("--batch-size", type=int, default=1, help="Training batch size.")
    parser.add_argument("--num-workers", type=int, default=1, help="Dataloader worker count.")
    parser.add_argument("--num-epochs", type=int, default=1, help="Total training epochs.")
    parser.add_argument("--learning-rate", type=float, default=1e-4, help="Optimizer learning rate.")
    parser.add_argument("--checkpoint-dir", type=str, default="checkpoints", help="Directory to save checkpoints.")
    parser.add_argument("--resume-checkpoint", type=str, default=None,
                        help="Path to checkpoint to resume from.")
    parser.add_argument("--save-interval", type=int, default=1,
                        help="Save checkpoint every N epochs.")
    parser.add_argument("--max-bbox-n", type=int, default=10,
                        help="Max number of objects per frame (must match preprocessing).")
    parser.add_argument("--device", type=str, default=None,
                        help="Device to use (default: auto detect).")
    return parser.parse_args()


def main():
    args = parse_args()
    device = torch.device(args.device) if args.device else torch.device(
        "cuda" if torch.cuda.is_available() else "cpu"
    )
    print(f"Using {device} device to run.")

    # --- 创建保存Checkpoint的目录 ---
    os.makedirs(args.checkpoint_dir, exist_ok=True)

    # --- 创建数据集和加载器 ---
    train_dataset = MMAU_TRAIN(root_path=args.dataset_root, phase="train", data_aug=True)
    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=torch.cuda.is_available(),
    )

    # --- 实例化模型 ---
    model = LightDrivingRiskPredictor(max_bbox_n=args.max_bbox_n).to(device)

    # --- 定义损失函数和优化器 ---
    criterion = nn.BCELoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.learning_rate)
    
    # --- 加载Checkpoint (如果存在) ---
    start_epoch = 0
    if args.resume_checkpoint and os.path.exists(args.resume_checkpoint):
        print(f"Resuming training from checkpoint: {args.resume_checkpoint}")
        checkpoint = torch.load(args.resume_checkpoint, map_location=device)
        
        model.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        start_epoch = checkpoint['epoch'] + 1  # 从下一个epoch开始
        
        print(f"Resumed from epoch {checkpoint['epoch']}. Starting training at epoch {start_epoch}.")
    else:
        print("Starting training from scratch.")

    # --- 简单的训练示例 ---
    for epoch in range(start_epoch, args.num_epochs):
        model.train()
        total_loss = 0
        for i, batch in enumerate(train_loader):
            # 将数据移动到GPU
            for key, value in batch.items():
                if isinstance(value, torch.Tensor):
                    batch[key] = value.to(device)

            # 1. 前向传播
            risk_predictions = model(batch)

            # 2. 计算损失
            labels = batch['label'].float()
            tai = batch['tai']
            
            target_risk = torch.zeros_like(risk_predictions)

            seq_len = risk_predictions.size(1)

            for j in range(len(labels)):
                if labels[j] == 1:
                    # accident_frame = int(tai[j])
                    # start_ramp_frame = max(0, accident_frame - 30)

                    accident_frame = min(int(tai[j]), seq_len - 1)  # 确保不超过序列长度
                    start_ramp_frame = max(0, accident_frame - 30)

                    # print(f"Sample {j}: accident_frame={accident_frame}, start_ramp_frame={start_ramp_frame}")

                    ramp_length = accident_frame - start_ramp_frame
                    if ramp_length > 0:
                        ramp = torch.linspace(0, 1, ramp_length).to(device)

                        end_frame = min(accident_frame, seq_len)
                        target_slice_length = end_frame - start_ramp_frame

                        if target_slice_length != ramp_length:
                            ramp = torch.linspace(0, 1, target_slice_length).to(device)
                        target_risk[j, start_ramp_frame:end_frame] = ramp

                    target_risk[j, accident_frame:] = 1.0

            loss = criterion(risk_predictions, target_risk)
            
            # 3. 反向传播和优化
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
            if (i+1) % 50 == 0:
                print(f"Epoch [{epoch+1}/{args.num_epochs}], Step [{i+1}/{len(train_loader)}], Loss: {loss.item():.4f}")

        avg_loss = total_loss / len(train_loader)
        print(f"Epoch [{epoch+1}/{args.num_epochs}] finished. Average Loss: {avg_loss:.4f}")

        # --- 保存Checkpoint ---
        if (epoch + 1) % args.save_interval == 0:
            checkpoint_path = os.path.join(args.checkpoint_dir, f"model_epoch_{epoch+1}.pth")
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'loss': avg_loss,
            }, checkpoint_path)
            print(f"Checkpoint saved to {checkpoint_path}")

if __name__ == "__main__":
    main()
