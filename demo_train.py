from mmau_train import MMAU_TRAIN
from torch.utils.data import DataLoader
import torch
from torch import nn
# from model import DrivingRiskPredictor
from model import LightDrivingRiskPredictor
from tqdm import tqdm
import os
import json

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

def main():
    # --- 参数设置 ---
    ROOT_PATH = "/home/msi/driving-risk-prediction/MMAU_TRAIN"
    BATCH_SIZE = 1
    NUM_WORKERS = 1
    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using {DEVICE} device to run.")

    # --- Checkpoint相关参数设置 ---
    CHECKPOINT_DIR = "checkpoints"  # 保存Checkpoint的目录
    RESUME_CHECKPOINT = None # 如果要从某个checkpoint恢复训练，请指定文件路径, e.g., "./checkpoints/model_epoch_5.pth"
    SAVE_INTERVAL = 1  # 每隔多少个epoch保存一次

    # --- 创建保存Checkpoint的目录 ---
    os.makedirs(CHECKPOINT_DIR, exist_ok=True)

    # --- 创建数据集和加载器 ---
    train_dataset = MMAU_TRAIN(root_path=ROOT_PATH, phase="train", data_aug=True)
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=NUM_WORKERS)

    # --- 实例化模型 ---
    # model = DrivingRiskPredictor(max_bbox_n=10).to(DEVICE)
    max_bbox_n = 10
    model = LightDrivingRiskPredictor(max_bbox_n=max_bbox_n).to(DEVICE)

    # --- 定义损失函数和优化器 ---
    criterion = nn.BCELoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)
    
    # --- 加载Checkpoint (如果存在) ---
    start_epoch = 0
    if RESUME_CHECKPOINT and os.path.exists(RESUME_CHECKPOINT):
        print(f"Resuming training from checkpoint: {RESUME_CHECKPOINT}")
        checkpoint = torch.load(RESUME_CHECKPOINT, map_location=DEVICE)
        
        model.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        start_epoch = checkpoint['epoch'] + 1  # 从下一个epoch开始
        
        print(f"Resumed from epoch {checkpoint['epoch']}. Starting training at epoch {start_epoch}.")
    else:
        print("Starting training from scratch.")

    # --- 简单的训练示例 ---
    # num_epochs = 10
    num_epochs = 1
    # 修改range以从正确的epoch开始
    for epoch in range(start_epoch, num_epochs):
        model.train()
        total_loss = 0
        for i, batch in enumerate(train_loader):
            # 将数据移动到GPU
            for key, value in batch.items():
                if isinstance(value, torch.Tensor):
                    batch[key] = value.to(DEVICE)

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
                    # if accident_frame > start_ramp_frame:
                    if ramp_length > 0:
                        # ramp = torch.linspace(0, 1, accident_frame - start_ramp_frame).to(DEVICE)
                        ramp = torch.linspace(0, 1, ramp_length).to(DEVICE)
                        # target_risk[j, start_ramp_frame:accident_frame] = ramp

                        end_frame = min(accident_frame, seq_len)
                        target_slice_length = end_frame - start_ramp_frame

                        if target_slice_length == ramp_length:
                            target_risk[j, start_ramp_frame:end_frame] = ramp
                        else:
                            # 如果不匹配，调整 ramp
                            adjusted_ramp = torch.linspace(0, 1, target_slice_length).to(DEVICE)
                            target_risk[j, start_ramp_frame:end_frame] = adjusted_ramp

                    target_risk[j, accident_frame:] = 1.0

            loss = criterion(risk_predictions, target_risk)
            
            # 3. 反向传播和优化
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
            if (i+1) % 50 == 0:
                print(f"Epoch [{epoch+1}/{num_epochs}], Step [{i+1}/{len(train_loader)}], Loss: {loss.item():.4f}")

        avg_loss = total_loss / len(train_loader)
        print(f"Epoch [{epoch+1}/{num_epochs}] finished. Average Loss: {avg_loss:.4f}")

        # --- 保存Checkpoint ---
        if (epoch + 1) % SAVE_INTERVAL == 0:
            checkpoint_path = os.path.join(CHECKPOINT_DIR, f"model_epoch_{epoch+1}.pth")
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'loss': avg_loss,
            }, checkpoint_path)
            print(f"Checkpoint saved to {checkpoint_path}")

if __name__ == "__main__":
    main()
