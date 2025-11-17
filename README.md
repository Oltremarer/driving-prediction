# Driving Risk Prediction Competition

https://www.codabench.org/competitions/11247/#/pages-tab

GPU: NVIDIA GeForce GTX 5070 Ti Laptop GPU

GPU from 50 series requires latest PyTorch

Code structure:
- requirements.txt
- mmau_train.py
- mmau_test.py
- demo_train.py
- demo_test.py
- model.py

Train dataset structure:
```
- detection
    - 1
    - 2
    - ...
- video
    - 1
    - 2
    - ...
- accident_type.txt
- train.txt
```

Test dataset structure:
```
- detection
    - 1
    - 2
    - ...
- video
    - 1
    - 2
    - ...
- testing.txt
```

1. Downlaod the MMAU dataset (train and test) from website

https://www.codabench.org/competitions/11247/#/pages-tab

2. Execute the following commands

Environment:
```bash
conda create -n mmau_env python=3.10 -y
conda activate mmau_env
# conda install pytorch torchvision torchaudio pytorch-cuda=12.1 -c pytorch -c nvidia -y
pip install -r requirements.txt
pip install torch torchvision torchaudio
```

Train & Test:
```bash
# Train
python demo_train.py
# Test / Eval / Submission export
python demo_test.py --help
```

## 无 GPU 本地调试 & 让同学代跑指南

1. **你本地准备**
   - 按需修改代码、验证逻辑后，推送到 GitHub 仓库，例如 `https://github.com/<yourname>/driving-risk-prediction-master`.
   - 在 README 中保持以下信息齐全：依赖、数据目录结构、训练/推理命令（如下所示）。

2. **同学在有 GPU 的机器上执行的步骤**
   ```bash
   git clone https://github.com/<yourname>/driving-risk-prediction-master.git
   cd driving-risk-prediction-master

   # 准备环境（根据显卡/驱动调整 CUDA 版 PyTorch）
   conda create -n mmau_env python=3.10 -y
   conda activate mmau_env
   pip install -r requirements.txt
   pip install torch torchvision torchaudio  # 或指定 cu117/cu121 版本

   # 训练：ROOT_PATH 指向包含 train.txt/video/detection 的 MMAU_TRAIN
   python demo_train.py \
       --dataset-root /path/to/MMAU_TRAIN \
       --checkpoint-dir checkpoints \
       --num-epochs 10

   # 推理/导出：dataset-root 指向测试集，checkpoint 换成实际路径
   python demo_test.py \
       --dataset-root /path/to/MMAU_TEST \
       --phase test \
       --checkpoint checkpoints/model_epoch_10.pth \
       --output-path prediction/prediction.json \
       --skip-eval

   # 若要在训练集上核对指标，可改:
   # --phase train --metrics-path prediction/train_metrics.json --skip-eval 去掉
   ```

3. **上传 Codabench**
   ```bash
   cd prediction
   zip submission.zip prediction.json
   ```
   将 `submission.zip` 上传至 Codabench 的提交入口即可。

Backup:
```bash
# MMAU_TRAIN
mv /home/msi/driving-risk-prediction/MMAU_TRAIN /home/msi/data
mv /home/msi/data/MMAU_TRAIN /home/msi/driving-risk-prediction
# DADA_Test_2025
mv /home/msi/driving-risk-prediction/DADA_Test_2025 /home/msi/data
mv /home/msi/data/DADA_Test_2025 /home/msi/driving-risk-prediction
```
