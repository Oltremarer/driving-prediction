# from sklearn.cluster import KMeans

# class Model:
#     def __init__(self):
#         self.kmeans = KMeans(n_clusters=3)

#     def fit(self, X, y):
#         self.kmeans.fit(X=X, y=y)

#     def predict(self, X):
#         return self.kmeans.predict(X)

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.models import resnet50, ResNet50_Weights
from transformers import BertModel, BertTokenizer

# # --- 1. 视觉特征编码器 ---
# # 使用预训练的ResNet50来提取图像特征
# class VisionEncoder(nn.Module):
#     def __init__(self, feature_dim=512):
#         super().__init__()
#         # 加载在ImageNet上预训练的ResNet50
#         weights = ResNet50_Weights.IMAGENET1K_V2
#         self.model = resnet50(weights=weights)
        
#         # 移除最后的分类层，我们只需要特征
#         self.model.fc = nn.Linear(self.model.fc.in_features, feature_dim)
        
#         # 适配Gaze Map（单通道）输入
#         # 创建一个新的卷积层，权重从原有多通道层平均得到
#         original_conv1 = self.model.conv1
#         self.gaze_conv1 = nn.Conv2d(1, 64, kernel_size=7, stride=2, padding=3, bias=False)
#         self.gaze_conv1.weight.data = original_conv1.weight.data.mean(dim=1, keepdim=True)
        
#     def forward(self, rgb_video, gaze_map):
#         # rgb_video: (Batch, Frames, 3, H, W)
#         # gaze_map: (Batch, Frames, 1, H, W)
#         batch_size, num_frames, _, h, w = rgb_video.shape
        
#         # 将视频帧序列当成一个大的batch来处理，提高效率
#         # RGB
#         rgb_flat = rgb_video.view(batch_size * num_frames, 3, h, w)

#         self.model.conv1 = self.model._modules['conv1'] # 确保使用原始的conv1
#         rgb_features = self.model(rgb_flat) # (B*F, feature_dim)
        
#         # Gaze Map
#         gaze_flat = gaze_map.view(batch_size * num_frames, 1, h, w)

#         self.model.conv1 = self.gaze_conv1 # 替换为单通道的conv1
#         gaze_features = self.model(gaze_flat) # (B*F, feature_dim)
        
#         # 将特征合并
#         vision_features = torch.cat([rgb_features, gaze_features], dim=1) # (B*F, feature_dim * 2)
        
#         # 恢复时序维度
#         vision_features = vision_features.view(batch_size, num_frames, -1) # (B, F, feature_dim * 2)
#         return vision_features

# # --- 2. 文本特征编码器 ---
# class TextEncoder(nn.Module):
#     def __init__(self, feature_dim=512):
#         super().__init__()
#         self.tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
#         self.model = BertModel.from_pretrained('bert-base-uncased')
#         # 添加一个线性层将BERT的输出[CLS] token维度（768）映射到我们需要的维度
#         self.fc = nn.Linear(768, feature_dim)

#     def forward(self, prompts):
#         # prompts: list of strings, len = Batch_size
#         inputs = self.tokenizer(prompts, return_tensors='pt', padding=True, truncation=True, max_length=128)
#         inputs = {key: val.to(self.model.device) for key, val in inputs.items()}
        
#         outputs = self.model(**inputs)
#         # 我们使用[CLS] token的输出来代表整个句子的语义
#         cls_token_output = outputs.last_hidden_state[:, 0, :]
#         text_features = self.fc(cls_token_output) # (B, feature_dim)
#         return text_features

# # --- 3. Bounding Box 编码器 ---
# class BboxEncoder(nn.Module):
#     def __init__(self, input_dim, feature_dim=128):
#         super().__init__()
#         # 将每个时间步的 (N, 4) bbx展平后，通过MLP进行编码
#         self.fc = nn.Sequential(
#             nn.Linear(input_dim, 256),
#             nn.ReLU(),
#             nn.Linear(256, feature_dim)
#         )

#     def forward(self, bbx):
#         # bbx: (Batch, Frames, N, 4)
#         batch_size, num_frames, N, _ = bbx.shape
#         bbx_flat = bbx.view(batch_size * num_frames, -1) # (B*F, N*4)
        
#         bbx_features = self.fc(bbx_flat) # (B*F, feature_dim)
        
#         # 恢复时序维度
#         bbx_features = bbx_features.view(batch_size, num_frames, -1) # (B, F, feature_dim)
#         return bbx_features

# # --- 4. 最终的多模态融合与风险预测模型 ---
# class DrivingRiskPredictor(nn.Module):
#     def __init__(self, vision_dim=512, text_dim=512, bbox_dim=128, max_bbox_n=10, lstm_hidden_dim=512):
#         super().__init__()
#         self.vision_encoder = VisionEncoder(feature_dim=vision_dim)
#         self.text_encoder = TextEncoder(feature_dim=text_dim)
#         # 输入维度是 N*4，因为每个框有4个坐标
#         self.bbox_encoder = BboxEncoder(input_dim=max_bbox_n * 4, feature_dim=bbox_dim)
        
#         # 融合后的特征维度
#         fused_dim = (vision_dim * 2) + text_dim + bbox_dim
        
#         # 使用LSTM来处理融合后的特征序列
#         self.temporal_model = nn.LSTM(
#             input_size=fused_dim,
#             hidden_size=lstm_hidden_dim,
#             num_layers=2,
#             batch_first=True, # 输入输出格式为 (Batch, Seq, Feature)
#             dropout=0.2
#         )
        
#         # 预测头，用于输出每一帧的风险值
#         self.prediction_head = nn.Sequential(
#             nn.Linear(lstm_hidden_dim, 256),
#             nn.ReLU(),
#             nn.Dropout(0.3),
#             nn.Linear(256, 1),
#             nn.Sigmoid() # 将输出压缩到0-1之间，代表风险概率
#         )

#     def forward(self, batch):
#         # 从数据加载器提供的batch字典中解包数据
#         rgb_video = batch['pixel_values']       # (B, F, C, H, W)
#         gaze_map = batch['map_values']          # (B, F, C, H, W)
#         prompts = batch['prompt']               # List of strings
#         c_prompts = batch['c_prompt']           # List of strings for cause
#         p_prompts = batch['p_prompt']           # List of strings for prevention
#         bbx = batch['bbx']                      # (B, F, N, 4)
        
#         # 1. 编码各个模态的特征
#         vision_features = self.vision_encoder(rgb_video, gaze_map) # (B, F, vision_dim * 2)
        
#         # 我们可以将三个文本prompt拼接起来，获取更丰富的语义信息
#         combined_prompts = [f"{p}. Cause: {c}. Prevention: {pre}" for p, c, pre in zip(prompts, c_prompts, p_prompts)]
#         text_features = self.text_encoder(combined_prompts) # (B, text_dim)
        
#         bbox_features = self.bbox_encoder(bbx) # (B, F, bbox_dim)
        
#         # 2. 融合特征
#         # 文本特征对于视频中的每一帧都是不变的，所以需要扩展
#         num_frames = vision_features.shape[1]
#         text_features_expanded = text_features.unsqueeze(1).repeat(1, num_frames, 1) # (B, F, text_dim)
        
#         # 沿特征维度拼接所有模态的特征
#         fused_features = torch.cat([vision_features, text_features_expanded, bbox_features], dim=2)
        
#         # 3. 通过时序模型进行推理
#         lstm_output, _ = self.temporal_model(fused_features) # (B, F, lstm_hidden_dim)
        
#         # 4. 预测每一帧的风险值
#         # lstm_output的形状是(B, F, hidden_dim)，我们需要对每一帧都进行预测
#         batch_size, num_frames, hidden_dim = lstm_output.shape
#         lstm_output_flat = lstm_output.reshape(batch_size * num_frames, hidden_dim)
#         risk_predictions_flat = self.prediction_head(lstm_output_flat)
        
#         risk_predictions = risk_predictions_flat.view(batch_size, num_frames) # (B, F)
        
#         return risk_predictions

class SimpleTokenizer:
    def __init__(self, vocab_size):
        self.vocab_size = vocab_size
        self.word_to_index = {"<PAD>": 0, "<UNK>": 1}
        self.index_to_word = {0: "<PAD>", 1: "<UNK>"}
        self.next_index = 2

    def tokenize(self, texts):
        # texts: list of strings
        tokenized_texts = []
        max_len = 0
        for text in texts:
            tokens = []
            for word in text.split():
                if word in self.word_to_index:
                    tokens.append(self.word_to_index[word])
                else:
                    if self.next_index < self.vocab_size:
                        self.word_to_index[word] = self.next_index
                        self.index_to_word[self.next_index] = word
                        self.next_index += 1
                        tokens.append(self.word_to_index[word])
                    else:
                        tokens.append(1)  # <UNK>
            tokenized_texts.append(tokens)
            max_len = max(max_len, len(tokens))

        # Padding
        padded_texts = []
        for tokens in tokenized_texts:
            padded_tokens = tokens + [0] * (max_len - len(tokens))  # Pad with <PAD>
            padded_texts.append(padded_tokens)

        return torch.tensor(padded_texts, dtype=torch.long)

# --- 轻量级视觉特征编码器 ---
class LightVisionEncoder(nn.Module):
    def __init__(self, feature_dim=128):
        super().__init__()
        # 使用简单的CNN替代ResNet50
        self.rgb_encoder = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=3, stride=2, padding=1),  # 下采样
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1), # 再次下采样
            nn.ReLU(),
            nn.AdaptiveAvgPool2d((4, 4)),  # 固定大小输出
            nn.Flatten(),
            nn.Linear(64 * 4 * 4, feature_dim)
        )
        
        self.gaze_encoder = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=3, stride=2, padding=1),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d((4, 4)),
            nn.Flatten(),
            nn.Linear(64 * 4 * 4, feature_dim)
        )
        
    def forward(self, rgb_video, gaze_map):
        batch_size, num_frames, _, h, w = rgb_video.shape
        
        # 处理RGB
        rgb_flat = rgb_video.view(batch_size * num_frames, 3, h, w)
        rgb_features = self.rgb_encoder(rgb_flat)
        
        # 处理Gaze Map
        gaze_flat = gaze_map.view(batch_size * num_frames, 1, h, w)
        gaze_features = self.gaze_encoder(gaze_flat)
        
        # 合并特征
        vision_features = torch.cat([rgb_features, gaze_features], dim=1)
        vision_features = vision_features.view(batch_size, num_frames, -1)
        
        return vision_features

# --- 轻量级文本编码器 ---
class LightTextEncoder(nn.Module):
    def __init__(self, vocab_size=10000, feature_dim=128, max_length=128):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, 64)
        self.conv1d = nn.Conv1d(64, 64, kernel_size=3, padding=1)
        self.pool = nn.AdaptiveAvgPool1d(1)
        self.fc = nn.Linear(64, feature_dim)
        self.tokenizer = SimpleTokenizer(vocab_size)
        
    def forward(self, texts):
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        # text_tokens: (B, seq_len) - 需要预先tokenize
        text_tokens = self.tokenizer.tokenize(texts)
        text_tokens = text_tokens.to(device)
        x = self.embedding(text_tokens)  # (B, seq_len, 64)
        x = x.transpose(1, 2)  # (B, 64, seq_len)
        x = F.relu(self.conv1d(x))
        x = self.pool(x).squeeze(-1)  # (B, 64)
        text_features = self.fc(x)  # (B, feature_dim)
        return text_features

# --- 轻量级Bbox编码器 ---
class LightBboxEncoder(nn.Module):
    def __init__(self, max_bbox_n=10, feature_dim=64):
        super().__init__()
        input_dim = max_bbox_n * 4
        self.fc = nn.Sequential(
            nn.Linear(input_dim, 128),
            nn.ReLU(),
            nn.Linear(128, feature_dim)
        )

    def forward(self, bbx):
        batch_size, num_frames, N, _ = bbx.shape
        bbx_flat = bbx.view(batch_size * num_frames, -1)
        # print(self.max_bbox_n)
        bbx_features = self.fc(bbx_flat)
        bbx_features = bbx_features.view(batch_size, num_frames, -1)
        return bbx_features

# --- 最终轻量级模型 ---
class LightDrivingRiskPredictor(nn.Module):
    # def __init__(self, vision_dim=128, text_dim=128, bbox_dim=64, lstm_hidden_dim=256):
    def __init__(self, vision_dim=128, text_dim=128, bbox_dim=64, max_bbox_n=10, lstm_hidden_dim=256):
        # vision_dim=512, text_dim=512, bbox_dim=128, max_bbox_n=10, lstm_hidden_dim=512
        super().__init__()
        self.vision_encoder = LightVisionEncoder(feature_dim=vision_dim)
        self.text_encoder = LightTextEncoder(feature_dim=text_dim)
        # 输入维度是 N*4，因为每个框有4个坐标
        self.bbox_encoder = LightBboxEncoder(max_bbox_n=max_bbox_n, feature_dim=bbox_dim)
        
        # 融合特征维度
        fused_dim = (vision_dim * 2) + text_dim + bbox_dim
        
        # 使用GRU替代LSTM，更轻量
        self.temporal_model = nn.GRU(
            input_size=fused_dim,
            hidden_size=lstm_hidden_dim,
            num_layers=1,  # 减少层数
            batch_first=True,
            dropout=0.1
        )
        
        # 简化预测头
        self.prediction_head = nn.Sequential(
            nn.Linear(lstm_hidden_dim, 64),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(64, 1),
            nn.Sigmoid()
        )

    def forward(self, batch):
        # pixel_values, map_values, prompt, c_prompt, p_prompt, bbx, accident_type, accident_id, tai, label
        # print(f"batch = {batch}")
        rgb_video = batch['pixel_values']
        gaze_map = batch['map_values']
        # text_tokens = batch['prompt']
        combined_prompts = [f"{p}. Cause: {c}. Prevention: {pre}" for p, c, pre in zip(batch['prompt'], batch['c_prompt'], batch['p_prompt'])]
        bbx = batch['bbx']
        # print(f"bbx.shape = {bbx.shape}")
        
        # 编码特征
        vision_features = self.vision_encoder(rgb_video, gaze_map)
        # text_features = self.text_encoder(text_tokens)
        text_features = self.text_encoder(combined_prompts)
        bbox_features = self.bbox_encoder(bbx)
        
        # 扩展文本特征
        num_frames = vision_features.shape[1]
        text_features_expanded = text_features.unsqueeze(1).repeat(1, num_frames, 1)
        
        # 融合特征
        fused_features = torch.cat([vision_features, text_features_expanded, bbox_features], dim=2)
        
        # 时序建模
        gru_output, _ = self.temporal_model(fused_features)
        
        # 预测
        batch_size, num_frames, hidden_dim = gru_output.shape
        risk_predictions = self.prediction_head(
            gru_output.reshape(batch_size * num_frames, hidden_dim)
        ).view(batch_size, num_frames)
        
        return risk_predictions
    