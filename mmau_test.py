import os
import random
import json
import numpy as np
import cv2
import torch
from torchvision import transforms
from torch.utils.data import Dataset, DataLoader
import glob
from PIL import Image
# import decord
# decord.bridge.set_bridge('torch')

from torch.utils.data import Dataset
from einops import rearrange

#数据集读取示例,仅供参考
#视频信息必须使用，驾驶员注视图信息，文本和bbx信息可选择使用
#有四种模态：RGB图像，驾驶员注视图信息，文本信息，bounding box物体检测框信息
class MMAU_TEST(Dataset):
    def __init__(
            self, 
            root_path, 
            phase,
            data_aug=True
        ):
        self.root_path = root_path
        self.data_aug = data_aug
        self.fps = 30
        self.phase = phase
        #文件名，事故标签，起始帧，结束帧，事故发生前的时刻，文本
        self.data_list, self.label, self.start, self.end, self.tai, self.text, self.cause_texts, self.prevention_texts = self.get_data_list()
        #detection 路径
        # self.bbx_path = r".../detection"
        self.bbx_path = r"/home/msi/driving-risk-prediction/MMAU_TRAIN/detection"

    def get_data_list(self):
        if self.phase == "test":
            list_file = os.path.join(self.root_path + "/" + 'testing.txt')
            # print(f"list_file = {list_file}")
            assert os.path.exists(list_file), "File does not exist! %s" % (list_file)
            fileIDs, labels, start_ids, end_ids, tais, factual_texts, cause_texts, prevention_texts = [], [], [], [], [], [], [], []
            with open(list_file, 'r', encoding='utf-8') as f:
                # for ids, line in enumerate(f.readlines()):
                # print(f"f = {f}")
                for ids, line in enumerate(f.readlines()):
                    parts = line.strip().split('//')
                    ID, label, start, end, tai = parts[0].split(',')[0].split(' ')
                    text= parts[0].split(',')[1]
                    fileIDs.append(ID)
                    labels.append(label)
                    start_ids.append(start)
                    end_ids.append(end)
                    tais.append(tai)
                    factual_texts.append(text)
                    cause_texts.append(parts[1])
                    prevention_texts.append(parts[2])
            return fileIDs, labels, start_ids, end_ids, tais, factual_texts, cause_texts, prevention_texts
        # elif self.phase == "test":
        #     pass

    def __len__(self):
        return len(self.data_list)
    
    #图像尺寸可根据实际需要进行修改，若此处修改图像尺寸，recalculate_box_and_verify_if_valid也要作相应修改
    def pross_video_data(self, video):
        video_datas = []
        for fid in range(len(video)):
            video_data = video[fid]
            video_data = Image.open(video_data)
            video_data = video_data.resize((224, 224))
            video_data = np.asarray(video_data, np.float32)

            video_datas.append(video_data)
        video_data = np.array(video_datas, dtype=np.float32)  # 4D tensor
        video_data = rearrange(video_data, 'f w h c -> f c w h')
        return video_data

    def read_normal_rgbvideo(self, video_file):
        video_data = self.pross_video_data(video_file)
        return video_data

        # 图像尺寸可根据实际需要进行修改，若此处修改图像尺寸，recalculate_box_and_verify_if_valid也要作相应修改

    def pross_map_data(self, video):
        video_datas = []
        for fid in range(len(video)):
            video_data = video[fid]
            video_data = Image.open(video_data)
            video_data = video_data.resize((224, 224))
            video_data = np.asarray(video_data, np.float32)
            video_data = video_data[:,:,None]
            video_datas.append(video_data)
        video_data = np.array(video_datas, dtype=np.float32)  # 4D tensor
        video_data = rearrange(video_data, 'f w h c -> f c w h')
        return video_data

    def read_mapvideo(self, video_file):
        video_data = self.pross_map_data(video_file)
        return video_data

    def pdbbx(self, bbx, max_N):
        N = bbx.shape[0]
        if N < max_N:
            pad_objects = torch.zeros(max_N - N, 4)
            bbx = torch.cat([bbx, pad_objects], dim=0)
        elif N > max_N:
            bbx = bbx[:max_N, :]
        return bbx

    def to_valid(self,x0, y0, x1, y1, image_size, min_box_size):
        valid = True

        if x0 > image_size or y0 > image_size or x1 < 0 or y1 < 0:
            valid = False  # no way to make this box vide, it is completely cropped out
            return valid, (None, None, None, None)

        x0 = max(x0, 0)
        y0 = max(y0, 0)
        x1 = min(x1, image_size)
        y1 = min(y1, image_size)

        if (x1 - x0) * (y1 - y0) / (image_size * image_size) < min_box_size:
            valid = False
            return valid, (None, None, None, None)

        return valid, (x0, y0, x1, y1)

    def recalculate_box_and_verify_if_valid(self, normalized_bbx, original_image_size, target_image_size,image_size):
        # normalized_bbx = (x1, y1, x2, y2)
        x1_orig, y1_orig, x2_orig, y2_orig = normalized_bbx
        # Scale coordinates from original image size to target image size
        x1_target = x1_orig * (target_image_size[0] / original_image_size[0])
        y1_target = y1_orig * (target_image_size[1] / original_image_size[1])
        x2_target = x2_orig * (target_image_size[0] / original_image_size[0])
        y2_target = y2_orig * (target_image_size[1] / original_image_size[1])
        valid, (x0, y0, x1, y1) = self.to_valid(x1_target,y1_target,x2_target,y2_target,image_size, min_box_size=0.01)

        # if valid:
        #     # we also perform random flip.
        #     # Here boxes are valid, and are based on image_size
        #     # if trans_info["performed_flip"]:
        #         x0, x1 = image_size - x1, image_size - x0

        return valid, (x0, y0, x1, y1)

    def mapping_caption(self, category_number):
        category_mapping = {
            '{}': 0, 
            'motorcycle': 1, 
            'truck': 2, 
            'bus': 3, 
            'traffic light': 4,
            'person': 5, 
            'bicycle': 6, 
            'car': 7
        }
        if category_number in category_mapping.values():
            caption = next(key for key, value in category_mapping.items() if value == category_number)
            # print(caption)
            return caption

    def extract_masks(self, video_frames, bounding_boxes):
        masks = []
        for i in range(video_frames.shape[0]):
            frame = video_frames[i]
            frame_masks = np.zeros_like(frame)  # Create a blank mask with the same dimensions as the frame

            for j in range(bounding_boxes.shape[1]):
                bbx = bounding_boxes[i, j].int()  # Convert the bounding box coordinates to integers
                x1, y1, x2, y2 = bbx

                # 保留 bounding box 区域内的像素
                frame_masks[..., y1:y2, x1:x2] = frame[..., y1:y2, x1:x2]  # 复制 bounding box 区域内的像素到 mask

            masks.append(frame_masks)
        return np.array(masks)

    def bbx_caption_process(self, bbx_info, original_image_size, target_image_size, max_N,image_size):
        caption = bbx_info[:, 0]  # Add a new dimension
        caption_text = [self.mapping_caption(category_number.item()) for category_number in caption]
        caption_text = ", ".join(caption_text)
        # # Extract the second to fifth elements of each row and keep them as (N, 4) tensor
        bbx = bbx_info[:, 1:]
        areas=[]
        all_boxes=[]
        all_boxes_ = []
        all_masks=[]
        for i in range(bbx.shape[0]):
            row_bbx = bbx[i]
            valid, (x0, y0, x1, y1) = self.recalculate_box_and_verify_if_valid(row_bbx, original_image_size, target_image_size,image_size)
            if valid:
                areas.append((x1 - x0) * (y1 - y0))
                all_boxes.append(torch.tensor([x0, y0, x1, y1]) / image_size)  # scale to 0-1
                all_boxes_.append(torch.tensor([x0, y0, x1, y1]))
                all_masks.append(1)
        wanted_idxs = torch.tensor(areas).sort(descending=True)[1]
        wanted_idxs = wanted_idxs[0:max_N]
        new_boxes = torch.zeros(max_N, 4)
        new_boxes_ = torch.zeros(max_N, 4)
        masks = torch.zeros(max_N)

        for i, idx in enumerate(wanted_idxs):
            new_boxes[i] = all_boxes[idx]
            new_boxes_[i]=all_boxes_[idx]
            masks[i] = all_masks[idx]
        new_bbx = self.pdbbx(new_boxes, max_N)
        new_bbx_=self.pdbbx(new_boxes_, max_N)
        image_masks = masks
        text_masks = masks

        return new_bbx,caption_text,image_masks,text_masks,new_bbx_

    def gather_info(self, index):
        # accident_id = int(self.data_list[index].split('/')[0])
        accident_identifier = self.data_list[index]
        video_id = int(self.data_list[index].split('/')[1])
        catagroy=int(self.data_list[index].split('/')[0])
        text = self.text[index]
        c_text=self.cause_texts[index]
        p_texts=self.prevention_texts[index]
        return accident_identifier, video_id, text,catagroy,c_text,p_texts

    def enhance_start_end(self,label, start, end, tai, total_frames):
        if label == 1:
            if total_frames > end + 30:
                new_end=end + 30
                start = random.randint(0,new_end-150)
                end = start + 150
        elif label == 0:
            if end > 151:
                start = random.randint(0, end-150)
                end = start + 150
        return start, end

    def __getitem__(self, index):
        accident_label = int(self.label[index])
        start = int(self.start[index])
        end = int(self.end[index])
        tai = int(self.tai[index])
        # video_path = os.path.join(self.root_path+"/",self.data_list[index]+"/"+"images")
        # map_path = os.path.join(self.root_path+"/",self.data_list[index]+"/"+"maps")
        video_path = os.path.join(self.root_path+"/video/",self.data_list[index]+"/"+"images")
        map_path = os.path.join(self.root_path+"/video/",self.data_list[index]+"/"+"maps")
        #数据增强，固定tai，start和end取不同，有差异的区间
        if self.data_aug:
            start, end = self.enhance_start_end(accident_label, start, end, tai, len(os.listdir(video_path)))
        v_r = [video_path + "/" + f'{i:06d}' + ".jpg" for i in range(start, end)]
        m_r = [map_path + "/" + f'{i:06d}' + ".png" for i in range(start, end)]
        accident_identifier, video_id, text, catagroy, c_text, p_text = self.gather_info(index)
        vr = self.read_normal_rgbvideo(v_r)
        mr = self.read_mapvideo(m_r)
        bbx_info_list=[]
        bbx_path=os.path.join(self.bbx_path+"/",self.data_list[index])
        bbx_path = [bbx_path + "/" + f'{i:06d}' + ".json" for i in range(start,end)]
        for bbx_file in bbx_path:
            with open(bbx_file) as json_file:
                lines = json.load(json_file)
                if not lines or len(lines) == 0 or all(line.isspace() for line in lines):
                    filtered_datas = torch.zeros(1, 5, dtype=torch.float32)
                else:
                    bbx_info = lines["bboxes"]
                    scores = lines["scores"]
                    label = lines["labels"]
                    filtered_data = [[lbl, *info] for info, scr, lbl in zip(bbx_info, scores, label) if scr > 0.2]
                    if not filtered_data or len(filtered_data) == 0:
                        filtered_datas = torch.stack(
                            [torch.zeros(1, 5, dtype=torch.float32) for _ in range(len(bbx_path))]
                        ).squeeze(1)
                    else:
                        filtered_datas = torch.stack(
                            [torch.tensor(list(map(float, line)), dtype=torch.float32) for line in filtered_data]
                        )
                #默认每帧最多取10个objects，可根据需要修改
                new_bbx,caption_text,image_masks,text_masks,new_bbx_ = self.bbx_caption_process(
                    filtered_datas, 
                    original_image_size=(1560, 660),
                    target_image_size=(224, 224), 
                    max_N=10,
                    image_size=224
                )
                bbx_info_list.append(new_bbx)
        boxes = torch.stack(bbx_info_list)

        example = {
            "pixel_values": vr,       # rgb videos:(batch,frames,channels,height,width)
            "map_values":mr,          # map videos:(batch,frames,channels,height,width)
            "prompt": text,           # scene description
            "c_prompt":c_text,        # cause prompt
            "p_prompt":p_text,        # prevention prompt
            "bbx": boxes,             # (batch,frames,N,4)
            "accident_type":catagroy, # accident type, you can check the accident_type.txt
            "accident_id":video_id,   # numeric video id
            "tai":tai,                # time to accident (time period before accident)
            "label":accident_label,   # 1-accident 0-normal
            "video_name":accident_identifier,  # e.g., "1/009334"
        }

        return example
