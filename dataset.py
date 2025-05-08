import os
import json
import torch
import numpy as np
from PIL import Image
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms

def iclevr_collate_fn(batch):
    pixel_values = torch.stack([b['pixel_values'] for b in batch])     # (B,3,64,64)
    labels       = torch.stack([b['labels'] for b in batch])           # (B,24)
    obj_names    = [b['obj_names'] for b in batch]                     # List[List[str]]
    return {'pixel_values': pixel_values,
            'labels': labels,
            'obj_names': obj_names}

class ICLEVRDataset(Dataset):
    def __init__(self, data_dir, json_file, img_dir=None, transform=None, img_size=64):
        """
        初始化ICLEVR數據集
        
        參數:
            data_dir (str): 資料目錄
            json_file (str): 標籤json檔案名稱
            img_dir (str, optional): 圖片目錄，訓練模式需提供
            transform: 圖像轉換函數
        """
        self.data_dir = data_dir
        self.transform = transform
        self.img_dir = img_dir
        self.img_size = img_size
        
        # 讀取物件映射
        with open(os.path.join(data_dir, 'objects.json'), 'r') as f:
            self.obj2idx = json.load(f)
            self.idx2obj = {v: k for k, v in self.obj2idx.items()}
        
        # 讀取數據
        with open(os.path.join(data_dir, json_file), 'r') as f:
            if 'train' in json_file:
                self.data = json.load(f)
                self.filenames = list(self.data.keys())
            else:
                self.data = json.load(f)
                self.filenames = [f"{i}.png" for i in range(len(self.data))]
                
        self.is_train = 'train' in json_file
                
    def __len__(self):
        return len(self.filenames)
    
    def __getitem__(self, idx):
        filename = self.filenames[idx]
        
        # 處理標籤
        if self.is_train:
            labels = self.data[filename]
        else:
            labels = self.data[idx]
            
        # 創建one-hot標籤
        one_hot = torch.zeros(len(self.obj2idx))
        for obj in labels:
            one_hot[self.obj2idx[obj]] = 0.95

        # 增加標籤平滑化
        one_hot[one_hot == 0] = 0.01 / (len(self.obj2idx) - len(labels))
            
        # 讀取圖像(訓練)
        if self.is_train and self.img_dir is not None:
            img_path = os.path.join(self.img_dir, filename)
            image = Image.open(img_path).convert('RGB')
            if self.transform:
                image = self.transform(image)
            return {'pixel_values': image, 'labels': one_hot, 'obj_names': labels}
        
        # 測試模式只返回標籤
        dummy_img = torch.zeros(3, self.img_size, self.img_size)
        return {'pixel_values': dummy_img, 'labels': one_hot, 'obj_names': labels}
    
    def get_dataloader(self, batch_size=32, shuffle=True, num_workers=4, collate_fn=iclevr_collate_fn):
        return DataLoader(
            self, 
            batch_size=batch_size,
            shuffle=shuffle,
            num_workers=num_workers,
            pin_memory=True,
            collate_fn=collate_fn
        )


def get_transforms():
    """獲取標準的圖像轉換"""
    return transforms.Compose([
        transforms.Resize((64, 64)),
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])


def get_inverse_transforms():
    """獲取逆轉換，用於可視化"""
    return transforms.Compose([
        transforms.Normalize((-1, -1, -1), (2, 2, 2)),  # [-1,1] to [0,1]
    ])


def get_dataset(data_dir, json_file='train.json', img_dir=None, transform=None):
    """獲取數據集實例"""
    if transform is None:
        transform = get_transforms()
    return ICLEVRDataset(data_dir, json_file, img_dir, transform)