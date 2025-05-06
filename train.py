import os
import torch
import argparse
import numpy as np
from tqdm import tqdm
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
from dataset import get_dataset, get_transforms, iclevr_collate_fn
from model import ConditionalLDM
from utils import save_checkpoint, load_checkpoint, save_images, setup_logging, log_images

def parse_args():
    parser = argparse.ArgumentParser(description='Train Conditional LDM')
    parser.add_argument('--data_dir', type=str, default='./', help='Data directory')
    parser.add_argument('--img_dir', type=str, default='./iclevr', help='Image directory')
    parser.add_argument('--output_dir', type=str, default='./output', help='Output directory')
    parser.add_argument('--vae_model', type=str, default='stabilityai/sd-vae-ft-mse', help='VAE model path/name')
    parser.add_argument('--batch_size', type=int, default=64, help='Batch size')
    parser.add_argument('--epochs', type=int, default=100, help='Number of epochs')
    parser.add_argument('--lr', type=float, default=1e-4, help='Learning rate')
    parser.add_argument('--save_every', type=int, default=10, help='Save checkpoint every n epochs')
    parser.add_argument('--resume', type=str, default=None, help='Resume from checkpoint')
    parser.add_argument('--seed', type=int, default=42, help='Random seed')
    parser.add_argument('--num_workers', type=int, default=4, help='Number of workers')
    parser.add_argument('--fp16', action='store_true', help='Use mixed precision training')
    return parser.parse_args()

def train(args):
    # 設置隨機種子
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    
    # 創建輸出目錄
    os.makedirs(args.output_dir, exist_ok=True)
    
    # 設置日誌
    logger = setup_logging(args)
    
    # 準備數據集
    transform = get_transforms()
    train_dataset = get_dataset(args.data_dir, 'train.json', args.img_dir, transform)
    train_dataloader = train_dataset.get_dataloader(
        args.batch_size, 
        num_workers=args.num_workers
    )
    
    # 獲取設備
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # 準備模型
    model = ConditionalLDM(num_labels=24, vae_model_path=args.vae_model)
    model = model.to(device)
    
    # 載入檢查點(如果有)
    start_epoch = 0
    if args.resume:
        logger.info(f"Resuming from checkpoint: {args.resume}")
        start_epoch = load_checkpoint(args.resume, model)
    
    # 準備優化器
    optimizer = AdamW(model.parameters(), lr=args.lr)
    lr_scheduler = CosineAnnealingLR(optimizer, T_max=args.epochs)
    
    # 混合精度訓練
    scaler = torch.cuda.amp.GradScaler() if args.fp16 else None
    
    # 開始訓練
    logger.info("Starting training...")
    
    for epoch in range(start_epoch, args.epochs):
        model.train()
        epoch_loss = 0.0
        
        with tqdm(train_dataloader, desc=f"Epoch {epoch+1}/{args.epochs}") as pbar:
            for batch in pbar:
                # 移動數據到設備
                pixel_values = batch['pixel_values'].to(device)
                labels = batch['labels'].to(device)
                
                # 清除梯度
                optimizer.zero_grad()
                
                # 前向傳播
                if args.fp16:
                    with torch.cuda.amp.autocast():
                        loss = model(pixel_values, labels)
                    
                    # 反向傳播
                    scaler.scale(loss).backward()
                    scaler.step(optimizer)
                    scaler.update()
                else:
                    loss = model(pixel_values, labels)
                    
                    # 反向傳播
                    loss.backward()
                    optimizer.step()
                
                # 更新進度條
                epoch_loss += loss.item()
                pbar.set_postfix(loss=loss.item())
        
        # 更新學習率
        lr_scheduler.step()
        
        # 計算平均損失
        avg_loss = epoch_loss / len(train_dataloader)
        logger.info(f"Epoch {epoch+1}/{args.epochs}, Loss: {avg_loss:.6f}")
        
        # 保存檢查點
        if (epoch + 1) % args.save_every == 0 or epoch == args.epochs - 1:
            checkpoint_path = os.path.join(args.output_dir, f"checkpoint_epoch_{epoch+1}.pth")
            save_checkpoint(checkpoint_path, model, optimizer, epoch + 1)
            logger.info(f"Checkpoint saved at {checkpoint_path}")
            
            # 生成樣本
            if (epoch + 1) % args.save_every == 0:
                generate_samples(model, train_dataset, args.output_dir, epoch+1, device)
    
    logger.info("Training completed!")
    
def generate_samples(model, dataset, output_dir, epoch, device, num_samples=4):
    """生成樣本圖像"""
    model.eval()
    
    # 隨機選擇一些標籤
    indices = np.random.choice(len(dataset), num_samples, replace=False)
    samples = [dataset[i] for i in indices]
    labels = torch.stack([sample['labels'] for sample in samples]).to(device)
    
    # 生成圖像
    with torch.no_grad():
        images = model.generate(labels, batch_size=num_samples, guidance_scale=7.5)
    
    # 保存圖像
    sample_dir = os.path.join(output_dir, 'samples')
    os.makedirs(sample_dir, exist_ok=True)
    save_images(images, [s['obj_names'] for s in samples], os.path.join(sample_dir, f"samples_epoch_{epoch}.png"))

if __name__ == "__main__":
    args = parse_args()
    train(args)