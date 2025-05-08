import os
import torch
import argparse
import numpy as np
import wandb
from tqdm import tqdm
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR, LinearLR, SequentialLR
from torchvision import transforms
from dataset import get_dataset, get_transforms, iclevr_collate_fn
from model import ConditionalLDM
from evaluate import evaluate_model
from utils import save_checkpoint, load_checkpoint, save_images, setup_logging, log_images
from config import Config

def parse_args():
    parser = argparse.ArgumentParser(description='Train Conditional LDM')
    parser.add_argument('--data_dir', type=str, default=None, help='Data directory')
    parser.add_argument('--img_dir', type=str, default=None, help='Image directory')
    parser.add_argument('--output_dir', type=str, default=None, help='Output directory')
    parser.add_argument('--vae_model', type=str, default=None, help='VAE model path/name')
    parser.add_argument('--batch_size', type=int, default=None, help='Batch size')
    parser.add_argument('--epochs', type=int, default=None, help='Number of epochs')
    parser.add_argument('--lr', type=float, default=None, help='Learning rate')
    parser.add_argument('--save_every', type=int, default=None, help='Save checkpoint every n epochs')
    parser.add_argument('--eval_every', type=int, default=None, help='Evaluate model every n epochs')
    parser.add_argument('--resume', type=str, default=None, help='Resume from checkpoint')
    parser.add_argument('--seed', type=int, default=None, help='Random seed')
    parser.add_argument('--num_workers', type=int, default=None, help='Number of workers')
    parser.add_argument('--fp16', action='store_true', help='Use mixed precision training')
    parser.add_argument('--use_wandb', action='store_true', help='Use Weights & Biases for logging')
    parser.add_argument('--wandb_project', type=str, default=None, help='Wandb project name')
    parser.add_argument('--wandb_name', type=str, default=None, help='Wandb run name')
    parser.add_argument('--guidance_scale', type=float, default=None, help='Guidance scale for sampling')
    parser.add_argument('--inference_steps', type=int, default=None, help='Number of inference steps')
    parser.add_argument('--config', type=str, default=None, help='Path to config file to load')
    return parser.parse_args()

def train(args):    
    if args.config:
        # 從文件加載配置
        Config.load_config(args.config)
    
    # 更新配置
    Config.update_from_args(args)
    
    # 更新路徑
    Config.update_paths(wandb_id=Config.WANDB_ID if Config.USE_WANDB else None)
    
    # 創建目錄
    Config.create_directories()
    
    # 保存配置
    config_path = Config.save_config()
    print(f"保存配置到: {config_path}")

    # 設置隨機種子
    torch.manual_seed(Config.SEED)
    np.random.seed(Config.SEED)
    
    # 設置日誌
    logger = setup_logging(Config.LOG_DIR, Config.RUN_ID)
    
    # 準備數據集
    transform = get_transforms()
    train_dataset = get_dataset(Config.DATA_DIR, Config.TRAIN_JSON, Config.IMG_DIR, transform)
    train_dataloader = train_dataset.get_dataloader(
        Config.BATCH_SIZE, 
        num_workers=Config.NUM_WORKERS
    )
    
    # 準備模型
    model = ConditionalLDM(num_labels=Config.NUM_CLASSES, vae_model_path=Config.VAE_MODEL)
    model = model.to(Config.DEVICE)
    
    # 載入檢查點(如果有)
    start_epoch = 0
    if Config.RESUME:
        logger.info(f"Resuming from checkpoint: {Config.RESUME}")
        start_epoch = load_checkpoint(Config.RESUME, model)
    
    # 準備優化器
    optimizer = AdamW(
        model.parameters(), 
        lr=Config.LEARNING_RATE, 
        weight_decay=Config.WEIGHT_DECAY
    )
    # lr_scheduler = CosineAnnealingLR(optimizer, T_max=Config.NUM_EPOCHS)
    # 創建預熱+余弦退火排程
    warmup_scheduler = LinearLR(
        optimizer, 
        start_factor=0.1, 
        end_factor=1.0, 
        total_iters=int(0.05 * Config.NUM_EPOCHS)  # 5%時間預熱
    )
    cosine_scheduler = CosineAnnealingLR(
        optimizer, 
        T_max=Config.NUM_EPOCHS - int(0.05 * Config.NUM_EPOCHS)
    )
    lr_scheduler = SequentialLR(
        optimizer, 
        schedulers=[warmup_scheduler, cosine_scheduler],
        milestones=[int(0.05 * Config.NUM_EPOCHS)]
    )
    
    # 混合精度訓練
    scaler = torch.amp.GradScaler('cuda') if Config.FP16 else None

    if Config.DEVICE.type == 'cuda':
        torch.backends.cudnn.benchmark = True  # 為固定大小輸入優化CUDNN
        logger.info("CUDNN Benchmarking enabled")

    # 開始訓練
    logger.info("Starting training...")
    
    for epoch in range(start_epoch, Config.NUM_EPOCHS):
        model.train()
        epoch_loss = 0.0
        
        with tqdm(train_dataloader, desc=f"Epoch {epoch+1}/{Config.NUM_EPOCHS}") as pbar:
            for batch in pbar:
                # 移動數據到設備
                pixel_values = batch['pixel_values'].to(Config.DEVICE)
                labels = batch['labels'].to(Config.DEVICE)
                
                # 清除梯度
                optimizer.zero_grad()
                
                # 前向傳播
                if Config.FP16:
                    with torch.amp.autocast('cuda'):
                        loss = model(pixel_values, labels)
                    
                    # 反向傳播
                    scaler.scale(loss).backward()
                    scaler.step(optimizer)
                    scaler.update()
                else:
                    loss = model(pixel_values, labels)
                    
                    # 反向傳播
                    loss.backward()

                    # 梯度裁剪
                    if Config.GRAD_CLIP:
                        torch.nn.utils.clip_grad_norm_(model.parameters(), Config.GRAD_CLIP)

                    optimizer.step()
                
                # 更新進度條
                epoch_loss += loss.item()
                pbar.set_postfix(loss=loss.item())

                # 記錄批次損失到wandb
                if Config.USE_WANDB:
                    wandb.log({"batch_loss": loss.item()})
        
        # 更新學習率
        lr_scheduler.step()
        
        # 計算平均損失
        avg_loss = epoch_loss / len(train_dataloader)
        logger.info(f"Epoch {epoch+1}/{Config.NUM_EPOCHS}, Loss: {avg_loss:.6f}")

        # 記錄到wandb
        if Config.USE_WANDB:
            wandb.log({
                "epoch": epoch + 1,
                "loss": avg_loss,
                "learning_rate": optimizer.param_groups[0]['lr']
            })
        
        # 保存檢查點
        if (epoch + 1) % Config.SAVE_EVERY == 0 or epoch == Config.NUM_EPOCHS - 1:
            checkpoint_path = os.path.join(Config.CHECKPOINT_DIR, f"epoch_{epoch+1}.pth")
            save_checkpoint(checkpoint_path, model, optimizer, epoch + 1)
            logger.info(f"Checkpoint saved at {checkpoint_path}")

        if (epoch + 1) % Config.EVAL_EVERY == 0:
            logger.info("Evaluating model...")

            results = evaluate_model(
                model=model,
                data_dir=Config.DATA_DIR, 
                test_files=[Config.TEST_JSON, Config.NEW_TEST_JSON],
                guidance_scale=Config.GUIDANCE_SCALE,
                num_steps=Config.NUM_INFERENCE_STEPS,
                device=Config.DEVICE,
                save_dir=Config.RUN_DIR,
                run_id=Config.RUN_ID
            )

            for test_file, acc in results.items():
                logger.info(f"{test_file} 評估準確率: {acc:.4f}")

            # 記錄評估結果到wandb
            if Config.USE_WANDB:
                for test_file, acc in results.items():
                    test_name = test_file.split('.')[0]
                    wandb.log({f"evaluate/{test_name}_accuracy": acc})
                    
    logger.info("Training completed!")

    # 關閉wandb
    if Config.USE_WANDB:
        wandb.finish()
    
def generate_samples(model, dataset, output_dir, epoch, device, num_samples=4):
    """生成樣本圖像"""
    model.eval()
    
    # 隨機選擇一些標籤
    indices = np.random.choice(len(dataset), num_samples, replace=False)
    samples = [dataset[i] for i in indices]
    labels = torch.stack([sample['labels'] for sample in samples]).to(device)
    
    # 生成圖像
    with torch.no_grad():
        images = model.generate(
            labels, 
            batch_size=num_samples, 
            guidance_scale=Config.GUIDANCE_SCALE
        )
    
    # 保存圖像
    sample_dir = os.path.join(output_dir, 'samples')
    os.makedirs(sample_dir, exist_ok=True)
    save_path = os.path.join(sample_dir, f"samples_epoch_{epoch}.png")
    save_images(images, [s['obj_names'] for s in samples], save_path)

    # 轉換為PIL圖像列表
    pil_images = []
    for img in images:
        # 轉換為PIL圖像
        img = (img.cpu() + 1) / 2  # [-1,1] -> [0,1]
        img = torch.clamp(img, 0, 1)
        pil_img = transforms.ToPILImage()(img)
        pil_images.append(pil_img)
    
    # 準備標題
    captions = [", ".join(s['obj_names']) for s in samples]
    
    return pil_images, captions

if __name__ == "__main__":
    args = parse_args()
    train(args)