import os
import sys
import argparse
import torch
from config import Config
from train import train
from generate import generate_images
from evaluate import evaluate_model, visualize_denoising_process
import wandb

def parse_args():
    parser = argparse.ArgumentParser(description='條件式擴散模型訓練與評估')
    
    # 基本命令參數
    parser.add_argument('action', choices=['train', 'evaluate'], help='要執行的操作: train 或 evaluate')
    
    # 配置相關
    parser.add_argument('--config', type=str, default=None, help='載入的配置檔案路徑')
    parser.add_argument('--checkpoint', type=str, default=None, help='模型檢查點路徑 (用於評估或繼續訓練)')
    
    # 資料與輸出路徑
    parser.add_argument('--data_dir', type=str, default=None, help='資料目錄')
    parser.add_argument('--img_dir', type=str, default=None, help='圖像目錄')
    parser.add_argument('--output_dir', type=str, default=None, help='輸出目錄')
    
    # 模型參數
    parser.add_argument('--vae_model', type=str, default=None, help='VAE模型路徑/名稱')
    parser.add_argument('--guidance_scale', type=float, default=None, help='CFG引導強度')
    parser.add_argument('--classifier_scale', type=float, default=None, help='分類器引導強度')
    parser.add_argument('--inference_steps', type=int, default=None, help='推理步數')
    
    # 訓練參數
    parser.add_argument('--batch_size', type=int, default=None, help='批次大小')
    parser.add_argument('--epochs', type=int, default=None, help='訓練輪數')
    parser.add_argument('--lr', type=float, default=None, help='學習率')
    parser.add_argument('--save_every', type=int, default=None, help='每N個epoch儲存一次檢查點')
    parser.add_argument('--eval_every', type=int, default=None, help='每N個epoch評估一次模型')
    parser.add_argument('--seed', type=int, default=None, help='隨機種子')
    
    # 硬體設定
    parser.add_argument('--num_workers', type=int, default=None, help='數據加載工作進程數')
    parser.add_argument('--fp16', action='store_true', help='使用混合精度訓練')
    
    # Wandb設定
    parser.add_argument('--use_wandb', action='store_true', help='使用Weights & Biases進行日誌記錄')
    parser.add_argument('--wandb_project', type=str, default=None, help='Wandb專案名稱')
    parser.add_argument('--wandb_name', type=str, default=None, help='Wandb運行名稱')
    
    # 評估特定參數
    parser.add_argument('--visualize_denoising', action='store_true', help='可視化去噪過程')
    parser.add_argument('--test_files', nargs='+', default=None, help='評估用的測試檔案')
    
    return parser.parse_args()

def main():
    # 解析命令行參數
    args = parse_args()
    
    # 如果指定了配置檔案，則加載
    if args.config:
        if os.path.exists(args.config):
            Config.load_config(args.config)
            print(f"已從 {args.config} 載入配置")
        else:
            print(f"錯誤: 配置檔案 {args.config} 不存在")
            sys.exit(1)

    # 設置Wandb (如果啟用)
    if Config.USE_WANDB:
        wandb.init(
            project=Config.WANDB_PROJECT,
            name=Config.WANDB_NAME or Config.RUN_ID,
        )
        Config.WANDB_ID = wandb.run.id
    
    # 根據命令行參數更新配置
    Config.update_from_args(args)
    
    # 根據動作執行相應功能
    if args.action == 'train':
        # 更新路徑
        Config.update_paths(wandb_id=Config.WANDB_ID if Config.USE_WANDB else None)
        
        # 創建目錄
        Config.create_directories()
        
        # 儲存運行配置
        config_path = Config.save_config()
        print(f"配置已儲存到: {config_path}")
        
        # 設置隨機種子
        torch.manual_seed(Config.SEED)
        torch.cuda.manual_seed_all(Config.SEED)
        
        # 執行訓練
        print(f"開始訓練 - 運行ID: {Config.RUN_ID}")
        train(args)
        
    elif args.action == 'evaluate':
        if not args.checkpoint:
            print("錯誤: 評估模式需要指定模型檢查點 (--checkpoint)")
            sys.exit(1)
            
        # 更新路徑
        Config.update_paths()
        
        # 創建目錄
        Config.create_directories()
        
        # 儲存運行配置
        config_path = Config.save_config()
        print(f"配置已儲存到: {config_path}")
        
        # 設置隨機種子
        torch.manual_seed(Config.SEED)
        torch.cuda.manual_seed_all(Config.SEED)
        
        # 執行評估
        print(f"開始評估 - 運行ID: {Config.RUN_ID}")
        
        # 生成測試集圖像並評估
        results = evaluate_model(
            data_dir=Config.DATA_DIR,
            test_files=[Config.TEST_JSON, Config.NEW_TEST_JSON],
            guidance_scale=Config.GUIDANCE_SCALE,
            classifier_scale=Config.CLASSIFIER_SCALE,
            num_steps=Config.NUM_INFERENCE_STEPS,
            device=Config.DEVICE,
            checkpoint=args.checkpoint,
            vae_model=Config.VAE_MODEL,
            save_dir=Config.RUN_DIR,
            run_id=Config.RUN_ID
        )
        
        # 可視化去噪過程
        if args.visualize_denoising:
            print("開始可視化去噪過程...")
            visualize_denoising_process(
                data_dir=Config.DATA_DIR,
                save_dir=Config.RUN_DIR,
                guidance_scale=Config.GUIDANCE_SCALE,
                num_steps=Config.NUM_INFERENCE_STEPS,
                device=Config.DEVICE,
                checkpoint=args.checkpoint,
                vae_model=Config.VAE_MODEL,
                seed=Config.SEED
            )
        
        print("評估完成!")

if __name__ == "__main__":
    main()