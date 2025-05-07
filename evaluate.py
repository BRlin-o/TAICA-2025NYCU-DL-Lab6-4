import os
import torch
import numpy as np
from tqdm import tqdm
from PIL import Image
from torchvision import transforms
from torch.utils.data import DataLoader

from model import ConditionalLDM, ClassifierGuidedLDM
from dataset import get_dataset, get_transforms
from utils import save_images, setup_logging, load_checkpoint
from evaluator import evaluation_model
from config import Config

def normalize_for_evaluator(images):
    """正確標準化張量以用於評估器"""
    # 確保值在[-1,1]範圍內
    images = torch.clamp(images, -1.0, 1.0)
    return images

def evaluate_model(
    model=None,
    data_dir=None,
    test_files=None,
    guidance_scale=7.5,
    classifier_scale=0.0,
    num_steps=50,
    batch_size=32,
    device='cuda',
    checkpoint=None,
    vae_model=None,
    save_dir=None,
    run_id=None
):
    """
    評估模型在測試集上的性能
    
    參數:
        model: 預訓練模型 (如果不提供，則從checkpoint載入)
        data_dir: 數據目錄
        test_files: 測試檔案列表，如果為None則使用默認設置
        guidance_scale: CFG引導強度
        classifier_scale: 分類器引導強度
        num_steps: 擴散步數
        batch_size: 批次大小
        device: 運行設備
        checkpoint: 檢查點路徑 (當model為None時使用)
        vae_model: VAE模型路徑
        save_dir: 保存生成圖像的目錄
        run_id: 運行ID (用於日誌)
    
    返回:
        results: 包含每個測試檔案準確率的字典
    """
    # 設置日誌
    if run_id and save_dir:
        logger = setup_logging(os.path.join(save_dir, "logs"), run_id)
    else:
        import logging
        logger = logging.getLogger(__name__)
    
    # 如果沒有提供模型實例，則根據檢查點創建一個
    if model is None:
        if checkpoint is None:
            raise ValueError("必須提供model或checkpoint之一")
        
        logger.info(f"從檢查點載入模型: {checkpoint}")
        model = ConditionalLDM(num_labels=24, vae_model_path=vae_model or 'stabilityai/sd-vae-ft-mse')
        model = model.to(device)
        
        # 載入檢查點
        load_checkpoint(checkpoint, model)
    
    # 確保模型處於評估模式
    model.eval()
    
    # 加載評估器
    logger.info("初始化評估器...")
    evaluator = evaluation_model()
    
    # 如果沒有提供測試文件，則使用默認的
    if test_files is None:
        test_files = ['test.json', 'new_test.json']
    
    # 如果提供了分類器引導強度，則使用分類器引導模型
    if classifier_scale > 0:
        logger.info(f"使用分類器引導 (強度: {classifier_scale})")
        model = ClassifierGuidedLDM(model, evaluator, guidance_scale, classifier_scale)
    
    results = {}
    
    for test_file in test_files:
        logger.info(f"評估 {test_file}...")
        
        # 加載測試數據
        test_dataset = get_dataset(data_dir, test_file)
        test_dataloader = test_dataset.get_dataloader(
            batch_size=batch_size, 
            shuffle=False, 
            num_workers=2
        )
        
        all_images = []
        all_labels = []
        
        # 生成圖像
        with torch.no_grad():
            for batch in tqdm(test_dataloader, desc=f"生成 {test_file} 圖像"):
                labels = batch['labels'].to(device)
                
                # 生成圖像
                if classifier_scale > 0:
                    images = model.generate(
                        labels, 
                        batch_size=labels.size(0),
                        num_inference_steps=num_steps
                    )
                else:
                    images = model.generate(
                        labels, 
                        batch_size=labels.size(0),
                        guidance_scale=guidance_scale,
                        num_inference_steps=num_steps
                    )
                
                all_images.append(images)
                all_labels.append(labels)
        
        # 串聯所有批次
        all_images = torch.cat(all_images)[:len(test_dataset)]
        all_labels = torch.cat(all_labels)[:len(test_dataset)]
        
        # 標準化圖像用於評估
        normalized_images = normalize_for_evaluator(all_images)
        
        # 獲取評估器設備
        eva_device = next(evaluator.resnet18.parameters()).device
        
        # 評估準確率
        acc = evaluator.eval(normalized_images.to(eva_device), all_labels.to(eva_device))
        results[test_file] = acc
        
        logger.info(f"{test_file} 準確率: {acc:.4f}")
        
        # 保存生成的圖像
        if save_dir:
            test_name = test_file.split('.')[0]
            
            # 保存網格圖像
            eval_dir = os.path.join(save_dir, "eval")
            os.makedirs(eval_dir, exist_ok=True)
            grid_path = os.path.join(eval_dir, f"{test_name}_grid.png")
            save_images(all_images[:min(32, len(all_images))], None, grid_path, nrow=4)
            logger.info(f"網格圖像已保存至: {grid_path}")
            
            # 保存單獨圖像
            img_dir = os.path.join(save_dir, "images", test_name)
            os.makedirs(img_dir, exist_ok=True)
            
            for i, img in enumerate(all_images):
                if i < len(test_dataset):
                    img_path = os.path.join(img_dir, f"{i}.png")
                    img_normalized = (img.cpu() + 1) / 2
                    img_normalized = torch.clamp(img_normalized, 0, 1)
                    pil_image = transforms.ToPILImage()(img_normalized)
                    pil_image.save(img_path)
            
            logger.info(f"單獨圖像已保存至: {img_dir}")
    
    # 將結果保存為JSON
    if save_dir:
        import json
        result_path = os.path.join(save_dir, "eval", "results.json")
        with open(result_path, "w") as f:
            json.dump(results, f, indent=4)
        logger.info(f"評估結果已保存至: {result_path}")
    
    return results

def visualize_denoising_process(
    model=None,
    data_dir=None,
    save_dir=None,
    specific_objects=["red sphere", "cyan cylinder", "cyan cube"],
    guidance_scale=7.5,
    num_steps=20,
    device='cuda',
    checkpoint=None,
    vae_model=None,
    seed=42
):
    """
    可視化去噪過程
    
    參數:
        model: 預訓練模型 (如果不提供，則從checkpoint載入)
        data_dir: 數據目錄 (用於讀取objects.json)
        save_dir: 保存生成圖像的目錄
        specific_objects: 特定物件列表
        guidance_scale: CFG引導強度
        num_steps: 擴散步數
        device: 運行設備
        checkpoint: 檢查點路徑 (當model為None時使用)
        vae_model: VAE模型路徑
        seed: 隨機種子
    
    返回:
        denoising_path: 保存的去噪過程圖像路徑
    """
    # 設置隨機種子
    torch.manual_seed(seed)
    np.random.seed(seed)
    
    # 如果沒有提供模型實例，則根據檢查點創建一個
    if model is None:
        if checkpoint is None:
            raise ValueError("必須提供model或checkpoint之一")
        
        print(f"從檢查點載入模型: {checkpoint}")
        model = ConditionalLDM(num_labels=24, vae_model_path=vae_model or 'stabilityai/sd-vae-ft-mse')
        model = model.to(device)
        
        # 載入檢查點
        load_checkpoint(checkpoint, model)
    
    # 確保模型處於評估模式
    model.eval()
    
    print("開始可視化去噪過程...")
    
    # 創建特定標籤
    with open(os.path.join(data_dir, 'objects.json'), 'r') as f:
        obj2idx = json.load(f)
    
    # 創建one-hot標籤
    label = torch.zeros(24)
    for obj in specific_objects:
        label[obj2idx[obj]] = 1.0
    
    label = label.unsqueeze(0).to(device)
    
    # 設置採樣器
    model.sampler.set_timesteps(num_steps, device=device)
    timesteps = model.sampler.timesteps
    
    # 創建隨機潛變量
    generator = torch.Generator(device).manual_seed(seed)
    latents = torch.randn(
        (1, model.unet.config.in_channels, 16, 16),
        generator=generator,
        device=device
    )
    
    # 保存去噪過程中的圖像
    denoising_images = []
    
    # 紀錄起始雜訊狀態
    with torch.no_grad():
        noise_img = model.decode(latents)
        denoising_images.append(noise_img.cpu())
    
    print(f"使用 {num_steps} 步擴散進行去噪...")
    
    # 準備條件嵌入 (只計算一次)
    condition_embedding = model.prepare_condition(label)
    uncond_embedding = model.prepare_condition(torch.zeros_like(label))
    
    # 採樣循環
    latents_t = latents.clone()
    
    for i, t in enumerate(tqdm(timesteps)):
        # 每步完成後清除快取
        torch.cuda.empty_cache()
        
        # 擴展潛變量
        latent_model_input = torch.cat([latents_t] * 2)
        latent_model_input = model.sampler.scale_model_input(latent_model_input, t)
        
        # 擴展條件
        encoder_hidden_states = torch.cat([condition_embedding, uncond_embedding])
        
        # 預測噪聲
        with torch.no_grad():
            noise_pred = model.unet(
                latent_model_input, 
                t,
                encoder_hidden_states=encoder_hidden_states
            ).sample
        
        # 執行CFG
        noise_pred_cond, noise_pred_uncond = noise_pred.chunk(2)
        noise_pred = noise_pred_uncond + guidance_scale * (noise_pred_cond - noise_pred_uncond)
        
        # 更新採樣
        latents_t = model.sampler.step(noise_pred, t, latents_t).prev_sample
        
        # 儲存中間結果
        with torch.no_grad():
            mid_img = model.decode(latents_t)
            denoising_images.append(mid_img.cpu())
            
        # 釋放記憶體
        del noise_pred, latent_model_input
        torch.cuda.empty_cache()
    
    # 創建輸出目錄
    if save_dir:
        os.makedirs(os.path.join(save_dir, "eval"), exist_ok=True)
        
        # 保存去噪過程網格
        denoising_grid_path = os.path.join(save_dir, "eval", "denoising_process.png")
        
        # 將所有圖像串聯
        all_denoise_images = torch.cat(denoising_images)
        
        # 保存為網格
        save_images(all_denoise_images, None, denoising_grid_path, nrow=len(denoising_images))
        
        # 保存為逐步圖像
        steps_dir = os.path.join(save_dir, "eval", "denoising_steps")
        os.makedirs(steps_dir, exist_ok=True)
        
        for i, img in enumerate(denoising_images):
            step_name = "noise" if i == 0 else f"step_{i}"
            img_path = os.path.join(steps_dir, f"{step_name}.png")
            img_normalized = (img.squeeze() + 1) / 2
            img_normalized = torch.clamp(img_normalized, 0, 1)
            pil_image = transforms.ToPILImage()(img_normalized)
            pil_image.save(img_path)
        
        print(f"去噪過程可視化已保存至: {denoising_grid_path}")
        return denoising_grid_path
    
    return None