import os
import json
import torch
import argparse
import numpy as np
from tqdm import tqdm
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image
from dataset import get_dataset, get_transforms, get_inverse_transforms
from model import ConditionalLDM, ClassifierGuidedLDM
from utils import save_checkpoint, load_checkpoint, save_images, setup_logging, log_images
from evaluator import evaluation_model

def parse_args():
    parser = argparse.ArgumentParser(description='Generate images with LDM')
    parser.add_argument('--data_dir', type=str, default='./', help='Data directory')
    parser.add_argument('--output_dir', type=str, default='./output', help='Output directory')
    parser.add_argument('--checkpoint', type=str, required=True, help='Model checkpoint')
    parser.add_argument('--vae_model', type=str, default='stabilityai/sd-vae-ft-mse', help='VAE model path/name')
    parser.add_argument('--batch_size', type=int, default=16, help='Batch size')
    parser.add_argument('--guidance_scale', type=float, default=7.5, help='CFG scale')
    parser.add_argument('--cls_scale', type=float, default=1.0, help='Classifier guidance scale')
    parser.add_argument('--steps', type=int, default=50, help='Inference steps')
    parser.add_argument('--seed', type=int, default=42, help='Random seed')
    parser.add_argument('--test_files', nargs='+', default=['test.json', 'new_test.json'], help='Test files')
    parser.add_argument('--visualize_denoising', action='store_true', help='Visualize denoising')
    return parser.parse_args()

def normalize_for_evaluator(images):
    """正確標準化張量以用於評估器"""
    # 確保值在[0,1]範圍內 (從[-1,1]轉換)
    images = torch.clamp((images + 1) / 2, 0, 1)
    
    # 應用與get_transforms()中相同的標準化
    # transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    # 這將[0,1]映射回[-1,1]
    images = images * 2 - 1
    
    return images

def generate_images(args):
    """生成測試集圖像並評估"""
    # 設置隨機種子
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    
    # 創建輸出目錄
    os.makedirs(args.output_dir, exist_ok=True)
    for test_file in args.test_files:
        test_name = test_file.split('.')[0]
        os.makedirs(os.path.join(args.output_dir, 'images', test_name), exist_ok=True)
    
    # 獲取設備
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # 準備模型
    model = ConditionalLDM(num_labels=24, vae_model_path=args.vae_model)
    
    # 載入檢查點
    load_checkpoint(args.checkpoint, model, map_location=device)
    model = model.to(device)
    model.eval()
    
    # 載入評估器
    evaluator = evaluation_model()
    
    # 使用分類器引導
    guided_model = ClassifierGuidedLDM(model, evaluator, args.guidance_scale, args.cls_scale)
    
    # 獲取轉換
    transform = get_transforms()
    inverse_transform = get_inverse_transforms()
    
    # 為每個測試文件生成圖像
    accuracies = {}
    for test_file in args.test_files:
        print(f"Generating images for {test_file}...")
        test_name = test_file.split('.')[0]
        test_dataset = get_dataset(args.data_dir, test_file)
        test_dataloader = test_dataset.get_dataloader(args.batch_size, shuffle=False)
        
        all_images = []
        all_labels = []
        
        for batch_idx, batch in enumerate(tqdm(test_dataloader)):
            # 移動數據到設備
            labels = batch['labels'].to(device)
            
            # 生成圖像
            with torch.no_grad():
                if args.cls_scale > 0:
                    images = guided_model.generate(
                        labels, 
                        batch_size=labels.size(0),
                        num_inference_steps=args.steps
                    )
                else:
                    images = model.generate(
                        labels, 
                        batch_size=labels.size(0),
                        guidance_scale=args.guidance_scale,
                        num_inference_steps=args.steps
                    )
            
            # 保存個別圖像
            for i, image in enumerate(images):
                idx = batch_idx * args.batch_size + i
                if idx < len(test_dataset):
                    img_path = os.path.join(args.output_dir, 'images', test_name, f"{idx}.png")
                    
                    # 轉換為PIL並保存
                    pil_image = transforms.ToPILImage()(torch.clamp(image * 0.5 + 0.5, 0, 1))
                    pil_image.save(img_path)
            
            # 收集評估用圖像和標籤
            all_images.append(images)
            all_labels.append(labels)
        
        # 串聯所有批次
        all_images = torch.cat(all_images)[:len(test_dataset)]
        all_labels = torch.cat(all_labels)[:len(test_dataset)]
        
        # 評估準確率
        # normalized_images = transform(inverse_transform(all_images))
        normalized_images = normalize_for_evaluator(all_images)
        eva_device = next(evaluator.resnet18.parameters()).device
        acc = evaluator.eval(normalized_images.to(eva_device), all_labels.to(eva_device))
        accuracies[test_file] = acc
        
        print(f"Accuracy for {test_file}: {accuracies[test_file]:.4f}")
        
        # 生成網格圖像
        grid_path = os.path.join(args.output_dir, f"{test_name}_grid.png")
        save_images(all_images[:min(32, len(all_images))], None, grid_path, nrow=4)
    
    # 保存結果
    with open(os.path.join(args.output_dir, 'accuracies.json'), 'w') as f:
        json.dump(accuracies, f, indent=4)
    
    print("Generation completed!")
    
    # 特定標籤的去噪過程
    if args.visualize_denoising:
        visualize_denoising_process(args, model, device)


# def visualize_denoising_process(args, model, device, timesteps=[0, 250, 500, 750, 999]):
#     """可視化去噪過程"""
#     print("Visualizing denoising process...")
    
#     # 創建特定標籤
#     with open(os.path.join(args.data_dir, 'objects.json'), 'r') as f:
#         obj2idx = json.load(f)
    
#     # 特定標籤集
#     specific_objects = ["red sphere", "cyan cylinder", "cyan cube"]
    
#     # 創建one-hot標籤
#     label = torch.zeros(24)
#     for obj in specific_objects:
#         label[obj2idx[obj]] = 1.0
    
#     label = label.unsqueeze(0).to(device)
    
#     # 設置採樣器
#     model.sampler.set_timesteps(1000, device=device)
    
#     # 生成隨機潛變量
#     generator = torch.Generator(device).manual_seed(args.seed)
#     latents = torch.randn(
#         (1, model.unet.config.in_channels, 16, 16),
#         generator=generator,
#         device=device
#     )
    
#     # 準備條件嵌入
#     condition_embedding = model.prepare_condition(label)
    
#     # 準備無條件嵌入
#     uncond_label = torch.zeros_like(label)
#     uncond_embedding = model.prepare_condition(uncond_label)
    
#     # 保存去噪過程中的圖像
#     denoising_images = []
    
#     # 完整時間步
#     all_timesteps = model.sampler.timesteps
    
#     # 選擇視覺化的時間步
#     vis_timesteps = [all_timesteps[i] for i in timesteps]
#     print(f"將保存以下時間步的圖像: {vis_timesteps}")
    
#     # 採樣循環
#     latents_t = latents.clone()
#     for i, t in enumerate(tqdm(all_timesteps)):
#         # 擴展潛變量
#         latent_model_input = torch.cat([latents_t] * 2)
#         latent_model_input = model.sampler.scale_model_input(latent_model_input, t)
        
#         # 擴展條件
#         encoder_hidden_states = torch.cat([condition_embedding, uncond_embedding])
        
#         # 預測噪聲
#         noise_pred = model.unet(
#             latent_model_input, 
#             t,
#             encoder_hidden_states=encoder_hidden_states
#         ).sample
        
#         # 執行CFG
#         noise_pred_cond, noise_pred_uncond = noise_pred.chunk(2)
#         noise_pred = noise_pred_uncond + args.guidance_scale * (noise_pred_cond - noise_pred_uncond)
        
#         # 更新採樣
#         latents_t = model.sampler.step(noise_pred, t, latents_t).prev_sample
        
#         # 保存特定時間步的圖像
#         if t in vis_timesteps:
#             # 立即清除 CUDA 快取並釋放記憶體
#             torch.cuda.empty_cache()
            
#             # 解碼並保存該時間步的圖像
#             with torch.no_grad():
#                 image_t = model.decode(latents_t)
#             # denoising_images.append(image_t)
#             # 轉移到CPU以節省GPU記憶體
#             denoising_images.append(image_t.cpu())

#     # 完成後移回GPU進行處理
#     denoising_images = [img.to(device) for img in denoising_images]
    
#     # 保存去噪過程網格
#     denoising_grid_path = os.path.join(args.output_dir, "denoising_process.png")
#     denoising_images = torch.cat(denoising_images)
#     save_images(denoising_images, None, denoising_grid_path, nrow=len(denoising_images))
    
#     print(f"Denoising process visualization saved to {denoising_grid_path}")

#     # 釋放記憶體
#     del denoising_images, denoised_tensor
#     torch.cuda.empty_cache()

def visualize_denoising_process(args, model, device, num_steps=100):
    """極度優化的去噪可視化"""
    print("使用極簡模式可視化去噪過程...")
    
    # 創建特定標籤
    with open(os.path.join(args.data_dir, 'objects.json'), 'r') as f:
        obj2idx = json.load(f)
    
    specific_objects = ["red sphere", "cyan cylinder", "cyan cube"]
    
    # 創建one-hot標籤
    label = torch.zeros(24)
    for obj in specific_objects:
        label[obj2idx[obj]] = 1.0
    
    label = label.unsqueeze(0).to(device)
    
    # 直接使用少量步數的DDIM採樣
    # 這會極大地減少記憶體消耗
    model.sampler.set_timesteps(num_steps, device=device)
    timesteps = model.sampler.timesteps
    
    # 創建隨機潛變量
    generator = torch.Generator(device).manual_seed(args.seed)
    latents = torch.randn(
        (1, model.unet.config.in_channels, 16, 16),
        generator=generator,
        device=device
    )
    
    # 保存進程圖像
    denoising_images = []
    
    # 準備條件嵌入 (只計算一次)
    condition_embedding = model.prepare_condition(label)
    uncond_embedding = model.prepare_condition(torch.zeros_like(label))
    
    # 進行極簡採樣
    latents_t = latents.clone()
    
    # 紀錄起始雜訊狀態
    with torch.no_grad():
        # 轉換到CPU以節省GPU記憶體
        noise_img = model.decode(latents_t).cpu()
        denoising_images.append(noise_img)
    
    print(f"將使用{num_steps}步DDIM採樣進行可視化")
    
    # 極簡採樣循環
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
        noise_pred = noise_pred_uncond + args.guidance_scale * (noise_pred_cond - noise_pred_uncond)
        
        # 更新採樣
        latents_t = model.sampler.step(noise_pred, t, latents_t).prev_sample
        
        # 儲存中間結果 (每個時間步都儲存)
        with torch.no_grad():
            # 轉換到CPU以節省GPU記憶體
            mid_img = model.decode(latents_t).cpu()
            denoising_images.append(mid_img)
            
        # 主動釋放記憶體
        del noise_pred, latent_model_input
        torch.cuda.empty_cache()
    
    # 所有圖像都在CPU上，安全地串接
    denoised_tensor = torch.cat(denoising_images)
    
    # 保存去噪過程網格
    denoising_grid_path = os.path.join(args.output_dir, "denoising_process.png")

    # 創建臨時目錄存放部分圖像
    temp_dir = os.path.join(args.output_dir, "temp_denoising_parts")
    os.makedirs(temp_dir, exist_ok=True)
    
    # 保存部分網格圖像
    part_paths = []
    
    # 將圖像批量移回GPU以創建網格，但一次只移動一部分
    batch_size = 5  # 每次移動2張圖片
    total_images = len(denoising_images)
    
    for i in range(0, total_images, batch_size):
        end_idx = min(i + batch_size, total_images)
        batch_imgs = torch.cat([img.to(device) for img in denoising_images[i:end_idx]])
        
        # 保存部分網格
        part_path = os.path.join(temp_dir, f"part_{i//batch_size}.png")
        part_paths.append(part_path)
        save_images(batch_imgs, None, part_path, nrow=batch_size)
        
        # 釋放記憶體
        del batch_imgs
        torch.cuda.empty_cache()
    
    # 使用PIL合併所有部分圖像
    from PIL import Image
    
    # 讀取所有部分圖像
    part_images = [Image.open(path) for path in part_paths]
    
    # 計算總高度和最大寬度
    total_height = sum(img.height for img in part_images)
    max_width = max(img.width for img in part_images)
    
    # 創建新圖像
    combined_image = Image.new('RGB', (max_width, total_height))
    
    # 依次垂直拼接
    y_offset = 0
    for img in part_images:
        combined_image.paste(img, (0, y_offset))
        y_offset += img.height
    
    # 保存最終圖像
    combined_image.save(denoising_grid_path)
    
    # 清理臨時文件
    for path in part_paths:
        os.remove(path)
    os.rmdir(temp_dir)
    
    print(f"去噪過程可視化已完成並保存至 {denoising_grid_path}")
    
    # 釋放記憶體
    del denoising_images, part_images, combined_image
    torch.cuda.empty_cache()


if __name__ == "__main__":
    args = parse_args()
    generate_images(args)