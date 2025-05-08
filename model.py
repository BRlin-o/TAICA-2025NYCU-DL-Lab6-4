import os
import torch
import torch.nn as nn
import torch.nn.functional as F
from diffusers import AutoencoderKL, UNet2DConditionModel
from diffusers import DDPMScheduler, DDIMScheduler
from diffusers.utils import BaseOutput
from dataclasses import dataclass
from typing import Optional, Tuple, Union
from config import Config

# 自定義擴散模型輸出格式
@dataclass
class LDMOutput(BaseOutput):
    loss: Optional[torch.FloatTensor] = None
    pred_original_sample: Optional[torch.FloatTensor] = None
    sample: Optional[torch.FloatTensor] = None


class ConditionalLDM(nn.Module):
    def __init__(
        self,
        num_labels=None,
        condition_dim=None,
        vae_model_path=None,
        use_config=True
    ):
        """
        條件式潛擴散模型
        
        參數:
            num_labels (int): 標籤數量
            condition_dim (int): 條件嵌入維度
            vae_model_path (str): 預訓練VAE模型路徑/名稱
            use_config (bool): 是否使用Config類中的參數
        """
        super().__init__()
        
        # 從Config讀取參數或使用提供的參數
        if use_config:
            self.num_labels = num_labels or Config.NUM_CLASSES
            self.condition_dim = condition_dim or Config.CONDITION_DIM
            self.vae_model_path = vae_model_path or Config.VAE_MODEL
            self.num_train_timesteps = Config.NUM_TRAIN_TIMESTEPS
            self.beta_schedule = Config.BETA_SCHEDULE
            self.prediction_type = Config.PREDICTION_TYPE
        else:
            self.num_labels = num_labels or 24
            self.condition_dim = condition_dim or 64
            self.vae_model_path = vae_model_path or "stabilityai/sd-vae-ft-mse"
            self.num_train_timesteps = 1000
            self.beta_schedule = "squaredcos_cap_v2"
            self.prediction_type = "v_prediction"
        
        # 1. 加載預訓練VAE
        self.vae = AutoencoderKL.from_pretrained(self.vae_model_path)
        
        # 凍結VAE參數
        for param in self.vae.parameters():
            param.requires_grad = False
            
        # 獲取VAE潛空間維度
        latent_channels = self.vae.config.latent_channels  # 通常為4
        
        # 2. 條件嵌入層
        self.condition_embedding = nn.Sequential(
            nn.Linear(num_labels, 512),
            nn.SiLU(),
            nn.LayerNorm(512),  # 增加層標準化
            nn.Dropout(0.1),
            nn.Linear(512, 512),
            nn.SiLU(),
            nn.Linear(512, self.condition_dim),
            nn.SiLU(),
        )
        
        # 3. UNet擴散模型
        self.unet = UNet2DConditionModel(
            sample_size=16,  # 64/4=16 (VAE降維)
            in_channels=latent_channels,
            out_channels=latent_channels,
            layers_per_block=3,
            block_out_channels=(128, 256, 384, 512),
            down_block_types=(
                "CrossAttnDownBlock2D",
                "CrossAttnDownBlock2D",
                "CrossAttnDownBlock2D",
                "DownBlock2D",
            ),
            up_block_types=(
                "UpBlock2D",
                "CrossAttnUpBlock2D",
                "CrossAttnUpBlock2D",
                "CrossAttnUpBlock2D",
            ),
            attention_head_dim=32,  # 定義注意力頭數
            cross_attention_dim=self.condition_dim,
        )
        
        # 4. 噪聲調度器
        self.noise_scheduler = DDPMScheduler(
            num_train_timesteps=self.num_train_timesteps,
            beta_schedule=self.beta_schedule,
            prediction_type=self.prediction_type,
            clip_sample=False
        )
        
        # 5. 採樣調度器
        self.sampler = DDIMScheduler(
            num_train_timesteps=self.num_train_timesteps,
            beta_schedule=self.beta_schedule,
            prediction_type=self.prediction_type,
            clip_sample=False,
        )

        def _init_weights(module):
            if isinstance(module, (nn.Linear, nn.Conv2d)):
                torch.nn.init.kaiming_normal_(module.weight, nonlinearity='relu')
                if module.bias is not None:
                    torch.nn.init.zeros_(module.bias)

        self.apply(_init_weights)
        
    def encode(self, pixel_values):
        """圖像編碼為潛變量"""
        with torch.no_grad():
            latent_dist = self.vae.encode(pixel_values)
            latents = latent_dist.latent_dist.sample() * 0.18215
        return latents
    
    def decode(self, latents):
        """潛變量解碼為圖像"""
        with torch.no_grad():
            latents = 1 / 0.18215 * latents
            images = self.vae.decode(latents).sample
        return images
    
    def prepare_condition(self, labels):
        """處理條件標籤"""
        return self.condition_embedding(labels).unsqueeze(1)
    
    def forward(self, pixel_values, labels):
        """
        模型前向傳播
        
        參數:
            pixel_values: 標準化的圖像張量 [B, 3, 64, 64]
            labels: 標籤one-hot向量 [B, 24]
        
        返回:
            loss: 擴散損失
        """
        batch_size = pixel_values.shape[0]
        device = pixel_values.device
        
        # 1. 將圖像編碼為潛變量
        latents = self.encode(pixel_values)
        
        # 2. 為潛變量添加噪聲
        noise = torch.randn_like(latents)
        timesteps = torch.randint(
            0, self.noise_scheduler.config.num_train_timesteps, 
            (batch_size,), device=device
        ).long()
        noisy_latents = self.noise_scheduler.add_noise(latents, noise, timesteps)
        
        # 3. 準備條件嵌入
        condition_embedding = self.prepare_condition(labels)
        
        # 4. 預測添加的噪聲
        noise_pred = self.unet(
            noisy_latents, 
            timesteps, 
            encoder_hidden_states=condition_embedding
        ).sample

        if self.prediction_type == "v_prediction":
            target = self.noise_scheduler.get_velocity(latents, noise, timesteps)
        else:                                       # 兼容 epsilon 與 x0 設定
            target = noise
        
        # 5. 計算損失
        # loss = F.mse_loss(noise_pred, noise)
        # latent_loss = 0.1 * torch.mean(latents.pow(2))
        # loss = F.mse_loss(noise_pred, noise) + latent_loss
        loss = F.mse_loss(noise_pred.float(), target.float(), reduction="mean")
        
        return loss
    
    @torch.no_grad()
    def generate(
        self, 
        labels, 
        batch_size=1, 
        generator=None,
        guidance_scale=None,
        num_inference_steps=None
    ):
        """
        條件圖像生成
        
        參數:
            labels: 標籤one-hot向量 [B, 24]
            batch_size: 批次大小
            generator: 隨機數生成器
            guidance_scale: CFG引導強度
            num_inference_steps: 推理步數
            
        返回:
            images: 生成的圖像
        """
        device = self.unet.device

        guidance_scale = guidance_scale or Config.GUIDANCE_SCALE
        num_inference_steps = num_inference_steps or Config.NUM_INFERENCE_STEPS
        
        # 準備採樣器
        self.sampler.set_timesteps(num_inference_steps, device=device)
        timesteps = self.sampler.timesteps
        
        # 準備潛變量
        latents = torch.randn(
            (batch_size, self.unet.config.in_channels, 16, 16),
            generator=generator,
            device=device
        )
        
        # 準備條件嵌入
        condition_embedding = self.prepare_condition(labels)
        
        # 準備無條件嵌入(用於CFG)
        uncond_labels = torch.zeros_like(labels)
        uncond_embedding = self.prepare_condition(uncond_labels)
        
        # 採樣循環
        for i, t in enumerate(timesteps):
            # 擴展潛變量
            latent_model_input = torch.cat([latents] * 2)
            latent_model_input = self.sampler.scale_model_input(latent_model_input, t)
            
            # 擴展條件
            encoder_hidden_states = torch.cat([condition_embedding, uncond_embedding])
            
            # 預測噪聲
            noise_pred = self.unet(
                latent_model_input, 
                t,
                encoder_hidden_states=encoder_hidden_states
            ).sample
            
            # 執行CFG
            noise_pred_cond, noise_pred_uncond = noise_pred.chunk(2)
            noise_pred = noise_pred_uncond + guidance_scale * (noise_pred_cond - noise_pred_uncond)
            
            # 更新採樣
            latents = self.sampler.step(noise_pred, t, latents).prev_sample
            
        # 解碼生成的潛變量
        images = self.decode(latents)
        
        # 裁剪到[0,1]範圍
        images = torch.clamp(images, -1.0, 1.0)
        
        return images


class ClassifierGuidedLDM(nn.Module):
    def __init__(self, ldm_model, classifier, cfg_scale=None, cls_scale=None):
        """
        帶分類器引導的LDM模型
        
        參數:
            ldm_model: 基本LDM模型
            classifier: 分類器模型
            cfg_scale: CFG引導強度
            cls_scale: 分類器引導強度
        """
        super().__init__()
        self.ldm = ldm_model
        self.classifier = classifier
        self.cfg_scale = cfg_scale if cfg_scale is not None else Config.GUIDANCE_SCALE
        self.cls_scale = cls_scale if cls_scale is not None else Config.CLASSIFIER_SCALE
        
    @torch.no_grad()
    def generate(
        self, 
        labels, 
        batch_size=1, 
        generator=None,
        num_inference_steps=None
    ):
        """
        帶分類器引導的圖像生成
        
        參數:
            labels: 標籤one-hot向量 [B, 24]
            batch_size: 批次大小
            generator: 隨機數生成器
            num_inference_steps: 推理步數
            
        返回:
            images: 生成的圖像
        """
        device = self.ldm.unet.device

        num_inference_steps = num_inference_steps or Config.NUM_INFERENCE_STEPS
        
        # 準備採樣器
        self.ldm.sampler.set_timesteps(num_inference_steps, device=device)
        timesteps = self.ldm.sampler.timesteps
        
        # 準備潛變量
        latents = torch.randn(
            (batch_size, self.ldm.unet.config.in_channels, 16, 16),
            generator=generator,
            device=device
        )
        
        # 準備條件嵌入
        condition_embedding = self.ldm.prepare_condition(labels)
        
        # 準備無條件嵌入(用於CFG)
        uncond_labels = torch.zeros_like(labels)
        uncond_embedding = self.ldm.prepare_condition(uncond_labels)
        
        # 採樣循環
        for i, t in enumerate(timesteps):
            # 清理記憶體
            torch.cuda.empty_cache()
            # 擴展潛變量
            latent_model_input = torch.cat([latents] * 2)
            latent_model_input = self.ldm.sampler.scale_model_input(latent_model_input, t)
            
            # 擴展條件
            encoder_hidden_states = torch.cat([condition_embedding, uncond_embedding])
            
            # 預測噪聲
            noise_pred = self.ldm.unet(
                latent_model_input, 
                t,
                encoder_hidden_states=encoder_hidden_states
            ).sample
            
            # 執行CFG
            noise_pred_cond, noise_pred_uncond = noise_pred.chunk(2)
            noise_pred = noise_pred_uncond + self.cfg_scale * (noise_pred_cond - noise_pred_uncond)
            
            # 分類器引導部分改進
            if self.cls_scale > 0 and i > len(timesteps) * 0.2:  # 在生成過程的後80%應用
                # 自適應調整分類器引導強度 (後期更強)
                progress = 1.0 - i / len(timesteps)  # 從0到1，表示接近完成度
                adaptive_cls_scale = self.cls_scale * (0.5 + 1.5 * progress)  # 從0.5到2倍的cls_scale
                
                with torch.enable_grad():
                    latents_cls = latents.detach().requires_grad_(True)
                    
                    # 解碼為圖像
                    latents_scaled = 1 / 0.18215 * latents_cls
                    with torch.no_grad():
                        images = self.ldm.vae.decode(latents_scaled).sample
                    
                    # 標準化
                    images_norm = torch.clamp(images, -1.0, 1.0)
                    
                    # 應用分類器
                    cls_device = next(self.classifier.resnet18.parameters()).device
                    cls_score = self.classifier.resnet18(images_norm.to(cls_device))
                    
                    # 改進損失函數：加權二元交叉熵 + 餘弦相似度
                    # 1. 加權BCE
                    bce_loss = F.binary_cross_entropy_with_logits(
                        cls_score, labels.to(cls_device),
                        pos_weight=torch.ones_like(labels).to(cls_device) * 2.0  # 加權正例
                    )
                    
                    # 2. 計算餘弦相似度損失 (讓預測向量方向更接近標籤)
                    pred_norm = torch.sigmoid(cls_score)
                    target_norm = labels.to(cls_device)
                    cosine_loss = 1.0 - F.cosine_similarity(pred_norm, target_norm, dim=1).mean()
                    
                    # 組合損失
                    cls_loss = bce_loss + cosine_loss * 0.5
                    
                    # 計算梯度
                    try:
                        grad = torch.autograd.grad(cls_loss, latents_cls, allow_unused=True)[0]
                        if grad is None:
                            grad = torch.zeros_like(latents_cls)
                        
                        # 梯度規範化
                        grad_norm = torch.norm(grad)
                        if grad_norm > 0:
                            grad = grad / grad_norm * min(grad_norm, 1.0)
                        
                        # 應用梯度
                        noise_pred = noise_pred - adaptive_cls_scale * grad
                        
                        # 調試信息
                        if i % 10 == 0:
                            print(f"時間步 {t}: BCE={bce_loss.item():.4f}, Cosine={cosine_loss.item():.4f}, 梯度範數={grad_norm.item():.4f}")
                    
                    except Exception as e:
                        print(f"分類器引導出錯 (時間步 {t}): {e}")
            
            # 更新採樣
            latents = self.ldm.sampler.step(noise_pred, t, latents).prev_sample
            
        # 解碼生成的潛變量
        images = self.ldm.decode(latents)
        
        # 裁剪到[-1,1]範圍
        images = torch.clamp(images, -1.0, 1.0)
        
        return images