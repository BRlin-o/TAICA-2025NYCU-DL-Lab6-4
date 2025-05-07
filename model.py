import os
import torch
import torch.nn as nn
import torch.nn.functional as F
from diffusers import AutoencoderKL, UNet2DConditionModel
from diffusers import DDPMScheduler, DDIMScheduler
from diffusers.utils import BaseOutput
from dataclasses import dataclass
from typing import Optional, Tuple, Union

# 自定義擴散模型輸出格式
@dataclass
class LDMOutput(BaseOutput):
    loss: Optional[torch.FloatTensor] = None
    pred_original_sample: Optional[torch.FloatTensor] = None
    sample: Optional[torch.FloatTensor] = None


class ConditionalLDM(nn.Module):
    def __init__(
        self,
        num_labels=24,
        condition_dim=64,
        vae_model_path="stabilityai/sd-vae-ft-mse",
    ):
        """
        條件式潛擴散模型
        
        參數:
            num_labels (int): 標籤數量
            condition_dim (int): 條件嵌入維度
            vae_model_path (str): 預訓練VAE模型路徑/名稱
        """
        super().__init__()
        
        # 1. 加載預訓練VAE
        self.vae = AutoencoderKL.from_pretrained(vae_model_path)
        
        # 凍結VAE參數
        for param in self.vae.parameters():
            param.requires_grad = False
            
        # 獲取VAE潛空間維度
        latent_channels = self.vae.config.latent_channels  # 通常為4
        
        # 2. 條件嵌入層
        self.condition_embedding = nn.Sequential(
            nn.Linear(num_labels, 256),
            nn.SiLU(),
            nn.Linear(256, condition_dim),
            nn.SiLU(),
        )
        
        # 3. UNet擴散模型
        self.unet = UNet2DConditionModel(
            sample_size=16,  # 64/4=16 (VAE降維)
            in_channels=latent_channels,
            out_channels=latent_channels,
            layers_per_block=2,
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
            cross_attention_dim=condition_dim,
        )
        
        # 4. 噪聲調度器
        self.noise_scheduler = DDPMScheduler(
            num_train_timesteps=1000,
            beta_schedule="squaredcos_cap_v2",
            prediction_type="epsilon",
        )
        
        # 5. 採樣調度器
        self.sampler = DDIMScheduler(
            num_train_timesteps=1000,
            beta_schedule="squaredcos_cap_v2",
            prediction_type="epsilon",
        )
        
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
        
        # 5. 計算損失
        loss = F.mse_loss(noise_pred, noise)
        
        return loss
    
    @torch.no_grad()
    def generate(
        self, 
        labels, 
        batch_size=1, 
        generator=None,
        guidance_scale=7.5,
        num_inference_steps=50
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
    def __init__(self, ldm_model, classifier, cfg_scale=7.5, cls_scale=1.0):
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
        self.cfg_scale = cfg_scale
        self.cls_scale = cls_scale
        
    @torch.no_grad()
    def generate(
        self, 
        labels, 
        batch_size=1, 
        generator=None,
        num_inference_steps=50
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
            
            # 分類器引導 (如果啟用)
            if self.cls_scale > 0 and t < timesteps[0] * 0.8:  # 只在後期階段應用分類器引導
                # 需要計算梯度
                with torch.enable_grad():
                    latents_cls = latents.detach().requires_grad_(True)
                    
                    # 臨時解碼為圖像
                    images = self.ldm.decode(latents_cls)
                    
                    # 標準化以符合分類器
                    images_norm = torch.clamp(images, -1.0, 1.0)
                    
                    # 計算分類器得分和梯度
                    cls_device = next(self.classifier.resnet18.parameters()).device
                    cls_score  = self.classifier.resnet18(images_norm.to(cls_device))
                    cls_loss = F.binary_cross_entropy_with_logits(cls_score, labels.to(cls_device))
                    
                    # 計算梯度
                    try:
                        grad = torch.autograd.grad(cls_loss, latents_cls)[0]
                    except RuntimeError:
                        # 使用allow_unused參數處理未連接的梯度
                        grad = torch.autograd.grad(cls_loss, latents_cls, allow_unused=True)[0]
                        if grad is None:
                            grad = torch.zeros_like(latents_cls)
                    
                    # 應用分類器梯度
                    noise_pred = noise_pred - self.cls_scale * grad
            
            # 更新採樣
            latents = self.ldm.sampler.step(noise_pred, t, latents).prev_sample
            
        # 解碼生成的潛變量
        images = self.ldm.decode(latents)
        
        # 裁剪到[-1,1]範圍
        images = torch.clamp(images, -1.0, 1.0)
        
        return images