import os
import json
import torch
import datetime

class Config:
    # 基礎路徑設定
    DATA_DIR = "./"                   # 資料檔案基礎目錄
    IMG_DIR = "./iclevr"              # 訓練圖像存放目錄
    OUTPUT_DIR = "output"             # 輸出結果的基礎目錄
    RUN_ID = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")  # 執行ID，用時間戳記

    # Wandb設定
    USE_WANDB = True
    WANDB_ID = None  # 將在使用wandb時設置
    WANDB_PROJECT = "conditional-ddpm(v4)"
    WANDB_NAME = "RTX4080"
    
    # 動態生成的目錄路徑
    RUN_DIR = None                    # 將被設為: output/{RUN_ID}({WANDB_ID})
    CHECKPOINT_DIR = None             # 檢查點儲存目錄
    IMAGES_DIR = None                 # 生成圖像儲存目錄
    EVAL_DIR = None                   # 評估結果儲存目錄
    LOG_DIR = None                    # 日誌檔案儲存目錄
    
    # 資料集檔案
    TRAIN_JSON = "train.json"         # 訓練資料標籤文件
    TEST_JSON = "test.json"           # 測試資料標籤文件 
    NEW_TEST_JSON = "new_test.json"   # 額外測試資料標籤文件
    OBJECTS_JSON = "objects.json"     # 物件定義文件
    IMAGE_SIZE = 64                   # 圖像大小 (64x64)
    
    # 模型架構設定
    VAE_MODEL = "stabilityai/sd-vae-ft-mse"  # 預訓練VAE模型
    LATENT_CHANNELS = 4               # 潛在空間的通道數
    CONDITION_DIM = 256               # 條件嵌入維度
    NUM_CLASSES = 24                  # 物件類別數 (24 = 8色 x 3形)
    LAYERS_PER_BLOCK = 3              # 每個Block的層數
    BLOCK_OUT_CHANNELS = (128, 256, 384, 512)
    DOWN_BLOCK_TYPES = (
        "CrossAttnDownBlock2D",
        "CrossAttnDownBlock2D",
        "CrossAttnDownBlock2D",
        "DownBlock2D",
    )
    UP_BLOCK_TYPES = (
        "UpBlock2D",
        "CrossAttnDownBlock2D",
        "CrossAttnDownBlock2D",
        "CrossAttnDownBlock2D",
    )
    ATTENTION_HEAD_DIM = 32
    
    # 擴散過程參數
    NUM_TRAIN_TIMESTEPS = 1000         # 訓練時擴散時間步數
    NUM_INFERENCE_STEPS = 50           # DDIM採樣步數
    BETA_SCHEDULE = "squaredcos_cap_v2"  # beta噪聲排程方式
    PREDICTION_TYPE = "v_prediction"  # 預測類型: "epsilon" or "v_prediction"
    CFG_DROPOUT_P = 0.1
    
    # 訓練參數
    RESUME = None                     # 恢復訓練的檢查點路徑
    # RESUME = "output/2023-10-01_12-00-00/checkpoints/epoch_100.pth"    
    BATCH_SIZE = 64                  # 批次大小
    NUM_EPOCHS = 150                  # 訓練輪數
    LEARNING_RATE = 3e-5              # 學習率
    WEIGHT_DECAY = 1e-5               # 權重衰減係數
    FP16 = False                      # 混合精度訓練開關
    GRAD_CLIP = 0.5                   # 梯度裁剪閾值
    SEED = 42                         # 隨機種子
    SAVE_EVERY = 20                   # 每10輪儲存一次檢查點
    EVAL_EVERY = 2                   # 每10輪評估一次模型
    
    # 硬體相關
    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    NUM_WORKERS = 12                   # 數據載入的工作線程數
    PIN_MEMORY = True  # 添加此設定

    
    # 採樣參數
    GUIDANCE_SCALE = 7.5              # 無條件引導強度(CFG) Classifier-free guidance
    CLASSIFIER_SCALE = 1.5            # 分類器引導強度
    
    @classmethod
    def update_paths(cls, wandb_id=None):
        """更新所有路徑，設置運行ID"""
        if wandb_id:
            cls.WANDB_ID = wandb_id
            cls.RUN_DIR = os.path.join(cls.OUTPUT_DIR, f"{cls.RUN_ID}({cls.WANDB_ID})")
        else:
            cls.RUN_DIR = os.path.join(cls.OUTPUT_DIR, cls.RUN_ID)
            
        cls.CHECKPOINT_DIR = os.path.join(cls.RUN_DIR, "checkpoints")
        cls.IMAGES_DIR = os.path.join(cls.RUN_DIR, "images")
        cls.EVAL_DIR = os.path.join(cls.RUN_DIR, "eval")
        cls.LOG_DIR = os.path.join(cls.RUN_DIR, "logs")
    
    @classmethod
    def create_directories(cls):
        """創建必要的目錄"""
        os.makedirs(cls.RUN_DIR, exist_ok=True)
        os.makedirs(cls.CHECKPOINT_DIR, exist_ok=True)
        os.makedirs(cls.IMAGES_DIR, exist_ok=True)
        os.makedirs(cls.EVAL_DIR, exist_ok=True)
        os.makedirs(cls.LOG_DIR, exist_ok=True)
        os.makedirs(os.path.join(cls.IMAGES_DIR, "samples"), exist_ok=True)
        os.makedirs(os.path.join(cls.IMAGES_DIR, "test"), exist_ok=True)
        os.makedirs(os.path.join(cls.IMAGES_DIR, "new_test"), exist_ok=True)
    
    @classmethod
    def save_config(cls):
        """將配置保存為JSON"""
        config_dict = {k: v for k, v in cls.__dict__.items() 
                      if not k.startswith('__') and not callable(getattr(cls, k))}
        
        # 處理不可JSON序列化的類型
        for key, value in config_dict.items():
            if isinstance(value, torch.device):
                config_dict[key] = str(value)
                
        config_path = os.path.join(cls.RUN_DIR, "config.json")
        with open(config_path, 'w') as f:
            json.dump(config_dict, f, indent=4)
        
        return config_path
    
    @classmethod
    def load_config(cls, config_path):
        """從JSON加載配置"""
        with open(config_path, 'r') as f:
            config_dict = json.load(f)
            
        for key, value in config_dict.items():
            if key in cls.__dict__:
                if key == "DEVICE":
                    setattr(cls, key, torch.device(value))
                else:
                    setattr(cls, key, value)
    
    @classmethod
    def update_from_args(cls, args):
        """從命令行參數更新配置"""
        args_dict = vars(args)
        for key, value in args_dict.items():
            if value is not None:  # 只更新非None的值
                upper_key = key.upper()
                if hasattr(cls, upper_key):
                    setattr(cls, upper_key, value)