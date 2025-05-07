import os
import json
import torch
import datetime

class Config:
    # 資料和路徑設定
    DATA_DIR = "./"
    IMG_DIR = "./iclevr"
    OUTPUT_DIR = "output"
    RUN_ID = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")

    # Wandb設定
    USE_WANDB = True
    WANDB_ID = None  # 將在使用wandb時設置
    WANDB_PROJECT = "conditional-ddpm(v4)"
    WANDB_NAME = ""
    
    # 目錄設定 (將在run_id確定後更新)
    # RUN_DIR = os.path.join(OUTPUT_DIR, f"{RUN_ID}" + f"({WANDB_ID})" if USE_WANDB else "")  # 將被設為: output/{RUN_ID}({WANDB_ID})
    RUN_DIR = None # 將被設為: output/{RUN_ID}({WANDB_ID})
    CHECKPOINT_DIR = None
    IMAGES_DIR = None
    EVAL_DIR = None
    LOG_DIR = None
    
    # 資料集文件
    TRAIN_JSON = "train.json"
    TEST_JSON = "test.json"
    NEW_TEST_JSON = "new_test.json"
    OBJECTS_JSON = "objects.json"
    IMAGE_SIZE = 64
    
    # 模型架構參數
    VAE_MODEL = "stabilityai/sd-vae-ft-mse"
    LATENT_CHANNELS = 4  # 潛在空間的通道數
    CONDITION_DIM = 256  # 條件嵌入維度
    NUM_CLASSES = 24  # 物件類別數
    
    # 擴散過程參數
    NUM_TRAIN_TIMESTEPS = 100
    NUM_INFERENCE_STEPS = 50  # DDIM採樣步數
    BETA_SCHEDULE = "squaredcos_cap_v2"  # beta排程
    PREDICTION_TYPE = "v_prediction"  # 預測類型
    
    # 訓練參數
    RESUME = None  # 恢復檢查點的路徑
    # RESUME = "output/2023-10-01_12-00-00/checkpoints/epoch_100.pth"
    BATCH_SIZE = 128
    NUM_EPOCHS = 300
    LEARNING_RATE = 3e-5
    WEIGHT_DECAY = 1e-5
    FP16 = False  # 使用混合精度訓練
    GRAD_CLIP = 1.0  # 梯度裁剪
    SEED = 42
    SAVE_EVERY = 10  # 每N個epoch儲存檢查點
    EVAL_EVERY = 10  # 每N個epoch評估模型
    
    # 硬體設定
    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    NUM_WORKERS = 4
    
    # 採樣參數
    GUIDANCE_SCALE = 7.5  # Classifier-free guidance強度
    CLASSIFIER_SCALE = 1.0  # 分類器引導強度
    
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