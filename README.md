# Lab6

## 環境設定

### 虛擬環境建立與啟用

使用 Python 虛擬環境可避免套件衝突。請根據作業平台依序執行以下步驟：

- **MacOS / Linux:**

    ```bash
    ## env create
    python -m venv .venv
    source ./.venv/bin/activate

    ## pip update
    pip install --upgrade pip

    ## Normal
    pip install -r requirements.txt
    ## Using CUDA
    pip install -r requirements_CUDA.txt
    ```

- **Windows**

    ```bash
    ## env create
    python -m venv .venv
    .\.venv\Scripts\activate.bat

    ## pip update
    python -m pip install --upgrade pip

    ## Normal
    pip install -r requirements.txt
    ## Using CUDA
    pip install -r requirements_CUDA.txt
    ```

## 訓練模型

```bash
python train.py --data_dir ./ --img_dir ./iclevr --output_dir ./output --vae_model stabilityai/sd-vae-ft-mse --batch_size 64 --epochs 100 --lr 1e-4 --save_every 10 --fp16
```

## 生成圖像

```bash
python generate.py --data_dir ./ --output_dir ./output --checkpoint ./output/checkpoint_epoch_10.pth --vae_model stabilityai/sd-vae-ft-mse --guidance_scale 7.5 --cls_scale 1.0 --steps 50 --visualize_denoising
```

## 使用特定檢查點生成圖像

```bash
python main.py --mode sample --checkpoint path/to/checkpoint.pth
```