# Real-ESRGAN-streamlit

基于[Real-ESRGAN](https://github.com/xinntao/Real-ESRGAN)的web交互图片超分辨率工具。

## 1 效果

![](./image/1.gif)

------------------

<p align="center">
  <img src="./image/3.jpg" alt="原图" width="500"/>
  <br>
  <span>原图</span>
  <br>
  <img src="./image/2.png" alt="输出图" width="500"/>
  <br>
  <span>输出图</span>
</p>


## 2 安装

* python3.10
```bash
conda create -n SR python=3.10
conda activate SR
```

* 安装依赖
```bash
pip install torch==2.1.1 torchvision==0.16.1  --index-url https://download.pytorch.org/whl/cu118
pip install -r requirements.txt
python setup.py develop
```

## 3 模型

存放在`./weights`下

* [RealESRGAN_x4plus](https://github.com/xinntao/Real-ESRGAN/releases/download/v0.1.0/RealESRGAN_x4plus.pth)：X4 模型用于一般图像
* [RealESRGAN_x2plus](https://github.com/xinntao/Real-ESRGAN/releases/download/v0.2.1/RealESRGAN_x2plus.pth)：X2 模型用于一般图像
* [RealESRNet_x4plus](https://github.com/xinntao/Real-ESRGAN/releases/download/v0.1.1/RealESRNet_x4plus.pth)：使用均方误差（MSE）作为损失函数，可能导致过度平滑的效果
* [RealESRGAN_x4plus_anime_6B](https://github.com/xinntao/Real-ESRGAN/releases/download/v0.2.2.4/RealESRGAN_x4plus_anime_6B.pth)：针对动漫图像进行了优化


## 4 使用

```bash
streamlit run web.py
```
