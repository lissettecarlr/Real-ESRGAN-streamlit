import streamlit as st
import cv2
import os
import tempfile
from realesrgan import RealESRGANer
import time
import random
import torch


# 由于torchvision版本不同，可能会导致报错:https://github.com/xinntao/Real-ESRGAN/issues/859
import sys
import types
try:
    # Check if `torchvision.transforms.functional_tensor` and `rgb_to_grayscale` are missing
    from torchvision.transforms.functional_tensor import rgb_to_grayscale
except ImportError:
    # Import `rgb_to_grayscale` from `functional` if it’s missing in `functional_tensor`
    from torchvision.transforms.functional import rgb_to_grayscale
    # Create a module for `torchvision.transforms.functional_tensor`
    functional_tensor = types.ModuleType("torchvision.transforms.functional_tensor")
    functional_tensor.rgb_to_grayscale = rgb_to_grayscale
    # Add this module to `sys.modules` so other imports can access it
    sys.modules["torchvision.transforms.functional_tensor"] = functional_tensor

from basicsr.archs.rrdbnet_arch import RRDBNet


@st.cache_resource
def load_model(model_name, device="cpu", tile=0):
    model_configs = {
        'RealESRGAN_x4plus': (RRDBNet(num_in_ch=3, num_out_ch=3, num_feat=64, num_block=23, num_grow_ch=32, scale=4), 4),
        'RealESRNet_x4plus': (RRDBNet(num_in_ch=3, num_out_ch=3, num_feat=64, num_block=23, num_grow_ch=32, scale=4), 4),
        'RealESRGAN_x4plus_anime_6B': (RRDBNet(num_in_ch=3, num_out_ch=3, num_feat=64, num_block=6, num_grow_ch=32, scale=4), 4),
        'RealESRGAN_x2plus': (RRDBNet(num_in_ch=3, num_out_ch=3, num_feat=64, num_block=23, num_grow_ch=32, scale=2), 2)
    }

    if model_name not in model_configs:
        raise ValueError(f'不支持的模型名称 {model_name}')

    model, netscale = model_configs[model_name]
    model_path = os.path.join('weights', model_name + '.pth')

    if not os.path.isfile(model_path):
        raise FileNotFoundError(f'未找到模型文件 {model_path}，请先下载')
    
    print(f'使用模型 {model_name}')

    half = device != 'cpu'

    return RealESRGANer(
        scale=netscale,
        model_path=model_path,
        model=model,
        tile=tile,
        tile_pad=10,
        pre_pad=0,
        half=half,
        device=device
    )

def main():
    # 输出文件夹
    output_folder = 'output'
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)
    else:
        # 清空输出文件夹里的所有文件
        for file in os.listdir(output_folder):
            os.remove(os.path.join(output_folder, file))
    
    st.title("基于Real-ESRGAN的图片超分辨率")
    model_name = st.selectbox(
        "选择模型",
        ['RealESRGAN_x4plus', 'RealESRNet_x4plus', 'RealESRGAN_x4plus_anime_6B', 'RealESRGAN_x2plus']
    )
    device_option = st.selectbox(
        "选择设备",
        ['cuda:0' if torch.cuda.is_available() else 'cpu', 'cpu']
    )
    tile = st.number_input("Tile 参数（切分原图，降低GPU内存使用，0为不切分）", min_value=0, max_value=512, value=0, step=1)

    if 'model_handler' not in st.session_state:
        st.session_state.model_handler = None

    if st.button('加载模型'):
        st.session_state.model_handler = load_model(model_name, device=device_option, tile=tile)
        st.write(f"模型 {model_name} 已加载，设备：{device_option}, Tile: {tile}")

    uploaded_file = st.file_uploader("上传图像", type=["png", "jpg", "jpeg"])
    if uploaded_file is not None:
        with tempfile.NamedTemporaryFile(delete=False) as temp_file:
            temp_file.write(uploaded_file.read())
            input_image_path = temp_file.name

        img = cv2.imread(input_image_path, cv2.IMREAD_UNCHANGED)
        st.image(input_image_path, caption='原始图像', use_container_width=True)
        if img is not None:
            if st.button('开始转换'):
                if st.session_state.model_handler is None:
                    st.error("请先加载模型！")
                else:
                    with st.spinner('转换中，请稍候...'):
                        output, _ = st.session_state.model_handler.enhance(img, outscale=4)
                        # 根据时间和随机数创建文件名字
                        filename = f"{int(time.time())}_{random.randint(0, 1000)}.png"
                        output_image_path = os.path.join('output', filename)
                        cv2.imwrite(output_image_path, output)

                        st.image(output_image_path, caption='转换后图像', use_container_width=True)
                        # 提供下载按钮
                        with open(output_image_path, "rb") as file:
                            btn = st.download_button(
                                label="下载图像",
                                data=file,
                                file_name=filename,
                                mime="image/png"
                            )
        else:
            st.write("无法读取上传的图片!")

if __name__ == "__main__":
    main()
