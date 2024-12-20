import torch
from torchvision import transforms
from model_setup import get_model  # 替换为你的模型加载函数
from PIL import Image, ImageDraw, ImageFont
import os
import numpy as np

def recognize_and_annotate_images(model, folder_path, output_folder, device,flag):
    """
    识别文件夹中的数字或字母，并将识别结果写入图片。

    参数：
    - model: 预训练的 PyTorch 模型。
    - folder_path: 包含图片的文件夹路径。
    - output_folder: 保存注释后图片的文件夹路径。
    - transform: 对图片进行预处理的转换操作。
    - device: 使用的设备（CPU 或 GPU）。
    """
    os.makedirs(output_folder, exist_ok=True)  # 确保输出文件夹存在

    for filename in os.listdir(folder_path):
        if filename.endswith(('.png', '.jpg', '.jpeg')):
            image_path = os.path.join(folder_path, filename)
            image = Image.open(image_path).convert('RGB')
            original_image = image.copy()  # 保留原始图像用于绘制

            image_G = Image.open(image_path).convert('L').resize((28, 28))  # 转换为灰度
            image_array = np.array(image_G, dtype=np.float32).T / 255.0  # 转换为数组并归一化
            if flag == 1:
                image_array = 1 - image_array
            processed_image = torch.tensor(image_array).float().unsqueeze(0).unsqueeze(0).to(device)  # 转换为Tensor并添加维度和通道

            # 使用模型预测
            model.eval()
            with torch.no_grad():
                outputs = model(processed_image)
                predicted = torch.argmax(outputs, dim=1).item()

            # 在图片上写入识别结果
            draw = ImageDraw.Draw(original_image)
            font = ImageFont.load_default()  # 使用默认字体
            text = f"{predicted}"  # 识别结果
            draw.text((10, 10), text, fill=(0), font=font)  # 在左上角写入文字

            # 保存图片到输出文件夹
            output_path = os.path.join(output_folder, filename)
            original_image.save(output_path)
            print(f"Processed and saved: {output_path}")

if __name__ == "__main__":
    # 参数设置
    folder_path = "./data/pic"  # 输入文件夹
    output_folder = "./data/annotated_images"  # 输出文件夹
    label_flag = 0  # 标志类别
    model_name = "en_torch_cnn"  # 使用的模型名称
    flag = 1 # flag 为1 时 黑字白底，flag为0时白字黑底
    # 加载模型
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = get_model(model_name, label_flag)
    model_path = f"./models/test/{model_name}_{label_flag}.pt"

    if not os.path.exists(model_path):
        raise FileNotFoundError(f"模型文件 {model_path} 不存在，请检查。")

    model.load_state_dict(torch.load(model_path))
    model.to(device)

    # 定义预处理
    # 调用函数进行识别
    recognize_and_annotate_images(
        model=model,
        folder_path=folder_path,
        output_folder=output_folder,
        device=device,
        flag = flag
    )
