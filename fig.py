import torch
from PIL import Image
import os
from model_setup import get_model
import numpy as np
from torch.utils.data import DataLoader, TensorDataset
from tqdm import tqdm

def load_and_process_image(image_path,flag):
    """
    读取图片，调整为28x28大小，进行灰度化，并确保格式与EMNIST数据一致
    :param image_path: 图片路径
    :return: 处理后的Tensor
    """
    image = Image.open(image_path).convert('L')  # 转换为灰度
    print(np.array(image))
    image = image.resize((28, 28))  # 调整尺寸
    print(np.array(image))
    image_array = np.array(image, dtype=np.float32).T / 255.0  # 转换为数组并归一化\
    if flag == 1 :
        image_array = 1-image_array
    print(np.array(image_array))
    image_tensor = torch.tensor(image_array).float().unsqueeze(0)  # 转换为Tensor并添加维度
    test_dataset = TensorDataset(image_tensor)
    test_loader = DataLoader(test_dataset, batch_size=32, shuffle=True)
    print(image_tensor.shape)
    return test_loader

if __name__ == "__main__":
    image_path = "./data/pic/1.png"  # 图片路径
    label_flag = 0  # 设置的label标志 (digits/letters/byclass)
    model_name = input("输入要使用的模型 (torch_cnn/torch_mlp/resnet18): ").strip().lower()
    flag = 1 #翻转标志

    # 初始化模型
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = get_model(model_name, label_flag)
    model_path = f"./models/test/{model_name}_{label_flag}.pt"

    if not os.path.exists(model_path):
        raise FileNotFoundError(f"模型文件 {model_path} 不存在，请检查。")

    # 加载预训练的模型状态
    model.load_state_dict(torch.load(model_path))
    model.to(device)

    # 处理图片
    image_tensor = load_and_process_image(image_path,flag)
    # 预测结果
    with torch.no_grad():
        with tqdm(image_tensor, desc="Validating", unit="batch") as pbar:
            for inputs in pbar:
                print(inputs[0].to(device).shape)
                outputs = model(inputs[0].to(device).float().unsqueeze(0))
                preds = torch.argmax(outputs, dim=1)

    print(f"图片中的预测数字或字母是： {preds}")
