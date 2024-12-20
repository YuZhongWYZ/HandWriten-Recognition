import numpy as np
import torch
from sklearn.metrics import accuracy_score, classification_report, f1_score, precision_score, recall_score
from torch.utils.data import DataLoader, TensorDataset
from data import load_and_preprocess_data
from model_setup import get_model
import os
from PIL import Image, ImageDraw, ImageFont
from tqdm import tqdm
from joblib import dump, load
import pandas as pd  # 导入 pandas 用于处理 Excel 文件

# 训练 PyTorch 模型的函数
def train_torch_model(model, train_loader, epochs=10, lr=0.001):
    """
    训练 PyTorch 模型。

    参数：
    - model: 要训练的 PyTorch 模型。
    - train_loader: 用于训练的数据加载器。
    - val_loader: 用于验证的数据加载器。
    - epochs: 训练的总轮数，默认为 10。
    - lr: 学习率，默认为 0.001。

    返回：
    - metrics: 包含验证集的评价指标（准确率、精确率、召回率、F1分数）。
    """
    model_path = "./models/save"
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)  # 使用 Adam 优化器
    criterion = torch.nn.CrossEntropyLoss()  # 使用交叉熵损失函数
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lambda x: 1 - x / epochs, last_epoch=-1)
    train_loss_save = []

    for epoch in range(epochs):
        model.train()  # 确保模型处于训练模式
        train_loss = 0  # 初始化整个 epoch 的损失

        # 使用 tqdm 添加进度条
        with tqdm(train_loader, desc=f"Epoch {epoch + 1}/{epochs}", unit="batch") as pbar:
            for batch_idx, (inputs, labels) in enumerate(pbar, start=1):  # batch_idx 从 1 开始
                optimizer.zero_grad()
                print(inputs.shape)
                outputs = model(inputs)
                loss = criterion(outputs, labels)
                loss.backward()
                optimizer.step()

                # 累计损失
                train_loss += loss.item()
                current_lr = optimizer.param_groups[0]['lr']
                # 更新进度条中的描述，包含损失和学习率
                pbar.set_postfix(loss=loss.item(), lr=current_lr)

            scheduler.step()

        # 打印每个 epoch 的最终平均损失
        avg = train_loss / len(train_loader)
        print(f"Epoch {epoch + 1}/{epochs}, Loss: {avg:.4f}")
        train_loss_save.append(avg)
        torch.save(model.state_dict(), os.path.join(model_path, f"{model_name}_epoch_{epoch+1}_{label_flag}.pt"))

    print("Train end.")
    return train_loss_save
def test_torch_model(model, val_loader):

    model.eval()
    val_preds, val_targets = [], []
    with torch.no_grad():
        with tqdm(val_loader, desc="Validating", unit="batch") as pbar:
            for inputs, labels in pbar:
                outputs = model(inputs)
                preds = torch.argmax(outputs, dim=1)
                val_preds.extend(preds.cpu().numpy())
                val_targets.extend(labels.cpu().numpy())

    # 计算评价指标
    accuracy = accuracy_score(val_targets, val_preds)  # 准确率
    precision = precision_score(val_targets, val_preds, average='weighted')  # 精确率
    recall = recall_score(val_targets, val_preds, average='weighted')  # 召回率
    f1 = f1_score(val_targets, val_preds, average='weighted')  # F1 分数

    metrics = {
        "accuracy": accuracy,
        "precision": precision,
        "recall": recall,
        "f1": f1
    }
    print("Test end.")
    return metrics
# 训练非 PyTorch 模型（KNN 或 SVM）的函数
def train_non_torch_model(model, x_train, y_train):
    """
    训练非 PyTorch 模型（KNN 或 SVM）。

    参数：
    - model: sklearn 模型（KNN 或 SVM）。
    - x_train: 训练集特征。
    - y_train: 训练集标签。
    """
    # 使用 tqdm 添加进度条
    with tqdm(total=len(x_train), desc="Training", unit="sample") as pbar:
        model.fit(x_train, y_train)
        pbar.update(len(x_train))
    print("Train end.")

def test_non_torch_model(model, x_val, y_val):
    val_preds = model.predict(x_val)  # 在验证集上预测
    print(val_preds)
    # 计算评价指标
    accuracy = accuracy_score(y_val, val_preds)  # 准确率
    precision = precision_score(y_val, val_preds, average='weighted')  # 精确率
    recall = recall_score(y_val, val_preds, average='weighted')  # 召回率
    f1 = f1_score(y_val, val_preds, average='weighted')  # F1 分数

    metrics = {
        "accuracy": accuracy,
        "precision": precision,
        "recall": recall,
        "f1": f1
    }
    print("Test end.")

    return metrics

# 识别并写入图片的函数
def recognize_and_annotate_images(model, folder_path, output_folder, transform, device):
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
            image = Image.open(image_path).convert('RGB')  # 转换为 RGB 格式
            original_image = image.copy()  # 保留原始图像用于绘制

            # 预处理图片
            processed_image = transform(image).unsqueeze(0).to(device)

            # 使用模型预测
            model.eval()
            with torch.no_grad():
                outputs = model(processed_image)
                predicted = torch.argmax(outputs, dim=1).item()

            # 在图片上写入识别结果
            draw = ImageDraw.Draw(original_image)
            font = ImageFont.load_default()  # 使用默认字体
            text = f"{predicted}"  # 识别结果
            draw.text((10, 10), text, fill=(255, 0, 0), font=font)  # 在左上角写入文字

            # 保存图片到输出文件夹
            output_path = os.path.join(output_folder, filename)
            original_image.save(output_path)
            print(f"Processed and saved: {output_path}")

if __name__ == "__main__":
    # 加载并预处理数据
    k = 10
    lr = 0.0005
    label_flag = 3
    epoch = 20
    train_loss = []
    train_acc = []
    train_precision = []
    train_recall =[]
    train_f1 = []

    # k:K折次数  lr:初始学习率  flag: flag = 1 时 大小写合并，分类为36种 ; flag = 0 时 大小写不合并，分类为62种。

    x_min_max, x_mean, y, kf_splits = load_and_preprocess_data(k=k,label_deal_flag=label_flag)

    # 选择一组 K 折数据进行训练和验证
    train_idx, val_idx = kf_splits[0]
    x_train, x_val = x_min_max[train_idx], x_min_max[val_idx]
    y_train, y_val = y[train_idx], y[val_idx]
    # 初始化模型
    model_name = input("请输入要使用的模型 (torch_cnn/torch_mlp/knn/svm): ").strip().lower()
    model = get_model(model_name,label_flag = label_flag)
    model_path = "./models/save"
    os.makedirs(model_path, exist_ok=True)
    data_save_path = "./data/save"
    os.makedirs(data_save_path, exist_ok=True)

    if isinstance(model, torch.nn.Module):  # 判断是否为 PyTorch 模型
        # 转换为 PyTorch 张量
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model.to(device)

        x_train_tensor = torch.tensor(x_train).float().unsqueeze(1).to(device)  # 数据迁移到设备
        x_val_tensor = torch.tensor(x_val).float().unsqueeze(1).to(device)
        y_train_tensor = torch.tensor(y_train).long().to(device)
        y_val_tensor = torch.tensor(y_val).long().to(device)

        # DataLoader 本身不会改变设备，因此需要在训练时对数据迁移
        train_dataset = TensorDataset(x_train_tensor, y_train_tensor)
        val_dataset = TensorDataset(x_val_tensor, y_val_tensor)
        train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
        val_loader = DataLoader(val_dataset, batch_size=32)

        train_loss = train_torch_model(model, train_loader, epochs=epoch,lr=lr)

        torch.save(model.state_dict(), os.path.join(model_path, f"{model_name}_{label_flag}.pt"))
        print(f"PyTorch 模型已保存至 {os.path.join(model_path, f'{model_name}_{label_flag}.pt')}")
        loaded_model = get_model(model_name,label_flag=label_flag)
        loaded_model.load_state_dict(torch.load(os.path.join(model_path, f"{model_name}_{label_flag}.pt")))
        loaded_model.to(device)

        print("\n开始对每个 epoch 的模型进行测试并输出准确率：\n")
        for epoch_num in range(1, epoch + 1):
            # 构建每个 epoch 的模型文件路径
            model_file = os.path.join(model_path, f"{model_name}_epoch_{epoch_num}_{label_flag}.pt")

            if not os.path.exists(model_file):
                print(f"模型文件 {model_file} 不存在。跳过此 epoch。")
                continue

            # 初始化一个新的模型实例
            loaded_model = get_model(model_name, label_flag=label_flag)
            loaded_model.load_state_dict(torch.load(model_file))
            loaded_model.to(device)

            # 测试模型
            metrics = test_torch_model(loaded_model, val_loader)

            # 输出准确率并存储
            print(f"Epoch {epoch_num} 模型的准确率: {metrics['accuracy']:.4f}\n")
            train_acc.append(metrics['accuracy'])
            train_precision.append(metrics['precision'])
            train_recall.append(metrics['recall'])
            train_f1.append(metrics['f1'])

        # 将 train_loss 和 train_acc 以及 metrics 保存到 Excel
        excel_data = {
#            "Epoch": list(range(1, len(train_acc)+1)),
            "Train Loss": train_loss,
            "Train Accuracy": train_acc,
            "Final Precision": train_precision,
            "Final Recall": train_recall,
            "Final f1": train_f1
        }
        # 创建 DataFrame
        df_train = pd.DataFrame(excel_data)
        # 保存到 Excel
        excel_filename = f"{model_name}_label_{label_flag}.xlsx"
        excel_filepath = os.path.join(data_save_path, excel_filename)

        with pd.ExcelWriter(excel_filepath) as writer:
            df_train.to_excel(writer, sheet_name="Train", index=False)
    else:

        # 非 PyTorch 模型（如 KNN 或 SVM）
        train_non_torch_model(model, x_train.reshape(len(x_train), -1), y_train)
        # 保存非 PyTorch 模型
        dump(model, os.path.join(model_path, f"{model_name}_{label_flag}.pkl"))
        print(f"非 PyTorch 模型已保存至 {os.path.join(model_path, f'{model_name}_{label_flag}.pkl')}")
        loaded_model = load(os.path.join(model_path, f"{model_name}_{label_flag}.pkl"))

        metrics = test_non_torch_model(loaded_model,x_val.reshape(len(x_val), -1), y_val)



    # 输出最终验证集评价指标
    print("Final Validation Metrics:")
    for key, value in metrics.items():
        print(f"{key.capitalize()}: {value * 100:.2f}%")

# 可选：识别 upload 文件夹中的图片，并保存带注释的图片
#    recognize_option = input("是否识别 upload 文件夹中的图片并保存结果？(yes/no): ").strip().lower()
#    if recognize_option == "yes":
#        upload_folder = "./upload"  # upload 文件夹路径
#        output_folder = "./output"  # output 文件夹路径
#        transform = Compose([
#            Resize((64, 64)),  # 将图片调整为 64x64
#            ToTensor()        # 转换为 PyTorch 张量
#        ])
#        recognize_and_annotate_images(model, upload_folder, output_folder, transform, "gpu")