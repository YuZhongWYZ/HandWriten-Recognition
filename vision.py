import os
import torch
import numpy as np
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
from matplotlib.backends.backend_pdf import PdfPages
from torch.utils.data import DataLoader, TensorDataset
from model_setup import get_model
from data import load_and_preprocess_data

# 全局变量
device = torch.device("cpu")
model_path = "./models/test/"  # 模型保存路径
data_path = "./data/save"  # 数据保存路径

def load_model(model_name, label_flag, input_channels=1):
    print(f"Inside load_model: {model_name}, label_flag={label_flag}")

    num_classes_map = {0: 10, 1: 26, 2: 36, 3: 62}
    if label_flag not in num_classes_map:
        raise KeyError(f"Invalid label_flag: {label_flag}. Supported values are {list(num_classes_map.keys())}")
    num_classes = num_classes_map[label_flag]

    model = get_model(model_name, label_flag, input_channels=input_channels, num_classes=num_classes)
    model_file = f"{model_path}/{model_name}_{label_flag}.pt"

    print(f"Attempting to load model from: {model_file}")
    try:
        model.load_state_dict(torch.load(model_file, map_location=torch.device('cpu')))
        print(f"Model {model_name} loaded successfully.")
    except FileNotFoundError:
        print(f"Error: Model file not found: {model_file}")
        raise
    except Exception as e:
        print(f"Error loading model: {e}")
        raise

    model.to(device)
    model.eval()
    return model



def visualize_confusion_matrix(model, val_loader, num_classes):
    """生成混淆矩阵，支持动态分类数"""
    val_preds, val_targets = [], []
    with torch.no_grad():
        for inputs, labels in val_loader:
            inputs = inputs.to(device)
            outputs = model(inputs)
            preds = torch.argmax(outputs, dim=1)
            val_preds.extend(preds.cpu().numpy())
            val_targets.extend(labels.numpy())

    # 动态生成混淆矩阵
    cm = confusion_matrix(val_targets, val_preds, labels=range(num_classes))
    return cm

def visualize_feature_distribution(model, val_loader):
    """生成 t-SNE 特征分布"""
    all_features = []
    all_labels = []
    with torch.no_grad():
        for inputs, labels in val_loader:
            inputs = inputs.to(device)
            features = model.conv_layers(inputs)  # 提取卷积层特征
            all_features.append(features.cpu().view(features.size(0), -1).numpy())
            all_labels.append(labels.numpy())
    all_features = np.concatenate(all_features, axis=0)
    all_labels = np.concatenate(all_labels, axis=0)
    tsne = TSNE(n_components=2, random_state=42)
    features_2d = tsne.fit_transform(all_features)
    return features_2d, all_labels


def visualize_samples(val_loader, model, correct=True, num_samples=6):
    """生成分类样本的可视化子图"""
    val_preds, val_targets, val_inputs = [], [], []
    with torch.no_grad():
        for inputs, labels in val_loader:
            inputs = inputs.to(device)
            outputs = model(inputs)
            preds = torch.argmax(outputs, dim=1)
            val_preds.extend(preds.cpu().numpy())
            val_targets.extend(labels.numpy())
            val_inputs.extend(inputs.cpu().numpy())
    val_inputs = np.array(val_inputs)
    indices = [i for i, (p, t) in enumerate(zip(val_preds, val_targets)) if (p == t) == correct]
    indices = indices[:num_samples]
    return [(val_inputs[i, 0], val_targets[i], val_preds[i]) for i in indices]

def process_models_to_pdfs(data_path, epoch_number=20):
    """处理所有模型并生成多个PDF文件"""
    files = [f for f in os.listdir(data_path) if f.endswith(".xlsx")]
    print("Files in data_path:", files)  # 打印文件列表以调试

    processed_models = []
    skipped_files = []

    # 初始化PDF文件
    confusion_matrix_pdf = PdfPages("confusion_matrices.pdf")
    tsne_pdf = PdfPages("tsne_feature_distributions.pdf")
    incorrect_samples_pdf = PdfPages("incorrect_samples.pdf")
    correct_samples_pdf = PdfPages("correct_samples.pdf")

    for file in files:
        try:
            # 解析模型名称和 label_flag
            model_name, label_flag = file.split("_label_")[0], int(file.split("_label_")[1].split(".")[0])
            print(f"Processing file: {file}, Parsed model_name: {model_name}, label_flag: {label_flag}")

            num_classes_map = {0: 10, 1: 26, 2: 36, 3: 62}
            if label_flag not in num_classes_map:
                print(f"Skipping unsupported label_flag: {label_flag}")
                skipped_files.append((file, f"Unsupported label_flag: {label_flag}"))
                continue

            num_classes = num_classes_map[label_flag]

            # 加载数据
            x_min_max, x_mean, y, kf_splits = load_and_preprocess_data(k=10, label_deal_flag=label_flag)
            train_idx, val_idx = kf_splits[0]
            x_val = x_min_max[val_idx]
            y_val = y[val_idx]
            x_val_tensor = torch.tensor(x_val).float().unsqueeze(1)
            y_val_tensor = torch.tensor(y_val).long()
            val_dataset = TensorDataset(x_val_tensor, y_val_tensor)
            val_loader = DataLoader(val_dataset, batch_size=32)

            # 加载模型
            model = load_model(model_name, label_flag)
            print(f"Model loaded: {model_name}, label_flag={label_flag}")

            processed_models.append((model_name, label_flag))

            # 混淆矩阵
            print(f"Generating confusion matrix for model: {model_name}, label_flag: {label_flag}")
            cm = visualize_confusion_matrix(model, val_loader, num_classes=num_classes)
            fig, ax = plt.subplots(figsize=(8, 8))  # 设置单张图大小
            disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=range(num_classes))
            disp.plot(ax=ax, cmap="Blues", colorbar=True)
            ax.set_title(f"Confusion Matrix\nModel: {model_name}, Label: {label_flag}")
            confusion_matrix_pdf.savefig(fig)
            plt.close(fig)

            # 特征分布图（t-SNE）
            print(f"Generating t-SNE for model: {model_name}, label_flag: {label_flag}")
            features_2d, labels = visualize_feature_distribution(model, val_loader)
            fig_tsne, ax_tsne = plt.subplots(figsize=(8, 8))
            scatter = ax_tsne.scatter(features_2d[:, 0], features_2d[:, 1], c=labels, cmap="viridis", s=5)
            ax_tsne.set_title(f"t-SNE Distribution\nModel: {model_name}, Label: {label_flag}")
            tsne_pdf.savefig(fig_tsne)
            plt.close(fig_tsne)

            # 错误分类样本
            print(f"Generating incorrect samples for model: {model_name}, label_flag: {label_flag}")
            incorrect_samples = visualize_samples(val_loader, model, correct=False, num_samples=25)
            fig_incorrect, axs_incorrect = plt.subplots(5, 5, figsize=(16, 16))  # 5x5布局
            for i, (img, true_label, pred_label) in enumerate(incorrect_samples):
                ax = axs_incorrect[i // 5, i % 5]
                ax.imshow(img, cmap="gray")
                ax.set_title(f"T: {true_label}\nP: {pred_label}", fontsize=8)
                ax.axis("off")
            fig_incorrect.suptitle(f"Incorrect Samples\nModel: {model_name}, Label: {label_flag}")
            incorrect_samples_pdf.savefig(fig_incorrect)
            plt.close(fig_incorrect)

            # 正确分类样本
            print(f"Generating correct samples for model: {model_name}, label_flag: {label_flag}")
            correct_samples = visualize_samples(val_loader, model, correct=True, num_samples=25)
            fig_correct, axs_correct = plt.subplots(5, 5, figsize=(16, 16))  # 5x5布局
            for i, (img, true_label, pred_label) in enumerate(correct_samples):
                ax = axs_correct[i // 5, i % 5]
                ax.imshow(img, cmap="gray")
                ax.set_title(f"T: {true_label}\nP: {pred_label}", fontsize=8)
                ax.axis("off")
            fig_correct.suptitle(f"Correct Samples\nModel: {model_name}, Label: {label_flag}")
            correct_samples_pdf.savefig(fig_correct)
            plt.close(fig_correct)

        except Exception as e:
            print(f"Error processing file {file}: {e}")
            skipped_files.append((file, f"Error: {e}"))
            continue

    # 关闭PDF文件
    confusion_matrix_pdf.close()
    tsne_pdf.close()
    incorrect_samples_pdf.close()
    correct_samples_pdf.close()

    # 输出处理过的模型信息
    print("\nProcessed Models:")
    for model_name, label_flag in processed_models:
        print(f"Model: {model_name}, Label Flag: {label_flag}")

    # 输出被跳过的文件信息
    print("\nSkipped Files:")
    for file, reason in skipped_files:
        print(f"File: {file}, Reason: {reason}")


if __name__ == "__main__":
    process_models_to_pdfs(data_path)
    print("PDFs saved successfully.")
