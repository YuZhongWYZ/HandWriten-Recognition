#%%
import torch
import torch.nn as nn
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
import torch.nn.functional as F
#%%

# Define PyTorch CNN model
class TorchCNN(nn.Module):
    def __init__(self, input_channels, num_classes):
        """
        PyTorch CNN 模型
        参数：
        - input_channels: 输入通道数，例如 MNIST 的灰度图像为 1
        - num_classes: 输出类别数，例如 MNIST 为 10 类
        """
        super(TorchCNN, self).__init__()
        self.conv_layers = nn.Sequential(
            nn.Conv2d(input_channels, 32, kernel_size=3, stride=1, padding=1),  # 卷积层，输出通道数 32
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),  # 最大池化，减少尺寸
            nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1),  # 卷积层，输出通道数 64
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2)  # 最大池化
        )
        self.fc_layers = nn.Sequential(
            nn.Flatten(),
            nn.Linear(64 * 7 * 7, 128),  # 全连接层，输出维度 128
            nn.ReLU(),
            nn.Linear(128, num_classes)  # 全连接层，输出维度为类别数
        )

    def forward(self, x):
        x = self.conv_layers(x)
        x = self.fc_layers(x)
        return x
#%%

# Define sklearn KNN model
def create_knn_model(n_neighbors=5):
    """
    sklearn KNN 模型
    参数：
    - n_neighbors: 邻居数量，默认 5
    """
    return KNeighborsClassifier(n_neighbors=n_neighbors)

# Define sklearn SVM model
def create_svm_model(kernel='linear', C=1.0):
    """
    sklearn SVM 模型
    参数：
    - kernel: 核函数类型，例如 'linear', 'rbf' 等
    - C: 正则化参数，默认为 1.0
    """
    return SVC(kernel=kernel, C=C, probability=True)

#%%
#模型拓展，这里以MLP为例
class TorchMLP(torch.nn.Module):
    def __init__(self, input_size, hidden_size, num_classes):
        super(TorchMLP, self).__init__()
        self.model = torch.nn.Sequential(
            torch.nn.Flatten(),
            torch.nn.Linear(input_size, hidden_size),
            torch.nn.ReLU(),
            torch.nn.Linear(hidden_size, num_classes)
        )

    def forward(self, x):
        return self.model(x)


#%%

# Define PyTorch Enhanced CNN model
class EnhancedTorchCNN(nn.Module):
    def __init__(self, input_channels, num_classes):
        """
        改进的 PyTorch CNN 模型
        参数：
        - input_channels: 输入通道数，例如 MNIST 的灰度图像为 1
        - num_classes: 输出类别数，例如 MNIST 为 10 类
        """
        super(EnhancedTorchCNN, self).__init__()
        self.conv_layers = nn.Sequential(
            nn.Conv2d(input_channels, 32, kernel_size=3, stride=1, padding=1),  # 卷积层 1
            nn.BatchNorm2d(32),  # 批量归一化
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),  # 最大池化

            nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1),  # 卷积层 2
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),

            nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1),  # 卷积层 3
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2)
        )
        self.fc_layers = nn.Sequential(
            nn.Flatten(),
            nn.Linear(128 * 3 * 3, 256),  # 全连接层 1
            nn.ReLU(),
            nn.Dropout(0.5),  # Dropout 防止过拟合
            nn.Linear(256, num_classes)  # 全连接层 2
        )

    def forward(self, x):
        x = self.conv_layers(x)
        x = self.fc_layers(x)
        return x

#%%

#%%
# Define a ResNet Basic Block
class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, in_channels, out_channels, stride=1, downsample=None):
        super(BasicBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.downsample = downsample

    def forward(self, x):
        identity = x
        if self.downsample is not None:
            identity = self.downsample(x)

        out = self.conv1(x)
        out = self.bn1(out)
        out = F.relu(out, inplace=True)
        out = self.conv2(out)
        out = self.bn2(out)

        out += identity
        out = F.relu(out, inplace=True)

        return out

#%%
# Define the ResNet Model
class ResNet(nn.Module):
    def __init__(self, block, layers, input_channels=1, num_classes=62):
        super(ResNet, self).__init__()
        self.in_channels = 64

        self.conv1 = nn.Conv2d(input_channels, 64, kernel_size=7, stride=2, padding=3, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

        self.layer1 = self._make_layer(block, 64, layers[0])
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2)
        self.layer3 = self._make_layer(block, 256, layers[2], stride=2)
        self.layer4 = self._make_layer(block, 512, layers[3], stride=2)

        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(512 * block.expansion, num_classes)

    def _make_layer(self, block, out_channels, blocks, stride=1):
        downsample = None
        if stride != 1 or self.in_channels != out_channels * block.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(self.in_channels, out_channels * block.expansion,
                          kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(out_channels * block.expansion),
            )

        layers = []
        layers.append(block(self.in_channels, out_channels, stride, downsample))
        self.in_channels = out_channels * block.expansion
        for _ in range(1, blocks):
            layers.append(block(self.in_channels, out_channels))

        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.fc(x)

        return x

# Wrapper to create models based on input
def get_model(model_type,label_flag, **kwargs):
    """
    根据指定类型创建模型
    参数：
    - model_type: 模型类型，支持 'torch_cnn', 'knn', 'svm', 'enhanced_torch_cnn'
    - kwargs: 传递给具体模型的参数
    """
    num_class = 0
    if label_flag == 0:
        num_class = 10
    elif label_flag == 1:
        num_class = 26
    elif label_flag == 2:
        num_class = 36
    elif label_flag == 3:
        num_class = 62
    else:
        print("ERROR: Unavailable num_classes.")

    if model_type == "torch_cnn":
        input_channels = kwargs.get("input_channels", 1)  # 输入通道数，默认 1
        num_classes = kwargs.get("num_classes", num_class)  # 类别数，默认 10
        return TorchCNN(input_channels, num_classes)

    elif model_type == "knn":
        n_neighbors = kwargs.get("n_neighbors", 5)  # KNN 的邻居数量，默认 5
        return create_knn_model(n_neighbors)

    elif model_type == "svm":
        kernel = kwargs.get("kernel", 'linear')  # SVM 的核函数类型，默认 'linear'
        C = kwargs.get("C", 1.0)  # SVM 的正则化参数，默认 1.0
        return create_svm_model(kernel, C)

    elif model_type == "torch_mlp":
        input_size = kwargs.get("input_size", 28 * 28)
        hidden_size = kwargs.get("hidden_size", 128)
        num_classes = kwargs.get("num_classes", num_class)  # 类别数，默认 10
        return TorchMLP(input_size, hidden_size, num_classes)

    elif model_type == "en_torch_cnn":
        input_channels = kwargs.get("input_channels", 1)
        num_classes = kwargs.get("num_classes", num_class)  # 类别数，默认 10
        return EnhancedTorchCNN(input_channels, num_classes)

    else:
        raise ValueError(f"Unsupported model type: {model_type}")

#%%
if __name__ == "__main__":
    # Example usage
    torch_model = get_model("torch_cnn", input_channels=1, num_classes=62)
    print(torch_model)

    knn_model = get_model("knn", n_neighbors=3)
    print(knn_model)

    svm_model = get_model("svm", kernel='rbf', C=2.0)
    print(svm_model)

    mlp_model = get_model("torch_mlp", input_size=28 * 28, hidden_size=128, num_classes=62)
    print(mlp_model)

    enhanced_cnn_model = get_model("en_torch_cnn", input_channels=1, num_classes=62)
    print(enhanced_cnn_model)
    # Example usage
