# V2项目简介
该项目以EMNIST为基础数据集进行训练，实现了一个简单的手写数字识别系统。主要功能包括：
- 数据预处理
- 模型训练
- 模型评估
- 模型预测
- 模型保存与加载
- 训练结果可视化
 
# 环境依赖  
  requirements.txt文件给出基本的依赖库，可以通过以下命令安装：  
  

    pip install requirements.txt
 本项目基于torch，并设置GPU运算，如果本地GPU为N卡则无需修改，若本地使用A卡，应在```main.ipynb```文档中“cuda”处进行修改。此处也可选择使用CPU，但训练速度较慢。
 
# 使用说明
## 添加模型
若要添加模型，须在model_setup.py文件中添加模型的类，并在model_setup.py文件中的```get_model```函数中添加模型调用的接口。
```python
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

def get_model(model_type, **kwargs):
    """
    根据指定类型创建模型
    参数：
    - model_type: 模型类型，支持 'torch_cnn', 'knn', 'svm'
    - kwargs: 传递给具体模型的参数
    """

    # 模型拓展，在上文添加模型后，在此处注册
    elif model_type == "torch_mlp":
        input_size = kwargs.get("input_size", 28 * 28)
        hidden_size = kwargs.get("hidden_size", 128)
        num_classes = kwargs.get("num_classes", 36)
        return TorchMLP(input_size, hidden_size, num_classes)

    else:
        raise ValueError(f"Unsupported model type: {model_type}")
```
其余模型添加方式类似，只需先添加对应的函数，然后在```get_model```函数中添加对应的接口，即可从```main```文档中调用/
## 模型调参
```get_model```函数定义了不同模型的参数，可在此处进行调整。
## 更改数据集 
本项目数据处理部分于```data_processing_torch.py```文件中，其中
```python
    mnist_dataset = datasets.MNIST(root='./data', train=True, download=True, transform=data_transform)
    mnist_test_dataset = datasets.MNIST(root='./data', train=False, download=True, transform=data_transform)
```
定义了使用的数据集为MNIST手写数据集，MNIST数据集来自于torchvision库，若要更改数据集，只需更改数据集的调用方式即可。推荐使用torchvision库中的数据集，若使用其他数据集，需要更改数据处理部分的代码。

# 文件说明
## ./data/data_processing_torch.py
数据处理部分，包括数据集的下载、数据集的预处理、数据集的加载等。
该文件中定义```label_deal_flag```用于区分使用的数据集。```label_deal_flag=0```表示只使用数字作为数据集，```label_deal_flag=1```表示只使用大小写合并的字母作为数据集，```label_deal_flag=2```表示使用大小写合并字母和数字作为数据集，```label_deal_flag=3```表示使用全部分类作为数据集。
```python    
if label_deal_flag == 1:
        emnist_dataset = datasets.EMNIST(root='./data', split='letters', train=True, download=False,transform=data_transform)
        emnist_test_dataset = datasets.EMNIST(root='./data', split='letters', train=False, download=False,transform=data_transform)
        x = torch.cat([emnist_dataset.data, emnist_test_dataset.data], dim=0).float()
        y = torch.cat([emnist_dataset.targets, emnist_test_dataset.targets], dim=0)-1

    elif label_deal_flag == 0:
        emnist_dataset = datasets.EMNIST(root='./data', split='digits', train=True, download=False, transform=data_transform)
        emnist_test_dataset = datasets.EMNIST(root='./data', split='digits', train=False, download=False, transform=data_transform)
        x = torch.cat([emnist_dataset.data, emnist_test_dataset.data], dim=0).float()
        y = torch.cat([emnist_dataset.targets, emnist_test_dataset.targets], dim=0)

    elif label_deal_flag == 2 or label_deal_flag == 3:
        emnist_dataset = datasets.EMNIST(root='./data', split='byclass', train=True, download=False, transform=data_transform)
        emnist_test_dataset = datasets.EMNIST(root='./data', split='byclass', train=False, download=False, transform=data_transform)
        x = torch.cat([emnist_dataset.data, emnist_test_dataset.data], dim=0).float()
        y = torch.cat([emnist_dataset.targets, emnist_test_dataset.targets], dim=0)


    if label_deal_flag == 2:
        # Define a mapping to merge uppercase and lowercase labels
        label_mapping = {}
        for i in range(10):  # Numbers 0-9 remain unchanged
            label_mapping[i] = i
        for i in range(26):  # A-Z and a-z are mapped to the same labels
            label_mapping[10 + i] = 10 + i  # Map A-Z
            label_mapping[36 + i] = 10 + i  # Map a-z
        # Remap the labels
        y = torch.tensor([label_mapping[label.item()] for label in y], dtype=torch.long)
    print(x.shape)
```
## ./model_setup.py
见上文“添加模型”部分。
## main.py
主程序，包括模型的训练、评估、预测、保存与加载等功能。其中模型的保存路径```model_path = "./models/save"```，训练结果的保存路径```data_save_path = "./data/save"```。
该文件将保存训练结果。同时，该文件还会保存训练过程中的损失值和准确率的变化情况，以便后续的可视化。

## vision(混淆矩阵和t-sne).py
该模型将读取main文件生成的模型文件和数据文件，生成混淆矩阵和t-sne图，并输出正确分类样本和错误分类样本，其路径和main一致。在运行该项目前请保证```model_path```和```data_path```中模型一一对应，即模型文件必须要有对应的数据文件。
```python   
confusion_matrix_pdf = PdfPages("confusion_matrices.pdf")
tsne_pdf = PdfPages("tsne_feature_distributions.pdf")
incorrect_samples_pdf = PdfPages("incorrect_samples.pdf")
correct_samples_pdf = PdfPages("correct_samples.pdf")
```
## 损失函数和准确率绘制.py
该文件将读取main文件生成的模型文件，然后以测试模式生成准确率和损失函数曲线。
```python
    # 保存训练过程中的损失值和准确率
    loss_fig.savefig(os.path.join(data_save_path, f"loss_{model_name}_{label_flag}.png"))
    accuracy_fig.savefig(os.path.join(data_save_path, f"accuracy_{model_name}_{label_flag}.png"))
```python
    # 初始化模型
    model_name = "torch_mlp"  #此处更改模型名
    model = get_model(model_name, label_flag=label_flag)
    model_path = "./models/save"
    os.makedirs(model_path, exist_ok=True)
```
加载模型部分代码如下：
```python
    # 加载预训练的 PyTorch 模型
    model.load_state_dict(torch.load(os.path.join(model_path, f"{model_name}_epoch_{epoch_number}_{label_flag}.pt")))
    model.to(device)
    model.eval()
```
## 单个字符处理.py
该文件实现单个字符的识别，关键实现部分如下：
```python
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

            width, height = original_image.size
            font_size = max(width // 5, 10)  # 根据图片宽度动态调整字体大小，最小为10
            try:
                font = ImageFont.truetype("arial.ttf", font_size)  # 使用系统字体
            except IOError:
                font = ImageFont.load_default()  # 如果系统字体加载失败，使用默认字体


            # 在图片上写入识别结果
            draw = ImageDraw.Draw(original_image)
            predicted = decode_predictions(predicted)
            text = f"{predicted}"  # 识别结果
            draw.text((10, 10), text, fill=(0), font=font)  # 在左上角写入文字

            # 保存图片到输出文件夹
            output_path = os.path.join(output_folder, filename)
            original_image.save(output_path)
            print(f"Processed and saved: {output_path}")
```
需要注意的是该文件读取的路径和标签要匹配
```python
    folder_path = "./data/pic"  # 输入文件夹
    output_folder = "./data/annotated_images"  # 输出文件夹
    label_flag = 3  # 标志类别
    model_name = "en_torch_cnn"  # 使用的模型名称
    flag = 1 # flag 为1 时 黑字白底，flag为0时白字黑底
```
## 字符串识别.py
该文件实现字符串的识别，运行时需要输入图片路径，模型名称，标签类别，模型路径。
```python
    image_path = "./data/pic_cs/1.jpg"  # 替换为你的图片路径
    model_name = "en_torch_cnn"  # 替换为你的模型名称
    label_flag = 3  # 预测模式（根据你的模型设置）
    model_path = "./models/test/en_torch_cnn_3.pt"  # 替换为你的模型路径
    main(image_path, model_name, label_flag, model_path)
```
