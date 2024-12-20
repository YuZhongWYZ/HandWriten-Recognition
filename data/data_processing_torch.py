#%%
import numpy as np
import torch
from torchvision import datasets, transforms
from sklearn.model_selection import KFold

def load_and_preprocess_data(k,label_deal_flag):
    # Step 1: Load the EMNIST ByClass dataset using PyTorch
    data_transform = transforms.Compose([
        transforms.ToTensor()
    ])

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

    # Step 2: Normalize the data
    # Min-Max normalization
    x_min_max = x / 255.0


    # Mean normalization
#    mean = torch.mean(x, dim=(1, 2), keepdim=True)
#    std = torch.std(x, dim=(1, 2), keepdim=True) + 1e-8
#    x_mean = (x - mean) / std
    x_mean = 0
    # Step 3: Prepare K-fold splitting
    kf = KFold(n_splits=k, shuffle=True, random_state=42)

    # Store K-fold split indices for later use
    kf_splits = []
    for train_idx, val_idx in kf.split(x):
        kf_splits.append((train_idx.tolist(), val_idx.tolist()))



    return x_min_max, x_mean, y, kf_splits

if __name__ == "__main__":
    x_min_max, x_mean, y, kf_splits = load_and_preprocess_data(k=10,label_deal_flag=0)

    # Save processed data and splits to be reused in the main script
    torch.save({
        'x_min_max': x_min_max,
        'x_mean': x_mean,
        'y': y,
        'kf_splits': kf_splits
    }, "processed_data.pth")

    # Display summary
    print("Data normalization completed.")
    print(f"Min-Max normalized shape: {x_min_max.shape}")
    print(f"Number of splits: {len(kf_splits)}")
