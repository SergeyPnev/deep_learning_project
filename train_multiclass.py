import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.models import resnet18

import warnings
import numpy as np
import matplotlib
matplotlib.use('TkAgg')  # Use the TkAgg backend
import matplotlib.pyplot as plt
import os
import tempfile
import PIL
import sklearn
from sklearn.metrics import classification_report
import random
import json
from omegaconf import OmegaConf

from monai.apps import download_and_extract
from monai.config import print_config
from monai.data import decollate_batch, DataLoader
from monai.metrics import ROCAUCMetric
from monai.networks.nets import DenseNet121
from monai.transforms import (
    Activations,
    EnsureChannelFirst,
    AsDiscrete,
    Compose,
    LoadImage,
    RandFlip,
    RandRotate,
    RandZoom,
    ScaleIntensity,
    ToTensor,
    EnsureType,
)
from monai.utils import set_determinism

class SimpleCNN(nn.Module):
    def __init__(self, num_classes):
        super(SimpleCNN, self).__init__()
        self.conv1 = nn.Conv2d(1, 32, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(32)
        self.relu1 = nn.ReLU()
        self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2)

        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(64)
        self.relu2 = nn.ReLU()
        self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2)

        self.fc1 = nn.Linear(64 * 16 * 16, 256)
        self.bn3 = nn.BatchNorm1d(256)
        self.relu3 = nn.ReLU()

        self.fc2 = nn.Linear(256, 128)
        self.bn4 = nn.BatchNorm1d(128)
        self.relu4 = nn.ReLU()

        self.fc3 = nn.Linear(128, num_classes)

    def forward(self, x):
        x = self.pool1(self.relu1(self.bn1(self.conv1(x))))
        x = self.pool2(self.relu2(self.bn2(self.conv2(x))))
        x = x.view(-1, 64 * 16 * 16)
        x = self.relu3(self.bn3(self.fc1(x)))
        x = self.relu4(self.bn4(self.fc2(x)))
        x = self.fc3(x)
        return x

def init_gaussian(model):
    for layer in model.modules():
        if isinstance(layer, nn.Linear):
            nn.init.normal_(layer.weight, mean=0.0, std=0.02)
            if layer.bias is not None:
                nn.init.constant_(layer.bias, 0.0)
    return model

def init_uniform(model):
    for layer in model.modules():
        if isinstance(layer, nn.Linear):
            nn.init.uniform_(layer.weight, a=-0.1, b=0.1)
            if layer.bias is not None:
                nn.init.constant_(layer.bias, 0.0)
    return model

def init_he(model):
    for layer in model.modules():
        if isinstance(layer, nn.Linear):
            nn.init.kaiming_normal_(layer.weight, mode='fan_in', nonlinearity='relu')
            if layer.bias is not None:
                nn.init.constant_(layer.bias, 0.0)
    return model

def init_xavier(model):
    for layer in model.modules():
        if isinstance(layer, nn.Linear):
            nn.init.xavier_normal_(layer.weight)
            if layer.bias is not None:
                nn.init.constant_(layer.bias, 0.0)
    return model

class MedNISTDataset(torch.utils.data.Dataset):
    def __init__(self, image_files, labels, transforms):
        self.image_files = image_files
        self.labels = labels
        self.transforms = transforms

    def __len__(self):
        return len(self.image_files)

    def __getitem__(self, index):
        return self.transforms(self.image_files[index]), self.labels[index]


def create_imbalanced_dataset(balanced_x, balanced_y, imbalance_ratios):
        datasets = []
        for ratio in imbalance_ratios:
            imbalanced_x = []
            imbalanced_y = []

            for i in range(num_class):
                class_images = [balanced_x[j] for j in range(len(balanced_x)) if balanced_y[j] == i]
                num_images_to_select = int(num_images_per_class * ratio[i])  # Select based on the imbalance ratio
                imbalanced_x.extend(random.sample(class_images, num_images_to_select))
                imbalanced_y.extend([i] * num_images_to_select)

            datasets.append((imbalanced_x, imbalanced_y))

        return datasets

train_transforms = Compose([
            LoadImage(image_only=True),
            EnsureChannelFirst(channel_dim="no_channel"),
            ScaleIntensity(0, 1),
            # RandRotate(range_x=np.pi / 12, prob=0.5, keep_size=True),
            RandFlip(spatial_axis=0, prob=0.5),
            EnsureType(),
        ])
test_transforms = val_transforms = Compose([
            LoadImage(image_only=True),
            EnsureChannelFirst(channel_dim="no_channel"),
            ScaleIntensity(0, 1),
            EnsureType(),
        ])

def convert_to_serializable(obj):
    if isinstance(obj, np.float32):
        return float(obj)
    if isinstance(obj, np.integer):
        return int(obj)
    return obj

def model_prep(name, init):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(device)

    if name == 'simple':
        simple_model = SimpleCNN(num_classes=6)
        model = simple_model.to(device)

    elif name == 'densnet':
        densnet = DenseNet121(
            spatial_dims=2,
            in_channels=1,
            out_channels=6,
        )
        model = densnet.to(device
                          )
    elif name == 'resnet':
        resnet = resnet18(weights=None)
        resnet.fc = nn.Linear(resnet.fc.in_features, 6)
        resnet.conv1 = nn.Conv2d(1, 64, kernel_size=(7, 7), stride=(2, 2), padding=(3, 3), bias=False)
        model = resnet.to(device)

    else:
        warnings.warn("Model's name must be simple, denset or resnet", UserWarning)

    if init == 'he':
        model = init_he(model)
    elif init == 'xavier':
        model = init_xavier(model)
    elif init == 'uniform':
        model = init_uniform(model)
    elif init == 'gaussian':
        model = init_gaussian(model)
    else:
        warnings.warn("Init must be he, xavier, uniform or gaussian.", UserWarning)

    return model

def get_optim(name, lr=0.001):
    if name == 'SGD':
        return torch.optim.SGD(model.parameters(), lr=lr, momentum=0)  # Momentum to 0 to get a very simple SGD
    elif name == 'momentum':
        return torch.optim.SGD(model.parameters(), lr=lr, momentum=0.9)
    elif name == 'ADAM':
        return torch.optim.Adam(model.parameters(), lr=lr)
    elif name == 'Adagrad':
        return torch.optim.Adagrad(model.parameters(), lr=lr)
    else:
        warnings.warn("Optimizer must be SGD, momentum ADAM or Adagrad", UserWarning)


if __name__ == "__main__":

    config = OmegaConf.load('config.yml')
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # print(device)
    unbalance = config.unbalance
    model_name = config.model
    init = config.init
    optim_name = config.optim
    name = model_name + '_' + optim_name + '_' + init + '_' + str(unbalance)

    '---- MONAI stuff and image download ----'
    directory = os.environ.get("MONAI_DATA_DIRECTORY")
    if directory is not None:
        os.makedirs(directory, exist_ok=True)
    root_dir = tempfile.mkdtemp() if directory is None else directory

    resource = "https://github.com/Project-MONAI/MONAI-extra-test-data/releases/download/0.8.1/MedNIST.tar.gz"
    md5 = "0bc7306e7427e00ad1c5526a6677552d"

    compressed_file = os.path.join(root_dir, "MedNIST.tar.gz")
    data_dir = os.path.join(root_dir, "MedNIST")
    if not os.path.exists(data_dir):
        download_and_extract(resource, compressed_file, root_dir, md5)
	
    
    class_names = sorted(x for x in os.listdir(data_dir) if os.path.isdir(os.path.join(data_dir, x)))
    print(class_names)
    num_class = len(class_names)
    image_files = [
        [os.path.join(data_dir, class_names[i], x) for x in os.listdir(os.path.join(data_dir, class_names[i]))]
        for i in range(num_class)
    ]

    num_each = [len(image_files[i]) for i in range(num_class)]
    image_files_list = []
    image_class = []
    for i in range(num_class):
        image_files_list.extend(image_files[i])
        image_class.extend([i] * num_each[i])

    num_total = len(image_class)
    image_width, image_height = PIL.Image.open(image_files_list[0]).size

    # print(f"Total image count: {num_total}")
    # print(f"Image dimensions: {image_width} x {image_height}")
    # print(f"Label names: {class_names}")
    # print(f"Label counts: {num_each}")

    val_transforms = Compose([LoadImage(image_only=True), EnsureChannelFirst(), ScaleIntensity()])

    y_pred_trans = Compose([Activations(softmax=True)])
    y_trans = Compose([AsDiscrete(to_onehot=num_class)])

    model = model_prep(model_name, init)
    loss_function = torch.nn.CrossEntropyLoss()
    optimizer = get_optim(optim_name, lr=0.001)

    val_frac = 0.1
    test_frac = 0.1
    length = len(image_files_list)
    indices = np.arange(length)
    np.random.shuffle(indices)

    test_split = int(test_frac * length)
    val_split = int(val_frac * length) + test_split
    test_indices = indices[:test_split]
    val_indices = indices[test_split:val_split]
    train_indices = indices[val_split:]

    train_x = [image_files_list[i] for i in train_indices]
    train_y = [image_class[i] for i in train_indices]
    val_x = [image_files_list[i] for i in val_indices]
    val_y = [image_class[i] for i in val_indices]
    test_x = [image_files_list[i] for i in test_indices]
    test_y = [image_class[i] for i in test_indices]

    print(f"Training count: {len(train_x)}, Validation count: " f"{len(val_x)}, Test count: {len(test_x)}")

    val_ds = MedNISTDataset(val_x, val_y, val_transforms)
    val_loader = DataLoader(val_ds, batch_size=300, num_workers=2)

    test_ds = MedNISTDataset(test_x, test_y, val_transforms)
    test_loader = DataLoader(test_ds, batch_size=300, num_workers=2)

    # Step 1: Create a balanced dataset from train_x and train_y
    num_images_per_class = min(np.bincount(train_y))  # Use the minimum number of images in any class
    balanced_x = []
    balanced_y = []

    for i in range(num_class):
        class_images = [train_x[j] for j in range(len(train_x)) if train_y[j] == i]
        balanced_x.extend(class_images[:num_images_per_class])  # Select the first num_images_per_class images
        balanced_y.extend([i] * num_images_per_class)

    # Example imbalance ratios for 6 classes
    imbalance_ratios = [
        [0.2, 0.2, 0.2, 0.1, 0.2, 0.1],  # Moderate Imbalance
        [0.6, 0.2, 0.05, 0.05, 0.05, 0.05],  # Extreme Imbalance
        [1/6, 1/6, 1/6, 1/6, 1/6, 1/6]  # Balanced
    ]

    # Create imbalanced datasets
    imbalanced_datasets = create_imbalanced_dataset(balanced_x, balanced_y, imbalance_ratios)

    # Step 3: Create DataLoaders for each imbalanced dataset
    data_loaders = []
    batch_size = 300

    for i, (x, y) in enumerate(imbalanced_datasets):
        # Create dataset and DataLoader
        dataset = MedNISTDataset(x, y, train_transforms)  # Use appropriate transforms
        data_loader = DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=2)
        data_loaders.append(data_loader)

        # Print class distribution for each imbalanced dataset
        print(f"Imbalanced Dataset {i + 1}:")
        print(f"Total images: {len(x)}")
        print(f"Class distribution: {np.bincount(y)}")  # Print class distribution

    train_loader = data_loaders[unbalance]

    epoch_loss_values = []
    metric_values = []
    epoch_loss_values = []
    gradients_values = []
    weights_values = []

    evaluation_metrics = {
        "accuracy": [],
        "precision": [],
        "recall": [],
        "f1": []
    }
    max_epochs = config.epochs
    val_interval = 1
    auc_metric = ROCAUCMetric()

    for epoch in range(max_epochs):
        print("-" * 10)
        print(f"epoch {epoch + 1}/{max_epochs}")
        model.train()
        epoch_loss = 0
        step = 0
        for batch_data in train_loader:
            step += 1
            inputs, labels = batch_data[0].to(device), batch_data[1].to(device)
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = loss_function(outputs, labels)
            loss.backward()

            gradients = []
            for param in model.parameters():
                if param.grad is not None:
                    gradients.append(param.grad.view(-1))
            gradients_values.append(torch.cat(gradients).cpu().detach().numpy())

            weights = []
            for param in model.parameters():
                weights.append(param.view(-1))
            weights_values.append(torch.cat(weights).cpu().detach().numpy())

            optimizer.step()
            epoch_loss += loss.item()
            # print(f"{step}/{len(x) // train_loader.batch_size}, " f"train_loss: {loss.item():.4f}")
            epoch_len = len(x) // train_loader.batch_size
        epoch_loss /= step
        epoch_loss_values.append(epoch_loss)
        print(f"epoch {epoch + 1} average loss: {epoch_loss:.4f}")

    if epoch % val_interval == 0:
        model.eval()
        with torch.no_grad():
            y_pred = []
            y = []

            # Initialize counters for accuracy by class
            correct_predictions = torch.zeros(num_class, dtype=torch.int64, device=device)
            total_samples = torch.zeros(num_class, dtype=torch.int64, device=device)

            for val_data in val_loader:
                val_images, val_labels = (
                    val_data[0].to(device),
                    val_data[1].to(device),
                )
                # Get model predictions
                outputs = model(val_images)  # Shape should be (batch_size, num_classes)
                y_pred.append(outputs)  # Store outputs for later concatenation
                y.append(val_labels)  # Store labels for later concatenation

                # Update total samples for each class
                for label in range(num_class):
                    total_samples[label] += (val_labels == label).sum().item()

                # Calculate correct predictions
                acc_value = torch.eq(outputs.argmax(dim=1), val_labels)
                for label in range(num_class):
                    correct_predictions[label] += acc_value[val_labels == label].sum().item()

            # Concatenate predictions and labels after the loop
            y_pred = torch.cat(y_pred, dim=0)  # Shape should be (total_samples, num_classes)
            y = torch.cat(y, dim=0)  # Shape should be (total_samples,)

            y_onehot = [y_trans(i) for i in decollate_batch(y, detach=False)]
            y_pred_act = [y_pred_trans(i) for i in decollate_batch(y_pred)]
            auc_metric(y_pred_act, y_onehot)
            result = auc_metric.aggregate()
            auc_metric.reset()
            del y_pred_act, y_onehot

            # Calculate overall accuracy
            overall_acc = correct_predictions.sum().item() / len(y)

            # Calculate accuracy by class
            class_accuracies = correct_predictions.float() / total_samples.float()

            # Print results
            print(f"current epoch: {epoch + 1} current AUC: {result:.4f}")
            print(f"current validation accuracy: {overall_acc:.4f}")
            for i in range(num_class):
                if total_samples[i] > 0:  # Avoid division by zero
                    print(f"Accuracy for class {i}: {class_accuracies[i].item():.4f}")
                else:
                    print(f"Accuracy for class {i}: N/A (no samples)")

    plt.figure("train")
    plt.title("Epoch Average Loss")
    x = [i + 1 for i in range(len(epoch_loss_values))]
    y = epoch_loss_values
    plt.xlabel("epoch")
    plt.plot(x, y)
    plt.savefig('results/'+name + '.png')
    # plt.show()

    model.eval()
    y_true, y_pred = [], []
    with torch.no_grad():
        for test_data, test_target in test_loader:
            test_data = test_data.to(device)
            test_target = test_target.to(device)
            test_output = F.softmax(model(test_data), dim=1)
            test_pred = (test_output > 0.5).float()
            y_true.extend(test_target.cpu().numpy())
            y_pred.extend(test_pred.cpu().numpy())

    y_pred = [np.argmax(i) for i in y_pred]
    test_accuracy = sklearn.metrics.accuracy_score(y_true, y_pred)
    test_precision = sklearn.metrics.precision_score(y_true, y_pred, average='weighted', zero_division=0)
    test_recall = sklearn.metrics.recall_score(y_true, y_pred, average='weighted', zero_division=0)
    test_f1 = sklearn.metrics.f1_score(y_true, y_pred, average='weighted', zero_division=0)

    gradients_values = np.array(gradients_values)
    weights_values = np.array(weights_values)
    gradients_expectation = gradients_values.mean()
    gradients_variance = gradients_values.var()
    weights_expectation = weights_values.mean()
    weights_variance = weights_values.var()

    print("Test Results:")
    print(
        f"Accuracy: {test_accuracy:.4f}, Precision: {test_precision:.4f}, Recall: {test_recall:.4f}, F1: {test_f1:.4f}")
    print(f"Gradients Expectation: {gradients_expectation:.4e}\n")
    print(f"Gradients Variance: {gradients_variance:.4e}\n")
    print(f"Weights Expectation: {weights_expectation:.4e}\n")
    print(f"Weights Variance: {weights_variance:.4e}\n")

    report_dict = classification_report(y_true, y_pred, target_names=['AbdomenCT', 'BreastMRI', 'CXR', 'ChestCT', 'Hand', 'HeadCT'], output_dict=True)
    report_dict["gradients_expectation"] = gradients_expectation
    report_dict["gradients_variance"] = gradients_variance
    report_dict["weights_expectation"] = weights_expectation
    report_dict["weights_variance"] = weights_variance
    report_dict["unbalanced"] = imbalance_ratios[unbalance]
    report_dict["optimizer"] = optim_name
    report_dict["model"] = model_name
    report_dict["init"] = init

    experiment_name = name
    save_path = 'results/'
    # os.makedirs(save_path, exist_ok=True)

    with open(save_path + f"/{experiment_name}_classification_report.json", "w") as f:
        print("saved to: ", save_path + f"/{experiment_name}_classification_report.json")
        json.dump(report_dict, f, indent=4, default=convert_to_serializable)
