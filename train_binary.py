import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import transforms
import torchvision.models as models
from torchsummary import summary

import numpy as np
import matplotlib.pyplot as plt
import os
import gc
import json
import random
import shutil
import tempfile
import PIL
import sklearn
from sklearn.metrics import classification_report

from monai.apps import download_and_extract
from monai.config import print_config
from monai.data import decollate_batch, DataLoader, MetaTensor
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

print_config()

train_transforms = Compose(
    [
        LoadImage(image_only=True),
        EnsureChannelFirst(channel_dim="no_channel"),
        ScaleIntensity(0, 1),
        RandFlip(spatial_axis=0, prob=0.5),
        EnsureType(),
    ]
)

test_transforms = val_transforms = Compose(
    [
        LoadImage(image_only=True),
        EnsureChannelFirst(channel_dim="no_channel"),
        ScaleIntensity(0, 1),
        EnsureType(),
    ]
)

class MedNISTDataset(torch.utils.data.Dataset):
    def __init__(self, data, transforms):
        self.image_files = [item[0] for item in data]  # Extract image file paths
        self.labels = [item[1] for item in data]  # Extract labels
        self.transforms = transforms

    def __len__(self):
        return len(self.image_files)

    def __getitem__(self, index):
        image = self.transforms(self.image_files[index])
        if isinstance(image, MetaTensor):
            image = image.as_tensor()
        return image, self.labels[index]

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

init_list = [init_gaussian, init_uniform, init_he, init_xavier]


def convert_to_serializable(obj):
    if isinstance(obj, np.float32):
        return float(obj)
    if isinstance(obj, np.integer):
        return int(obj)
    return obj


def main(init_list, ratios, val_loader, test_loader, num_class):
    for init in init_list:
        for ratio in ratios:
            for leading_class in ["HeadCT", "ChectCT"]:
                epoch_loss_values = []
                gradients_values = []
                weights_values = []

                evaluation_metrics = {
                    "accuracy": [],
                    "precision": [],
                    "recall": [],
                    "f1": []
                }

                if leading_class == "ChestCT":
                    head_length = 1000
                    chest_length = head_length * ratio
                else:
                    chest_length = 1000
                    head_length = chest_length * ratio

                training_images = np.concatenate([hand_files[:head_length], chest_files[:chest_length]])
                training_labels = [0] * head_length + [1] * chest_length

                training_data = list(zip(training_images, training_labels))
                train_dataset = MedNISTDataset(training_data, train_transforms)
                train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=4, pin_memory=torch.cuda.is_available())

                if num_class > 2:
                    model = models.resnet18(pretrained=False)
                    model.conv1 = nn.Conv2d(1, model.conv1.out_channels, kernel_size=(7, 7), stride=(2, 2), padding=(3, 3), bias=False)
                    model.fc = nn.Linear(model.fc.in_features, num_class)
                else:
                    model = SimpleCNN(num_classes=2)

                model = init(model)
#                 model = torch.compile(model)
                model.to(device)
                summary(model, (1, 64, 64))
                loss_function = torch.nn.CrossEntropyLoss()
                optimizer = torch.optim.SGD(model.parameters(), lr=0.001, momentum=0)

#                 for epoch in range(max_epochs):
#                     print("-" * 10)
#                     print(f"epoch {epoch + 1}/{max_epochs}")
#                     model.train()
#                     epoch_loss = 0
#                     step = 0
#
#                     for batch_idx, (data, target) in enumerate(train_loader):
#                         data = data.to(device)
#                         target = target.to(device)
#
#                         optimizer.zero_grad()
#                         output = model(data)
#                         loss = loss_function(output, target)
#                         loss.backward()
#
#                         gradients = []
#                         for param in model.parameters():
#                             if param.grad is not None:
#                                 gradients.append(param.grad.view(-1))
#                         gradients_values.append(torch.cat(gradients).cpu().detach().numpy())
#
#                         weights = []
#                         for param in model.parameters():
#                             weights.append(param.view(-1))
#                         weights_values.append(torch.cat(weights).cpu().detach().numpy())
#
#                         optimizer.step()
#                         epoch_loss += loss.item()
#                         step += 1
#
#                     epoch_loss /= step
#                     epoch_loss_values.append(epoch_loss)
#
#                     model.eval()
#                     y_true, y_pred = [], []
#                     with torch.no_grad():
#                         for val_data, val_target in val_loader:
#                             val_data = val_data.to(device)
#                             val_target = val_target.to(device)
#                             val_output = F.softmax(model(val_data))
#                             val_pred = (val_output > 0.5).float()
#                             y_true.extend(val_target.cpu().numpy())
#                             y_pred.extend(val_pred.cpu().numpy())
#
#                     y_pred = [np.argmax(i) for i in y_pred]
#                     accuracy = sklearn.metrics.accuracy_score(y_true, y_pred)
#                     precision = sklearn.metrics.precision_score(y_true, y_pred)
#                     recall = sklearn.metrics.recall_score(y_true, y_pred)
#                     f1 = sklearn.metrics.f1_score(y_true, y_pred)
#
#                     evaluation_metrics["accuracy"].append(accuracy)
#                     evaluation_metrics["precision"].append(precision)
#                     evaluation_metrics["recall"].append(recall)
#                     evaluation_metrics["f1"].append(f1)
#
#                     print(f"Validation Accuracy: {accuracy:.4f}, Precision: {precision:.4f}, Recall: {recall:.4f}, F1: {f1:.4f}")
#
#                 model.eval()
#                 y_true, y_pred = [], []
#                 with torch.no_grad():
#                     for test_data, test_target in test_loader:
#                         test_data = test_data.to(device)
#                         test_target = test_target.to(device)
#                         test_output = F.softmax(model(test_data))
#                         test_pred = (test_output > 0.5).float()
#                         y_true.extend(test_target.cpu().numpy())
#                         y_pred.extend(test_pred.cpu().numpy())
#
#                 y_pred = [np.argmax(i) for i in y_pred]
#                 test_accuracy = sklearn.metrics.accuracy_score(y_true, y_pred)
#                 test_precision = sklearn.metrics.precision_score(y_true, y_pred)
#                 test_recall = sklearn.metrics.recall_score(y_true, y_pred)
#                 test_f1 = sklearn.metrics.f1_score(y_true, y_pred)
#
#                 gradients_values = np.array(gradients_values)
#                 weights_values = np.array(weights_values)
#                 gradients_expectation = gradients_values.mean()
#                 gradients_variance = gradients_values.var()
#                 weights_expectation = weights_values.mean()
#                 weights_variance = weights_values.var()
#
#                 print("Test Results:")
#                 print(f"Accuracy: {test_accuracy:.4f}, Precision: {test_precision:.4f}, Recall: {test_recall:.4f}, F1: {test_f1:.4f}")
#                 print(f"Gradients Expectation: {gradients_expectation:.4e}\n")
#                 print(f"Gradients Variance: {gradients_variance:.4e}\n")
#                 print(f"Weights Expectation: {weights_expectation:.4e}\n")
#                 print(f"Weights Variance: {weights_variance:.4e}\n")
#
#                 report_dict = classification_report(y_true, y_pred, target_names=["HeadCT", "ChestCT"], output_dict=True)
#                 report_dict["gradients_expectation"] = gradients_expectation
#                 report_dict["gradients_variance"] = gradients_variance
#                 report_dict["weights_expectation"] = weights_expectation
#                 report_dict["weights_variance"] = weights_variance
#
#                 experiment_name = f"{init.__name__}_{ratio}_{leading_class}_260125"
#                 save_path = project_root
#                 #os.makedirs(save_path, exist_ok=True)
#
#                 with open(save_path + f"/{experiment_name}_classification_report.json", "w") as f:
#                     print("saved to: ", save_path + f"/{experiment_name}_classification_report.json")
#                     json.dump(report_dict, f, indent=4, default=convert_to_serializable)
#
#                 torch.cuda.empty_cache()
#                 gc.collect()


if __name__ == "__main__":
    directory = "/data/sergei.pnev/MONAI"
    os.environ["MONAI_DATA_DIRECTORY"] = directory
    if directory is not None:
        os.makedirs(directory, exist_ok=True)
    root_dir = tempfile.mkdtemp() if directory is None else directory
    print(root_dir)

    resource = "https://github.com/Project-MONAI/MONAI-extra-test-data/releases/download/0.8.1/MedNIST.tar.gz"
    md5 = "0bc7306e7427e00ad1c5526a6677552d"

    compressed_file = os.path.join(root_dir, "MedNIST.tar.gz")
    data_dir = os.path.join(root_dir, "MedNIST")
    if not os.path.exists(data_dir):
        download_and_extract(resource, compressed_file, root_dir, md5)

    class_names = sorted(x for x in os.listdir(data_dir) if os.path.isdir(os.path.join(data_dir, x)))

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

    print(f"Total image count: {num_total}")
    print(f"Image dimensions: {image_width} x {image_height}")
    print(f"Label names: {class_names}")
    print(f"Label counts: {num_each}")

#     hand = os.path.join(data_dir, 'HeadCT')
    hand = os.path.join(data_dir, 'AbdomenCT')
    chest = os.path.join(data_dir, 'ChestCT')

    hand_files = np.array([os.path.join(hand, x) for x in os.listdir(hand)])
    chest_files = np.array([os.path.join(chest, x) for x in os.listdir(chest)])

    ratio_list = []
    ratio = 1
    head_length = 1000
    chest_length = head_length * ratio

    #indices = np.arange(8000, 10000)
    #np.random.shuffle(indices)
    indices = np.load(directory + "/indices.npy")
    val_indices = indices[:1000]
    test_indices = indices[1000:]

    training_images = np.concatenate([hand_files[:head_length], chest_files[:chest_length]])
    val_images = np.concatenate([hand_files[val_indices], chest_files[val_indices]])
    test_images = np.concatenate([hand_files[test_indices], chest_files[test_indices]])

    training_labels = [0] * head_length + [1] * chest_length
    val_labels = [0] * 1000 + [1] * 1000

    training_data = list(zip(training_images, training_labels))
    val_data = list(zip(val_images, val_labels))
    test_data = list(zip(test_images, val_labels))

    val_dataset = MedNISTDataset(val_data, val_transforms)
    test_dataset = MedNISTDataset(test_data, test_transforms)

    project_root = "drive/MyDrive/deep_learning_project"
    os.makedirs(project_root, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    max_epochs = 10
    batch_size = 300

    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=4, pin_memory=torch.cuda.is_available())
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=4, pin_memory=torch.cuda.is_available())
    ratios = [1, 3, 8]

    main(init_list, ratios, val_loader, test_loader, num_class=2)