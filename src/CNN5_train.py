#%%
import torch
import pytorch_lightning as pl
import torchmetrics
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, models
from PIL import Image
import os
import numpy as np
from tqdm import tqdm
import cv2
import torch.optim as optim
import torch.nn as nn
import torch.nn.functional as F
from pytorch_lightning.loggers import TensorBoardLogger
import argparse

#%%
def parse_args():
    parser = argparse.ArgumentParser(description="Train YOLO model with frozen layers")
    parser.add_argument('-p', '--path', type=str, required=True, help="Path to the base directory")
    return parser.parse_args()

class CustomDataset(Dataset):
    def __init__(self, image_folders, label_folders, transform=None):
        self.image_folders = image_folders
        self.label_folders = label_folders
        self.transform = transform
        self.image_paths = []
        self.label_paths = []

        for image_folder, label_folder in zip(image_folders, label_folders):
            image_names = [f for f in os.listdir(image_folder) if f.endswith('.jpg')]
            for image_name in image_names:
                self.image_paths.append(os.path.join(image_folder, image_name))
                self.label_paths.append(os.path.join(label_folder, os.path.splitext(image_name)[0] + '.txt'))

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        image_path = self.image_paths[idx]
        label_path = self.label_paths[idx]

        image = Image.open(image_path)
        with open(label_path, 'r') as file:
            label = int(file.readline().strip())

        if self.transform:
            image = self.transform(image)

        return image, label


#%%
class SimpleCNN(pl.LightningModule):
    def __init__(self, num_classes=4, learning_rate=0.0001):
        super(SimpleCNN, self).__init__()
        self.save_hyperparameters()
        self.train_acc = torchmetrics.classification.Accuracy(task="multiclass", num_classes=num_classes)
        self.valid_acc = torchmetrics.classification.Accuracy(task="multiclass", num_classes=num_classes)
        self.criterion = nn.CrossEntropyLoss()
        self.lr = learning_rate
        self.train_losses = []
        self.conv1 = nn.Conv2d(3, 32, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(32)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(64)
        self.conv3 = nn.Conv2d(64, 128, kernel_size=3, padding=1)
        self.bn3 = nn.BatchNorm2d(128)
        self.conv4 = nn.Conv2d(128, 256, kernel_size=3, padding=1)
        self.bn4 = nn.BatchNorm2d(256)
        self.conv5 = nn.Conv2d(256, 512, kernel_size=3, padding=1)
        self.bn5 = nn.BatchNorm2d(512)
        self.fc1 = nn.Linear(512 * 8 * 8, 1024)
        self.dropout = nn.Dropout(0.5)
        self.fc2 = nn.Linear(1024, 4)

    def forward(self, x):
        x = F.relu(self.bn1(self.conv1(x)))
        x = F.max_pool2d(x, 2)
        x = F.relu(self.bn2(self.conv2(x)))
        x = F.max_pool2d(x, 2)
        x = F.relu(self.bn3(self.conv3(x)))
        x = F.max_pool2d(x, 2)
        x = F.relu(self.bn4(self.conv4(x)))
        x = F.max_pool2d(x, 2)
        x = F.relu(self.bn5(self.conv5(x)))
        x = F.max_pool2d(x, 2)
        x = x.view(x.size(0), -1)
        x = F.relu(self.fc1(x))
        x = self.dropout(x)
        x = self.fc2(x)
        return x
    
    def training_step(self, batch, batch_idx):
        images, labels = batch
        outputs = self.forward(images)
        loss = self.criterion(outputs, labels)
        batch_value = self.train_acc(outputs, labels)
        self.log('train_acc_step', batch_value, on_step=True, on_epoch=False, prog_bar=True)
        self.log('train_loss', loss, on_step=True, on_epoch=False, prog_bar=True)
        self.train_losses.append(loss)
        return loss

    def on_train_epoch_end(self):
        self.train_acc.reset()

    def validation_step(self, batch, batch_idx):
        images, labels = batch
        outputs = self.forward(images)
        loss = self.criterion(outputs, labels)
        self.valid_acc.update(outputs, labels)
        self.log('val_loss', loss, on_epoch=True, on_step=False, prog_bar=True)
        return loss

    def on_validation_epoch_end(self):
        self.log('valid_acc_epoch', self.valid_acc.compute(), on_step=False, on_epoch=True, prog_bar=True)
        self.valid_acc.reset()

    def configure_optimizers(self):
        return optim.Adam(self.parameters(), lr=self.lr)

#%%
def main(): 
    args = parse_args()
    
    base_path = args.path

    transform = transforms.Compose([
        transforms.Resize((256, 256)),
        transforms.RandomHorizontalFlip(),  
        transforms.ToTensor(),
    ])

    transform_t = transforms.Compose([
        transforms.Resize((256, 256)),
        transforms.ToTensor(),
    ])
    fall_train_path = os.path.join(base_path, "data/classifier_train_data/fall/train_images")
    fall_val_path = os.path.join(base_path, "data/classifier_train_data/fall/val_images")
    origin_train_path = os.path.join(base_path, "data/classifier_train_data/origin/train_images")
    origin_val_path = os.path.join(base_path, "data/classifier_train_data/origin/val_images")
    fall_train_label_path = os.path.join(base_path, "data/classifier_train_data/fall/train_labels")
    fall_val_label_path = os.path.join(base_path, "data/classifier_train_data/fall/val_labels")
    origin_train_label_path = os.path.join(base_path, "data/classifier_train_data/origin/train_labels")
    origin_val_label_path = os.path.join(base_path, "data/classifier_train_data/origin/val_labels")
    train_image_folders = [fall_train_path, origin_train_path]
    train_label_folders = [fall_train_label_path, origin_train_label_path]
    val_image_folders = [fall_val_path, origin_val_path]
    val_label_folders = [fall_val_label_path, origin_val_label_path]


    train_dataset = CustomDataset(train_image_folders, train_label_folders, transform=transform)
    val_dataset = CustomDataset(val_image_folders, val_label_folders, transform=transform_t)
    train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=64, shuffle=False)

    trainer = pl.Trainer(
        max_epochs=20, 
        accelerator='gpu' if torch.cuda.is_available() else 'cpu', 
        devices=1
    )
    model = SimpleCNN()

    trainer.fit(model, train_loader, val_loader)

    model_path = os.path.join(base_path, "results/CNN5/weights/best_model.ckpt")
    
    trainer.save_checkpoint(model_path)

if __name__ == "__main__":
    main()
