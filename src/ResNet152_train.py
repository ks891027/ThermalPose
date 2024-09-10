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

#%%
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

transform = transforms.Compose([
    transforms.Resize((256, 256)),
    transforms.RandomHorizontalFlip(),  
    transforms.ToTensor(),
])

transform_t = transforms.Compose([
    transforms.Resize((256, 256)),
    transforms.ToTensor(),
])

train_image_folders = ['/home/yang/Documents/Thermal_Pose/ThermalPose/data/classifier_train_data/fall/train_images', '/home/yang/Documents/Thermal_Pose/ThermalPose/data/classifier_train_data/origin/train_images']
train_label_folders = ['/home/yang/Documents/Thermal_Pose/ThermalPose/data/classifier_train_data/fall/train_labels', '/home/yang/Documents/Thermal_Pose/ThermalPose/data/classifier_train_data/origin/train_labels']
val_image_folders = ['/home/yang/Documents/Thermal_Pose/ThermalPose/data/classifier_train_data/fall/val_images', '/home/yang/Documents/Thermal_Pose/ThermalPose/data/classifier_train_data/origin/val_images']
val_label_folders = ['/home/yang/Documents/Thermal_Pose/ThermalPose/data/classifier_train_data/fall/val_labels', '/home/yang/Documents/Thermal_Pose/ThermalPose/data/classifier_train_data/origin/val_labels']


train_dataset = CustomDataset(train_image_folders, train_label_folders, transform=transform)
val_dataset = CustomDataset(val_image_folders, val_label_folders, transform=transform_t)
train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=64, shuffle=False)

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
        self.model = models.resnet152(pretrained=True)
        self.model.fc = nn.Linear(self.model.fc.in_features, num_classes)

    def forward(self, x):
        return self.model(x)
    
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
logger = TensorBoardLogger("tb_logs", name="resnet152_model")
trainer = pl.Trainer(
    max_epochs=20, 
    accelerator='gpu' if torch.cuda.is_available() else 'cpu', 
    devices=1,
    logger=logger
)
model = SimpleCNN()

# 訓練模型
trainer.fit(model, train_loader, val_loader)


# 保存最好的模型
trainer.save_checkpoint("/home/yang/Documents/Thermal_Pose/ThermalPose/results/ResNet152/best_model.ckpt")

# %%
# tensorboard --logdir=tb_logs