#%%
import os
import numpy as np
import pytorch_lightning as pl
import cv2
from ultralytics import YOLO
import numpy as np
import torch
from torchvision import transforms, models
from PIL import Image
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix
import argparse

#%%
def parse_args():
    parser = argparse.ArgumentParser(description="Train YOLO model with frozen layers")
    parser.add_argument('-v', '--video', type=str, required=True, help="Path to the base directory")
    return parser.parse_args()

class SimpleCNNLightning(pl.LightningModule):
    def __init__(self, num_classes=4, learning_rate=0.0001):
        super(SimpleCNNLightning, self).__init__()
        self.save_hyperparameters()
        self.criterion = nn.CrossEntropyLoss()
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

#%%
def predict_cnn(image):
    image = Image.fromarray(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
    image = transform(image).unsqueeze(0).to(device)
    with torch.no_grad():
        output = cnn_model(image)
        _, predicted = torch.max(output, 1)
    return predicted.item()

#%%
def main():

    args = parse_args()
    video_path = args.video

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    cnn_path = "./checkpoints/CNN5_model.ckpt"
    cnn_model = SimpleCNNLightning.load_from_checkpoint(cnn_path)
    cnn_model = cnn_model.to(device)
    cnn_model.eval() 

    transform = transforms.Compose([
        transforms.Resize((256, 256)),  # 调整图像大小
        transforms.ToTensor(),
    ])

    #%%
    yolo_path = "./checkpoints/best.pt"
    model = YOLO(yolo_path)
    cap = cv2.VideoCapture(video_path) # Open the video file
    num = 0 # count the number of frames
    while True:
        success, frame = cap.read() # Read the frame
        if success:
            num += 1
            if frame.shape[0] != 480 or frame.shape[1] != 640:
                frame = cv2.resize(frame, (640, 480))
            results = model.predict(frame, conf=0.25) # YOLOv8 Track the object in the frame
            if results[0].boxes.shape[0] > 0: # If the object is detected
                for i, box in enumerate(results[0].boxes.xyxy.cpu().numpy().astype(int)): # Draw the bounding box of the object
                    x_min, y_min, x_max, y_max = box
                    cropped_img = frame[y_min:y_max, x_min:x_max]
                    cropped_img = cv2.resize(cropped_img, (128, 128))
                    pose_pred = predict_cnn(cropped_img)
                    x_center = (x_min + x_max) / 2 / frame.shape[1]
                    y_center = (y_min + y_max) / 2 / frame.shape[0]
                    bbox_width = (x_max - x_min) / frame.shape[1]
                    bbox_height = (y_max - y_min) / frame.shape[0]
                                    
                    match pose_pred:
                        case 0:
                            id_char = 'bending'
                        case 1:
                            id_char = 'stand'
                        case 2:
                            id_char = 'lie'
                        case 3:
                            id_char = 'fall'
                    color = {
                            0: (0, 0, 255),   # Red for bending
                            1: (0, 255, 0),   # Green for stand
                            2: (255, 0, 0),   # Blue for lie
                            3: (0, 255, 255)  # Yellow for fall
                        }[pose_pred]
                        
                    cv2.rectangle(frame, (box[0], box[1]), (box[2], box[3]), color, 2)
                    cv2.putText(frame, f"{id_char}", (box[0], box[1] - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
            cv2.imshow("YOLOv8 Tracking", frame)
            if cv2.waitKey(1) & 0xFF == ord("q"):
                break
        else:
            break
    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
