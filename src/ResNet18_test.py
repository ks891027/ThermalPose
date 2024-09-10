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

#%%
class SimpleCNNLightning(pl.LightningModule):
    def __init__(self, num_classes=4, learning_rate=0.0001):
        super(SimpleCNNLightning, self).__init__()
        self.save_hyperparameters()
        self.criterion = nn.CrossEntropyLoss()
        self.lr = learning_rate
        self.train_losses = []
        self.model = models.resnet18(pretrained=True)
        self.model.fc = nn.Linear(self.model.fc.in_features, num_classes)

    def forward(self, x):
        return self.model(x)

#%%
def predict_cnn(image):
    image = Image.fromarray(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
    image = transform(image).unsqueeze(0).to(device)
    with torch.no_grad():
        output = cnn_model(image)
        _, predicted = torch.max(output, 1)
    return predicted.item()

def calculate_iou(box1, box2):
    """计算两个bounding box的IoU，忽略标签部分"""
    x1_min = box1[1] - box1[3] / 2
    y1_min = box1[2] - box1[4] / 2
    x1_max = box1[1] + box1[3] / 2
    y1_max = box1[2] + box1[4] / 2

    x2_min = box2[1] - box2[3] / 2
    y2_min = box2[2] - box2[4] / 2
    x2_max = box2[1] + box2[3] / 2
    y2_max = box2[2] + box2[4] / 2

    inter_x_min = max(x1_min, x2_min)
    inter_y_min = max(y1_min, y2_min)
    inter_x_max = min(x1_max, x2_max)
    inter_y_max = min(y1_max, y2_max)

    inter_area = max(0, inter_x_max - inter_x_min) * max(0, inter_y_max - inter_y_min)

    box1_area = (x1_max - x1_min) * (y1_max - y1_min)
    box2_area = (x2_max - x2_min) * (y2_max - y2_min)

    iou = inter_area / (box1_area + box2_area - inter_area)
    return iou

def greedy_match(pred_boxes, gt_boxes, iou_threshold=0.2):
    """
    使用贪心算法为预测框和 ground truth 框进行匹配
    """
    # 初始化 IoU 矩阵
    iou_matrix = np.zeros((len(pred_boxes), len(gt_boxes)))
    # 计算 IoU 矩阵
    for i, pred_box in enumerate(pred_boxes):
        for j, gt_box in enumerate(gt_boxes):
            iou_matrix[i, j] = calculate_iou(pred_box, gt_box)

    matches = []
    unmatched_predictions = set(range(len(pred_boxes)))
    unmatched_ground_truths = set(range(len(gt_boxes)))

    # 贪心匹配
    while np.max(iou_matrix) > 0:
        max_iou_idx = np.unravel_index(np.argmax(iou_matrix), iou_matrix.shape)
        pred_idx, gt_idx = max_iou_idx

        max_iou = iou_matrix[pred_idx, gt_idx]
        
        if max_iou >= iou_threshold:
            matches.append((pred_idx, gt_idx, max_iou))
            unmatched_ground_truths.discard(gt_idx)
            unmatched_predictions.discard(pred_idx)

        # 将选中的行和列置为负值以避免再次选择
        iou_matrix[pred_idx, :] = -1
        iou_matrix[:, gt_idx] = -1
    unmatched_predictions = list(unmatched_predictions)
    unmatched_ground_truths = list(unmatched_ground_truths)
    # 将未匹配的预测框也加入到未匹配列表中
    # unmatched_predictions.extend(np.where(np.max(iou_matrix, axis=1) == -1)[0])

    return matches, unmatched_predictions, unmatched_ground_truths

def calculate_yolo_metrics(matches, unmatched_predictions, unmatched_ground_truths, pred_labels, gt_labels):
    """根据匹配结果计算一张图片的TP, FP, FN"""
    TP, FP, FN = 0, 0, 0
    pred = []
    gt = []
    for match in matches:
        TP += 1
        pred.append(1)
        gt.append(1)
    for i in range(len(unmatched_predictions)):
        FP += 1
        pred.append(1)
        gt.append(0)
    for i in range(len(unmatched_ground_truths)):
        FN += 1
        pred.append(0)
        gt.append(1)
    return TP, FP, FN, pred, gt

def calculate_image_metrics(matches, unmatched_predictions, unmatched_ground_truths, pred_labels, gt_labels):
    """根据匹配结果计算一张图片的TP, FP, FN"""
    TP, FP, FN = 0, 0, 0
    pred = []
    gt = []

    for match in matches:
        pred_idx, gt_idx, _ = match
        if pred_labels[pred_idx][0] == None:
            FN += 1
            pred.append(-1)
            gt.append(gt_labels[gt_idx][0])
        elif gt_labels[gt_idx][0] == pred_labels[pred_idx][0]:  # pose_pred相同
            TP += 1
            pred.append(pred_labels[pred_idx][0])
            gt.append(gt_labels[gt_idx][0])
        else:
            FP += 1
            pred.append(pred_labels[pred_idx][0])
            gt.append(gt_labels[gt_idx][0])

    for i in range(len(unmatched_ground_truths)):
        FN += 1
        pred.append(-1)
        gt.append(gt_labels[unmatched_ground_truths[i]][0])
        
    return TP, FP, FN, pred, gt

def calculate_total_metrics(matches, unmatched_predictions, unmatched_ground_truths, pred_labels, gt_labels):
    """根据匹配结果计算一张图片的TP, FP, FN"""
    TP, FP, FN = 0, 0, 0
    pred = []
    gt = []
    
    for match in matches:
        pred_idx, gt_idx, _ = match
        if gt_labels[gt_idx][0] == pred_labels[pred_idx][0]:  # pose_pred相同
            TP += 1
            pred.append(pred_labels[pred_idx][0])
            gt.append(gt_labels[gt_idx][0])
        else:
            FP += 1
            pred.append(pred_labels[pred_idx][0])
            gt.append(gt_labels[gt_idx][0])

    for i in range(len(unmatched_ground_truths)):
        FN += 1
        pred.append(-1)
        gt.append(gt_labels[unmatched_ground_truths[i]][0])
    for i in range(len(unmatched_predictions)):
        FP += 1
        pred.append(pred_labels[unmatched_predictions[i]][0])
        gt.append(-1)
    
    return TP, FP, FN, pred, gt

def calculate_overall_metrics(TP, FP, FN):
    """计算总体的precision, recall, accuracy"""
    precision = TP / (TP + FP) if TP + FP > 0 else 0
    recall = TP / (TP + FN) if TP + FN > 0 else 0
    accuracy = TP / (TP + FP + FN) if TP + FP + FN > 0 else 0
    return precision, recall, accuracy

def draw_boxes(image, boxes, color, label=None):
    """
    在图像上绘制边界框
    :param image: 输入的图像
    :param boxes: 边界框列表，格式为 [x_center, y_center, width, height]
    :param color: 边界框颜色
    :param label: 标签（可选），用于在边界框上显示文本
    """
    for box in boxes:
        x_center, y_center, width, height = box[1:5]
        x_min = int((x_center - width / 2) * image.shape[1])
        y_min = int((y_center - height / 2) * image.shape[0])
        x_max = int((x_center + width / 2) * image.shape[1])
        y_max = int((y_center + height / 2) * image.shape[0])
        
        cv2.rectangle(image, (x_min, y_min), (x_max, y_max), color, 2)
        
        if label is not None:
            cv2.putText(image, label, (x_min, y_min - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, color, 2)

def visualize_iou_matching(image, gt_boxes, pred_boxes, matches, output_path):
    """
    可视化 IoU 匹配
    :param image: 输入的图像
    :param gt_boxes: Ground Truth 边界框列表
    :param pred_boxes: 预测边界框列表
    :param matches: 匹配结果，格式为 (pred_idx, gt_idx, iou)
    """
    # 复制图像用于绘制
    image_with_boxes = image.copy()
    
    # 绘制所有的 GT 框
    draw_boxes(image_with_boxes, gt_boxes, (0, 255, 0), label="GT")
    
    # 绘制所有的预测框
    draw_boxes(image_with_boxes, pred_boxes, (255, 0, 0), label="Pred")
    
    # 绘制匹配的框，并标注 IoU 值
    for pred_idx, gt_idx, iou in matches:
        pred_box = pred_boxes[pred_idx]
        gt_box = gt_boxes[gt_idx]
        
        # 选取匹配的框使用不同的颜色
        draw_boxes(image_with_boxes, [gt_box], (0, 255, 255), label=f"GT IoU={iou:.2f}")
        draw_boxes(image_with_boxes, [pred_box], (255, 255, 0), label=f"Pred IoU={iou:.2f}")

    cv2.imwrite(output_path, image_with_boxes)

#%%
images_folder = '/home/yang/Documents/Thermal_Pose/ThermalPose/data/test_data/images'
labels_folder = '/home/yang/Documents/Thermal_Pose/ThermalPose/data/test_data/labels'

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

cnn_model = SimpleCNNLightning.load_from_checkpoint("/home/yang/Documents/Thermal_Pose/ThermalPose/results/ResNet18/best_model.ckpt")
cnn_model = cnn_model.to(device)
# 进行推理
cnn_model.eval()  # 设为评估模式

transform = transforms.Compose([
    transforms.Resize((256, 256)),  # 调整图像大小
    transforms.ToTensor(),
])

model = YOLO('/home/yang/Documents/Thermal_Pose/ThermalPose/results/transfer/weights/best.pt') # Load the YOLO model
image_files = sorted([f for f in os.listdir(images_folder) if f.endswith('.jpg')])
output_path = '/home/yang/Documents/Thermal_Pose/ThermalPose/data/classifier_test_results/ResNet18'

yolo_TP, yolo_FP, yolo_FN = 0, 0, 0
class_TP, class_FP, class_FN = 0, 0, 0
total_TP, total_FP, total_FN = 0, 0, 0
yolo_true_labels = []
yolo_pred_labels = []

class_true_labels = []
class_pred_labels = []

total_true_labels = []
total_pred_labels = []

for image_file in image_files:
    image_path = os.path.join(images_folder, image_file)
    name = image_file.split('.')[0]
    frame = cv2.imread(image_path)
    if frame.shape[0] != 480 or frame.shape[1] != 640:
        frame = cv2.resize(frame, (640, 480))
    results = model.predict(frame, conf=0.25) # YOLOv8 Track the object in the frame
    if results[0].boxes.shape[0] > 0: # If the object is detected
        gt_boxes = []
        pred_boxes = []

        GroundTruthPath = os.path.join(labels_folder, name + '.txt')
        with open(GroundTruthPath, 'r') as label_file:
            ground_truths = label_file.readlines()
        
        for ground_truth in ground_truths:
            values = ground_truth.strip().split()
            converted_values = [int(values[0])] + [float(v) for v in values[1:]]            
            gt_boxes.append(converted_values)

        for i, box in enumerate(results[0].boxes.xyxy.cpu().numpy().astype(int)): # Draw the bounding box of the object
            x_min, y_min, x_max, y_max = box
            cropped_img = frame[y_min:y_max, x_min:x_max]
            cropped_img = cv2.resize(cropped_img, (128, 128))
            pose_pred = predict_cnn(cropped_img)
            x_center = (x_min + x_max) / 2 / frame.shape[1]
            y_center = (y_min + y_max) / 2 / frame.shape[0]
            bbox_width = (x_max - x_min) / frame.shape[1]
            bbox_height = (y_max - y_min) / frame.shape[0]
            pred_boxes.append([pose_pred, x_center, y_center, bbox_width, bbox_height])
                
        matches, unmatched_predictions, unmatched_ground_truths = greedy_match(pred_boxes, gt_boxes)
        TP, FP, FN, yolo_pred_labels, yolo_true_labels = calculate_yolo_metrics(matches, unmatched_predictions, unmatched_ground_truths, pred_boxes, gt_boxes)
        yolo_TP += TP
        yolo_FP += FP
        yolo_FN += FN

        TP, FP, FN, class_pred_labels, class_true_labels = calculate_image_metrics(matches, unmatched_predictions, unmatched_ground_truths, pred_boxes, gt_boxes)
        class_TP += TP
        class_FP += FP
        class_FN += FN

        TP, FP, FN, total_pred_labels, total_true_labels = calculate_total_metrics(matches, unmatched_predictions, unmatched_ground_truths, pred_boxes, gt_boxes)
        total_TP += TP
        total_FP += FP
        total_FN += FN
        visualize_iou_matching(frame, gt_boxes, pred_boxes, matches, os.path.join(output_path, image_file))
    
    else: # If the object is not detected
        GroundTruthPath = os.path.join(labels_folder, name + '.txt')
        with open(GroundTruthPath, 'r') as label_file:
            ground_truths = label_file.readlines()
        for i in range(len(ground_truths)):
            total_FN += 1

yolo_precision, yolo_recall, yolo_accuracy = calculate_overall_metrics(yolo_TP, yolo_FP, yolo_FN)
class_precision, class_recall, class_accuracy = calculate_overall_metrics(class_TP, class_FP, class_FN)
precision, recall, accuracy = calculate_overall_metrics(total_TP, total_FP, total_FN)

print(f"YOLOv8 Precision: {yolo_precision:.4f}, Recall: {yolo_recall:.4f}, Accuracy: {yolo_accuracy:.4f}\nTP: {yolo_TP}, FP: {yolo_FP}, FN: {yolo_FN}")            
print(f"Classification Precision: {class_precision:.4f}, Recall: {class_recall:.4f}, Accuracy: {class_accuracy:.4f}\nTP: {class_TP}, FP: {class_FP}, FN: {class_FN}")
print(f"Precision: {precision:.4f}, Recall: {recall:.4f}, Accuracy: {accuracy:.4f}\nTP: {total_TP}, FP: {total_FP}, FN: {total_FN}")

# yolo_cm = confusion_matrix(yolo_true_labels, yolo_pred_labels)
# resnet_cm = confusion_matrix(class_true_labels, class_pred_labels)
# combined_cm = confusion_matrix(total_true_labels, total_pred_labels)

# print("YOLOv8 Confusion Matrix:")
# print(yolo_cm)

# print("\nResNet Classifier Confusion Matrix:")
# print(resnet_cm)

# print("\nCombined Confusion Matrix:")
# print(combined_cm)