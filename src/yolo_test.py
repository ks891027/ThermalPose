from ultralytics import YOLO

# Load a model
# model = YOLO("/home/yang/Documents/Thermal_Pose/ThermalPose/results/transfer/weights/best.pt")  # load a pretrained model (recommended for training)
model = YOLO("yolov8x.bt")

# Train the model
model.val(data="/home/yang/Documents/Thermal_Pose/ThermalPose/test_data.yaml", project="/home/yang/Documents/Thermal_Pose/ThermalPose/results", name='transfer_test_boundingbox')