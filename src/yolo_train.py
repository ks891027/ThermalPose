import argparse
from ultralytics import YOLO
import os

def parse_args():
    parser = argparse.ArgumentParser(description="Train YOLO model")
    parser.add_argument('-p', '--path', type=str, required=True, help="Path to the base directory")
    return parser.parse_args()

def main():
    
    args = parse_args()
    
    base_path = args.path

    # Load a model
    model = YOLO("yolov8x.pt")  

    # Train the model
    data_path = os.path.join(base_path, "documents/openthermalpose_data.yaml")
    results_path = os.path.join(base_path, "results")
    model.train(data=data_path, epochs=100, imgsz=640, project=results_path, name='Thermal')

if __name__ == "__main__":
    main()