import argparse
from ultralytics import YOLO
import os

def parse_args():
    parser = argparse.ArgumentParser(description="Train YOLO model with frozen layers")
    parser.add_argument('-p', '--path', type=str, required=True, help="Path to the base directory")
    return parser.parse_args()

def freeze_layer(trainer):
    model = trainer.model
    num_freeze = 3
    print(f"Freezing {num_freeze} layers")
    freeze = [f'model.{x}.' for x in range(num_freeze)]
    for k, v in model.named_parameters():
        v.requires_grad = True
        if any(x in k for x in freeze):
            print(f'Freezing {k}')
            v.requires_grad = False
    print(f"{num_freeze} layers are frozen.")

def main():
    
    args = parse_args()
    
    base_path = args.path

    # Load a pretrained model
    model_path = os.path.join(base_path, "results/OpenTermalPose/weights/best.pt")
    model = YOLO(model_path)

    model.add_callback("on_train_start", freeze_layer)

    # Train the model
    data_path = os.path.join(base_path, "documents/transfer_data.yaml")
    results_path = os.path.join(base_path, "results")
    model.train(data=data_path, epochs=50, imgsz=640, project=results_path, name="transfer")

if __name__ == "__main__":
    main()
