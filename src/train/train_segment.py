import os
import sys
import argparse
from pathlib import Path

# Add project root to sys.path
project_root = str(Path(__file__).resolve().parents[2])
if project_root not in sys.path:
    sys.path.append(project_root)
from ultralytics import YOLO
from src.utils.download_weights import ensure_yolo11n_seg, ensure_yolo12n

def parse_args():
    parser = argparse.ArgumentParser(description="Train Fish Segmentation Model (YOLO-Seg)")
    parser.add_argument("--data_yaml", type=str, required=True, help="Path to dataset.yaml for segmentation")
    parser.add_argument("--model", type=str, default="yolo11n-seg.pt", help="Pretrained weights (e.g., yolo11n-seg.pt or yolo12n-seg.pt)")
    parser.add_argument("--imgsz", type=int, default=1024, help="Image size")
    parser.add_argument("--epochs", type=int, default=300, help="Number of epochs")
    parser.add_argument("--batch", type=int, default=8, help="Batch size")
    parser.add_argument("--device", type=str, default="0", help="Device (0, 1, or 'cpu')")
    parser.add_argument("--project", type=str, default="results/segment", help="Project name")
    parser.add_argument("--name", type=str, default="finetune_seg", help="Run name")
    return parser.parse_args()

def main():
    args = parse_args()
    
    if not os.path.exists(args.data_yaml):
        print(f"Error: dataset.yaml not found at {args.data_yaml}")
        return

    # Ensure weights exist
    if args.model == "yolo11n-seg.pt":
        ensure_yolo11n_seg()
    elif args.model == "yolo12n-seg.pt":
        # Note: yolo12n-seg is not in the automatic map yet, but we check common ones
        pass

    # Initialize YOLO segmentation model
    model = YOLO(args.model)
    
    print(f"Starting Segmentation Training...")
    model.train(
        data=args.data_yaml,
        epochs=args.epochs,
        imgsz=args.imgsz,
        batch=args.batch,
        device=args.device,
        project=os.path.abspath(args.project),
        name=args.name,
        patience=30,
        task='segment'
    )
    print(f"Training finished. Results saved to {args.project}/{args.name}")

if __name__ == "__main__":
    main()
