import os
import argparse
from ultralytics import YOLO

def parse_args():
    parser = argparse.ArgumentParser(description="Evaluate Fish Detection/Segmentation Model")
    parser.add_argument("--model", type=str, required=True, help="Path to best.pt")
    parser.add_argument("--data", type=str, required=True, help="Path to data.yaml")
    parser.add_argument("--imgsz", type=int, default=1024, help="Image size")
    parser.add_argument("--device", type=str, default="0", help="Device")
    parser.add_argument("--split", type=str, default="val", help="Dataset split to evaluate (val, test)")
    return parser.parse_args()

def main():
    args = parse_args()
    
    model = YOLO(args.model)
    print(f"Evaluating model: {args.model} on {args.split} set...")
    
    results = model.val(
        data=args.data,
        imgsz=args.imgsz,
        device=args.device,
        split=args.split
    )
    
    print("Evaluation Results:")
    print(results)

if __name__ == "__main__":
    main()
