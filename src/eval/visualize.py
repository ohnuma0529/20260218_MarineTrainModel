import os
import argparse
import cv2
from ultralytics import YOLO
from pathlib import Path

def parse_args():
    parser = argparse.ArgumentParser(description="Visualize Inference Results")
    parser.add_argument("--model", type=str, required=True, help="Path to best.pt")
    parser.add_argument("--source", type=str, required=True, help="Source directory or image path")
    parser.add_argument("--output", type=str, default="outputs/viz", help="Output directory")
    parser.add_argument("--conf", type=float, default=0.25, help="Confidence threshold")
    parser.add_argument("--imgsz", type=int, default=1024, help="Image size")
    parser.add_argument("--device", type=str, default="0", help="Device")
    return parser.parse_args()

def main():
    args = parse_args()
    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    model = YOLO(args.model)
    print(f"Running inference on {args.source}...")
    
    results = model.predict(
        source=args.source,
        conf=args.conf,
        imgsz=args.imgsz,
        device=args.device,
        save=True,
        project=os.path.abspath(output_dir.parent),
        name=output_dir.name,
        exist_ok=True
    )
    
    print(f"Visualization results saved to {output_dir}")

if __name__ == "__main__":
    main()
