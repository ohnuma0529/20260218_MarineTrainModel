import os
import argparse
from pathlib import Path
from ultralytics import YOLO
import yaml

def parse_args():
    parser = argparse.ArgumentParser(description="Train Fish Detection Model (B1: All-in-One)")
    parser.add_argument("--data_dir", type=str, default="data/detect_dataset", help="Dataset directory")
    parser.add_argument("--model", type=str, default="yolo12n.pt", help="Pretrained model weight")
    parser.add_argument("--imgsz", type=int, default=1024, help="Image size")
    parser.add_argument("--epochs", type=int, default=300, help="Number of epochs")
    parser.add_argument("--batch", type=int, default=8, help="Batch size")
    parser.add_argument("--device", type=str, default="0", help="Device (0, 1, or 'cpu')")
    parser.add_argument("--project", type=str, default="results/detect", help="Project name")
    parser.add_argument("--name", type=str, default="b1_all_in_one", help="Run name")
    return parser.parse_args()

def gather_images(data_dir, subdirs):
    images = []
    for sd in subdirs:
        path = Path(data_dir) / sd / "images"
        if path.exists():
            images.extend([str(img.absolute()) for img in path.glob("*") if img.suffix.lower() in [".jpg", ".png", ".jpeg"]])
    return images

def create_temp_yaml(images, nc=1, names=["fish"]):
    import tempfile
    data = {
        "train": images,  # List of paths or path to txt
        "val": images[:max(1, int(len(images)*0.1))], # Simple split for val
        "nc": nc,
        "names": names
    }
    # For YOLO train, it's better to save lists to txt files
    tmp_dir = Path("tmp_splits")
    tmp_dir.mkdir(exist_ok=True)
    train_txt = tmp_dir / "train.txt"
    val_txt = tmp_dir / "val.txt"
    
    with open(train_txt, 'w') as f:
        f.write("\n".join(images))
    with open(val_txt, 'w') as f:
        f.write("\n".join(data["val"]))
        
    yaml_data = {
        "train": str(train_txt.absolute()),
        "val": str(val_txt.absolute()),
        "nc": nc,
        "names": names
    }
    yaml_path = tmp_dir / "data.yaml"
    with open(yaml_path, 'w') as f:
        yaml.dump(yaml_data, f)
    return yaml_path

def main():
    args = parse_args()
    subdirs = ['1_auto_fish', '2_FIB', '3_fish_tray', '4_local', '5_syn']
    
    print(f"Gathering images from {subdirs} in {args.data_dir}...")
    all_images = gather_images(args.data_dir, subdirs)
    print(f"Total images found: {len(all_images)}")
    
    if not all_images:
        print("Error: No images found. Check your data_dir.")
        return

    yaml_path = create_temp_yaml(all_images)
    print(f"Created temporary training config: {yaml_path}")

    model = YOLO(args.model)
    print(f"Starting training (B1 mode)...")
    model.train(
        data=str(yaml_path),
        epochs=args.epochs,
        imgsz=args.imgsz,
        batch=args.batch,
        device=args.device,
        project=args.project,
        name=args.name,
        patience=30
    )
    print(f"Training finished. Results saved to {args.project}/{args.name}")

if __name__ == "__main__":
    main()
