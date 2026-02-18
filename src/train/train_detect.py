import os
import sys
import argparse
from pathlib import Path

# Add project root to sys.path
project_root = str(Path(__file__).resolve().parents[2])
if project_root not in sys.path:
    sys.path.append(project_root)
from ultralytics import YOLO
import yaml
from src.utils.download_weights import ensure_yolo12n, ensure_yolo11n_seg, ensure_sam2

def parse_args():
    parser = argparse.ArgumentParser(description="Train Fish Detection Model (B1: All-in-One)")
    parser.add_argument("--config", type=str, default="configs/detect_config.yaml", help="Path to config YAML")
    parser.add_argument("--data_dir", type=str, help="Dataset directory (overrides config)")
    parser.add_argument("--model", type=str, help="Pretrained model weight (overrides config)")
    parser.add_argument("--epochs", type=int, help="Number of epochs (overrides config)")
    parser.add_argument("--batch", type=int, help="Batch size (overrides config)")
    parser.add_argument("--device", type=str, help="Device (overrides config)")
    parser.add_argument("--name", type=str, help="Run name (overrides config)")
    return parser.parse_args()

def load_config(config_path):
    if not os.path.exists(config_path):
        return {}
    with open(config_path, 'r') as f:
        return yaml.safe_load(f)

def gather_images(data_dir, subdirs, exclude_list=None):
    images = []
    exclude_set = set(exclude_list) if exclude_list else set()
    for sd in subdirs:
        path = Path(data_dir) / sd / "images"
        if path.exists():
            for img in path.glob("*"):
                if img.suffix.lower() in [".jpg", ".png", ".jpeg"]:
                    abs_path = str(img.absolute())
                    if abs_path not in exclude_set:
                        images.append(abs_path)
    return images

def read_split_file(file_path):
    if not file_path:
        return []
    # Relative path support: if not absolute, assume relative to project root (CWD)
    abs_path = os.path.abspath(file_path)
    if not os.path.exists(abs_path):
        print(f"Warning: Split file not found at {abs_path}")
        return []
    
    with open(abs_path, 'r') as f:
        lines = [line.strip() for line in f if line.strip()]
        # Each line in split file might also be relative
        results = []
        for line in lines:
            if os.path.isabs(line):
                results.append(line)
            else:
                # Resolve relative to project root
                results.append(os.path.abspath(line))
        return results

def create_temp_yaml(train_images, val_images, nc=1, names=["fish"]):
    tmp_dir = Path("tmp_splits")
    tmp_dir.mkdir(exist_ok=True)
    train_txt = tmp_dir / "train.txt"
    val_txt = tmp_dir / "val.txt"
    
    with open(train_txt, 'w') as f:
        f.write("\n".join(train_images))
    with open(val_txt, 'w') as f:
        f.write("\n".join(val_images))
        
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
    
    # Ensure weights exist
    if args.model == "yolo12n.pt":
        ensure_yolo12n()
    elif args.model == "yolo11n-seg.pt":
        ensure_yolo11n_seg()

    # Initialize YOLO model
    config = load_config(args.config)
    
    # Extract data config
    data_cfg = config.get("data", {})
    data_dir = args.data_dir or data_cfg.get("data_dir", "data/detect_dataset")
    subdirs = data_cfg.get("subdirs", ['1_auto_fish', '2_FIB', '3_fish_tray', '4_local', '5_syn'])
    test_split_file = data_cfg.get("test_split_file")
    
    # Extract train params
    train_cfg = config.get("train_params", {})
    model_path = args.model or train_cfg.get("model", "weights/yolo12n.pt")
    epochs = args.epochs or train_cfg.get("epochs", 300)
    batch = args.batch or train_cfg.get("batch", 8)
    device = args.device or train_cfg.get("device", "0")
    project = train_cfg.get("project", "results/detect")
    name = args.name or train_cfg.get("name", "b1_all_in_one")
    imgsz = train_cfg.get("imgsz", 1024)

    # 1. Load test split (Validation/Test data)
    val_images = read_split_file(test_split_file)
    print(f"Loaded {len(val_images)} images for validation from {test_split_file}")

    # 2. Gather training images (excluding val_images)
    print(f"Gathering images from {subdirs} in {data_dir}...")
    train_images = gather_images(data_dir, subdirs, exclude_list=val_images)
    print(f"Total training images found: {len(train_images)}")
    
    if not train_images:
        print("Error: No training images found. Check your data_dir and subdirs.")
        return

    # 3. Create YAML
    yaml_path = create_temp_yaml(train_images, val_images)
    print(f"Created temporary training config: {yaml_path}")

    # 4. Run Training
    model = YOLO(model_path)
    print(f"Starting training (B1 mode)...")
    model.train(
        data=str(yaml_path),
        epochs=epochs,
        imgsz=imgsz,
        batch=batch,
        device=device,
        project=os.path.abspath(project),
        name=name,
        patience=30
    )
    print(f"Training finished. Results saved to {project}/{name}")

if __name__ == "__main__":
    main()
