import os
import requests
from pathlib import Path
from tqdm import tqdm

def download_weight(filename, url):
    target_path = Path("weights") / filename
    if target_path.exists():
        return str(target_path.absolute())
    
    print(f"Downloading {filename} from {url}...")
    os.makedirs("weights", exist_ok=True)
    
    response = requests.get(url, stream=True)
    total_size = int(response.headers.get('content-length', 0))
    
    with open(target_path, 'wb') as file, tqdm(
        desc=filename,
        total=total_size,
        unit='iB',
        unit_scale=True,
        unit_divisor=1024,
    ) as bar:
        for data in response.iter_content(chunk_size=1024):
            size = file.write(data)
            bar.update(size)
    
    return str(target_path.absolute())

def ensure_sam2():
    url = "https://github.com/ultralytics/assets/releases/download/v8.3.0/sam2_b.pt"
    return download_weight("sam2_b.pt", url)

def ensure_yolo12n():
    url = "https://github.com/ultralytics/assets/releases/download/v8.3.0/yolo12n.pt"
    return download_weight("yolo12n.pt", url)

def ensure_yolo11n_seg():
    url = "https://github.com/ultralytics/assets/releases/download/v8.3.0/yolo11n-seg.pt"
    return download_weight("yolo11n-seg.pt", url)

if __name__ == "__main__":
    ensure_yolo12n()
    ensure_yolo11n_seg()
    ensure_sam2()
