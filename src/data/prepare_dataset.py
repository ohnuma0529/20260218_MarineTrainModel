import os
import sys
import cv2
import numpy as np
import glob
from tqdm import tqdm
import shutil
import argparse
from pathlib import Path

# Add project root to sys.path
project_root = str(Path(__file__).resolve().parents[2])
if project_root not in sys.path:
    sys.path.append(project_root)
from ultralytics import SAM
from src.utils.download_weights import ensure_sam2

def parse_args():
    parser = argparse.ArgumentParser(description="Prepare dataset for Fish Detection (Detection & Segmentation)")
    parser.add_argument("--root_dir", type=str, default="data", help="Root directory containing raw datasets")
    parser.add_argument("--output_dir", type=str, default="data/processed", help="Output directory for processed dataset")
    parser.add_argument("--sam_model", type=str, default="sam2_b.pt", help="Path to SAM2 model weight")
    parser.add_argument("--expand_ratio", type=float, default=1.1, help="Ratio to expand crop around bbox")
    parser.add_argument("--min_area", type=int, default=200000, help="Minimum bbox area for local dataset filtering")
    parser.add_argument("--debug_count", type=int, default=10, help="Number of images to save for debug visualization")
    return parser.parse_args()

def create_dir_structure(base_path, sub_dirs):
    for sub in sub_dirs:
        os.makedirs(os.path.join(base_path, sub, 'images'), exist_ok=True)
        os.makedirs(os.path.join(base_path, sub, 'labels'), exist_ok=True)

def draw_translucent_mask(image, polygon_norm, color=(0, 255, 0), alpha=0.4):
    h, w = image.shape[:2]
    overlay = image.copy()
    poly_px = np.array(polygon_norm).reshape(-1, 2)
    poly_px[:, 0] *= w
    poly_px[:, 1] *= h
    poly_px = poly_px.astype(np.int32)
    cv2.fillPoly(overlay, [poly_px], color)
    return cv2.addWeighted(overlay, alpha, image, 1 - alpha, 0)

def load_yolo_labels(label_path):
    if not os.path.exists(label_path):
        return []
    labels = []
    with open(label_path, 'r') as f:
        for line in f:
            parts = list(map(float, line.strip().split()))
            if len(parts) >= 5:
                labels.append(parts)
    return labels

def load_yolo_polygons(label_path):
    if not os.path.exists(label_path):
        return []
    polygons = []
    with open(label_path, 'r') as f:
        for line in f:
            parts = list(map(float, line.strip().split()))
            if len(parts) < 5: continue
            class_id = int(parts[0])
            rest = parts[1:]
            # Assume format: class [bbox_cx cy w h] x1 y1 x2 y2 ...
            # Current dataset seems to have 4 bbox coords before polygon points if len >= 6
            if len(rest) >= 6 and len(rest) % 2 == 0:
                coords = rest[4:]
            else:
                coords = rest
            points = [[coords[i], coords[i+1]] for i in range(0, len(coords), 2)]
            if len(points) >= 3:
                polygons.append((class_id, points))
    return polygons

def get_crop_coords(bbox, img_w, img_h, expand_ratio):
    xc, yc, w, h = bbox
    xc_px, yc_px = xc * img_w, yc * img_h
    w_px, h_px = w * img_w * expand_ratio, h * img_h * expand_ratio
    
    x1 = max(0, int(xc_px - w_px / 2))
    y1 = max(0, int(yc_px - h_px / 2))
    x2 = min(img_w, int(xc_px + w_px / 2))
    y2 = min(img_h, int(yc_px + h_px / 2))
    return x1, y1, x2, y2

def process_dataset(name, root_dir, output_dir, is_local, args, sam_model=None):
    print(f"Processing {'Local' if is_local else 'Open'} Dataset: {name}")
    img_dir = os.path.join(root_dir, 'detect_dataset', name, 'images')
    label_dir = os.path.join(root_dir, 'detect_dataset', name, 'labels')
    seg_label_dir = os.path.join(root_dir, 'seg_dataset', name, 'labels')
    
    out_img_dir = os.path.join(output_dir, name, 'images')
    out_label_dir = os.path.join(output_dir, name, 'labels')
    debug_dir = os.path.join(output_dir, 'debug', name)
    os.makedirs(debug_dir, exist_ok=True)
    
    image_files = glob.glob(os.path.join(img_dir, '*.jpg')) + glob.glob(os.path.join(img_dir, '*.png'))
    processed_count = 0
    debug_count = 0

    for img_path in tqdm(image_files):
        img_name = os.path.basename(img_path)
        label_name = os.path.splitext(img_name)[0] + '.txt'
        det_label_path = os.path.join(label_dir, label_name)
        
        if not os.path.exists(det_label_path): continue
        img = cv2.imread(img_path)
        if img is None: continue
        h_img, w_img = img.shape[:2]
        
        detections = load_yolo_labels(det_label_path)
        seg_polygons = [] if is_local else load_yolo_polygons(os.path.join(seg_label_dir, label_name))
        
        for i, det in enumerate(detections):
            cls_id, xc, yc, w, h = det
            if is_local and (w * w_img) * (h * h_img) <= args.min_area: continue
            
            x1, y1, x2, y2 = get_crop_coords((xc, yc, w, h), w_img, h_img, args.expand_ratio)
            crop_img = img[y1:y2, x1:x2]
            crop_h, crop_w = crop_img.shape[:2]
            if crop_h == 0 or crop_w == 0: continue

            best_poly = None
            if not is_local and seg_polygons:
                # Use existing polygon
                det_center = (xc * w_img, yc * h_img)
                candidates = []
                for s_cls, poly_pts in seg_polygons:
                    poly_px = np.array([[p[0]*w_img, p[1]*h_img] for p in poly_pts], dtype=np.int32)
                    if cv2.pointPolygonTest(poly_px, det_center, False) >= 0:
                        mask = np.zeros((h_img, w_img), dtype=np.uint8)
                        cv2.fillPoly(mask, [poly_px], 1)
                        crop_mask = mask[y1:y2, x1:x2]
                        if np.sum(crop_mask) > 0:
                            conts, _ = cv2.findContours(crop_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                            if conts:
                                l_cont = max(conts, key=cv2.contourArea)
                                norm_p = [coord for pt in l_cont for coord in [pt[0][0]/crop_w, pt[0][1]/crop_h]]
                                candidates.append((s_cls, norm_p, cv2.contourArea(l_cont)))
                if candidates:
                    candidates.sort(key=lambda x: x[2], reverse=True)
                    best_poly = (candidates[0][0], candidates[0][1])

            elif is_local and sam_model:
                # Use SAM2
                bx1, by1 = max(0, xc*w_img - w*w_img/2 - x1), max(0, yc*h_img - h*h_img/2 - y1)
                bx2, by2 = min(crop_w, xc*w_img + w*w_img/2 - x1), min(crop_h, yc*h_img + h*h_img/2 - y1)
                try:
                    res = sam_model.predict(crop_img, bboxes=[[bx1, by1, bx2, by2]], verbose=False)
                    if res and res[0].masks is not None:
                        pts = res[0].masks.xy[0]
                        if len(pts) > 0:
                            norm_p = [coord for pt in pts for coord in [pt[0]/crop_w, pt[1]/crop_h]]
                            best_poly = (int(cls_id), norm_p)
                except Exception as e:
                    print(f"SAM2 Error on {img_name}: {e}")

            if best_poly:
                save_name = f"{os.path.splitext(img_name)[0]}_crop_{i}"
                cv2.imwrite(os.path.join(out_img_dir, save_name + '.jpg'), crop_img)
                with open(os.path.join(out_label_dir, save_name + '.txt'), 'w') as f:
                    f.write(f"{best_poly[0]} " + " ".join(map(str, best_poly[1])) + "\n")
                if debug_count < args.debug_count:
                    vis = draw_translucent_mask(crop_img, best_poly[1])
                    cv2.imwrite(os.path.join(debug_dir, save_name + '.jpg'), vis)
                    debug_count += 1
                processed_count += 1

def main():
    args = parse_args()
    open_ds = ['1_auto_fish', '2_FIB', '3_fish_tray']
    local_ds = ['4_local']
    
    if os.path.exists(args.output_dir): shutil.rmtree(args.output_dir)
    create_dir_structure(args.output_dir, open_ds + local_ds)
    
    if any(local_ds):
        if args.sam_model == "sam2_b.pt":
            ensure_sam2()
        sam_model = SAM(args.sam_model)
    else:
        sam_model = None
    
    for ds in open_ds:
        process_dataset(ds, args.root_dir, args.output_dir, False, args)
    for ds in local_ds:
        process_dataset(ds, args.root_dir, args.output_dir, True, args, sam_model)
    print("Dataset preparation complete!")

if __name__ == "__main__":
    main()
