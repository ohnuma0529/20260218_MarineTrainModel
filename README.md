# 魚検出（Detection & Segmentation）訓練リポジトリ

このリポジトリは、YOLOv12およびYOLO-Segを用いた魚の検出・セグメンテーションモデルを訓練・評価するための統合環境です。

## 機能
- **環境構築**: `setup.sh` による自動セットアップ（venv使用）
- **データ準備**: SAM2を用いた自動セグメンテーションラベル生成とクロップ済みデータセット作成
- **Detection訓練**: YOLOv12による全データ一括訓練（B1形式）
- **Segmentation訓練**: YOLO-Segによるファインチューニング
- **検証・可視化**: 精度評価（mAP）と推論結果の可視化

## ディレクトリ構成
```text
.
├── README.md               # このファイル
├── setup.sh                # 環境構築用スクリプト
├── requirements.txt        # 依存パッケージ一覧
├── configs/                # 各種設定ファイル（YAML等）
├── src/
│   ├── data/
│   │   └── prepare_dataset.py  # データセット準備（SAM2活用）
│   ├── train/
│   │   ├── train_detect.py    # Detection訓練（B1形式）
│   │   └── train_segment.py   # Segmentation訓練
│   └── eval/
│       ├── evaluate.py        # 精度評価
│       └── visualize.py       # 推論結果可視化
└── data/                   # 学習データ（シンボリックリンク等で配置）
```

## セットアップ
1. **仮想環境の構築**
   ```bash
   bash setup.sh
   source venv/bin/activate
   ```

## 使い方

### 1. データセットの配置
`data/detect_dataset` および `data/seg_dataset` に元データを配置してください。

### 2. データセットの準備（SAM2によるラベル生成等）
```bash
python src/data/prepare_dataset.py --root_dir data --output_dir data/processed
```

### 3. Detection訓練（B1形式: 全データ一括）
```bash
python src/train/train_detect.py --data_dir data/detect_dataset --model yolo12n.pt
```

### 4. Segmentation訓練
```bash
python src/train/train_segment.py --data_yaml data/processed/dataset.yaml --model yolo11n-seg.pt
```

### 5. 評価・可視化
```bash
# 評価
python src/eval/evaluate.py --model results/detect/b1_all_in_one/weights/best.pt --data path/to/data.yaml
# 可視化
python src/eval/visualize.py --model results/detect/b1_all_in_one/weights/best.pt --source data/test_images
```

## 注意事項
- GPU環境（CUDA）での実行を推奨します。
- SAM2の重み（`sam2_b.pt`等）は初回実行時に自動ダウンロードされます。
