# 魚検出（Detection & Segmentation）訓練リポジトリ

このリポジトリは、YOLOv12およびYOLO-Segを用いた魚の検出・セグメンテーションモデルを訓練・評価するための統合環境です。
ポータビリティに配慮しており、データセットを配置するだけで訓練を開始できるよう構成されています。

## 特徴
- **ポータビリティ**: すべてのパスはプロジェクトルートからの相対パスで管理されています。
- **自動化**: 軽量な重みはリポジトリに含まれており、大容量な重み（SAM2等）は初回実行時に自動ダウンロードされます。
- **統合パイプライン**: SAM2を用いたセグメンテーションラベル生成から、YOLOv12による一括訓練まで対応。

## ディレクトリ構成
```text
.
├── setup.sh                # 環境構築スクリプト
├── requirements.txt        # 依存パッケージ
├── configs/
│   └── detect_config.yaml  # Detection訓練の設定（データセット構成等）
├── src/
│   ├── data/
│   │   └── prepare_dataset.py  # データセット準備（SAM2活用）
│   ├── train/
│   │   ├── train_detect.py    # Detection訓練 (B1: All-in-One)
│   │   └── train_segment.py   # Segmentation訓練
│   ├── eval/
│   │   ├── evaluate.py        # 精度評価
│   │   └── visualize.py       # 推論結果可視化
│   └── utils/
│       └── download_weights.py # 重み自動ダウンロード
├── weights/                # 学習済みモデル（yolo12n.pt等）
├── data/                   # 【ユーザーが用意】
│   ├── detect_dataset/     # 検出用データセット
│   ├── seg_dataset/        # セグメンテーション用データセット
│   └── splits/             # 訓練/テスト分割リスト（リポジトリ内製）
└── results/                # 訓練結果出力先
```

## 1. セットアップ
初回のみ、以下のコマンドで環境構築を行います。
```bash
bash setup.sh
source venv/bin/activate
```

## 2. データセットの準備

### データの配置
`data/` ディレクトリ配下に、以下の構造でデータを配置してください。
※ `4_local` などのフォルダ名は `configs/detect_config.yaml` の設定と対応している必要があります。

```text
data/
├── detect_dataset/
│   ├── 1_auto_fish/
│   │   ├── images/
│   │   └── labels/
│   ├── 4_local/
│   │   ├── images/
│   │   └── labels/
│   └── ...
└── seg_dataset/
    └── ... (セグメンテーション用データ)
```

### クロップ済みデータの作成（SAM2使用）
Detection訓練で使用する、魚ごとにクロップされたデータセットを作成します。
`4_local` などのラベルがないデータに対しては、SAM2が自動でセグメンテーションラベルを付与します。
```bash
python src/data/prepare_dataset.py --root_dir data --output_dir data/processed
```
※ `sam2_b.pt` がない場合は自動でダウンロードされます。

## 3. 訓練の実行

### Detection訓練（B1形式: 全データ一括）
魚の検出モデルを訓練します。デフォルトでは `configs/detect_config.yaml` の設定に従い、`data/splits/local_test.txt` に記載された画像を除いた全データで学習します。
```bash
python src/train/train_detect.py --model yolo12n.pt --epochs 300 --batch 16
```
- 学習済みの重みは `results/detect/b1_all_in_one/weights/best.pt` に保存されます。

### Segmentation訓練
セグメンテーションモデルを訓練します。
```bash
python src/train/train_segment.py --data_yaml data/processed/dataset.yaml --model yolo11n-seg.pt
```

## 4. 評価・可視化

### 精度評価
```bash
python src/eval/evaluate.py --model results/detect/b1_all_in_one/weights/best.pt --data data/splits/data.yaml
```

### 結果の可視化
特定の画像フォルダに対して推論を行い、結果を可視化します。
```bash
python src/eval/visualize.py --model results/detect/b1_all_in_one/weights/best.pt --source data/test_images/
```

## 注意事項
- **GPU推奨**: 訓練にはCUDA環境を推奨します。
- **データ分割**: `data/splits/local_test.txt` を編集することで、テストに使用する画像を自由に変更可能です。
