# Google Colab 環境構築および実験実行手順

本プロジェクトでは、コード管理（GitHub）と大容量データ・モデル重み管理（Google Drive）を分離して運用する。本ドキュメントは、Google Colab 上で VLA モデルのファインチューニングを行う際の手順書である。

---

## 1. データの共有管理 (Google Drive)

1. Google Drive 上に共有フォルダ（例: `research-hand-sign-shared`）を配置する。
2. フォルダ構成は以下の通りとする。
   ```text
   research-hand-sign-shared/
   ├── data/
   │   ├── raw/          # ローカルで収録した生CSVデータ
   │   └── processed/    # エクスポートされた vla_discrete_64.jsonl 等
   └── weights/          # 学習済みモデル・量子化重み
   ```
3. ローカル環境でエクスポートしたデータセット（JSONL）をこのフォルダに配置して同期する。

---

## 2. Google Colab での実行手順

### 2.1 Google Drive のマウント
```python
from google.colab import drive
drive.mount('/content/drive')
```

### 2.2 リポジトリのクローン
```python
GITHUB_USER = "your_github_username"
GITHUB_TOKEN = "your_personal_access_token"
REPO_NAME = "research-hand-sign"

!git clone https://{GITHUB_USER}:{GITHUB_TOKEN}@github.com/{GITHUB_USER}/{REPO_NAME}.git
```

### 2.3 依存パッケージのインストール
```python
%cd /content/research-hand-sign
!pip install -r requirements.txt
!pip install "numpy<2.0.0" "transformers<5.0.0" peft accelerate datasets matplotlib bitsandbytes
```
※ インストール完了後、Colabメニューの「ランタイム」→「セッションを再起動」を実行すること。

### 2.4 共有データフォルダの接続
```python
SHARED_DRIVE_DIR = "/content/drive/MyDrive/research-hand-sign-shared"

# data フォルダおよび weights フォルダをシンボリックリンクで接続
!rm -rf data
!ln -s {SHARED_DRIVE_DIR}/data data

!rm -rf src/learning/weights
!ln -s {SHARED_DRIVE_DIR}/weights src/learning/weights
```

### 2.5 VLA LoRA訓練スクリプトの実行
```python
# 離散トークン形式でのファインチューニング実行
!python src/learning/vla_train_colab.py --dataset data/processed/vla_discrete.jsonl --approach discrete
```
学習完了後、学習済み重み `vla_lora_weights.pt` が Google Drive 側の `weights/` に保存される。
