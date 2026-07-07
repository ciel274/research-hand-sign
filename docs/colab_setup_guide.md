# 🤝 共同開発者向け Google Colab セットアップガイド

本プロジェクトは共同開発のため、**「コード管理（GitHub）」**と**「大容量データ管理（Google Drive）」**を切り離して管理します。これにより、マージコンフリクトやGitHubのファイル容量制限を回避し、安全にチームで開発を進められます。

---

## 📁 データの共有方法 (Google Drive)

1. メンバーの誰か一人が、Google Drive 上に共有フォルダ（例: `research-hand-sign-shared`）を作成し、共同開発メンバー全員を招待（共有）します。
2. この共有フォルダの中に、以下のフォルダ構造を作成します：
   ```text
   research-hand-sign-shared/
   ├── data/
   │   ├── raw/          # 各自がローカルUIで撮影したCSVを集約する場所
   │   └── processed/    # エクスポートした vla_discrete.jsonl 等を置く場所
   └── weights/          # 学習済みのモデルの重みを置く場所
   ```
3. 各自がローカルのWeb UIで撮影・生成した `vla_discrete.jsonl` や CSV ファイルは、定期的にこの共有ドライブの同じ場所にアップロードして同期します。

---

## 📓 Google Colab での実行手順（コピペ用コード）

Google Colab で学習を回す際は、新規ノートブックを作成し、以下のブロックを上から順に実行してください。

### ❶ 共有 Google Drive のマウント
```python
from google.colab import drive
drive.mount('/content/drive')
```
* ※マウントすると、Colab内の `/content/drive/MyDrive/` 以下に共有フォルダが見えるようになります。

### ❷ コード（GitHub）のクローン
共同開発リポジトリから最新のプログラムをクローンします。
```python
# あなたのGitHubユーザー名と発行したPersonal Access Token (PAT) を入力してください
GITHUB_USER = "your_github_username"
GITHUB_TOKEN = "your_personal_access_token"
REPO_NAME = "research-hand-sign"

# クローン実行
!git clone https://{GITHUB_USER}:{GITHUB_TOKEN}@github.com/{GITHUB_USER}/{REPO_NAME}.git
```

### ❸ 依存パッケージのインストール
```python
%cd /content/research-hand-sign
# ColabのGPU版PyTorch環境を壊さないよう、特定のパッケージのみ指定してインストール
!pip install -r requirements.txt
!pip install "numpy<2.0.0" "transformers<5.0.0" peft accelerate datasets matplotlib bitsandbytes
```
> [!IMPORTANT]
> **重要**: インストール後は、Google Colabのメニューバーから **「ランタイム」➔「セッションを再起動」** を必ず実行してください。再起動後、次のステップから進めてください。

### ❹ データのシンボリックリンク（紐付け）
GitHubから落としたコードに対して、共有Google Drive内の大容量データフォルダを接続します。
```python
# 共有ドライブのパス（ご自身の共有フォルダ名に合わせて修正してください）
SHARED_DRIVE_DIR = "/content/drive/MyDrive/research-hand-sign-shared"

# data フォルダへのショートカット（シンボリックリンク）を作成
!rm -rf data
!ln -s {SHARED_DRIVE_DIR}/data data

# weights フォルダへのショートカットを作成
!rm -rf src/learning/weights
!ln -s {SHARED_DRIVE_DIR}/weights src/learning/weights

print("Google Driveの共有データとの接続が完了しました！")
```

### ❺ VLA LoRAモデル訓練スクリプトの実行
これで、共有ドライブにある最新のデータセットを自動的に読み込んで学習が走ります。
```python
# 離散値アプローチBでの学習実行
!python src/learning/vla_train_colab.py --dataset data/processed/vla_discrete.jsonl --approach discrete
```
* **学習成果物**: 完了すると、モデルの重み `vla_lora_weights.pt` が自動的に Google Drive の共有フォルダ側の `weights/` に直接保存されます。
