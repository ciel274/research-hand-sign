# research-hand-sign

小規模データ（Few-shot / Sample-efficient）での効率的学習による手話アクションの獲得・創出をする、高効率なVLA（Vision-Language-Action）学習フレームワークの研究プロジェクト。

---

## 1. プロジェクト構造

```text
research-hand-sign/
├── README.md                   # プロジェクト概要・実行手順
├── requirements.txt            # Pythonライブラリ依存関係
├── .venv/                      # Python 仮想環境
│
├── docs/                       # ドキュメント類
│   ├── requirements_definition.md  # 要件定義書
│   └── system_architecture.md       # システム設計書
│
├── data/                       # 収集した手話データ（CSV）の保存先
│   ├── raw/                    # 各自レコーディングした生のCSV
│   └── processed/              # 前処理済みの統合データセット (dataset.csv)
│
└── src/                        # ソースコード
    ├── perception/             # 視覚・骨格トラッキング (Vision)
    │   ├── hand_tracking.py        # リアルタイム骨格描画テスト
    │   ├── hand_tracking_csv.py    # 手話ジェスチャーのCSVデータ収集
    │   └── realtime_inference.py   # 【速度静止検知付】リアルタイム文字判定デモ
    ├── learning/               # 機械学習モデル (Model)
    │   ├── model.py                # 学習モデル定義（SVM/MLP）
    │   └── train.py                # 分類モデルの訓練スクリプト
```

---

## 2. セットアップ (仮想環境の利用方法)

本プロジェクトでは、他の環境とライブラリの競合を防ぐため、Python の仮想環境 (`.venv`) を使用して開発を行います。

### 2.1 仮想環境の有効化（アクティベート）
ターミナルを開くたびに、最初に以下のコマンドを実行して仮想環境を有効化してください。

```bash
# 仮想環境のアクティベート
source .venv/bin/activate
```
* 有効になると、ターミナルの左端に `(.venv)` と表示されます。

### 2.2 ライブラリのインストール
仮想環境が有効な状態で、以下のコマンドを実行して必要なライブラリを一括インストールします。

```bash
pip install -r requirements.txt
```

---

## 3. 手話（ひらがな：あ〜お）判別の実行手順

### Step 1: データの収集
1. `src/perception/hand_tracking_csv.py` 内の `WORD_NAME` 変数を、記録したいラベル（例: `"a_1"`, `"i_1"` など）に変更します。
2. スクリプトを実行します。
   ```bash
   python3 src/perception/hand_tracking_csv.py
   ```
3. カメラウィンドウが開いたら、手の形を作り、**「スペースキー」** を押して3〜5秒間記録します（手を傾けたり、距離を変えたりしてバリエーションをつけます）。
4. 再度 **「スペースキー」** を押して一時保存します。
5. 各文字（あ〜お）ごとに最低 5〜10 パターン記録します。
6. 作成されたCSVファイルを、すべて **`data/raw/`** ディレクトリに移動します。
   * ※ファイル名が `sign_language_a_1.csv` のようになっていれば、自動的に `a` が正解ラベルとして解釈されます。

### Step 2: データの前処理（一括正規化）
収集したすべての生CSVを正規化し、1つのデータセットにまとめます。
```bash
python3 src/learning/preprocess.py
```
* 実行後、`data/processed/dataset.csv` が生成されます。

### Step 3: モデルの学習（訓練）
```bash
python3 src/learning/train.py
```
* テスト正解率が表示され、`src/learning/weights/hand_sign_model_svm.joblib` が生成されます。

### Step 4: リアルタイム判別デモの起動
学習したモデルを使って、実際に画面上で文字入力を行います。
```bash
python3 src/perception/realtime_inference.py
```
* カメラの前で手を「ピタッ」と止めると、文字が認識され画面にスタック（追記）されていきます。
* `Space` キーでテキストの全消去、`Backspace` キーで最後の1文字を消去します。`q` キーで終了します。