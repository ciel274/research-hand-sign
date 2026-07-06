# research-hand-sign

小規模データ（Few-shot / Sample-efficient）での効率的学習による手話アクションの獲得・創出をする、高効率なVLA（Vision-Language-Action）学習フレームワークの研究プロジェクト。

---

## 1. プロジェクト構造

```text
research-hand-sign/
├── README.md                   # プロジェクト概要・実行手順
├── requirements.txt            # Pythonライブラリ依存関係
├── pixi.toml                   # ROS2環境管理用 (pixi)
│
├── docs/                       # ドキュメント類
│   ├── requirements_definition.md  # 要件定義書
│   └── system_architecture.md       # システム設計書
│
├── data/                       # 収集した手話データ（CSV）の保存先
│   ├── raw/
│   └── processed/
│
└── src/                        # ソースコード
    ├── perception/             # 視覚・骨格トラッキング (Vision)
    │   ├── hand_tracking.py        # リアルタイム骨格描画テスト
    │   └── hand_tracking_csv.py    # 手話ジェスチャーのCSVデータ収集
    ├── learning/               # 機械学習モデル (Model)
    └── control/                # ロボット制御・ROS2連携 (Action)
```

---

## 2. セットアップ

### 2.1 Pythonの依存ライブラリのインストール
以下のコマンドで、必要なPythonパッケージをインストールします。

```bash
pip install -r requirements.txt
```

### 2.2 ROS2 Humble のインストール (macOS用)
pixiを使用して、ユーザー権限内でROS2環境を初期化・構築します。

```bash
# pixiのインストール (未導入の場合)
curl -fsSL https://pixi.sh/install.sh | sh
source ~/.zshrc

# ROS2環境の初期化と依存パッケージ追加
pixi init ros2 -c conda-forge -c robostack-humble
cd ros2
pixi add ros-humble-desktop ros-humble-turtlesim
```

---

## 3. 使用方法

### 3.1 骨格トラッキングのテスト
MediaPipeを用いた手のトラッキングが正常に動作するかテストします。

```bash
python src/perception/hand_tracking.py
```
* カメラ映像が表示され、手の骨格が検出されます。`q` キーで終了します。

### 3.2 手話ジェスチャーのデータ収集 (CSV出力)
学習用のデータセットを作成するために、動作データを収集します。

1. `src/perception/hand_tracking_csv.py` 内の `WORD_NAME` 変数を、記録したい単語（例: `forward`, `stop` 等）に変更します。
2. スクリプトを実行します。
   ```bash
   python src/perception/hand_tracking_csv.py
   ```
3. カメラウィンドウが開いたら、**「スペースキー」** を押すと録画（CSVへの書き込み）が始まります。
4. 手話アクションを行ったら、再度 **「スペースキー」** を押して録画を停止（一時保存）します。
5. 最低3回以上のレコーディングを行い、データセットを作成します。`q` キーでプログラムを終了します。
6. 生成されたCSVファイルは、必要に応じて `data/raw/` ディレクトリに移動させて保管します。

### 3.3 タートルシム (Turtlesim) の起動
ROS2のトピック通信を確認するためのシミュレータを起動します。

**ターミナル 1 (Turtlesim ノードの起動)**:
```bash
cd ros2
pixi shell
ros2 run turtlesim turtlesim_node
```

**ターミナル 2 (キーボードによる手動制御テスト)**:
```bash
cd ros2
pixi shell
ros2 run turtlesim turtle_teleop_key
```