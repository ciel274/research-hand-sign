# システム設計書：手話VLA学習システム

本ドキュメントは、MediaPipeを用いた手のトラッキングから、機械学習による手話理解、そしてROS2を介したロボット制御までの全体設計と各モジュール間のデータ接続仕様を定義します。

---

## 1. 全体アーキテクチャ

システムは「感知 (Vision)」「理解・推論 (Language/Model)」「実行 (Action)」の3層レイヤーで構成されます。

```mermaid
graph TD
    subgraph 1. Vision (Perception)
        Cam[RGBカメラ] -->|フレーム画像| CV[OpenCV]
        CV -->|画像処理| MP[MediaPipe Hands]
        MP -->|骨格座標 126dim| FE[特徴量抽出 & CSV保存]
    end

    subgraph 2. Model (Learning & Inference)
        FE -->|CSVデータセット| PL[前処理 / 座標の正規化・相対化]
        PL -->|入力テンソル| ML[学習モデル: LSTM / Transformer]
        ML -->|意味識別 / 軌道推論| OUT[アクション出力定義]
    end

    subgraph 3. Action (Control)
        OUT -->|制御コマンド| ROS[ROS2 Control Node]
        ROS -->|cmd_velトピック| TS[Turtlesim / Gazebo]
    end

    style Cam fill:#f9f,stroke:#333,stroke-width:2px
    style ML fill:#bbf,stroke:#333,stroke-width:2px
    style TS fill:#bfb,stroke:#333,stroke-width:2px
```

---

## 2. モジュール間インターフェース設計

### 2.1 データフォーマット（CSV構造）
`hand_tracking_csv.py` によって記録されるCSVデータは、時系列の各フレームごとに1行として書き込まれます。

- **ファイル名規則**: `sign_language_{WORD_NAME}.csv`
- **データ列（カラム）**:
  - `timestamp`: 浮動小数点数（`time.time()`値）
  - 右手の座標（`R_joint_0_x` 〜 `R_joint_20_z`、計 63 カラム）
  - 左手の座標（`L_joint_0_x` 〜 `L_joint_20_z`、計 63 カラム）
  - 合計: 1 + 63 + 63 = 127列

```csv
timestamp,R_joint_0_x,R_joint_0_y,R_joint_0_z,...,L_joint_20_x,L_joint_20_y,L_joint_20_z
1719000000.123,0.523,0.612,-0.05,...,0.412,0.590,-0.02
1719000000.156,0.525,0.615,-0.04,...,0.410,0.592,-0.02
```

### 2.2 前処理 (Preprocessing) の設計
手の大きさや、カメラと被写体（人物）の距離に依存しない頑健な学習を行うため、学習モデルに入力する前に以下の前処理を適用します。

1. **手首原点化（Relative Translation）**:
   - 右手のすべての関節 `(x, y, z)` から、手首（`R_joint_0`）の座標を減算し、手首を原点 `(0, 0, 0)` とする相対座標に変換。
   - 左手も同様に `L_joint_0` を原点とする。
2. **スケール正規化（Scale Normalization）**:
   - 手首から中指の付け根（`joint_9`）までの長さを算出し、その距離が `1.0` になるようにすべての座標値をスケーリング（除算）。これにより、カメラとの距離のばらつきを吸収。
3. **フレーム補間 / パディング (Sequence Padding)**:
   - 手話動作にかかる時間（フレーム数）は試行によって異なるため、すべてのシークエンスの長さを固定値（例: 50フレーム）にパディング（ゼロ埋め）、またはリサンプリングして統一する。

---

## 3. 推論モデル設計

極めて少ない（Few-shot）学習データに対して適応するため、以下の2つのアプローチを検討します。

### アプローチA：分類アプローチ（Classification + Action Mapping）
1. **モデルの役割**: 手話の時系列データから、定義された意味ラベル（`forward`, `backward`, `turn_left`, `turn_right`, `stop`）を分類する。
2. **モデル構成**: LSTM または 1次元CNN ＋ Fully Connected Layer。
3. **アクションマッピング**: 分類結果に基づき、対応する固定の速度指令（例：`forward` ➔ `linear.x = 2.0, angular.z = 0.0`）をルックアップテーブルで生成し、ROS2に送信する。

### アプローチB：エンドツーエンド回帰アプローチ（End-to-End Regression） [発展]
1. **モデルの役割**: 手の座標系列から、直接ロボットの目標速度 `linear.x` や `angular.z` を実数値として回帰予測する。
2. **モデル構成**: Transformer-Decoder または 連続軌道生成用のRNN。
3. **アクションマッピング**: 手の動く速さや移動量に応じて、亀の速度がダイナミックに変化する（例: 手を前に突き出すスピードに比例して急加速するなど）。

---

## 4. ROS2 制御ノードの設計

### 4.1 トピック通信仕様
Pythonで実装する ROS2 推論制御ノード（`src/control/ros2_controller.py`）は、モデルの推論結果を購読・または自ら推論を行い、ロボットシミュレータに指令を送ります。

- **パブリッシャー (Publisher)**:
  - トピック名: `/turtle1/cmd_vel` （Turtlesimの場合）
  - メッセージ型: `geometry_msgs/msg/Twist`
  - 送信頻度: 推論フレームごと（約 10Hz 〜 30Hz）

- **Twist メッセージパラメータのマッピング（アプローチAの場合）**:
  - `FORWARD`: `linear.x = 2.0`, `angular.z = 0.0`
  - `BACKWARD`: `linear.x = -2.0`, `angular.z = 0.0`
  - `LEFT`: `linear.x = 0.0`, `angular.z = 2.0`
  - `RIGHT`: `linear.x = 0.0`, `angular.z = -2.0`
  - `STOP`: `linear.x = 0.0`, `angular.z = 0.0`
