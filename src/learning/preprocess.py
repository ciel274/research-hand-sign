import os
import glob
import numpy as np
import pandas as pd


def normalize_hand_data(hand_coords):
    """
    1つの手(21箇所 * 3次元 = 63次元)の座標データを正規化します。
    - 手首(ID: 0)を原点(0, 0, 0)とする相対座標に変換
    - 手首(ID: 0)から中指の付け根(ID: 9)の距離が1.0になるようにスケール変換
    """
    # 21箇所 × 3次元(x, y, z) の行列に変換
    coords = hand_coords.reshape(21, 3)

    # 手が検出されていない(すべて0.0)の場合はそのまま返す
    if np.all(coords == 0.0):
        return coords.flatten()

    # 1. 手首を基準(原点)にする
    wrist = coords[0]
    relative_coords = coords - wrist

    # 2. スケール正規化 (手首から中指の付け根までの距離で割る)
    mcp_joint = relative_coords[9]
    distance = np.linalg.norm(mcp_joint)

    if distance > 0.0:
        normalized_coords = relative_coords / distance
    else:
        normalized_coords = relative_coords

    return normalized_coords.flatten()


def preprocess_csv(input_path, label):
    """
    生データのCSVを読み込み、正規化を施したデータリスト（各行にlabelを付与）を返します。
    """
    if not os.path.exists(input_path):
        print(f"エラー: ファイルが見つかりません {input_path}")
        return []

    df = pd.read_csv(input_path)
    normalized_rows = []

    # 右手と左手の列名リストを作成
    r_cols = [f"R_joint_{i}_{coord}" for i in range(21) for coord in ["x", "y", "z"]]
    l_cols = [f"L_joint_{i}_{coord}" for i in range(21) for coord in ["x", "y", "z"]]

    for idx, row in df.iterrows():
        # 右手と左手のデータを取得
        r_data = row[r_cols].values.astype(float)
        l_data = row[l_cols].values.astype(float)

        # それぞれ正規化
        r_norm = normalize_hand_data(r_data)
        l_norm = normalize_hand_data(l_data)

        # タイムスタンプ、正規化座標、ラベルを結合
        new_row = [row["timestamp"]] + list(r_norm) + list(l_norm) + [label]
        normalized_rows.append(new_row)

    return normalized_rows


def preprocess_all_raw_data(
    raw_dir="data/raw", output_csv="data/processed/dataset.csv"
):
    """
    raw_dir 内のすべての `sign_language_*.csv` を走査し、
    ファイル名からラベルを抽出して一括前処理を行い、1つのCSVファイルとして保存します。

    ファイル名形式の例:
    - `sign_language_a_1.csv` -> ラベル「a」
    - `sign_language_i.csv`   -> ラベル「i」
    """
    search_path = os.path.join(raw_dir, "sign_language_*.csv")
    csv_files = glob.glob(search_path)

    if not csv_files:
        print(
            f"お知らせ: {raw_dir} 内に sign_language_*.csv ファイルが見当たらなかったため、処理をスキップしました。データを data/raw に配置してから実行してください。"
        )
        return

    all_data = []

    # 正規化データのヘッダーを作成
    header = ["timestamp"]
    for i in range(21):
        header.extend([f"R_joint_{i}_x", f"R_joint_{i}_y", f"R_joint_{i}_z"])
    for i in range(21):
        header.extend([f"L_joint_{i}_x", f"L_joint_{i}_y", f"L_joint_{i}_z"])
    header.append("label")

    for file_path in csv_files:
        file_name = os.path.basename(file_path)

        # ファイル名からラベル部分を抽出
        # 例: "sign_language_a_1.csv" -> "a_1" -> 「_数字」があれば除外して「a」にする
        base_name = file_name.replace("sign_language_", "").replace(".csv", "")
        parts = base_name.split("_")

        if len(parts) > 1 and parts[-1].isdigit():
            label = "_".join(parts[:-1])  # 末尾の数値を削る (例: a_1 -> a)
        else:
            label = base_name

        print(f"解析中: {file_name} ➔ ラベル判別: '{label}'")
        rows = preprocess_csv(file_path, label)
        all_data.extend(rows)

    if all_data:
        # 保存先フォルダを作成して保存
        os.makedirs(os.path.dirname(output_csv), exist_ok=True)
        dataset_df = pd.DataFrame(all_data, columns=header)
        dataset_df.to_csv(output_csv, index=False)
        print(
            f"データセット作成完了: 全 {len(dataset_df)} フレームの正規化データを {output_csv} に統合しました。"
        )
    else:
        print("エラー: 処理できる有効なデータがありませんでした。")


if __name__ == "__main__":
    # このスクリプトが直接実行された場合、raw フォルダ内の全データを一括処理します
    preprocess_all_raw_data()
