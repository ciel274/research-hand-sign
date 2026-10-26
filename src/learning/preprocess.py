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


def add_extra_geometric_features(normalized_coords_flat):
    """
    すでに正規化（または回転）された63次元の相対座標から、
    56次元の追加幾何特徴量（指先間距離、3D関節角度、手のひら法線、仮想格子補間点）を算出し、
    結合して119次元にして返します。
    """
    coords = normalized_coords_flat.reshape(21, 3)
    if np.all(coords == 0.0):
        return np.zeros(119)
        
    extra_features = []
    fingertips = [4, 8, 12, 16, 20]
    
    # (a) 指先間の相互距離 (10ペア)
    for i in range(len(fingertips)):
        for j in range(i + 1, len(fingertips)):
            dist = np.linalg.norm(coords[fingertips[i]] - coords[fingertips[j]])
            extra_features.append(dist)
            
    # (b) 各指先から手首(0)までの距離 (5次元)
    for tip in fingertips:
        dist = np.linalg.norm(coords[tip] - coords[0])
        extra_features.append(dist)
        
    # (c) 各指先からベースジョイント(2, 5, 9, 13, 17)までの距離 (5次元)
    mcp_joints = [2, 5, 9, 13, 17]
    for tip, mcp in zip(fingertips, mcp_joints):
        dist = np.linalg.norm(coords[tip] - coords[mcp])
        extra_features.append(dist)

    # (d) 3D関節曲げ角度 (15次元)
    triplets = [
        # 親指
        (0, 1, 2), (1, 2, 3), (2, 3, 4),
        # 人差し指
        (0, 5, 6), (5, 6, 7), (6, 7, 8),
        # 中指
        (0, 9, 10), (9, 10, 11), (10, 11, 12),
        # 薬指
        (0, 13, 14), (13, 14, 15), (14, 15, 16),
        # 小指
        (0, 17, 18), (17, 18, 19), (18, 19, 20)
    ]
    for a, b, c in triplets:
        u = coords[a] - coords[b]
        v = coords[c] - coords[b]
        norm_u = np.linalg.norm(u)
        norm_v = np.linalg.norm(v)
        if norm_u > 1e-7 and norm_v > 1e-7:
            cos_angle = np.dot(u, v) / (norm_u * norm_v)
            cos_angle = np.clip(cos_angle, -1.0, 1.0)
            angle = np.arccos(cos_angle)
        else:
            angle = 0.0
        extra_features.append(angle)

    # (e) 手のひらの3D法線ベクトル (3次元)
    v1 = coords[5] - coords[0]
    v2 = coords[17] - coords[0]
    normal = np.cross(v1, v2)
    norm_n = np.linalg.norm(normal)
    if norm_n > 1e-7:
        normal = normal / norm_n
    else:
        normal = np.zeros(3)
    extra_features.extend(list(normal))

    # (f) 仮想格子補間点 (18次元: 5個のベース中点(15次元) + 手のひら中心(3次元))
    for mcp in mcp_joints:
        midpoint = (coords[0] + coords[mcp]) / 2.0
        extra_features.extend(list(midpoint))
    palm_center = (coords[0] + coords[5] + coords[17]) / 3.0
    extra_features.extend(list(palm_center))

    return np.concatenate([normalized_coords_flat, np.array(extra_features)])


def extract_hand_features(hand_coords):
    """
    手の生座標(63次元)から正規化座標と追加の幾何幾何特徴量（距離・3D関節角度・法線・仮想格子点）を抽出し、119次元の特徴ベクトルを生成します。
    """
    norm_coords = normalize_hand_data(hand_coords)
    return add_extra_geometric_features(norm_coords)


def rotate_landmarks_3d(coords_flat, roll, pitch, yaw):
    """
    フラット化された骨格座標 (63次元) を 3D 回転します。
    手首(ID: 0)はすでに(0,0,0)なので、そのまま原点中心で回転します。
    """
    coords = coords_flat.reshape(21, 3)
    if np.all(coords == 0.0):
        return coords.flatten()

    # 回転行列の作成
    Rx = np.array([
        [1, 0, 0],
        [0, np.cos(roll), -np.sin(roll)],
        [0, np.sin(roll), np.cos(roll)]
    ])
    Ry = np.array([
        [np.cos(pitch), 0, np.sin(pitch)],
        [0, 1, 0],
        [-np.sin(pitch), 0, np.cos(pitch)]
    ])
    Rz = np.array([
        [np.cos(yaw), -np.sin(yaw), 0],
        [np.sin(yaw), np.cos(yaw), 0],
        [0, 0, 1]
    ])
    R = Rz @ Ry @ Rx
    
    rotated_coords = coords @ R.T
    return rotated_coords.flatten()


def preprocess_csv(input_path, label, augment_factor=5):
    """
    生データのCSVを読み込み、正規化およびデータ拡張を施したデータリスト（各行にlabelを付与）を返します。
    - augment_factor: 元データ1件につき、微小ランダム回転させたデータを何件生成するか
    """
    if not os.path.exists(input_path):
        print(f"エラー: ファイルが見つかりません {input_path}")
        return []

    df = pd.read_csv(input_path)
    normalized_rows = []

    # 右手と左手の列名リストを作成
    r_cols = [f"R_joint_{i}_{coord}" for i in range(21) for coord in ["x", "y", "z"]]
    l_cols = [f"L_joint_{i}_{coord}" for i in range(21) for coord in ["x", "y", "z"]]

    # 再現性のためのシード値（ファイルパスに基づく）を設定
    seed = abs(hash(os.path.basename(input_path))) % (2**32)
    rng = np.random.default_rng(seed)

    for idx, row in df.iterrows():
        # 右手と左手のデータを取得
        r_data = row[r_cols].values.astype(float)
        l_data = row[l_cols].values.astype(float)

        # それぞれの幾何学特徴量を抽出 (83次元)
        r_feats = extract_hand_features(r_data)
        l_feats = extract_hand_features(l_data)

        timestamp = row["timestamp"]

        # 1. 元の正規化＋特徴量データ（Original）を格納
        original_row = [timestamp] + list(r_feats) + list(l_feats) + [label]
        normalized_rows.append(original_row)

        # 2. データ拡張（微小回転）を適用したデータを格納
        max_angle_rad = 15.0 * np.pi / 180.0

        for i in range(augment_factor):
            # 独立にロール、ピッチ、ヨーをランダム決定 (±15度)
            r_roll, r_pitch, r_yaw = rng.uniform(-max_angle_rad, max_angle_rad, 3)
            l_roll, l_pitch, l_yaw = rng.uniform(-max_angle_rad, max_angle_rad, 3)

            # 回転はベースの63次元座標に対して適用
            r_norm = normalize_hand_data(r_data)
            l_norm = normalize_hand_data(l_data)
            r_norm_aug = rotate_landmarks_3d(r_norm, r_roll, r_pitch, r_yaw)
            l_norm_aug = rotate_landmarks_3d(l_norm, l_roll, l_pitch, l_yaw)

            # スケール・微小ノイズ拡張 (±8%のサイズ変化と微少カメラノイズのシミュレーション)
            if np.any(r_norm_aug != 0.0):
                r_scale = rng.uniform(0.92, 1.08)
                r_noise = rng.normal(0.0, 0.005, size=r_norm_aug.shape)
                r_norm_aug = r_norm_aug * r_scale + r_noise
            if np.any(l_norm_aug != 0.0):
                l_scale = rng.uniform(0.92, 1.08)
                l_noise = rng.normal(0.0, 0.005, size=l_norm_aug.shape)
                l_norm_aug = l_norm_aug * l_scale + l_noise

            # 拡張後の座標から幾何学特徴量を計算して119次元化
            r_feats_aug = add_extra_geometric_features(r_norm_aug)
            l_feats_aug = add_extra_geometric_features(l_norm_aug)

            # タイムスタンプに拡張情報を付与
            aug_timestamp = f"{timestamp}_aug_{i+1}"
            aug_row = [aug_timestamp] + list(r_feats_aug) + list(l_feats_aug) + [label]
            normalized_rows.append(aug_row)

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

    # 幾何学特徴量（距離・3D角度・法線・仮想格子点）を追加した新しいヘッダーを作成 (合計238次元)
    header = ["timestamp"]
    # 右手の座標(63次元) + 幾何特徴量(56次元)
    for i in range(21):
        header.extend([f"R_joint_{i}_x", f"R_joint_{i}_y", f"R_joint_{i}_z"])
    for i in range(56):
        header.append(f"R_extra_{i}")
    # 左手の座標(63次元) + 幾何特徴量(56次元)
    for i in range(21):
        header.extend([f"L_joint_{i}_x", f"L_joint_{i}_y", f"L_joint_{i}_z"])
    for i in range(56):
        header.append(f"L_extra_{i}")
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
