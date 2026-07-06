import os
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

def preprocess_csv(input_path, output_path):
    """
    生データのCSVを読み込み、正規化したCSVを出力します。
    """
    if not os.path.exists(input_path):
        print(f"エラー: ファイルが見つかりません {input_path}")
        return
        
    df = pd.read_csv(input_path)
    normalized_rows = []
    
    # 右手と左手の列名リストを作成
    r_cols = [f"R_joint_{i}_{coord}" for i in range(21) for coord in ['x', 'y', 'z']]
    l_cols = [f"L_joint_{i}_{coord}" for i in range(21) for coord in ['x', 'y', 'z']]
    
    for idx, row in df.iterrows():
        timestamp = row['timestamp']
        
        # 右手と左手のデータを取得
        r_data = row[r_cols].values.astype(float)
        l_data = row[l_cols].values.astype(float)
        
        # それぞれ正規化
        r_norm = normalize_hand_data(r_data)
        l_norm = normalize_hand_data(l_data)
        
        # タイムスタンプと結合
        new_row = [timestamp] + list(r_norm) + list(l_norm)
        normalized_rows.append(new_row)
        
    # 新しいデータフレームを作成して保存
    new_df = pd.DataFrame(normalized_rows, columns=df.columns)
    
    # 出力先フォルダがない場合は作成
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    new_df.to_csv(output_path, index=False)
    print(f"正規化完了: {input_path} -> {output_path}")

if __name__ == "__main__":
    # 使用例：
    # preprocess_csv("data/raw/sign_language_hello.csv", "data/processed/sign_language_hello_normalized.csv")
    pass
