import os
import glob
import json
import numpy as np
import pandas as pd
import sys

# 相対インポートを有効にするためにパスを追加
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from preprocess import normalize_hand_data
from quantizer import PoseQuantizer

# 日本語ひらがなへの逆マッピング（instructionの文面生成用）
ROMAJI_TO_KANA = {
    "a": "あ", "i": "い", "u": "う", "e": "え", "o": "お",
    "ka": "か", "ki": "き", "ku": "く", "ke": "け", "ko": "こ",
    "sa": "さ", "si": "し", "su": "す", "se": "せ", "so": "そ",
    "ta": "た", "ti": "ち", "tu": "つ", "te": "て", "to": "と",
    "na": "な", "ni": "に", "nu": "ぬ", "ne": "ね", "no": "の",
    "ha": "は", "hi": "ひ", "hu": "ふ", "he": "へ", "ho": "ほ",
    "ma": "ま", "mi": "み", "mu": "む", "me": "め", "mo": "も",
    "ya": "や", "yu": "ゆ", "yo": "よ",
    "ra": "ら", "ri": "り", "ru": "る", "re": "れ", "ro": "ろ",
    "wa": "わ", "wo": "を", "nn": "ん"
}

def export_vla_datasets():
    BASE_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    RAW_DIR = os.path.join(BASE_DIR, "data", "raw")
    PROCESSED_DIR = os.path.join(BASE_DIR, "data", "processed")
    QUANTIZER_PATH = os.path.join(BASE_DIR, "src", "learning", "weights", "pose_quantizer.joblib")

    # 出力ファイルパス
    continuous_jsonl_path = os.path.join(PROCESSED_DIR, "vla_continuous.jsonl")
    discrete_jsonl_path = os.path.join(PROCESSED_DIR, "vla_discrete.jsonl")

    # CSVファイルの探索
    search_path = os.path.join(RAW_DIR, "sign_language_*.csv")
    csv_files = glob.glob(search_path)

    if not csv_files:
        print("エラー: 処理対象の生CSVファイルが data/raw/ 内に見つかりません。")
        return

    print(f"検出された生CSVファイル数: {len(csv_files)}")

    # 1. すべてのデータをロードし、フレーム単位で正規化する
    # (量子化器の事前学習、およびデータセット作成用)
    episodes_data = [] # 各エピソード（動画シーケンス）ごとのデータ
    all_normalized_frames = [] # クラスタ学習用の全フレーム平面配列

    # 列名リストの生成 (右手63次元 + 左手63次元)
    r_cols = [f"R_joint_{i}_{coord}" for i in range(21) for coord in ["x", "y", "z"]]
    l_cols = [f"L_joint_{i}_{coord}" for i in range(21) for coord in ["x", "y", "z"]]

    for file_path in csv_files:
        file_name = os.path.basename(file_path)
        base_name = file_name.replace("sign_language_", "").replace(".csv", "")
        parts = base_name.split("_")
        
        # ファイル名からラベルを抽出 (例: sign_language_a_1.csv -> 'a')
        if len(parts) > 1 and parts[-1].isdigit():
            label_romaji = "_".join(parts[:-1])
        else:
            label_romaji = base_name
            
        label_kana = ROMAJI_TO_KANA.get(label_romaji, label_romaji)

        # CSVの読み込みとフレームごとの正規化
        df = pd.read_csv(file_path)
        episode_frames = []

        for idx, row in df.iterrows():
            r_data = row[r_cols].values.astype(float)
            l_data = row[l_cols].values.astype(float)

            # 正規化
            r_norm = normalize_hand_data(r_data)
            l_norm = normalize_hand_data(l_data)
            
            # 126次元に結合
            combined = np.concatenate([r_norm, l_norm])
            episode_frames.append(combined)
            all_normalized_frames.append(combined)

        episodes_data.append({
            "romaji": label_romaji,
            "kana": label_kana,
            "frames": np.array(episode_frames),
            "file": file_name
        })

    # 2. 量子化器 (Quantizer) のロードまたは新規学習
    quantizer = PoseQuantizer(n_clusters=min(64, len(all_normalized_frames)))
    if os.path.exists(QUANTIZER_PATH):
        print(f"学習済みの量子化器をロードします: {QUANTIZER_PATH}")
        quantizer.load(QUANTIZER_PATH)
    else:
        print("量子化モデルが見つからないため、現在の全データで新規に学習します。")
        quantizer.fit(np.array(all_normalized_frames))
        quantizer.save(QUANTIZER_PATH)

    # 3. アプローチA (連続座標系列) & アプローチB (離散トークン系列) データセットの生成
    continuous_records = []
    discrete_records = []

    for ep in episodes_data:
        # 指示テキスト (プロンプト)
        instruction = f"ひらがなの『{ep['kana']}』を手話（指文字）で表現してください。"

        # --- アプローチA (連続値エクスポート) ---
        # 各フレームの126次元ベクトルをリストにする
        actions_list = ep["frames"].tolist()
        continuous_records.append({
            "instruction": instruction,
            "image": "dummy_hand_sign_start.jpg", # VLAの画像入力部分のプレースホルダー
            "actions": actions_list
        })

        # --- アプローチB (離散値エクスポート) ---
        # 1フレームずつ量子化
        token_ids = quantizer.tokenize(ep["frames"])
        # トークンIDの配列を文字列シーケンスに変換 (例: "<pose_12> <pose_45>")
        token_str = " ".join([f"<pose_{tid}>" for tid in token_ids])
        
        discrete_records.append({
            "instruction": instruction,
            "output": token_str
        })

    # 4. JSONLファイルへの書き出し
    os.makedirs(PROCESSED_DIR, exist_ok=True)

    with open(continuous_jsonl_path, "w", encoding="utf-8") as f:
        for rec in continuous_records:
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")
            
    with open(discrete_jsonl_path, "w", encoding="utf-8") as f:
        for rec in discrete_records:
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")

    print("\nVLAデータセットのエクスポートが完了しました！")
    print(f" - [アプローチA] 連続座標シーケンス: {continuous_jsonl_path} ({len(continuous_records)} 件)")
    print(f" - [アプローチB] 離散トークンシーケンス: {discrete_jsonl_path} ({len(discrete_records)} 件)")

if __name__ == "__main__":
    export_vla_datasets()
