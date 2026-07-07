import os
import pandas as pd
import numpy as np
import json
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report
import sys
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from model import get_model, save_model

def train_model(dataset_path="data/processed/dataset.csv", model_output_dir="src/learning/weights", model_type="svm"):
    """
    データセットを読み込んで分類モデルを訓練し、テストデータで精度を評価して保存します。
    さらに、可視化用のPCA（次元削減）データとクラス別認識精度を計算して JSON に保存します。
    """
    if not os.path.exists(dataset_path):
        print(f"エラー: 学習用データセットが見つかりません: {dataset_path}")
        print("あらかじめ data/raw 配下にCSVを収集し、preprocess.py を実行してデータセットを作成してください。")
        return
        
    # モードの判定 (ファイル名から hiragana または words を判別)
    mode = "hiragana"
    if "words" in dataset_path:
        mode = "words"

    # データフレームの読み込み
    df = pd.read_csv(dataset_path)
    print(f"データセットを読み込みました。総フレーム数: {len(df)}")
    
    # タイムスタンプとラベル以外のすべての特徴量(126次元)を X、ラベルを y とする
    feature_cols = [col for col in df.columns if col not in ['timestamp', 'label']]
    X = df[feature_cols].values
    y = df['label'].values
    
    # 各クラスのデータ数（フレーム数）の表示
    class_counts = df['label'].value_counts()
    print("\nクラス別データ内訳（フレーム数）:")
    for cls_name, count in class_counts.items():
        print(f"  - '{cls_name}': {count} フレーム")
        
    if len(class_counts) < 2:
        print("\n[エラー] 登録されているクラス（手話動作）が1つしかありません。")
        print("判別を行うには、最低2つ以上の異なる手話（例: 'a' と 'i'）のCSVを収集する必要があります。")
        return
        
    # データを訓練用(80%)とテスト用(20%)に分割
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    
    print(f"\nモデルの訓練を開始します ({model_type.upper()})...")
    model = get_model(model_type=model_type)
    model.fit(X_train, y_train)
    print("モデルの訓練が完了しました。")
    
    # テストデータでの評価
    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    print(f"\n【評価結果】テスト正解率 (Accuracy): {accuracy * 100:.2f}%")
    
    print("\n詳細評価レポート (Classification Report):")
    print(classification_report(y_test, y_pred))
    
    # モデルの保存
    os.makedirs(model_output_dir, exist_ok=True)
    model_path = os.path.join(model_output_dir, f"hand_sign_model_{model_type}.joblib")
    save_model(model, model_path)

    # --- 追加: 可視化用データの計算と保存 ---
    print("\n可視化用データの解析中...")
    
    # 1. クラス別認識精度の算出
    unique_labels = np.unique(y)
    class_accuracies = {}
    for label in unique_labels:
        idx_test = (y_test == label)
        if np.sum(idx_test) > 0:
            acc = accuracy_score(y_test[idx_test], y_pred[idx_test])
            class_accuracies[label] = float(acc)
        else:
            class_accuracies[label] = 1.0  # テストデータに含まれなかった場合は1.0とする

    # 2. PCA (主成分分析) による 2D 写像の計算 (126次元 ➔ 2次元)
    pca_points = []
    try:
        from sklearn.decomposition import PCA
        pca = PCA(n_components=2)
        X_pca = pca.fit_transform(X)
        
        # 全データ点から最大400点をランダムサンプリングしてフロント表示用に軽量化
        rng = np.random.default_rng(42)
        sample_size = min(len(X), 400)
        sample_indices = rng.choice(len(X), sample_size, replace=False)
        
        for idx in sample_indices:
            pca_points.append({
                "x": float(X_pca[idx, 0]),
                "y": float(X_pca[idx, 1]),
                "label": str(y[idx])
            })
        print(f"PCA計算完了 (サンプル点: {len(pca_points)}点)")
    except Exception as e:
        print(f"PCA計算エラー: {e}")

    # 3. 統計JSONとして保存
    stats_data = {
        "accuracy": float(accuracy),
        "class_accuracies": class_accuracies,
        "pca_points": pca_points
    }
    
    stats_output_path = os.path.join(os.path.dirname(dataset_path), f"train_stats_{mode}.json")
    try:
        with open(stats_output_path, "w", encoding="utf-8") as f:
            json.dump(stats_data, f, indent=4, ensure_ascii=False)
        print(f"可視化統計データを保存しました: {stats_output_path}")
    except Exception as e:
        print(f"統計データ保存エラー: {e}")

if __name__ == "__main__":
    # デフォルトのデータセットパスで訓練を実行
    train_model()
