import os
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report
import sys
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from model import get_model, save_model

def train_model(dataset_path="data/processed/dataset.csv", model_output_dir="src/learning/weights", model_type="svm"):
    """
    データセットを読み込んで分類モデルを訓練し、テストデータで精度を評価して保存します。
    """
    if not os.path.exists(dataset_path):
        print(f"エラー: 学習用データセットが見つかりません: {dataset_path}")
        print("あらかじめ data/raw 配下にCSVを収集し、preprocess.py を実行してデータセットを作成してください。")
        return
        
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

if __name__ == "__main__":
    # デフォルトのデータセットパスで訓練を実行
    train_model()
