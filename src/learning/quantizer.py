import os
import joblib
import numpy as np
import pandas as pd
from sklearn.cluster import KMeans

class PoseQuantizer:
    """
    手の関節座標(126次元)を離散的な代表ポーズ（トークンID）に変換する量子化器。
    K-Means クラスタリングを用いて代表ポーズ（コードブック）を決定します。
    """
    def __init__(self, n_clusters=64, random_state=42):
        self.n_clusters = n_clusters
        self.random_state = random_state
        self.kmeans = None
        self.cluster_centers_ = None

    def fit(self, X):
        """
        手の座標データの集合 X (N, 126) から、n_clusters 個の代表ポーズを学習します。
        """
        print(f"ポーズの量子化器を訓練中... (クラスタ数: {self.n_clusters})")
        self.kmeans = KMeans(n_clusters=self.n_clusters, random_state=self.random_state, n_init='auto')
        self.kmeans.fit(X)
        self.cluster_centers_ = self.kmeans.cluster_centers_
        return self

    def tokenize(self, X):
        """
        126次元の座標 X を、最も近い代表ポーズのトークンID（0からn_clusters-1の整数）に変換します。
        X: 2次元のnumpy配列 (N, 126)、または1次元の配列 (126,)
        """
        if self.kmeans is None:
            raise ValueError("KMeansがまだ学習されていません。fit()を先に実行してください。")
        
        # 1次元配列の場合はリシェイプ
        if len(X.shape) == 1:
            X = X.reshape(1, -1)
            
        labels = self.kmeans.predict(X)
        return labels

    def detokenize(self, tokens):
        """
        トークンID（整数）の配列から、対応する 126次元の代表ポーズ座標を復元します。
        """
        if self.cluster_centers_ is None:
            raise ValueError("KMeansがまだ学習されていないため、復元できません。")
            
        return self.cluster_centers_[tokens]

    def save(self, filepath):
        """
        学習した量子化モデルをディスクに保存します。
        """
        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        joblib.dump({
            'n_clusters': self.n_clusters,
            'random_state': self.random_state,
            'kmeans': self.kmeans,
            'cluster_centers_': self.cluster_centers_
        }, filepath)
        print(f"量子化モデルを保存しました: {filepath}")

    def load(self, filepath):
        """
        ディスクから学習済みの量子化モデルをロードします。
        """
        data = joblib.load(filepath)
        self.n_clusters = data['n_clusters']
        self.random_state = data['random_state']
        self.kmeans = data['kmeans']
        self.cluster_centers_ = data['cluster_centers_']
        return self

if __name__ == "__main__":
    # モジュールの動作検証用メイン処理
    BASE_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    DATASET_PATH = os.path.join(BASE_DIR, "data", "processed", "dataset.csv")
    MODEL_PATH = os.path.join(BASE_DIR, "src", "learning", "weights", "pose_quantizer.joblib")

    if not os.path.exists(DATASET_PATH):
        print(f"エラー: データセットが見つかりません: {DATASET_PATH}")
        print("あらかじめデータをWeb UIで収録し、「一括前処理」を行ってから実行してください。")
    else:
        # データセットの読み込み
        df = pd.read_csv(DATASET_PATH)
        feature_cols = [col for col in df.columns if col not in ['timestamp', 'label']]
        X = df[feature_cols].values

        print(f"読み込みデータ件数: {len(X)} フレーム")

        # 量子化器の訓練 (クラスタ数はデータの規模に合わせて調整可能)
        n_clusters = min(64, len(X))
        quantizer = PoseQuantizer(n_clusters=n_clusters)
        quantizer.fit(X)

        # モデルの保存
        quantizer.save(MODEL_PATH)

        # トークンIDへの変換と復元の検証（再構成誤差の評価）
        tokens = quantizer.tokenize(X)
        X_reconstructed = quantizer.detokenize(tokens)

        # 平均二乗誤差(MSE)の算出
        mse = np.mean((X - X_reconstructed) ** 2)
        print(f"\n【検証結果】量子化による再構成誤差 (Mean Squared Error): {mse:.6f}")
        print(f"トークン系列 of first 10 items: {tokens[:10]}")
        print("離散化テストが成功しました。")
