import joblib
from sklearn.svm import SVC
from sklearn.neural_network import MLPClassifier

def get_model(model_type="svm"):
    """
    分類器モデルを生成して返します。
    - 'svm': サポートベクターマシン (RBFカーネル)
    - 'rf':  ランダムフォレスト (アンサンブル決定木)
    - 'mlp': 多層パーセプトロン (全結合ニューラルネットワーク)
    - 'knn': K-近傍法 (K=3)
    """
    if model_type == "svm":
        return SVC(kernel='rbf', probability=True, C=10.0, gamma='scale', random_state=42)
    elif model_type == "rf":
        from sklearn.ensemble import RandomForestClassifier
        return RandomForestClassifier(n_estimators=100, random_state=42)
    elif model_type == "mlp":
        return MLPClassifier(hidden_layer_sizes=(128, 64), max_iter=500, activation='relu', random_state=42)
    elif model_type == "knn":
        from sklearn.neighbors import KNeighborsClassifier
        return KNeighborsClassifier(n_neighbors=3)
    else:
        raise ValueError(f"サポートされていないモデルタイプです: {model_type}")

def save_model(model, filepath):
    """
    学習済みモデルをディスクに保存します。
    """
    joblib.dump(model, filepath)
    print(f"モデルを保存しました: {filepath}")

def load_model(filepath):
    """
    ディスクから学習済みモデルを読み込みます。
    """
    return joblib.load(filepath)
