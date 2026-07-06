import joblib
from sklearn.svm import SVC
from sklearn.neural_network import MLPClassifier

def get_model(model_type="svm"):
    """
    分類器モデルを生成して返します。
    - 'svm': サポートベクターマシン (少数のサンプルで境界を作るのに適しています)
    - 'mlp': 多層パーセプトロン (全結合ニューラルネットワーク)
    """
    if model_type == "svm":
        # probability=True にすることで、判定確率（Confidence）を取得できるようにします
        return SVC(kernel='rbf', probability=True, C=10.0, gamma='scale', random_state=42)
    elif model_type == "mlp":
        return MLPClassifier(hidden_layer_sizes=(64, 32), max_iter=500, activation='relu', random_state=42)
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
