import asyncio
import cv2
import glob
import os
import time
import numpy as np
import mediapipe as mp
from fastapi import FastAPI, Response
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel
import threading

# 前処理と学習スクリプトをインポート
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from learning.preprocess import normalize_hand_data, preprocess_all_raw_data
from learning.model import load_model
from learning.train import train_model

app = FastAPI(title="Hand Sign Web Dashboard")

# 各種ディレクトリパスの定義
BASE_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
RAW_DATA_DIR = os.path.join(BASE_DIR, "data", "raw")
PROCESSED_DATA_DIR = os.path.join(BASE_DIR, "data", "processed")
MODEL_PATH = os.path.join(BASE_DIR, "src", "learning", "weights", "hand_sign_model_svm.joblib")
TEMPLATE_PATH = os.path.join(BASE_DIR, "src", "ui", "templates", "index.html")

os.makedirs(RAW_DATA_DIR, exist_ok=True)
os.makedirs(PROCESSED_DATA_DIR, exist_ok=True)

# グローバルな状態管理変数
state = {
    "is_recording": False,
    "active_label": None,
    "current_file_path": None,
    "velocity": 0.0,
    "pred_class": "--",
    "confidence": 0.0,
    "is_stable": False,
    "text_buffer": "",
}

# カメラ＆MediaPipe制御用のクラス
class CameraManager:
    def __init__(self):
        self.cap = None
        self.running = False
        self.thread = None
        self.latest_frame = None
        self.mp_hands = mp.solutions.hands
        self.mp_drawing = mp.solutions.drawing_utils
        self.model = None
        self.load_inference_model()
        self.last_prediction = None
        self.last_input_time = 0
        self.stability_counter = 0

    def load_inference_model(self):
        if os.path.exists(MODEL_PATH):
            try:
                self.model = load_model(MODEL_PATH)
                print(f"モデルを読み込みました: {MODEL_PATH}")
            except Exception as e:
                print(f"モデルロードエラー: {e}")
        else:
            self.model = None

    def start(self):
        if not self.running:
            # 利用可能なカメラIDを自動探索 (0, 1, 2)
            self.cap = None
            for cam_id in [0, 1, 2]:
                cap = cv2.VideoCapture(cam_id)
                if cap.isOpened():
                    self.cap = cap
                    print(f"カメラの起動に成功しました (カメラID: {cam_id})")
                    break
                else:
                    cap.release()
            
            if self.cap is None or not self.cap.isOpened():
                print("エラー: 利用可能なカメラデバイスが見つかりませんでした。")
                self.running = False
                return

            self.running = True
            self.thread = threading.Thread(target=self.loop, daemon=True)
            self.thread.start()

    def stop(self):
        self.running = False
        if self.cap:
            self.cap.release()

    def loop(self):
        # 速度計測・安定判定用
        prev_raw_coords = None
        VELOCITY_THRESHOLD = 0.008
        STABILITY_FRAMES = 8
        CONFIDENCE_THRESHOLD = 0.85
        INPUT_COOLDOWN = 1.0

        with self.mp_hands.Hands(
            max_num_hands=2,
            min_detection_confidence=0.7,
            min_tracking_confidence=0.7
        ) as hands:
            while self.running:
                success, frame = self.cap.read()
                if not success:
                    time.sleep(0.03)
                    continue

                # 画面反転、RGB変換
                frame = cv2.cvtColor(cv2.flip(frame, 1), cv2.COLOR_BGR2RGB)
                frame.flags.writeable = False
                results = hands.process(frame)
                frame.flags.writeable = True
                frame = cv2.cvtColor(frame, var_name := cv2.COLOR_RGB2BGR)

                right_hand_data = [0.0] * 63
                left_hand_data = [0.0] * 63
                has_hands = False

                if results.multi_hand_landmarks and results.multi_handedness:
                    has_hands = True
                    for hand_landmarks, handedness in zip(results.multi_hand_landmarks, results.multi_handedness):
                        self.mp_drawing.draw_landmarks(frame, hand_landmarks, self.mp_hands.HAND_CONNECTIONS)
                        
                        label = handedness.classification[0].label
                        temp_coords = []
                        for landmark in hand_landmarks.landmark:
                            temp_coords.extend([landmark.x, landmark.y, landmark.z])
                        
                        if label == "Right":
                            right_hand_data = temp_coords
                        elif label == "Left":
                            left_hand_data = temp_coords

                # 126次元の生座標
                current_raw_coords = np.array(right_hand_data + left_hand_data)

                # --- 速度（移動量）計算 ---
                velocity = 0.0
                if has_hands and prev_raw_coords is not None:
                    diff = current_raw_coords - prev_raw_coords
                    diff_reshaped = diff.reshape(42, 3)
                    valid_joints = np.any(prev_raw_coords.reshape(42, 3) != 0.0, axis=1)
                    if np.sum(valid_joints) > 0:
                        distances = np.linalg.norm(diff_reshaped[valid_joints], axis=1)
                        velocity = np.mean(distances)

                state["velocity"] = float(velocity)
                prev_raw_coords = current_raw_coords if has_hands else None

                # --- 静止判定ロジック ---
                current_time = time.time()
                is_stable = False
                if has_hands and velocity > 0.0:
                    if velocity < VELOCITY_THRESHOLD:
                        self.stability_counter += 1
                    else:
                        self.stability_counter = 0
                        is_stable = False
                    
                    if self.stability_counter >= STABILITY_FRAMES:
                        is_stable = True
                else:
                    self.stability_counter = 0
                    is_stable = False

                state["is_stable"] = is_stable

                # --- リアルタイム推論と文字スタック ---
                if is_stable and self.model is not None:
                    r_norm = normalize_hand_data(np.array(right_hand_data))
                    l_norm = normalize_hand_data(np.array(left_hand_data))
                    features = np.concatenate([r_norm, l_norm]).reshape(1, -1)

                    try:
                        probs = self.model.predict_proba(features)[0]
                        max_idx = np.argmax(probs)
                        pred_class = self.model.classes_[max_idx]
                        confidence = probs[max_idx]

                        state["pred_class"] = pred_class
                        state["confidence"] = float(confidence)

                        if confidence >= CONFIDENCE_THRESHOLD:
                            if pred_class != self.last_prediction or (current_time - self.last_input_time) > INPUT_COOLDOWN:
                                state["text_buffer"] += pred_class
                                self.last_prediction = pred_class
                                self.last_input_time = current_time
                                self.stability_counter = -5  # しばらく入力をロック
                    except Exception as e:
                        print(f"推論エラー: {e}")
                elif not has_hands:
                    state["pred_class"] = "--"
                    state["confidence"] = 0.0

                # --- 録画中のデータ保存 ---
                if state["is_recording"] and state["current_file_path"]:
                    try:
                        with open(state["current_file_path"], "a") as f:
                            # タイムスタンプ + 右手(63) + 左手(63)
                            row = [time.time()] + right_hand_data + left_hand_data
                            row_str = ",".join(map(str, row)) + "\n"
                            f.write(row_str)
                    except Exception as e:
                        print(f"録画書き込みエラー: {e}")

                # ストリーム用フレーム更新
                _, jpeg = cv2.imencode('.jpg', frame)
                self.latest_frame = jpeg.tobytes()
                time.sleep(0.03)  # 約30fpsに制限

camera = CameraManager()

@app.on_event("startup")
def startup_event():
    camera.start()

@app.on_event("shutdown")
def shutdown_event():
    camera.stop()

# --- WEBルート ---
@app.get("/", response_class=HTMLResponse)
def read_index():
    with open(TEMPLATE_PATH, "r", encoding="utf-8") as f:
        html_content = f.read()
    return HTMLResponse(content=html_content, status_code=200)

# --- カメラ配信エンドポイント ---
def gen(cam):
    while cam.running:
        if cam.latest_frame:
            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + cam.latest_frame + b'\r\n')
        time.sleep(0.03)

@app.get("/video_feed")
def video_feed():
    return Response(gen(camera), media_type="multipart/x-mixed-replace; boundary=frame")

# --- API エンドポイント ---

@app.get("/api/counts")
def get_dataset_counts():
    """
    data/raw 以下の各文字ごとの収録ファイル数をカウントして返します。
    """
    counts = {}
    # 全ての50音に対して0で初期化
    romaji_list = [
        'a', 'i', 'u', 'e', 'o', 'ka', 'ki', 'ku', 'ke', 'ko',
        'sa', 'si', 'su', 'se', 'so', 'ta', 'ti', 'tu', 'te', 'to',
        'na', 'ni', 'nu', 'ne', 'no', 'ha', 'hi', 'hu', 'he', 'ho',
        'ma', 'mi', 'mu', 'me', 'mo', 'ya', 'yu', 'yo', 'ra', 'ri',
        'ru', 're', 'ro', 'wa', 'wo', 'nn'
    ]
    for ro in romaji_list:
        counts[ro] = 0

    # 実際のファイル数を数える
    csv_files = glob.glob(os.path.join(RAW_DATA_DIR, "sign_language_*.csv"))
    for file_path in csv_files:
        file_name = os.path.basename(file_path)
        base_name = file_name.replace("sign_language_", "").replace(".csv", "")
        parts = base_name.split("_")
        
        # ラベルの特定
        if len(parts) > 1 and parts[-1].isdigit():
            label = "_".join(parts[:-1])
        else:
            label = base_name
            
        if label in counts:
            counts[label] += 1
            
    return counts

@app.post("/api/record/start/{label}")
def record_start(label: str):
    """
    録画を開始し、自動ナンバリングした新規CSVファイルを用意します。
    """
    state["is_recording"] = True
    state["active_label"] = label
    
    # 自動ナンバリング処理
    existing_files = glob.glob(os.path.join(RAW_DATA_DIR, f"sign_language_{label}_*.csv"))
    indices = []
    for f in existing_files:
        name = os.path.basename(f)
        try:
            idx = int(name.replace(f"sign_language_{label}_", "").replace(".csv", ""))
            indices.append(idx)
        except ValueError:
            pass
            
    next_idx = max(indices) + 1 if indices else 1
    file_name = f"sign_language_{label}_{next_idx}.csv"
    state["current_file_path"] = os.path.join(RAW_DATA_DIR, file_name)
    
    # ヘッダーを作成
    with open(state["current_file_path"], "w") as f:
        header = ["timestamp"]
        for i in range(21):
            header.extend([f"R_joint_{i}_x", f"R_joint_{i}_y", f"R_joint_{i}_z"])
        for i in range(21):
            header.extend([f"L_joint_{i}_x", f"L_joint_{i}_y", f"L_joint_{i}_z"])
        f.write(",".join(header) + "\n")
        
    print(f"録画開始: {file_name}")
    return {"status": "started", "file": file_name}

@app.post("/api/record/stop")
def record_stop():
    """
    録画を停止します。
    """
    print(f"録画停止: {os.path.basename(state['current_file_path']) if state['current_file_path'] else 'None'}")
    state["is_recording"] = False
    state["active_label"] = None
    state["current_file_path"] = None
    return {"status": "stopped"}

@app.get("/api/status")
def get_status():
    """
    現在のリアルタイム予測状態や速度、テキストバッファを返します。
    """
    return state

@app.post("/api/clear_text")
def clear_text():
    state["text_buffer"] = ""
    camera.last_prediction = None
    return {"status": "cleared"}

@app.post("/api/preprocess")
def run_preprocess():
    """
    データセットの一括前処理を行います。
    """
    try:
        preprocess_all_raw_data(raw_dir=RAW_DATA_DIR, output_csv=os.path.join(PROCESSED_DATA_DIR, "dataset.csv"))
        return {"status": "success", "message": "一括前処理（データの正規化と統合）が完了しました！"}
    except Exception as e:
        return {"status": "error", "message": f"前処理エラー: {e}"}

@app.post("/api/train")
def run_train():
    """
    モデルの学習を実行し、完了後に推論用モデルをリロードします。
    """
    try:
        train_model(
            dataset_path=os.path.join(PROCESSED_DATA_DIR, "dataset.csv"),
            model_output_dir=os.path.join(BASE_DIR, "src", "learning", "weights"),
            model_type="svm"
        )
        # 学習が終わったら推論モデルを最新の物にリロード
        camera.load_inference_model()
        return {"status": "success", "message": "学習が完了し、最新のモデルが推論用にロードされました！"}
    except Exception as e:
        return {"status": "error", "message": f"学習エラー: {e}"}

if __name__ == "__main__":
    import uvicorn
    # uvicorn dashboard:app --host 0.0.0.0 --port 8000 --reload
    uvicorn.run(app, host="0.0.0.0", port=8000)
